import argparse
import json
import os
import random
import time
import unittest
import unittest.mock

import fastchat.model.model_adapter
import shortuuid
import torch
from fastchat.llm_judge.common import load_questions
from fastchat.llm_judge.common import temperature_config
from fastchat.llm_judge.gen_model_answer import reorg_answer_file
from fastchat.llm_judge.gen_model_answer import run_eval
from fastchat.model import get_conversation_template
from fastchat.utils import str_to_torch_dtype
from tqdm import tqdm
from transformers import StoppingCriteria

import utils.common
from utils.dedup import drop_questions_already_processed_by_question_id


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stops=[]):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        for stop in self.stops:
            last_token = input_ids[0][-len(stop) :]
            if self.tokenizer.decode(last_token).endswith(stop.strip()):
                return True
        return False


def new_load_model(*args, **kwargs):
    model, tokenizer = fastchat.model.model_adapter.load_model(*args, **kwargs)
    model._generate = model.generate

    def new_generate(*args, **kwargs):
        kwargs["pad_token_id"] = tokenizer.pad_token_id
        kwargs["bos_token_id"] = tokenizer.bos_token_id
        kwargs["eos_token_id"] = tokenizer.eos_token_id
        return model._generate(*args, **kwargs)

    model.generate = new_generate
    return model, tokenizer


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    disable_strict_injection_check=False,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                disable_strict_injection_check=disable_strict_injection_check,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    disable_strict_injection_check=False,
):
    model, tokenizer = new_load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    questions = drop_questions_already_processed_by_question_id(questions, answer_file)

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                print(prompt)
                input_ids = tokenizer([prompt]).input_ids
                stop_list = [conv.sep + conv.roles[0], conv.sep + conv.roles[1]]
                stopping_criteria = StoppingCriteriaSub(
                    stops=stop_list,
                    tokenizer=tokenizer,
                )
                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                try:
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                        repetition_penalty=1.1,
                        stopping_criteria=[stopping_criteria],
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    for stop_str_original in stop_list:
                        for stop_str in [stop_str_original, stop_str_original.strip()]:
                            if output[-len(stop_str) :] == stop_str:
                                if disable_strict_injection_check:
                                    output = output[: -len(stop_str)]
                                else:
                                    output = ""

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print(e)
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR: " + str(e)

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=False, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="pfmt_bench_fin_ja",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=4096,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--disable-strict-injection-check",
        action="store_true",
        help="Disable strict injection check. If it is not set, the model output is ignored if it generate the next convesation inclusing conv template such as ### assistant:.",
    )

    args = parser.parse_args()
    if args.model_id is None:
        args.model_id = args.model_path.replace("/", "_")

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    with unittest.mock.patch("fastchat.model.load_model", new_load_model):
        run_eval(
            model_path=args.model_path,
            model_id=args.model_id,
            question_file=question_file,
            question_begin=args.question_begin,
            question_end=args.question_end,
            answer_file=answer_file,
            max_new_token=args.max_new_token,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            max_gpu_memory=args.max_gpu_memory,
            dtype=str_to_torch_dtype(args.dtype),
            revision=args.revision,
            disable_strict_injection_check=args.disable_strict_injection_check,
        )

    reorg_answer_file(answer_file)
