"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""

import argparse
import ast
import json
import re
import unittest
import unittest.mock
from concurrent.futures import ThreadPoolExecutor

import ftlangdetect
import numpy as np
from fastchat.llm_judge.common import TIE_DELTA
from fastchat.llm_judge.common import chat_completion_anthropic
from fastchat.llm_judge.common import check_data
from fastchat.llm_judge.common import get_model_list
from fastchat.llm_judge.common import load_judge_prompts
from fastchat.llm_judge.common import load_model_answers
from fastchat.llm_judge.common import load_questions
from fastchat.llm_judge.common import one_score_pattern
from fastchat.llm_judge.common import one_score_pattern_backup
from fastchat.llm_judge.common import play_a_match_pair
from fastchat.llm_judge.common import play_a_match_single
from fastchat.llm_judge.gen_judgment import make_judge_pairwise
from fastchat.llm_judge.gen_judgment import make_judge_single
from fastchat.llm_judge.gen_judgment import make_match
from fastchat.llm_judge.gen_judgment import make_match_all_pairs
from fastchat.llm_judge.gen_judgment import make_match_single
from fastchat.model.model_adapter import ANTHROPIC_MODEL_LIST
from fastchat.model.model_adapter import OPENAI_MODEL_LIST
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm

import utils.common
from utils.api import ANTHROPIC_MODEL_LIST_NEW
from utils.api import anthropic_chat_completion_new
from utils.api import chat_completion_openai
from utils.dedup import drop_matches_already_processed

NEED_REF_CATS = ["math"]


def reorg_judge_file(judge_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(judge_file, "r") as fin:
        for l in fin:
            x = json.loads(l)
            qid = x["question_id"] * 10 + x["turn"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(judge_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def check_data(questions, model_answers, ref_answers, models, judges):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q["category"] not in NEED_REF_CATS:
                continue
            assert (
                q["question_id"] in ref_answers[jg.model_name]
            ), f"Missing reference answer to Question {q['question_id']} for judge {jg.model_name}"


def run_judge_single(question, answer, judge, ref_answer, multi_turn=False):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    has_answer = False
    correct_language = True

    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_1=answer["choices"][0]["turns"][0],
            answer_2=answer["choices"][0]["turns"][1],
            **kwargs,
        )
        if answer["choices"][0]["turns"][1].strip() != "":
            has_answer = True
            if question["category"] != "translation":
                if (
                    ftlangdetect.detect(
                        answer["choices"][0]["turns"][1].strip()[:20].split("\n")[0],
                        low_memory=False,
                    )["lang"]
                    != "ja"
                ):
                    correct_language = False
            else:
                if (
                    ftlangdetect.detect(
                        answer["choices"][0]["turns"][1].strip().replace("\n", " "),
                        low_memory=False,
                    )["lang"]
                    != question["answer_lang"][1]
                ):
                    correct_language = False
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer=answer["choices"][0]["turns"][0],
            **kwargs,
        )
        if answer["choices"][0]["turns"][0].strip() != "":
            has_answer = True
            if question["category"] != "translation":
                if (
                    ftlangdetect.detect(
                        answer["choices"][0]["turns"][0].strip()[:20].split("\n")[0],
                        low_memory=False,
                    )["lang"]
                    != "ja"
                ):
                    correct_language = False
            else:
                if (
                    ftlangdetect.detect(
                        answer["choices"][0]["turns"][0].strip().replace("\n", " "),
                        low_memory=False,
                    )["lang"]
                    != question["answer_lang"][0]
                ):
                    correct_language = False

    rating = -1

    system_prompt = judge.prompt_template["system_prompt"]
    conv = get_conversation_template(model)
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if not has_answer:
        judgment = "No answer"
        rating = 0
    elif not correct_language:
        judgment = "Incorrect language"
        rating = 0
    else:
        for i in range(5):
            temperature = i * 0.001
            if model in OPENAI_MODEL_LIST:
                judgment = chat_completion_openai(
                    model, conv, temperature=temperature, max_tokens=2048
                )
            elif model in ANTHROPIC_MODEL_LIST_NEW:
                judgment = anthropic_chat_completion_new(
                    model, conv, temperature=temperature, max_tokens=1024
                )
            elif model in ANTHROPIC_MODEL_LIST:
                judgment = chat_completion_anthropic(
                    model, conv, temperature=temperature, max_tokens=1024
                )
            else:
                try:
                    judgment = chat_completion_openai(
                        model, conv, temperature=temperature, max_tokens=2048
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid judge model name or unknown error: {model}, {e}"
                    )

            if judge.prompt_template["output_format"] == "[[rating]]":
                match = re.search(one_score_pattern, judgment)
                if not match:
                    match = re.search(one_score_pattern_backup, judgment)

                if match:
                    rating = ast.literal_eval(match.groups()[0])
                    break
                else:
                    rating = -1
                    print("retry judge because score is missing")
            else:
                raise ValueError(
                    f"invalid output format: {judge.prompt_template['output_format']}"
                )

    return rating, user_prompt, judgment


def run_judge_pair(question, answer_a, answer_b, judge, ref_answer, multi_turn=False):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    has_answer_a = False
    has_answer_b = False

    if multi_turn:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_a_1=answer_a["choices"][0]["turns"][0],
            answer_b_1=answer_b["choices"][0]["turns"][0],
            answer_a_2=answer_a["choices"][0]["turns"][1],
            answer_b_2=answer_b["choices"][0]["turns"][1],
            **kwargs,
        )
        if answer_a["choices"][0]["turns"][1].strip() != "":
            has_answer_a = True
        if answer_b["choices"][0]["turns"][1].strip() != "":
            has_answer_b = True
    else:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer_a=answer_a["choices"][0]["turns"][0],
            answer_b=answer_b["choices"][0]["turns"][0],
            **kwargs,
        )
        if answer_a["choices"][0]["turns"][0].strip() != "":
            has_answer_a = True
        if answer_b["choices"][0]["turns"][0].strip() != "":
            has_answer_b = True

    winner = "error"

    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if not has_answer_a or not has_answer_b:
        if not has_answer_a and not has_answer_b:
            winner = "tie"
            judgment = "No answer from both assistants"
        elif has_answer_a:
            winner = "A"
            judgment = "No answer from assistant B"
        elif has_answer_b:
            winner = "B"
            judgment = "No answer from assistant A"
        else:
            raise AssertionError
    else:
        for i in range(5):
            temperature = i * 0.001
            if model in OPENAI_MODEL_LIST:
                conv.set_system_message(system_prompt)
                judgment = chat_completion_openai(
                    model, conv, temperature=temperature, max_tokens=2048
                )
            elif model in ANTHROPIC_MODEL_LIST_NEW:
                if system_prompt != "あなたは優秀なアシスタントです。":
                    user_prompt = "[指示]\n" + system_prompt + "\n\n" + user_prompt
                    conv.messages[0][1] = user_prompt
                judgment = anthropic_chat_completion_new(
                    model, conv, temperature=temperature, max_tokens=1024
                )
            elif model in ANTHROPIC_MODEL_LIST:
                if system_prompt != "あなたは優秀なアシスタントです。":
                    user_prompt = "[指示]\n" + system_prompt + "\n\n" + user_prompt
                    conv.messages[0][1] = user_prompt
                judgment = chat_completion_anthropic(
                    model, conv, temperature=temperature, max_tokens=1024
                )
            else:
                try:
                    conv.set_system_message(system_prompt)
                    judgment = chat_completion_openai(
                        model, conv, temperature=temperature, max_tokens=2048
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid judge model name or unknown error: {model}, {e}"
                    )

            if judge.prompt_template["output_format"] == "[[A]]":
                if "[[A]]" in judgment:
                    winner = "A"
                    break
                elif "[[B]]" in judgment:
                    winner = "B"
                    break
                elif "[[C]]" in judgment:
                    winner = "tie"
                    break
                else:
                    winner = "error"
                    print("retry judge because winner is missing")
            elif judge.prompt_template["output_format"] == "[[rating_a,rating_b]]":
                match = re.search(two_score_pattern, judgment)
                if not match:
                    match = re.search(two_score_pattern_backup, judgment)
                if match:
                    scores = [ast.literal_eval(s.strip()) for s in match.groups()]
                    if abs(scores[0] - scores[1]) <= TIE_DELTA:
                        winner = "tie"
                        break
                    elif scores[0] > scores[1]:
                        winner = "A"
                        break
                    else:
                        winner = "B"
                        break
                else:
                    winner = "error"
                    print("retry judge because winner is missing")
            else:
                raise ValueError(
                    f"invalid output format: {judge.prompt_template['output_format']}"
                )

    return winner, user_prompt, judgment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="pfmt_bench_fin_ja",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--baseline-model", type=str, default="gpt-35-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    args = parser.parse_args()

    if args.model_list:
        args.model_list = list(map(lambda x: x.replace("/", "_"), args.model_list))

    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)
    ref_answers[args.judge_model] = ref_answers["gpt-4o+human"]

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
        models = list(map(lambda x: x.replace("/", "_"), models))
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    def judge(_model, _output_file):
        check_data(questions, model_answers, ref_answers, _model, judges)

        question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
        question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

        # Make matches
        matches = []
        matches += make_match_func(
            question_default, _model, model_answers, judges["default"], baseline_model
        )
        matches += make_match_func(
            question_math,
            _model,
            model_answers,
            judges["math"],
            baseline_model,
            ref_answers,
        )
        matches += make_match_func(
            question_default,
            _model,
            model_answers,
            judges["default-mt"],
            baseline_model,
            multi_turn=True,
        )
        matches += make_match_func(
            question_math,
            _model,
            model_answers,
            judges["math-mt"],
            baseline_model,
            ref_answers,
            multi_turn=True,
        )

        matches = drop_matches_already_processed(
            matches=matches, output_file=_output_file
        )

        match_stat = {}
        match_stat["bench_name"] = args.bench_name
        match_stat["mode"] = args.mode
        match_stat["judge"] = args.judge_model
        match_stat["baseline"] = baseline_model
        match_stat["model_list"] = _model
        match_stat["total_num_questions"] = len(questions)
        match_stat["total_num_matches"] = len(matches)
        match_stat["output_path"] = _output_file

        with unittest.mock.patch(
            "fastchat.llm_judge.common.run_judge_single", run_judge_single
        ):
            with unittest.mock.patch(
                "fastchat.llm_judge.common.run_judge_pair", run_judge_pair
            ):
                # Play matches
                if args.parallel == 1:
                    for match in tqdm(matches):
                        play_a_match_func(match, output_file=_output_file)
                else:

                    def play_a_match_wrapper(match):
                        play_a_match_func(match, output_file=_output_file)

                    np.random.seed(0)
                    np.random.shuffle(matches)

                    with ThreadPoolExecutor(args.parallel) as executor:
                        for match in tqdm(
                            executor.map(play_a_match_wrapper, matches),
                            total=len(matches),
                        ):
                            pass
        reorg_judge_file(_output_file)

    if args.mode == "single":
        for model in models:
            print(model)
            output_file = f"data/{args.bench_name}/model_judgment/{args.judge_model}_single/{model}.jsonl"
            judge(_model=[model], _output_file=output_file)
    else:
        judge(_model=models, _output_file=output_file)
