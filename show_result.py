import argparse
import glob

import pandas as pd
from fastchat.llm_judge.show_result import display_result_pairwise


def display_result_single(args):
    if args.input_files is None:
        input_files = glob.glob(
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single/*.jsonl"
        )
    else:
        input_files = glob.glob(args.input_file)

    print(f"Input files: {input_files}")
    df_all = pd.concat(
        [pd.read_json(input_file, lines=True) for input_file in input_files]
    )
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if len(df[df["turn"] == 2]) > 0:
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="pfmt_bench_fin_ja")
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
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
    args = parser.parse_args()

    if args.model_list:
        args.model_list = list(map(lambda x: x.replace("/", "_"), args.model_list))

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)
