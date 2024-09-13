import argparse
import glob

import pandas as pd


def make_leaderboard(args):
    if args.input_files is None:
        input_files = glob.glob(
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single/*.jsonl"
        )
    else:
        input_files = glob.glob(args.input_file)
    question_file = f"data/{args.bench_name}/question.jsonl"
    df_category = pd.read_json(question_file, lines=True)[["question_id", "category"]]
    categories = list(df_category["category"].unique())

    print(f"Input files: {input_files}")
    df_all = pd.concat(
        [pd.read_json(input_file, lines=True) for input_file in input_files]
    )
    df_all = df_all.merge(df_category)
    df_all = df_all.drop_duplicates(subset=["model", "question_id", "turn"])
    df = df_all[["model", "score", "turn", "category"]]
    df = df[df["score"] != -1]

    df_results = (
        df.groupby(["model", "category"])
        .mean()[["score"]]
        .reset_index()
        .pivot(index="model", columns="category", values="score")
    )
    df_results["overall"] = df_results.mean(axis=1)
    df_results = df_results[["overall", *categories]].sort_values(
        by="overall", ascending=False
    )
    df_results.columns.name = None

    for model_name, _group in df.groupby("model"):
        if len(_group) != len(df_category) * 2:
            print(
                f"Warning: {model_name} has {len(_group)} / {len(df_category) * 2} answers"
            )

    df_results.index = df_results.index.str.replace("_", "/", 1)
    df_results.to_csv(f"data/{args.bench_name}/leaderboard_{args.judge_model}.csv")
    print(f"Leaderboard: data/{args.bench_name}/leaderboard_{args.judge_model}.csv")
    print(df_results.to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="pfmt_bench_fin_ja")
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    make_leaderboard(args)
