import argparse
import glob

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def make_leaderboard(args):
    if args.model_list is None:
        input_files = glob.glob(
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single/*.jsonl"
        )
    else:
        input_files = [
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single/{ x.replace('/', '_')}.jsonl"
            for x in args.model_list
        ]
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

    df_results.index = (
        df_results.index.str.replace("_", "/", 1)
        + " ("
        + df_results["overall"].map(lambda x: f"{x:.2f}")
        + ")"
    )
    fig = px.line_polar(
        df_results.stack()
        .reset_index()
        .rename(columns={0: "score", "level_0": "model", "level_1": "category"})
        .query("category != 'overall'"),
        r="score",
        theta="category",
        line_close=True,
        category_orders={"category": categories},
        color="model",
        markers=True,
        range_r=[0, 10],
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.write_html(f"data/{args.bench_name}/task_rader_{args.judge_model}.html")
    fig.write_image(f"data/{args.bench_name}/task_rader_{args.judge_model}.png")
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="pfmt_bench_fin_ja")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    args = parser.parse_args()

    make_leaderboard(args)
