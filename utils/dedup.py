import json
import os


def drop_questions_already_processed_by_question_id(questions, answer_file):
    if os.path.exists(answer_file):
        with open(answer_file, "r") as f:
            answers = list(map(lambda x: json.loads(x), f.readlines()))
        answer_ids = list(map(lambda x: x["question_id"], answers))
        questions = [q for q in questions if q["question_id"] not in answer_ids]
    return questions


def drop_matches_already_processed(matches, output_file):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            outputs = list(map(lambda x: json.loads(x), f.readlines()))
        output_ids = list(
            map(
                lambda x: (
                    x["question_id"],
                    x.get("model"),
                    x.get("model_1"),
                    x.get("model_2"),
                    x["judge"][0],
                ),
                filter(lambda x: x["turn"] == 2, outputs),
            )
        )
        matches = [
            m
            for m in matches
            if (
                m.question["question_id"],
                m.model if hasattr(m, "model") else None,
                m.model_1 if hasattr(m, "model_1") else None,
                m.model_2 if hasattr(m, "model_2") else None,
                m.judge.model_name,
            )
            not in output_ids
        ]
    return matches
