import json
from typing import List, Dict, Union

import git

import pathlib

import numpy as np
from tabulate import tabulate

from lm_eval.utils import handle_arg_string


def doc_to_target(doc) -> List[Union[str, float]]:
    return [
        handle_arg_string(a) for a in doc["answer"]
    ]


def _norm(text: str) -> str:
    # Replace non-breaking space
    return " ".join(text.split())


def _form_html_table(table: List[List]) -> str:
    html = tabulate(table, tablefmt="html")
    # Tabulate adds newlines between tags and additional spaces to make rows even and improve visuals.
    # This code removes visual effects to reduce unnecessary length.
    html = html.replace("\n", " ")
    end_td_tag = "</td>"
    html = end_td_tag.join([x.strip() for x in html.split(end_td_tag)])
    return html


def _map_answers(answers: Union[str, int, float, List[str]]) -> List[str]:
    if not isinstance(answers, list):
        return [str(answers)]
    else:
        return answers


def _map_sample(sample: Dict, keep_only_hypothetical: bool = False) -> List[Dict]:
    table = _form_html_table(sample["table"]["table"])
    paragraph = " ".join(j["text"] for j in sample["paragraphs"])
    # Some examples define the position (using phrases "the table above" or "the table below").
    # By default, this pre-processing assumes that the table is after the paragraph, unless
    # a phrase "the table above" appears.
    context = f"{paragraph}\n{table}"
    if "the table above" in paragraph:
        context = f"{table}\n{paragraph}"

    questions = [
        j["question"] for j in sample["questions"]
    ]
    answers = [
        # Answer field can be either ist of answers (e.g. few extracted values).
        _map_answers(j["answer"]) for j in sample["questions"]
    ]
    answer_types = [
        j["answer_type"] for j in sample["questions"]
    ]
    return [
        {
            "question": q,
            "answer": a,
            "context": _norm(context)
        } for q, a, t in zip(questions, answers, answer_types) if not keep_only_hypothetical or t == "counterfactual"
    ]


def preprocess_data():
    git_repo = git.Repo(".", search_parent_directories=True)
    project_root = pathlib.Path(git_repo.working_dir)
    data_paths = [
        project_root / "lm-eval-harness-data" / "tatqa",  # TAT-QA
        project_root / "lm-eval-harness-data" / "tathqa",  # TAT-HQA
    ]

    # Split TAT-HQA
    # We use dev set as test set and create a dev set from the training one.
    with (data_paths[1] / "raw" / f"tatqa_and_hqa_dataset_train.json").open() as f:
        data = json.load(f)
        np.random.shuffle(data)
        train, valid = np.split(np.array(data), [int(.8 * len(data))])
        with (data_paths[1] / "raw" / "train.json").open("w") as train_f:
            json.dump(train.tolist(), train_f)
        with (data_paths[1] / "raw" / "dev.json").open("w") as dev_f:
            json.dump(valid.tolist(), dev_f)

    for data_path in data_paths:
        for split in ["train", "dev", "test"]:

            with (data_path / "raw" / f"{split}.json").open() as f:
                data = json.load(f)
                preprocessed = []
                for sample in data:
                    examples = _map_sample(sample, keep_only_hypothetical="tathqa" in str(data_path))
                    preprocessed.extend(examples)
            with (data_path / f"{split}.jsonl").open("w") as f:
                for sample in preprocessed:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    preprocess_data()
