import logging
import pathlib
import re
import json
import math_verify

import git
from transformers import AutoTokenizer

from lm_eval.tasks.finance_leaderboard.finance_utils import value_to_float, equal_value as equal_value_finance

logger = logging.getLogger(__name__)


def _match_table(s):
    pattern = r"^## Table (\d+) ##$"
    match = re.search(pattern, s)
    if match:
        return int(match.group(1))
    else:
        return None


def _construct_context(instance):
    for i, line in enumerate(instance["paragraphs"]):
        table_idx = _match_table(line)
        if table_idx is not None:
            # replace the table placeholder in the paragraph with the actual table HTML
            instance["paragraphs"][i] = "\n" + instance["tables"][table_idx] + "\n"
        else:
            instance["paragraphs"][i] = line.strip()

    return " ".join(instance["paragraphs"]).strip()


def preprocess_data():
    git_repo = git.Repo(".", search_parent_directories=True)
    project_root = pathlib.Path(git_repo.working_dir)
    data_path = project_root / "lm-eval-harness-data" / "multihiertt"
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

    for split in ["train", "dev"]:
        with (data_path / "raw" / f"{split}.json").open() as f:
            data = json.load(f)
            preprocessed_easy = []
            preprocessed_hard = []
            for sample in data:
                try:
                    context = _construct_context(sample)
                    question = sample["qa"]["question"].strip()
                    answer = value_to_float(str(sample["qa"]["answer"]))

                    length = len(tokenizer.encode(f"{context}\n\nQuestion: {question}\n\nAnswer:"))

                    preprocessed_hard.append(dict(
                        answer=answer,
                        question=question,
                        context=context,
                    ))
                    if len(sample["tables"]) <= 3 and length < 4096:
                        preprocessed_easy.append(dict(
                            answer=answer,
                            question=question,
                            context=context,
                        ))

                except ValueError:
                    logger.warning(f"Non numerical answer: {sample['qa']['answer']}, skipping...")

        for mode, preprocessed in zip(["easy", "hard"], [preprocessed_easy, preprocessed_hard]):
            with (data_path / f"{split}_{mode}.jsonl").open("w") as f:
                for sample in preprocessed:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def equal_value(predictions, references, tolerance=0):
    """Wrapper for the YAML config."""
    return equal_value_finance(predictions, references, tolerance)

def equal_math_verify(predictions, references):
    assert len(predictions) == len(references) == 1
    prediction = math_verify.parse(predictions[0])
    reference = math_verify.parse(references[0])
    # reference goes first
    # .verify returns bool, we cast to int
    result = int(math_verify.verify(reference, prediction))
    return result

if __name__ == "__main__":
    preprocess_data()
