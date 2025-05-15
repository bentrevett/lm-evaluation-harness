import json
import logging
import math_verify

from lm_eval.tasks.aveni.utils import (
    value_to_float,
    equal_value as equal_value_finance,
)

logger = logging.getLogger(__name__)


def preprocess_data(path="../../../../lm-eval-harness-data/convfinqa"):
    # ConvFinQA
    for split in ["dev_turn", "train_turn"]:
        with open(f"{path}/raw/{split}.json", "r") as f_in:
            data = json.load(f_in)

        with open(f"{path}/{split}.jsonl", "w") as f_out:
            for d in data:
                current_dialogue_step = d["annotation"]["turn_ind"]
                preprocessed = dict(
                    pre_text=d["pre_text"],
                    post_text=d["post_text"],
                    annotation=dict(
                        amt_table=d["annotation"]["amt_table"],
                        dialogue_break=d["annotation"]["cur_dial"][:-1],
                        exe_ans_list=[
                            str(a)
                            for a in d["annotation"]["exe_ans_list"][
                                :current_dialogue_step
                            ]
                        ],
                    ),
                    qa=dict(
                        question=d["annotation"]["cur_dial"][-1],
                        answer=value_to_float(
                            d["annotation"]["exe_ans_list"][current_dialogue_step]
                        ),
                    ),
                )

                # ConvFinQA
                if preprocessed["qa"]["answer"]:
                    f_out.write(json.dumps(preprocessed, ensure_ascii=False) + "\n")
                else:
                    logger.warning(f"Ignoring a sample due to empty answer.")

    # FinQA (direct)
    for split in ["dev", "train"]:

        with open(f"{path}/raw/{split}.json", "r") as f_in:
            data = json.load(f_in)

        with open(f"{path}/{split}_direct.jsonl", "w") as f_out:
            for d in data:
                # Iterate over possible question dictionary fields
                for key in ["qa", "qa_0", "qa_1"]:
                    if key not in d or not d[key]["answer"]:
                        continue

                    preprocessed = dict(
                        pre_text=d["pre_text"],
                        post_text=d["post_text"],
                        annotation=dict(
                            amt_table=d["annotation"]["amt_table"],
                            dialogue_break=[],  # Direct QA setup (FinQA)
                            exe_ans_list=[],  # Direct QA setup (FinQA)
                        ),
                        qa=dict(
                            question=d[key]["question"],
                            answer=value_to_float(d[key]["exe_ans"]),
                        ),
                    )

                    # ConvFinQA
                    if preprocessed["qa"]["answer"]:
                        f_out.write(json.dumps(preprocessed, ensure_ascii=False) + "\n")
                    else:
                        logger.warning(f"Ignoring a sample due to empty answer.")


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

def _process_text_line(s):
    if s.strip() == "." or s.strip() == "":
        return ""
    else:
        return (
            s.replace(" .", ".")
            .replace(" ,", ",")
            .replace(" ;", ";")
            .replace(" :", ":")
            .replace(" ?", "?")
            .replace(" %", "%")
            .replace("( ", "(")
            .replace(" )", ")")
            .replace("$ ", "$")
            .capitalize()
        )


def _get_texts(ls):
    processed_ls = []
    for line in ls:
        result = _process_text_line(line)
        if result:
            processed_ls.append(result)
    if processed_ls:
        return " ".join(processed_ls).strip()
    else:
        return ""


def _doc_to_text(
    doc: dict,
    include_question_prompt: bool,
    include_answer_prompt: bool,
) -> str:
    pre_texts = _get_texts(doc["pre_text"])
    post_texts = _get_texts(doc["post_text"])

    questions = [_process_text_line(q) for q in doc["annotation"]["dialogue_break"]]
    answers = doc["annotation"]["exe_ans_list"]
    assert len(questions) == len(answers)

    context = (
        pre_texts + "\n" + doc["annotation"]["amt_table"].strip() + "\n" + post_texts
    ).strip()

    for i, (q, a) in enumerate(zip(questions, answers)):
        context += f"\n\nQuestion: {q}\n\nAnswer: {a}"

    final_context = f"{context}\n\n"

    if include_question_prompt:
        final_context += "Question: "
    final_context += f"{_process_text_line(doc['qa']['question'])}"

    if include_answer_prompt:
        final_context += "\n\nAnswer:"

    return final_context


def doc_to_text(doc):
    return _doc_to_text(
        doc,
        include_question_prompt=True,
        include_answer_prompt=True,
    )


def doc_to_text_sft(doc):
    # TODO: Figure out a way to add previous question/answer as conversation.
    return _doc_to_text(
        doc,
        include_question_prompt=True,
        include_answer_prompt=False,
    )


def doc_to_target(doc):
    return doc["qa"]["answer"]


if __name__ == "__main__":
    preprocess_data(path="../../../../lm-eval-harness-data/convfinqa")
