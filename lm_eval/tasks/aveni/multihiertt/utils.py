import math_verify

from lm_eval.tasks.aveni.utils import value_to_float, equal_value as equal_value_finance

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
