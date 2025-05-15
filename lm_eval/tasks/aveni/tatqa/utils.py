from typing import List, Union
from lm_eval.utils import handle_arg_string


def doc_to_target(doc) -> List[Union[str, float]]:
    return [
        handle_arg_string(a) for a in doc["answer"]
    ]