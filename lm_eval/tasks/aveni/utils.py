from typing import Union


def _str_to_bool(val: str) -> int:
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    Copied from distutils.util.strtobool as it is deprecated in Python 3.12
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def _currency_to_float(value: str) -> float:
    # Currently the implementation handles only $ and the following suffixes (ConvFinQA dataset has only such cases)
    suffix_to_value = {
        "million": 10e6,
        "thousand": 10e3,
    }
    assert value[0] == "$"
    value = value.replace("$", "").strip()
    try:
        # Currency without any suffix
        return float(value)
    except ValueError:
        matching_suffix = [x for x in suffix_to_value.keys() if x in value]
        if len(matching_suffix) != 1:
            raise ValueError(f"Cannot parse {value}, multiple suffixes.")
        matching_suffix = matching_suffix.pop()
        value = value.replace(matching_suffix, "").strip()
        return float(value) * suffix_to_value[matching_suffix]


def value_to_float(value: Union[str, float]) -> float:
    """Attempts to map an input value to float with a simple pre-processing.
    Additionally, maps percentages to floats.

    Returns:
        float/string: a float if the mapping was successful or the input value otherwise
    """
    if isinstance(value, float):
        return value
    elif isinstance(value, str):
        value = value.replace("\\\\n", "")
        value = value.replace("increased", "").replace("increase", "")
        value = value.replace("decreased", "").replace("decrease", "")
        value = value.strip()
        if value and value[-1] == "%":
            return float(value.replace("%", "")) / 100.
        elif value and value[0] == "$":
            return _currency_to_float(value)
        else:
            try:
                return float(value)
            except ValueError:
                return float(_str_to_bool(value))

    else:
        raise ValueError("Input must be str or float, but it's {}".format(type(value)))


def equal_value(predictions, references, tolerance=0):
    """ Checks if the prediction and the reference are equal numerically with a tolerance.
        The implemented mapping (see value_to_float) for predictions follows the cases
        that the ConvFinQA dataset has had in the string-formatted answers.

        Examples that are considered equal:
            1) 2005 and 2005
            2) 55.5% and 0.555
            3) 0.1234 and 0.123, if tolerance >= 0.0004

        It is possible to specify the tolerance value in the yaml file.

    Returns:
        int: either 1 or 0 indicating if the prediction and the reference match in value
    """
    # References should be floats.
    ref = float(references[0])

    try:
        pred = value_to_float(predictions[0])
        return abs(pred - ref) <= tolerance
    except ValueError:
        return 0
