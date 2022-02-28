from typing import Dict
import numpy as np
import syne_tune.config_space as cs


def encode_config(config_space: Dict, config: Dict, categorical_maps: Dict) -> np.array:
    """
    Encode a configuration into a vector that can be decoded back with `_decode_config`.
    :param config_space:
    :param config: configuration to be encoded
    :param categorical_maps: dictionary from categorical Hyperparameter name to a dictionary mapping categories to
    an integer index. For instance {"cell_type": {"conv3x3": 0, "skip": 1}}
    :return: encoded vector.
    """

    def numerize(value, domain, categorical_map):
        if isinstance(domain, cs.Categorical):
            res = np.zeros(len(domain))
            res[categorical_map[value]] = 1
            return res
        else:
            if hasattr(domain, "lower") and hasattr(domain, "upper"):
                return [(value - domain.lower) / (domain.upper - domain.lower)]
            else:
                return [value]

    return np.hstack([
        numerize(value=config[k], domain=v, categorical_map=categorical_maps.get(k, {}))
        for k, v in config_space.items()
        if isinstance(v, cs.Domain)
    ])


def decode_config(config_space: Dict, encoded_vector: np.array, inv_categorical_maps: Dict) -> Dict:
    """
    Return a config dictionary given an encoded vector.
    :param config_space:
    :param encoded_vector:
    :param inv_categorical_maps: dictionary from each categorical Hyperparameter name to a dictionary maping
    category index to category value. For instance {"cell_type": {0: "conv3x3", 1: "skip"}}
    :return:
    """

    def inv_numerize(values, domain, categorical_map):
        if not isinstance(domain, cs.Domain):
            # constant value
            return domain
        else:
            if isinstance(domain, cs.Categorical):
                values = 1.0 * (values == values.max())
                index = max(np.arange(len(domain)) * values)
                return categorical_map[index]
            else:
                if hasattr(domain, "lower") and hasattr(domain, "upper"):
                    return values[0] * (domain.upper - domain.lower) + domain.lower
                else:
                    return values[0]

    cur_pos = 0
    res = {}
    for k, domain in config_space.items():
        if hasattr(domain, "sample"):
            length = len(domain) if isinstance(domain, cs.Categorical) else 1
            res[k] = domain.cast(
                inv_numerize(
                    values=encoded_vector[cur_pos:cur_pos + length],
                    domain=domain,
                    categorical_map=inv_categorical_maps.get(k, {})
                )
            )
            cur_pos += length
        else:
            res[k] = domain
    return res
