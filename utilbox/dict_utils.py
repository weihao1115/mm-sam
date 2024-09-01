from typing import Union, Dict, List


def merge_flatten_dict(current_dict, keys=[]):
    def merge_dicts(dict1, dict2):
        for _key, _value in dict2.items():
            if _key in dict1 and isinstance(dict1[_key], dict) and isinstance(_value, dict):
                merge_dicts(dict1[_key], _value)
            else:
                dict1[_key] = _value

    new_dict = {}
    for key, value in current_dict.items():
        if isinstance(value, dict):
            new_sub_dict = merge_flatten_dict(value, keys + [key])
            for new_key, new_value in new_sub_dict.items():
                if new_key not in new_dict.keys():
                    new_dict[new_key] = {}
                merge_dicts(new_dict[new_key], new_value)
        else:
            new_key = keys[1:] + [key]
            sub_dict = new_dict
            for k in new_key:
                sub_dict = sub_dict.setdefault(k, {})
            sub_dict[keys[0]] = value
    return new_dict


def average_dicts(dicts: Union[Dict, List[Dict]]):
    if isinstance(dicts, Dict):
        dicts = list(dicts.values())

    def merge_dicts(input_dicts):
        merged = {}
        for d in input_dicts:
            for key, value in d.items():
                if key not in merged:
                    merged[key] = []
                merged[key].append(value)
        return merged

    def average_value(values):
        if isinstance(values[0], dict):
            return average_dicts(values)
        else:
            return sum(values) / len(values)

    merged_dict = merge_dicts(dicts)
    avg_dict = {key: average_value(values) for key, values in merged_dict.items()}

    return avg_dict


def nested_dict_to_columns(d, parent_keys=None, result=None):
    """
    Transform a nested dictionary into a flattened dictionary without prior knowledge of nesting levels.
    The output dictionary will have 'title' keys corresponding to the levels of nesting and the values
    will be aggregated into lists for each 'title' key.

    :param d: The original dictionary to transform.
    :param parent_keys: A list of parent keys that lead to the current value.
    :param result: The result dictionary being built up.
    :return: A dictionary with keys representing the nesting levels and values in lists.
    """
    if parent_keys is None:
        parent_keys = []
    if result is None:
        result = {}

    for key, value in d.items():
        new_keys = parent_keys + [key]

        if isinstance(value, dict):
            # Continue to recurse into the next level of the dictionary.
            nested_dict_to_columns(value, new_keys, result)
        else:
            # We've reached a leaf node, add the value and its path to the result.
            for i, key_part in enumerate(new_keys[:-1]):
                title_key = f'title{i+1}'
                if title_key not in result:
                    result[title_key] = []
                # Add the key part to the 'title' key list if it's not already there for this index.
                if len(result[title_key]) < len(result.get(new_keys[-1], [])) + 1:
                    result[title_key].append(key_part)

            # Add the value to the appropriate key in the result.
            if new_keys[-1] not in result:
                result[new_keys[-1]] = []
            result[new_keys[-1]].append(value)

    return result
