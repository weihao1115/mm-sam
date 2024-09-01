import os
import ruamel.yaml

from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarstring import PlainScalarString
from typing import Dict, List
from utilbox.regex_utils import regex_angle_bracket, has_nested_structure


def reform_config_dict(input_config):
    """
        Recursively reformat a configuration dictionary to convert rumel.yaml data types into normal Python data types.

        Args:
            input_config (Dict or List or ScalarFloat or PlainScalarString):
                The input configuration dictionary.

        Returns:
            Dict or List or float or str: The reformatted configuration dictionary with normal Python data types.

    """
    if isinstance(input_config, Dict):
        return {str(key): reform_config_dict(value) for key, value in input_config.items()}
    elif isinstance(input_config, List):
        return [reform_config_dict(item) for item in input_config]
    else:
        # Convert rumel.yaml data type into normal Python data type
        if isinstance(input_config, ScalarFloat):
            input_config = float(input_config)
        elif isinstance(input_config, PlainScalarString):
            input_config = str(input_config)
        return input_config


def remove_representer(parent_node, reference, curr_key=None):
    """
        Recursively removes representers from a configuration dictionary.

        Args:
            parent_node (Dict or List):
                The parent node in the configuration dictionary.
            reference (Dict):
                The reference dictionary for resolving references.
            curr_key (str, optional):
                The current key within the parent node. Defaults to None.

        Returns:
            Dict or List or any: The updated configuration dictionary with removed representers.

    """
    def get_reference_value(_ref: str):
        # Get the reference key
        ref_key = _ref[1: -1]
        if '[' in ref_key and ']' in ref_key:
            assert ref_key.count('[') == ref_key.count(']') and not has_nested_structure(ref_key)

            left_indices = [idx for idx, char in enumerate(ref_key) if char == '[']
            right_indices = [idx for idx, char in enumerate(ref_key) if char == ']']
            ref_key_main = ref_key[:left_indices[0]]
            ref_key_indices = [int(ref_key[left_idx + 1: right_idx]) for left_idx, right_idx in
                               zip(left_indices, right_indices)]

            reference_value = reference[ref_key_main]
            for idx in ref_key_indices:
                assert not hasattr(ref_key_indices, 'tag'), \
                    f"{ref_key_indices} should be clarified before {ref_key}!"
                reference_value = reference_value[idx]
        else:
            assert not hasattr(reference[ref_key], 'tag'), \
                f"{reference[ref_key]} should be clarified before {ref_key}!"
            reference_value = reference[ref_key]

        return reference_value

    child_node = parent_node[curr_key] if curr_key is not None else parent_node

    if isinstance(child_node, Dict):
        return {key: remove_representer(child_node, reference, key) for key, value in child_node.items()}
    elif isinstance(child_node, List):
        return [remove_representer(child_node, reference, i) for i, item in enumerate(child_node)]
    else:
        # For the item with an !-prefixed representer
        if hasattr(child_node, 'tag'):
            # For '!ref' representer
            if child_node.tag.value == '!ref':
                ref_string = child_node.value
                # Standalone '!ref' without <> reference or the anchor points where the reference is already done
                if regex_angle_bracket.search(ref_string) is None:
                    parent_node[curr_key] = parent_node[curr_key].value
                # '!ref' with <> reference
                else:
                    # <> reference-only without any additional information, retain the data type of the reference
                    if regex_angle_bracket.fullmatch(ref_string):
                        # get the reference key
                        parent_node[curr_key] = get_reference_value(ref_string)
                    # reference with additional information, data type will be str
                    else:
                        ref_matches = regex_angle_bracket.findall(ref_string)

                        # loop each match surrounded by <>
                        for ref in ref_matches:
                            # get the reference key
                            parent_node[curr_key].value = parent_node[curr_key].value.replace(ref, str(get_reference_value(ref)))
                        # assign the value to input_config after looping
                        parent_node[curr_key] = parent_node[curr_key].value

            # turn the string '(x, x, ..., x)' into tuple by '!tuple'
            elif child_node.tag.value == '!tuple':
                parent_node[curr_key] = tuple([int(i) if i.isnumeric() else i
                                               for i in parent_node[curr_key].value[1: -1].replace(' ', '').split(',')])

            # turn the string '[x, x, ..., x]' into list by '!list'
            elif child_node.tag.value == '!list':
                parent_node[curr_key] = [int(i) if i.isnumeric() else i
                                         for i in parent_node[curr_key].value[1: -1].replace(' ', '').split(',')]

            # turn the numerical value into string by '!str'
            elif child_node.tag.value == '!str':
                parent_node[curr_key] = str(parent_node[curr_key].value)

        return parent_node[curr_key]


def load_yaml(yaml_file) -> Dict:
    """
        Loads a YAML file and returns its content as a dictionary.

        Args:
            yaml_file (str or IO): The path to the YAML file or a file-like object.

        Returns:
            Dict: The parsed YAML content as a dictionary.

        Raises:
            AssertionError: If the input YAML file does not exist.

    """
    # turn the input file path into file IO stream
    if isinstance(yaml_file, str):
        assert os.path.exists(yaml_file), f"Your input .yaml file {yaml_file} doesn't exist!"
        yaml_file = open(yaml_file, mode='r', encoding='utf-8')

    # parse the yaml file with !-prefixed representers
    ruamel_yaml = ruamel.yaml.YAML()
    yaml_config = reform_config_dict(ruamel_yaml.load(yaml_file))

    # modify the value of each item in yaml_config in-place if there is an !-prefixed representer
    yaml_config = remove_representer(yaml_config, yaml_config)

    return yaml_config
