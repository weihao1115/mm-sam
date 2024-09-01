from typing import Dict


def dict_to_log_message(results: Dict, tab_num: int = 0):
    message = ""
    for result_key, result_value in results.items():
        message += "\t" * tab_num + f"{result_key}:"
        if isinstance(result_value, Dict):
            message += "\n" + dict_to_log_message(result_value, tab_num + 1)
        else:
            message += f" {result_value:.4%}\n"
    return message
