from typing import Union
import re

def are_numbers_close_enough(a: Union[float, int], b: Union[float, int]) -> bool:
    return a == b

def extract_final_number(response):
    # First try to find a number after "Final Answer:", handling commas and various dash types
    final_answer_match = re.search(r'Final Answer:\s*([–-]?[\d,]*\.?\d+)', response, re.IGNORECASE)
    if final_answer_match:
        # Replace em dash with regular minus sign and remove commas
        num_str = final_answer_match.group(1).replace('–', '-').replace(',', '')
        try:
            return float(num_str)
        except ValueError:
            pass

    # If that fails, look for the last number in the text
    numbers = re.findall(r'[–-]?[\d,]*\.?\d+', response)
    if numbers:
        try:
            return float(numbers[-1].replace('–', '-').replace(',', ''))
        except ValueError:
            pass
    return None