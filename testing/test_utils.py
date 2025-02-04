from typing import Union
import re

def are_numbers_close_enough(a: Union[float, int], b: Union[float, int]) -> bool:
    return a == b

def extract_final_number(response):
    """Extract the final number from the model's response."""
    # First try to find a number after "Final Answer:", handling commas
    final_answer_match = re.search(r'Final Answer:\s*(-?[\d,]*\.?\d+)', response, re.IGNORECASE)
    if final_answer_match:
        # Remove commas before converting to float
        num_str = final_answer_match.group(1).replace(',', '')
        return float(num_str)
    
    # If that fails, look for the last number in the text, handling commas
    numbers = re.findall(r'-?[\d,]*\.?\d+', response)
    if numbers:
        # Remove commas from the last number before converting
        return float(numbers[-1].replace(',', ''))
    return None