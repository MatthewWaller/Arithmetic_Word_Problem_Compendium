import json
import os
from datetime import datetime
import csv
import re
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv
from test_utils import are_numbers_close_enough
import time

# Load environment variables
load_dotenv()

class RemoteModel(ABC):
    """Abstract base class for remote model implementations."""
    
    @abstractmethod
    def setup(self) -> None:
        """Set up any necessary clients or configurations."""
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the model."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model."""
        pass

class OpenAIModel(RemoteModel):
    """OpenAI API implementation using Batch API for efficient processing."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", batch_size: int = 20):
        self._model_name = model_name
        self._client = None
        self._batch_size = min(batch_size, 50000)  # Batch API limit is 50,000 requests
    
    def setup(self) -> None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        self._client = OpenAI(api_key=api_key)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Single request generation - falls back to standard API."""
        if not self._client:
            self.setup()
        
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 500),
            stop=kwargs.get('stop', ["END OF RESPONSE", "Problem:", "Question:"])
        )
        return response.choices[0].message.content

    def _create_batch_input_file(self, messages_batch: List[List[Dict[str, str]]], **kwargs) -> str:
        """Create a JSONL file for batch processing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = f'testing/batch_inputs/batch_input_{timestamp}.jsonl'
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        
        with open(input_file, 'w') as f:
            for i, messages in enumerate(messages_batch):
                request = {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self._model_name,
                        "messages": messages,
                        "temperature": kwargs.get('temperature', 0.0),
                        "max_tokens": kwargs.get('max_tokens', 500),
                        "stop": kwargs.get('stop', ["END OF RESPONSE", "Problem:", "Question:"])
                    }
                }
                f.write(json.dumps(request) + '\n')
        
        return input_file

    def _wait_for_batch_completion(self, batch_id: str, timeout: int = 3600) -> Optional[str]:
        """Wait for batch completion and return output file ID."""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            batch = self._client.batches.retrieve(batch_id)
            print(f"\nBatch status: {batch.status}")
            print(f"Request counts: {batch.request_counts}")
            if batch.status == "completed":
                return batch.output_file_id
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch failed with status: {batch.status}")
            time.sleep(10)  # Check every 10 seconds
        return None

    def generate_batch(self, messages_batch: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """Generate responses using the Batch API."""
        if not self._client:
            self.setup()
        
        # Create and upload input file
        input_file_path = self._create_batch_input_file(messages_batch, **kwargs)
        print(f"\nCreated batch input file: {input_file_path}")
        
        with open(input_file_path, 'rb') as f:
            file = self._client.files.create(file=f, purpose="batch")
        print(f"Uploaded file with ID: {file.id}")
        
        # Create batch
        batch = self._client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"\nCreated batch with ID: {batch.id}")
        print(f"To check status: curl https://api.openai.com/v1/batches/{batch.id} -H 'Authorization: Bearer $OPENAI_API_KEY'")
        
        # Wait for completion
        print("\nWaiting for batch completion...")
        output_file_id = self._wait_for_batch_completion(batch.id)
        if not output_file_id:
            raise TimeoutError("Batch processing timed out")
        
        # Get results
        output = self._client.files.content(output_file_id)
        responses = []
        
        # Parse responses and maintain order
        response_dict = {}
        for line in output.text.strip().split('\n'):
            result = json.loads(line)
            custom_id = result['custom_id']
            if result.get('error'):
                response_text = f"Error: {result['error']['message']}"
            else:
                response_text = result['response']['body']['choices'][0]['message']['content']
            response_dict[custom_id] = response_text
        
        # Maintain original order
        for i in range(len(messages_batch)):
            responses.append(response_dict[f"request-{i}"])
        
        return responses
    
    @property
    def name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

def get_decimal_precision(numbers: List[float]) -> int:
    """
    Determine the required decimal precision based on the numbers in the problem.
    Takes the numbers array from the metadata and returns the maximum number of decimal places found.
    """
    max_decimals = 0
    for num in numbers:
        str_num = str(num)
        if '.' in str_num:
            decimals = len(str_num.split('.')[1].rstrip('0'))
            max_decimals = max(max_decimals, decimals)
    return max_decimals

def format_number(num: float, precision: Optional[int] = None) -> str:
    """
    Format a number with appropriate commas and decimal precision.
    """
    if isinstance(num, str):
        num = float(num)
    
    if precision is not None:
        num = round(num, precision)
    
    if isinstance(num, int) or (isinstance(num, float) and num.is_integer()):
        return "{:,}".format(int(num))
    else:
        if precision is not None:
            return "{:.{prec}f}".format(num, prec=precision)
        return str(num)

def format_messages(question: str) -> List[Dict[str, str]]:
    """Format the question into a chat-style prompt."""
    return [
        {
            "role": "system",
            "content": "You are a mathematical problem solver. Solve problems step by step and always end with 'Final Answer: ' followed by the numerical result formatted appropriately (with commas for large numbers and proper decimal precision). After giving the final answer, stop generating."
        },
        {
            "role": "user",
            "content": question + "\n\nSolve this step by step and end with 'Final Answer: [number]'. Stop after giving the final answer."
        }
    ]

def clean_response(text: str) -> str:
    """Clean the model's response."""
    text = re.sub(r'([!?.])(\1{2,})', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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

def evaluate_problem(model: RemoteModel, problem: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single problem using the specified remote model."""
    messages = format_messages(problem['question'])
    
    try:
        response_text = model.generate(messages)
        response_text = clean_response(response_text)
        
        predicted_answer = extract_final_number(response_text)
        
        if predicted_answer is None:
            return {
                'success': False,
                'predicted': None,
                'actual': problem['metadata']['solution'],
                'response': response_text,
                'problem_id': problem['id'],
                'question': problem['question']
            }
        
        actual_answer = problem['metadata']['solution']
        precision = get_decimal_precision(problem['metadata']['numbers'])
        
        is_correct = are_numbers_close_enough(predicted_answer, actual_answer)
        
        return {
            'success': is_correct,
            'predicted': format_number(predicted_answer, precision),
            'actual': format_number(actual_answer, precision),
            'response': response_text,
            'problem_id': problem['id'],
            'question': problem['question']
        }
        
    except Exception as e:
        print(f"Error evaluating problem {problem['id']}: {str(e)}")
        return {
            'success': False,
            'predicted': None,
            'actual': problem['metadata']['solution'],
            'response': f"Error: {str(e)}",
            'problem_id': problem['id'],
            'question': problem['question']
        }

def analyze_mistake(model: RemoteModel, problem_info: Dict[str, Any]) -> str:
    """Analyze why a mistake was made using the model itself."""
    analysis_prompt = [
        {
            "role": "system",
            "content": "You are analyzing mistakes made in mathematical problem solving. Evaluate whether the error was due to the question being confusing/ambiguous, or due to a simple mistake/misunderstanding. Be concise and direct."
        },
        {
            "role": "user",
            "content": f"""
Question: {problem_info['question']}
Model's Response: {problem_info['response']}
Predicted Answer: {problem_info['predicted']}
Correct Answer: {problem_info['actual']}

Was this error due to the question being confusing/ambiguous, or due to a simple mistake/misunderstanding? Explain briefly."""
        }
    ]
    
    try:
        analysis = model.generate(analysis_prompt, max_tokens=150)
        return analysis
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def save_results_to_csv(results: List[Dict[str, Any]], successes: int, total: int, model_name: str) -> str:
    """Save evaluation results to a CSV file and return the filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'testing/results/{model_name}_{timestamp}.csv'
    
    failed_problems = [
        {
            'problem_id': r['problem_id'],
            'question': r['question'],
            'predicted': r['predicted'],
            'actual': r['actual'],
            'response': r['response']
        }
        for r in results if not r['success']
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow(['Summary'])
        writer.writerow(['Model Type', model_name])
        writer.writerow(['Total Problems', total])
        writer.writerow(['Successful', successes])
        writer.writerow(['Failed', total - successes])
        writer.writerow(['Accuracy', f"{(successes/total*100):.2f}%"])
        writer.writerow([])
        
        if failed_problems:
            writer.writerow(['Failed Problems'])
            writer.writerow(['Problem ID', 'Question', 'Predicted Answer', 'Actual Answer', 'Model Response'])
            for prob in failed_problems:
                writer.writerow([
                    prob['problem_id'],
                    prob['question'],
                    prob['predicted'],
                    prob['actual'],
                    prob['response']
                ])
    
    print(f"\nResults saved to {filename}")
    return filename

def analyze_mistakes_from_csv(csv_filename: str, model: RemoteModel) -> None:
    """Analyze mistakes from an existing CSV file and create a new CSV with analysis."""
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"CSV file not found: {csv_filename}")
    
    # Read the original CSV
    problems = []
    summary_lines = []
    with open(csv_filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Store summary section
        for row in reader:
            if not row or row[0] == 'Failed Problems':
                summary_lines.append(row)  # Include the 'Failed Problems' row
                header_row = next(reader)  # Read and store the header row
                summary_lines.append(header_row)
                break
            summary_lines.append(row)
        
        # Read failed problems (we've already skipped the header)
        for row in reader:
            if len(row) >= 5:  # Ensure row has all required fields
                problems.append({
                    'problem_id': row[0],
                    'question': row[1],
                    'predicted': row[2],
                    'actual': row[3],
                    'response': row[4]
                })
    
    # Create new filename for analyzed results
    base, ext = os.path.splitext(csv_filename)
    analyzed_filename = f"{base}_analyzed{ext}"
    
    # Analyze each problem and write to new CSV
    with open(analyzed_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write original summary
        for row in summary_lines:
            writer.writerow(row)
        
        if problems:
            # Header was already written as part of summary_lines
            for prob in problems:
                print(f"\nAnalyzing problem {prob['problem_id']}...")
                analysis = analyze_mistake(model, prob)
                writer.writerow([
                    prob['problem_id'],
                    prob['question'],
                    prob['predicted'],
                    prob['actual'],
                    prob['response'],
                    analysis
                ])
    
    print(f"\nAnalysis results saved to {analyzed_filename}")

def evaluate_batch_problems(model: RemoteModel, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate a batch of problems using the specified remote model."""
    messages_batch = [format_messages(problem['question']) for problem in problems]
    
    try:
        responses = model.generate_batch(messages_batch)
        results = []
        
        for problem, response_text in zip(problems, responses):
            predicted_answer = extract_final_number(response_text)
            
            if predicted_answer is None:
                results.append({
                    'success': False,
                    'predicted': None,
                    'actual': problem['metadata']['solution'],
                    'response': response_text,
                    'problem_id': problem['id'],
                    'question': problem['question']
                })
                continue
            
            actual_answer = problem['metadata']['solution']
            
            is_correct = are_numbers_close_enough(predicted_answer, actual_answer)
            
            results.append({
                'success': is_correct,
                'predicted': predicted_answer,
                'actual': actual_answer,
                'response': response_text,
                'problem_id': problem['id'],
                'question': problem['question']
            })
            
    except Exception as e:
        # If batch processing fails, fall back to individual processing
        print(f"Batch processing failed: {str(e)}. Falling back to individual processing...")
        results = []
        for problem in problems:
            result = evaluate_problem(model, problem)
            results.append(result)
    
    return results

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run OpenAI model evaluation with customizable parameters')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo",
                      help='OpenAI model to use (default: gpt-3.5-turbo)')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Batch size for processing (default: 1000)')
    args = parser.parse_args()

    # Initialize the model with command line arguments
    model = OpenAIModel(args.model, batch_size=args.batch_size)
    
    try:
        # Set up the model (this will validate the API key)
        print(f"Setting up {model.name} with batch size {model.batch_size}...")
        model.setup()
        
        # Load just one problem for testing
        print("Loading test problem...")
        with open('sample.jsonl', 'r') as f:
            problems = [json.loads(line) for line in f]
        
        print("\nTest problem details:")
        print(f"Problem ID: {problems[0]['id']}")
        print(f"Question: {problems[0]['question']}")
        print(f"Expected solution: {problems[0]['metadata']['solution']}")
        
        # Process single problem
        print("\nProcessing test problem...")
        batch_results = evaluate_batch_problems(model, problems)
        
        # Print detailed results
        result = batch_results[0]
        print("\nResults:")
        print(f"Success: {result['success']}")
        print(f"Predicted: {result['predicted']}")
        print(f"Actual: {result['actual']}")
        print("\nFull response:")
        print(result['response'])
        
        # Save results to CSV
        successes = sum(1 for r in batch_results if r['success'])
        total = len(batch_results)
        print(f"\nOverall accuracy: {successes}/{total} ({successes/total*100:.2f}%)")
        results_file = save_results_to_csv(batch_results, successes, total, model.name)
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise  # Re-raise to see full traceback

if __name__ == "__main__":
    main()

# Example commands:
# python testing/test_openai.py --batch-size 1000 --model gpt-3.5-turbo
# python testing/test_openai.py --batch-size 1000 --model gpt-4-turbo