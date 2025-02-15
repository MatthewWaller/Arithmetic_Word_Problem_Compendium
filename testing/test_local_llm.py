import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import platform
import csv
import argparse
from datetime import datetime
from test_utils import are_numbers_close_enough, extract_final_number
from tqdm.auto import tqdm

torch.random.manual_seed(0)  # For reproducibility

def get_device():
    """Determine the optimal device for Mac."""
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS acceleration")
        return "mps"
    elif torch.cuda.is_available():
        print("Using CUDA acceleration")
        return "cuda"
    print("Using CPU (no GPU acceleration available)")
    return "cpu"

def format_messages(question):
    """Format the question into a chat-style prompt."""
    return [
        {"role": "system", "content": "You are a mathematical problem solver. Solve problems step by step as instructed and always end with 'Final Answer: ' followed by the numerical result formatted appropriately (with commas for large numbers and proper decimal precision). After giving the final answer, stop generating.\n\nEND OF RESPONSE"},
        {"role": "user", "content": question + "\n\nSolve this step by step and end with 'Final Answer: [number]'. Stop after giving the final answer."}
    ]

def evaluate_batch_problems(model_type, model_info, problems, use_lora_generate=True, batch_size=8):
    """Evaluate a batch of problems using either HF pipeline or MLX model."""
    messages_batch = [format_messages(p['question']) for p in problems]
    
    pipe = model_info
    generation_args = {
        "max_new_tokens": 500,
        "batch_size": batch_size,  # Add batch_size to generation args
        "pad_token_id": pipe.tokenizer.pad_token_id,  # Ensure padding token is used
        "padding": True,  # Enable padding for batching
    }
    outputs = pipe(messages_batch, **generation_args)
    responses = [output[0]['generated_text'][-1]['content'] for output in outputs]
    
    results = []
    for problem, response in zip(problems, responses):
        predicted_answer = extract_final_number(response)
        
        if predicted_answer is None:
            results.append({
                'success': False,
                'predicted': None,
                'actual': problem['metadata']['solution'],
                'response': response,
                'problem_id': problem['id'],
                'question': problem['question']
            })
            continue
        
        actual_answer = problem['metadata']['solution']
        
        # Use the number comparison function
        is_correct = are_numbers_close_enough(predicted_answer, actual_answer)
        
        results.append({
            'success': is_correct,
            'predicted': predicted_answer,
            'actual': actual_answer,
            'response': response,
            'problem_id': problem['id'],
            'question': problem['question']
        })
    
    return results

def save_results_to_csv(results, successes, total, model_type):
    """Save testing results to a CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'testing/results/{model_type}_{timestamp}.csv'
    
    # Prepare the failed problems data
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
    
    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write summary
        writer.writerow(['Summary'])
        writer.writerow(['Model Type', model_type])
        writer.writerow(['Total Problems', total])
        writer.writerow(['Successful', successes])
        writer.writerow(['Failed', total - successes])
        writer.writerow(['Accuracy', f"{(successes/total*100):.2f}%"])
        writer.writerow([])  # Empty row for separation
        
        # Write failed problems
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

def setup_llama_model(model_name):
    """Set up the model using PyTorch with MPS support."""
    print("Loading model...")
    device = get_device()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up padding configuration for batching
    if tokenizer.pad_token is None:
        # First set the pad token string
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for decoder-only model
    tokenizer.padding_side = 'left'
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.float16,
        padding=True,  # Enable padding
        truncation=True,  # Enable truncation
    )
    
    return pipe

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run local LLM testing with customizable parameters')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for processing (default: 4)')
    parser.add_argument('--model-name', type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                      help='Name or path of the model to use (default: meta-llama/Llama-3.2-1B-Instruct)')
    args = parser.parse_args()

    # Print system info
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    model_type = args.model_name.split('/')[-1].lower()  # Extract model type from name
    use_lora_generate = False  # Not needed for HF pipeline
    batch_size = args.batch_size
    
    # Set up the Llama model
    print(f"\nSetting up model {args.model_name}...")
    model_info = setup_llama_model(args.model_name)
    
    # Load the sample problems
    print("Loading problems...")
    with open('sample.jsonl', 'r') as f:
        problems = [json.loads(line) for line in f]

    # Process problems in batches
    results = []
    total_batches = (len(problems) + batch_size - 1) // batch_size
    successes = 0
    
    # Create progress bar
    pbar = tqdm(range(0, len(problems), batch_size), total=total_batches, desc="Evaluating batches")
    
    for i in pbar:
        batch = problems[i:i + batch_size]
        batch_results = evaluate_batch_problems(
            model_type, 
            model_info, 
            batch, 
            use_lora_generate=use_lora_generate,
            batch_size=batch_size
        )
        results.extend(batch_results)
        
        # Update success count and progress bar description
        batch_successes = sum(1 for r in batch_results if r['success'])
        successes += batch_successes
        processed = min(i + batch_size, len(problems))
        accuracy = (successes / processed) * 100
        pbar.set_description(f"Accuracy: {accuracy:.2f}% [{successes}/{processed}]")
        
        # Print detailed results for this batch
        for j, result in enumerate(batch_results, 1):
            print(f"\nProblem {i + j}:")
            print(f"Success: {result['success']}")
            print(f"Predicted: {result['predicted']}")
            print(f"Actual: {result['actual']}")
            if not result['success']:
                print("Model response:")
                print(result['response'])
            print("-" * 80)  # Add separator between problems
    
    # Calculate final statistics
    total = len(results)
    print(f"\nOverall accuracy: {successes}/{total} ({successes/total*100:.2f}%)")
    
    # Save results to CSV
    save_results_to_csv(results, successes, total, model_type)

if __name__ == "__main__":
    main()

# Example commands:
# python testing/test_local_llm.py --batch-size 8 --model-name meta-llama/Llama-3.2-1B-Instruct
# python testing/test_local_llm.py --batch-size 4 --model-name meta-llama/Llama-3.2-3B-Instruct