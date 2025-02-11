# Arithmetic Word Problem Compendium (AWPC)

A comprehensive dataset of mathematically correct, multi-step arithmetic word problems designed for training and evaluating Large Language Models (LLMs) in mathematical reasoning tasks.

## Dataset Description

The dataset is a comprehensive collection of mathematical word problems spanning multiple domains with rich metadata and natural language variations. The problems contain 1 - 5 steps of mathematical operations that are specifically designed to encourage showing work and maintaining appropriate decimal precision throughout calculations. 

The available data is a sample of 1,000 problems, and commerical options are available to procure datasets of 100,000 - 10 million problems, or to license the templating system that created the data for magnitudes more data or customizations like the number of mathematical steps involved, and the addition of domains. Contact hello@cephalopod.studio for more information.

### Key Features

- **Mathematically Verified**: All problems and solutions are mathematically verified for correctness
- **Multi-step Reasoning**: Problems require multiple steps of logical reasoning to solve
- **Real-world Contexts**: Problems are grounded in practical, real-world scenarios
- **Diverse Difficulty Levels**: Range from elementary to advanced problem-solving
- **Structured Format**: Data provided in JSONL format with clear problem-solution pairs
- **Rich Domains**: Problems span multiple real-world domains including:
  - Agriculture (soil temperature changes, etc.)
  - Athletics (training hours, distances, etc.)
  - Construction (elevation changes, work hours, etc.)
  - Culinary (cooking temperature changes, calories per serving, etc.)
  - Education (GPA changes, etc.)
  - Entertainment (show ratings, stage lighting, etc.)
  - Finance (stock prices, account balances, etc.)

## Data Format

The dataset is provided in JSONL format with the following files:
- `sample_train.jsonl`: Training dataset with 1,000 problems
- `sample_eval.jsonl`: Evaluation dataset with 1,000 problems

Each problem entry contains:
```json
{
    "id": "problem_X",
    "question": "Text of the math problem",
    "metadata": {
        "discrete": boolean,
        "domain": string,
        "numbers": number[],
        "object_type": string,
        "solution": number,
        "operators": string[],
        "decimals": number
    }
}
```

### Sample Problems

1. Finance (Account Management):
```
Question: "Jack sets up 19 bank accounts for clients. First the total rises to be 2 times greater than before. Following that, another 4 accounts were added."

Solution:
"Here's how we can solve this problem:
19 accounts times 2 equals 38
Addition step: 38 + 4 = 42 accounts

Based on these steps, the answer is 42."
```

2. Agriculture (Grain Storage):
```
Question: "Kevin oversees 14,457 metric tons of grain storage in the new concrete silo. In the beginning, the facility grows to 3 times its current measure of grain. Following that, the overall supply of grain grows by 1,514 tons."

Solution:
"Following these steps will give us the answer:
Multiplication operation: 14,457 tons * 3 = 43,371
Add 1514 to 43,371 tons: 44,885

Thus, we arrive at the answer: 44,885."
```

## Model Performance

Current benchmarks show varying performance across different model sizes (accuracy means the model's final answer is correctly rounded to the correct number of decimal places):

* GPT 3.5 Turbo: 84.50% accuracy
* GPT 4 Turbo: 93.20% accuracy
* o3-mini: 95.80% accuracy

## Usage

### Loading the Data
```python
import json

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Load training data
train_data = load_jsonl('sample_train.jsonl')
```

## Intended Uses & Limitations

**Intended Uses:**
The data can be used in 4 areas:
1. Pretraining
2. Instruction tuning 
3. Finetuning
4. Benchmarking existing models

All those areas are in service of:
- Training mathematical reasoning systems
- Developing step-by-step problem-solving capabilities
- Testing arithmetic operations across diverse real-world contexts
- Evaluating precision in decimal calculations

**Limitations:**
- Currently English-only
- Limited to specific mathematical operations
- Template-based generation may introduce structural patterns
- Focused on arithmetic operations with up to 5 numbers

## License

This dataset is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this dataset in your research, please cite:
```
@dataset{awpc2025,
    title = {Arithmetic Word Problem Compendium},
    author = {Waller, Matthew},
    year = {2025},
    publisher = {Cephalopod Studio},
    url = {https://www.kaggle.com/datasets/cephalopodstudio/arithmetic-word-problem-compendium}
}
```

## Contributing

Contributions to improve the dataset are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or feedback about the dataset, please open an issue in this repository or contact hello@cephalopod.studio.
