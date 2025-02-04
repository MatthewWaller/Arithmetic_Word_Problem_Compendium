import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Introduction
cells.append(nbf.v4.new_markdown_cell("""# Arithmetic Word Problem Compendium - Dataset Exploration

This notebook demonstrates how to work with the Arithmetic Word Problem Compendium dataset, exploring its structure and analyzing the problems it contains.

## Setup

First, let's install the required dependencies:"""))

# Setup
cells.append(nbf.v4.new_code_cell("""!pip install pandas numpy matplotlib seaborn"""))

# Imports
cells.append(nbf.v4.new_markdown_cell("## Import Dependencies"))
cells.append(nbf.v4.new_code_cell("""import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")"""))

# Load Data
cells.append(nbf.v4.new_markdown_cell("""## Load and Prepare Data

Let's load both the training and evaluation datasets:"""))
cells.append(nbf.v4.new_code_cell('''def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Load both training and evaluation datasets
train_data = load_jsonl('/kaggle/input/arithmetic-word-problem-compendium/sample_train.jsonl')
eval_data = load_jsonl('/kaggle/input/arithmetic-word-problem-compendium/sample_eval.jsonl')

# Convert to pandas DataFrames
train_df = pd.DataFrame(train_data)
eval_df = pd.DataFrame(eval_data)

print(f"Training set size: {len(train_df)}")
print(f"Evaluation set size: {len(eval_df)}")'''))

# Dataset Overview
cells.append(nbf.v4.new_markdown_cell("""## Dataset Overview

Let's examine the structure and contents of our dataset:"""))
cells.append(nbf.v4.new_code_cell("""# Display basic information about the training dataset
print("Training Dataset Info:")
train_df.info()

# Display first few examples
print("\nFirst few examples:")
pd.set_option('display.max_colwidth', None)
display(train_df.head(2))"""))

# Domain Analysis
cells.append(nbf.v4.new_markdown_cell("""## Analyzing Problem Domains

Let's visualize the distribution of problems across different domains:"""))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 6))
domain_counts = train_df['metadata'].apply(lambda x: x['domain']).value_counts()
sns.barplot(x=domain_counts.values, y=domain_counts.index)
plt.title('Distribution of Problem Domains')
plt.xlabel('Number of Problems')
plt.tight_layout()
plt.show()

# Print exact counts
print("\nDomain Distribution:")
for domain, count in domain_counts.items():
    print(f"{domain}: {count} problems")"""))

# Mathematical Operations
cells.append(nbf.v4.new_markdown_cell("""## Analyzing Mathematical Operations

Let's examine the types of mathematical operations used in the problems:"""))
cells.append(nbf.v4.new_code_cell("""def get_operators(metadata):
    return metadata['operators']

# Collect all operators
all_operators = [op for meta in train_df['metadata'] for op in get_operators(meta)]
operator_counts = Counter(all_operators)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(operator_counts.values()), y=list(operator_counts.keys()))
plt.title('Distribution of Mathematical Operations')
plt.xlabel('Number of Occurrences')
plt.tight_layout()
plt.show()

# Print exact counts
print("\nOperation Distribution:")
for op, count in operator_counts.most_common():
    print(f"{op}: {count} occurrences")"""))

# Example Problems
cells.append(nbf.v4.new_markdown_cell("""## Example Problems

Let's look at some example problems from different domains:"""))
cells.append(nbf.v4.new_code_cell("""def display_problem(problem):
    print(f"Domain: {problem['metadata']['domain']}")
    print(f"Question: {problem['question']}")
    print(f"Operations: {', '.join(problem['metadata']['operators'])}")
    print(f"Solution: {problem['metadata']['solution']}")
    print("-" * 80)

# Display one example from each domain
domains = set(train_df['metadata'].apply(lambda x: x['domain']))
for domain in sorted(domains):
    example = train_df[train_df['metadata'].apply(lambda x: x['domain'] == domain)].iloc[0]
    display_problem(example)"""))

# Problem Complexity
cells.append(nbf.v4.new_markdown_cell("""## Analyzing Problem Complexity

Let's analyze the complexity of problems based on the number of operations required:"""))
cells.append(nbf.v4.new_code_cell("""operation_counts = train_df['metadata'].apply(lambda x: len(x['operators']))

plt.figure(figsize=(10, 6))
sns.histplot(operation_counts, bins=range(1, max(operation_counts) + 2))
plt.title('Distribution of Number of Operations per Problem')
plt.xlabel('Number of Operations')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nOperation Count Statistics:")
print(operation_counts.describe())"""))

# Decimal Analysis
cells.append(nbf.v4.new_markdown_cell("""## Analyzing Decimal Precision

Let's examine the distribution of decimal places in the problems:"""))
cells.append(nbf.v4.new_code_cell("""decimal_places = train_df['metadata'].apply(lambda x: x['decimals'])

plt.figure(figsize=(10, 6))
sns.histplot(decimal_places, bins=range(0, max(decimal_places) + 2))
plt.title('Distribution of Decimal Places in Solutions')
plt.xlabel('Number of Decimal Places')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nDecimal Places Statistics:")
print(decimal_places.describe())"""))

# Cross-Domain Analysis
cells.append(nbf.v4.new_markdown_cell("""## Cross-Domain Analysis

Let's analyze how problem complexity varies across different domains:"""))
cells.append(nbf.v4.new_code_cell("""# Create a DataFrame with domain and operation count
domain_complexity = pd.DataFrame({
    'domain': train_df['metadata'].apply(lambda x: x['domain']),
    'num_operations': train_df['metadata'].apply(lambda x: len(x['operators'])),
    'decimal_places': train_df['metadata'].apply(lambda x: x['decimals'])
})

plt.figure(figsize=(12, 6))
sns.boxplot(data=domain_complexity, x='domain', y='num_operations')
plt.title('Number of Operations by Domain')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print summary statistics by domain
print("\nComplexity by Domain:")
print(domain_complexity.groupby('domain')['num_operations'].describe())"""))

# Usage Example
cells.append(nbf.v4.new_markdown_cell("""## Working with the Dataset

Here's a complete example of how to load and process problems from the dataset:"""))
cells.append(nbf.v4.new_code_cell('''def process_problem(problem):
    """Example function to process a single problem."""
    return {
        'domain': problem['metadata']['domain'],
        'num_operations': len(problem['metadata']['operators']),
        'has_decimals': problem['metadata']['decimals'] > 0,
        'question_length': len(problem['question'].split()),
        'solution': problem['metadata']['solution']
    }

# Process all problems
processed_problems = [process_problem(problem) for problem in train_data[:5]]
processed_df = pd.DataFrame(processed_problems)

print("Example of processed problems:")
display(processed_df)'''))

# Set notebook metadata
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.8.0"
    }
}

# Write the notebook
nbf.write(nb, "example_usage.ipynb") 