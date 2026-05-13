# Iterative Prompt Optimizer

This repository provides tools for iteratively optimizing LLM prompts. It consists of two main components: one for creating robust evaluation (judge) prompts and another for refining inference prompts based on those evaluation criteria.

## Features

### 1. Contrastive Eval Prompt Writer
The **Contrastive Eval Prompt Writer** focuses on creating and improving **Evaluation Prompts** (Judges). It uses a contrastive approach by comparing "good" and "poor" samples to identify what makes a response successful.

- **Location**: `contrastive-eval-prompt-writer/`
- **Method**: It analyzes sets of `good` and `poor` samples to understand the nuances of the desired output. Through iterative loops, it refines the judge's criteria until it can accurately distinguish between high and low-quality results.
- **Workflow**:
  1. **Initiate**: Set up initial evaluation prompts.
  2. **Batch Evaluate**: Run the current judge against the sample dataset.
  3. **Critic**: Analyze where the judge succeeded or failed to distinguish between good/poor samples.
  4. **Optimize**: Rewrite the evaluation prompt based on the critic's feedback.
- **Example Usage**:
  ```bash
  cd contrastive-eval-prompt-writer
  python run_contrastive_writer.py --version v0 --num_iterations 5
  ```

### 2. Inference Prompt Writer
The **Inference Prompt Writer** is designed to refine **Inference Prompts** once a reliable evaluation prompt is available. It follows an iterative "Critic-Optimize" loop to improve the quality of generated outputs.

- **Location**: `inference-prompt-writer/`
- **Method**: Given an evaluation prompt, it generates outputs using the current inference prompt, evaluates them, and then uses a critic to suggest improvements for the next version of the prompt.
- **Workflow**:
  1. **Batch Inference**: Generate responses using the current inference prompt.
  2. **Batch Evaluate**: Use the judge to score the generated responses.
  3. **Critic**: Identify weaknesses in the responses based on the evaluation scores.
  4. **Optimize**: Update the inference prompt to address the identified weaknesses.
- **Example Usage**:
  ```bash
  cd inference-prompt-writer
  python run_iterations.py --start_version v0 --num_iterations 10
  ```

## Project Structure

```text
.
├── contrastive-eval-prompt-writer/   # Tools for optimizing evaluation prompts
│   ├── eval_prompts/                 # History of generated judge prompts
│   ├── samples/                      # Good/Poor datasets for contrastive analysis
│   └── run_contrastive_writer.py     # Main orchestrator
├── inference-prompt-writer/           # Tools for optimizing inference prompts
│   ├── prompts/                      # History of generated inference prompts
│   └── run_iterations.py             # Main orchestrator
└── requirements.txt                  # Python dependencies
```

## Setup

1.  **Clone the repository**.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure API Keys**: Ensure your environment is configured for the Google GenAI API (e.g., `GOOGLE_API_KEY`).

## Getting Started

- Use the **Contrastive Eval Prompt Writer** first to establish a high-quality judge prompt.
- Once you have a reliable judge, use the **Inference Prompt Writer** to iteratively improve your production prompt's performance against that judge.
