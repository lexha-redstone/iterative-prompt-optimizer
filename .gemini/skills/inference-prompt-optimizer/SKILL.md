---
name: inference-prompt-optimizer
description: Refines LLM inference prompts iteratively based on a provided evaluation (judge) prompt. Use this to align production prompts with specific quality criteria through an automated Evaluate-Critic-Optimize loop.
---

# Inference Prompt Optimizer

## Overview
The Inference Prompt Optimizer is designed to take an initial prompt and iteratively improve its performance against a fixed set of evaluation criteria (the "Judge"). It uses numerical scoring and qualitative feedback to guide a critic in identifying weaknesses, which an optimizer then addresses by rewriting the prompt for the next iteration.

## Workflow

### 1. Prerequisites
- **A reliable Judge**: You should already have a judge prompt (e.g., `v7-judge.txt`) in `inference-prompt-writer/prompts/`.
- **Input Dataset**: `inference-prompt-writer/inputs/extracted_inputs.json` should contain the test cases.

See [DATA_FORMAT.md](references/DATA_FORMAT.md) for more details.

### 2. Initialization
Bootstraps the `v0` version of the inference prompt, critic, and optimizer.
```bash
python inference-prompt-writer/0_initiate.py
```

### 3. Running Iterations
Execute the main orchestrator to start the optimization loop.
```bash
python inference-prompt-writer/run_iterations.py --start_version v0 --num_iterations 10
```

## Core Guardrails
- **Prompt Alignment**: Whenever the Judge prompt structure changes (e.g., removing the `overall_score` field or adding new criteria), the **Critic** and **Optimizer** prompts must be updated to align with the new structure.
- **Data-Driven Updates**: The optimization process should always utilize the best-performing prompts and feedback identified in the `eval_logs` to ensure incremental improvement.
- **Scoring Logic**: Verify that the score calculation and logging logic in the scripts correctly reflect the requirements of the current Judge prompt.

## Best Practices
- **Fixed Judge**: Do not change the judge prompt during an inference optimization run unless you perform a full alignment of the Critic and Optimizer.
- **Diverse Inputs**: Ensure `extracted_inputs.json` covers a wide range of edge cases.
- **Convergence**: Review the critic's output in `generated/critic/` periodically to ensure the optimization is moving in the right direction.
