---
name: contrastive-judge-optimizer
description: Automates the iterative optimization of LLM evaluation prompts (judges) by analyzing contrastive "good" and "poor" sample datasets. Use this when you need to create or refine a judge prompt that can accurately distinguish high-quality responses from low-quality ones.
---

# Contrastive Judge Optimizer

## Overview
The Contrastive Judge Optimizer helps you build robust evaluation prompts (Judges) using a data-driven approach. By providing examples of what "good" and "poor" outputs look like, the system iteratively refines the judge's criteria and scoring logic until it can reliably differentiate between them.

## Workflow

### 1. Data Preparation
Before starting, ensure your dataset is organized in the `contrastive-eval-prompt-writer/samples/` directory:
- **Good Samples**: Place representative high-quality examples in `samples/good/`.
- **Poor Samples**: Place representative low-quality examples in `samples/poor/`.
- **Input Metadata**: Ensure `inputs/sample_inputs.json` contains the corresponding input data for these samples.

See [DATA_FORMAT.md](references/DATA_FORMAT.md) for detailed schema information.

### 2. Initialization
Run the initiation script to bootstrap the initial version (`v0`) of the judge, critic, and optimizer prompts.
```bash
python contrastive-eval-prompt-writer/0_initiate.py
```

### 3. Iterative Optimization
Execute the main orchestrator to run the Evaluate-Critic-Optimize loop.
```bash
python contrastive-eval-prompt-writer/run_contrastive_writer.py --version v0 --num_iterations 5
```

## Core Guardrails
- **Scoring Integrity**: The `overall_score` must always be calculated strictly as the **arithmetic mean** of individual field scores.
- **No Arbitrary Penalties**: Avoid introducing "Global caps" or penalties that override the calculated average regardless of individual field performance.
- **Criteria Consistency**: Ensure that rules added for specific score fields do not contradict the fundamental scoring logic.

## Best Practices
- **Balanced Samples**: Provide an equal number of good and poor samples if possible.
- **Representative Diversity**: Samples should cover the variety of cases the judge will encounter in production.
- **Incremental versions**: Start with a small number of iterations (e.g., 3-5) and review the critic's feedback before running more.
