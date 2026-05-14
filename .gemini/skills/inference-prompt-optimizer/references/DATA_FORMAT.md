# Data Format Specifications

The Inference Prompt Optimizer requires a structured environment within the `inference-prompt-writer/` directory.

## Directory Structure
```text
inference-prompt-writer/
├── prompts/
│   ├── v[N]-judge.txt         # The established evaluation prompt (required)
│   └── reference/              # Optional reference prompts
├── inputs/
│   └── extracted_inputs.json   # Test cases for batch inference
└── generated/                  # Outputs generated during iterations
    ├── create/                 # Responses from inference model
    ├── evaluate/               # Scores from evaluation model
    └── critic/                 # Analysis from critic model
```

## extracted_inputs.json Schema
This file contains the input data for the inference model to process during each iteration.

```json
[
  {
    "id": "input_001",
    "prompt_input": "The text or instructions to be processed",
    "metadata": {
      "difficulty": "high",
      "type": "creative"
    }
  }
]
```

## Performance Logs
The system tracks results in `logs/eval_logs.jsonl`, which includes:
- Version numbers
- Scores assigned by the judge
- Feedback for each generated response
