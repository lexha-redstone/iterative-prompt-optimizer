# Data Format Specifications

The Contrastive Judge Optimizer requires a specific file structure and data format to function correctly.

## Directory Structure
All data should be placed within the `contrastive-eval-prompt-writer/` directory.

```text
contrastive-eval-prompt-writer/
├── inputs/
│   └── sample_inputs.json        # Metadata and input text/parameters
└── samples/
    ├── good/                     # High-quality output samples
    │   ├── img-0.png
    │   └── ...
    └── poor/                     # Low-quality output samples
        ├── img-0.png
        └── ...
```

## sample_inputs.json Schema
This file maps the samples to their respective inputs. It should be an array of objects.

```json
[
  {
    "id": "sample_0",
    "input_text": "The prompt or input used to generate this sample",
    "good_sample_path": "samples/good/img-0.png",
    "poor_sample_path": "samples/poor/img-0.png",
    "metadata": {
      "category": "landscape",
      "style": "cinematic"
    }
  }
]
```

- `id`: Unique identifier for the test case.
- `input_text`: The actual prompt or context provided to the inference model.
- `good_sample_path`: Relative path to the known good output.
- `poor_sample_path`: Relative path to the known poor output.

## Sample Files
The system currently supports:
- **Images**: `.png`, `.jpg`, `.webp` (Multi-modal judges).
- **Text**: (If the scripts are adapted for text) `.txt` or `.md`.

Ensure that filenames match between `good/` and `poor/` directories if you are using the default mapping logic in `0_initiate.py`.
