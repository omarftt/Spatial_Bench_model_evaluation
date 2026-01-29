# Multi-Agent VQA Batch Inference

Run  inference on multi-agent visual question answering datasets with various VLM.


## Quick Start
```bash
python main.py --input data/annotations.jsonl --images data/images --model Qwen/Qwen2.5-VL-7B-Instruct
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `input_data.jsonl` | Path to input JSONL file with questions |
| `--images` | `./images` | Directory containing agent images |
| `--model` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model name |
| `--output` | `outputs` | Directory for saving results |
| `--max_tokens` | `128` | Maximum tokens for generation |

## Supported Models

The system **automatically detects** the correct inference script based on model name.

### Qwen Family (uses `scripts/infer_qwen.py`)

**1-4B Scale:**
- `Qwen/Qwen2-VL-2B-Instruct`
- `Qwen/Qwen2.5-VL-2B-Instruct`

**7-8B Scale:**
- `Qwen/Qwen2-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct` ⭐ *default*

**Larger:**
- `Qwen/Qwen2-VL-72B-Instruct`
- `Qwen/Qwen2.5-VL-72B-Instruct`

### LLaVA Family (uses `scripts/infer_llava.py`)

**7-8B Scale:**
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- `llava-hf/llava-onevision-qwen2-7b-si-hf`
- `llava-hf/LLaVA-Video-7B-hf`

**Larger:**
- `llava-hf/llava-onevision-qwen2-72b-ov-hf`
- `llava-hf/llava-onevision-qwen2-72b-si-hf`

### InternVL Family (uses `scripts/infer_internvl.py`)

**1-4B Scale:**
- `OpenGVLab/InternVL2-2B`

**7-8B Scale:**
- `OpenGVLab/InternVL2-8B`

**Larger:**
- `OpenGVLab/InternVL2-26B`
- `OpenGVLab/InternVL2-40B`
- `OpenGVLab/InternVL2-Llama3-76B`

## Usage Examples

**Default settings:**
```bash
python batch_runner.py
```

**Different model:**
```bash
python batch_runner.py --model llava-hf/llava-onevision-qwen2-7b-ov-hf
```

**Custom paths:**
```bash
python batch_runner.py \
  --input my_questions.jsonl \
  --images /path/to/images \
  --output results \
  --model OpenGVLab/InternVL2-8B
```

**Longer generation:**
```bash
python batch_runner.py --model Qwen/Qwen2-VL-7B-Instruct --max_tokens 256
```

## Input Format

JSONL file where each line contains:
```json
{"skill": "navigation", "images": ["img1.png", "img2.png"], "choices": ["A", "B", "C", "D"], "ground_truth": "A", "question": "Which agent...?"}
```

## Output

**Two files in output directory:**

1. **`predictions.jsonl`** - Original data + predictions
2. **`summary.json`** - Accuracy metrics

**Example summary:**
```json
{
  "overall": {"accuracy": 0.85, "correct": 85, "total": 100},
  "macro_skill_accuracy": 0.83,
  "per_skill": {
    "navigation": {"accuracy": 0.80, "correct": 40, "total": 50},
    "distance": {"accuracy": 0.90, "correct": 45, "total": 50}
  }
}
```

## System Architecture
```
batch_runner.py → Auto-detects → scripts/infer_qwen.py
                                  scripts/infer_llava.py
                                  scripts/infer_internvl.py
                                           ↓
                                    Subprocess isolation
                                    (No memory leaks!)
```

Each model family has its own inference script in the `scripts/` folder. The system automatically selects the correct one based on model name.