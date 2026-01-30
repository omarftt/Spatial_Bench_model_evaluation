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

## Supported Models

The system detects the correct inference script based on model name.

### 7-8B Scale (Primary Priority)

| Model Family | Model Name (for --model argument) | Status |
|--------------|-----------------------------------|--------|
| LLaVA-OneVision | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | ✅ Supported |
| Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` | ✅ Supported |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` | ✅ Supported |
| LLaVA-NeXT-Video | `llava-hf/LLaVA-NeXT-Video-7B-hf` | ✅ Supported |
| LLaVA-Video | `llava-hf/LLaVA-Video-7B-hf` | ❌ Not Supported |
| InternVL2 | `OpenGVLab/InternVL2-8B` | ✅ Supported |
| MiniCPM-V 2.6 | `openbmb/MiniCPM-V-2_6` | ❌ Not Supported |
| Molmo | `allenai/Molmo-7B-D-0924` | ✅ Supported |
| mPLUG-Owl | `mPLUG/mPLUG-Owl3-7B-240728` | ❌ Not Supported |
| VILA 1.5 | `Efficient-Large-Model/VILA1.5-7b` | ❌ Not Supported |
| Ovis 2 | `AIDC-AI/Ovis2-8B` | ❌ Not Supported |
| Oryx | `THU-MIG/Oryx-7B` | ❌ Not Supported |

### 1-4B Scale (Lesser Priority)

| Model Family | Model Name (for --model argument) | Status |
|--------------|-----------------------------------|--------|
| Qwen2-VL | `Qwen/Qwen2-VL-2B-Instruct` | ✅ Supported |
| InternVL2 | `OpenGVLab/InternVL2-2B` | ✅ Supported |
| Phi-3.5-Vision | `microsoft/Phi-3.5-vision-instruct` | ❌ Not Supported |

### Larger Scale (Lesser Priority)

| Model Family | Model Name (for --model argument) | Status |
|--------------|-----------------------------------|--------|
| Qwen2-VL 72B | `Qwen/Qwen2-VL-72B-Instruct` | ❌ Not Supported |
| LLaVA-OneVision 72B | `llava-hf/llava-onevision-qwen2-72b-ov-hf` | ❌ Not Supported |



## Input Format

JSONL file where each line contains:
```json
{"skill": "navigation", "images": ["img1.png", "img2.png"], "choices": ["A", "B", "C", "D"], "ground_truth": "A", "question": "Which agent...?"}
```
