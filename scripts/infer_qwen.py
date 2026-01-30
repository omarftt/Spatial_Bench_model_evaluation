import argparse
import sys
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from prompts import SYSTEM_PROMPT


def load_and_preprocess(path):
    """Open an image"""
    img = Image.open(path).convert("RGB")
    return img


def load_model(model_name, device):
    """Load appropriate model based on model name."""
    # Qwen2.5-VL models
    if "Qwen2.5-VL" in model_name or "Qwen2_5-VL" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
    # Qwen2-VL models
    elif "Qwen2-VL" in model_name:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def run_inference(image_paths, prompt_file, output_file, model_name, max_new_tokens):
    """Run inference on multiple images with the given prompt."""
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_name, device)
    
    # Load processor (works for all Qwen models)
    processor = AutoProcessor.from_pretrained(model_name)

    # Load and resize all images
    images = [load_and_preprocess(img_path) for img_path in image_paths]
    
    # Build message content with all images
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": content,  # your images + prompt text
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    # Move to device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    model.to(device)

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Write output to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text[0] if output_text else "")


def main():
    parser = argparse.ArgumentParser(description="Multi-agent VQA with Qwen VL models")
    parser.add_argument("--images", nargs='+', required=True, help="Paths to agent images")
    parser.add_argument("--prompt_file", required=True, help="Path to file containing prompt text")
    parser.add_argument("--output_file", required=True, help="Path to file for output text")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Qwen model name")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    
    run_inference(
        image_paths=args.images,
        prompt_file=args.prompt_file,
        output_file=args.output_file,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in infer_qwen.py: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)