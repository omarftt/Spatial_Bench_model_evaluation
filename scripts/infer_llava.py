import argparse
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from prompts import SYSTEM_PROMPT



def load_and_preprocess(path):
    """Open an image"""
    img = Image.open(path).convert("RGB")
    return img


def run_inference(image_paths, prompt_file, output_file, model_name, max_new_tokens):
    """Run inference on multiple images with the given prompt."""

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name_lower = model_name.lower()

    if "next-video" in model_name_lower:
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    else:
        try:
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
        except Exception:
            # Fallback for older LLaVA models or VILA models
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
        processor = AutoProcessor.from_pretrained(model_name)

    # Load images
    images = [load_and_preprocess(img_path) for img_path in image_paths]
    
    # Build conversation with images
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in images],
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Apply chat template
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(
        images=images,
        text=prompt_text,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    
    # Decode (strip input tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)


def main():
    parser = argparse.ArgumentParser(description="Multi-agent VQA with LLaVA and VILA models")
    parser.add_argument("--images", nargs='+', required=True, help="Paths to agent images")
    parser.add_argument("--prompt_file", required=True, help="Path to file containing prompt text")
    parser.add_argument("--output_file", required=True, help="Path to file for output text")
    parser.add_argument("--model", default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                       help="LLaVA or VILA model name")
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
        print(f"ERROR in infer_llava.py: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)