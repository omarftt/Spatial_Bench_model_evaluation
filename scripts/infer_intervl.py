"""
Inference script for InternVL2 family models.
Supports: InternVL2-2B, InternVL2-8B, InternVL2-26B, InternVL2-40B, InternVL2-Llama3-76B
"""

import argparse
import sys
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


def load_and_resize(path):
    """Open an image"""
    img = Image.open(path).convert("RGB")
    return img


def run_inference(image_paths, prompt_file, output_file, model_name, max_new_tokens):
    """Run inference"""
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load images
    images = [load_and_resize(img_path) for img_path in image_paths]
    
    # Build prompt with image tokens
    # InternVL uses <image> tokens for each image
    image_tokens = "".join([f"Image-{i+1}: <image>\n" for i in range(len(images))])
    full_prompt = f"{image_tokens}\n{prompt}"
    
    # Generate response
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=None,  # Will be processed from images
            images=images,
            question=full_prompt,
            generation_config={
                'max_new_tokens': max_new_tokens,
                'do_sample': False,
            }
        )
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response.strip())


def main():
    parser = argparse.ArgumentParser(description="Multi-agent VQA with InternVL2 models")
    parser.add_argument("--images", nargs='+', required=True, help="Paths to agent images")
    parser.add_argument("--prompt_file", required=True, help="Path to file containing prompt text")
    parser.add_argument("--output_file", required=True, help="Path to file for output text")
    parser.add_argument("--model", default="OpenGVLab/InternVL2-8B",
                       help="InternVL2 model name")
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
        print(f"ERROR in infer_internvl.py: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)