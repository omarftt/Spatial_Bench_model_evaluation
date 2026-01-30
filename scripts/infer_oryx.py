import argparse
import sys
import torch
from PIL import Image
from oryx.model.builder import load_pretrained_model
from oryx.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from oryx.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from oryx.conversation import conv_templates


SYSTEM_PROMPT = "You are an AI assistant performing an academic benchmark evaluation. The following question/proposition has 4 possible answers that are presented in alphabetical order. You must respond ONLY with the correct choice to the question with 'A', 'B', 'C', or 'D', where each letter corresponds to its respective answer choice and the text of the choice. DO NOT provide any explanation or reasoning, ONLY the selected choice in the specified format. The solution must be based only on the visual evidence in the two images. If multiple answers seem plausible, choose the most consistent with the given views."


def load_and_preprocess(path):
    """Open an image"""
    img = Image.open(path).convert("RGB")
    return img


def run_inference(image_paths, prompt_file, output_file, model_name, max_new_tokens):
    """Run inference on multiple images with the given prompt."""
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model using Oryx's custom builder
    model_name_short = get_model_name_from_path(model_name)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_name, 
        None, 
        model_name_short,
        device_map="auto" if device == "cuda" else None
    )
    
    # Load images
    images = [load_and_preprocess(img_path) for img_path in image_paths]
    
    # Process images using Oryx's processor
    image_tensor = process_images(images, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(device, dtype=torch.float16 if device == "cuda" else torch.float32) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
    
    # Build prompt with image tokens
    image_tokens = " ".join([DEFAULT_IMAGE_TOKEN for _ in images])
    full_prompt = f"{SYSTEM_PROMPT}\n\n{image_tokens}\n{prompt}"
    
    # Use Oryx conversation template
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt_text, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    
    # Decode
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)


def main():
    parser = argparse.ArgumentParser(description="Multi-agent VQA with Oryx models")
    parser.add_argument("--images", nargs='+', required=True, help="Paths to agent images")
    parser.add_argument("--prompt_file", required=True, help="Path to file containing prompt text")
    parser.add_argument("--output_file", required=True, help="Path to file for output text")
    parser.add_argument("--model", default="THUdyh/Oryx-7B", help="Oryx model name ")
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
        print(f"ERROR in infer_oryx.py: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)