import argparse
import sys
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


SYSTEM_PROMPT = "You are an AI assistant performing an academic benchmark evaluation. The following question/proposition has 4 possible answers that are presented in alphabetical order. You must respond ONLY with the correct choice to the question with 'A', 'B', 'C', or 'D', where each letter corresponds to its respective answer choice and the text of the choice. Do NOT provide any explanation or reasoning, ONLY the selected choice in the specified format. The solution must be based only on the visual evidence in the two images. If multiple answers seem plausible, choose the most consistent with the given views."


def load_and_preprocess(path):
    """Open an image"""
    img = Image.open(path).convert("RGB")
    return img


def run_inference(image_paths, prompt_file, output_file, model_name, max_new_tokens):
    """Run inference on multiple images with the given prompt."""
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    
    if device == "cuda":
        model = model.to(device)
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load images
    images = [load_and_preprocess(img_path) for img_path in image_paths]
    
    # Build conversation with system prompt and user content
    msgs = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": images + [prompt]  # Images followed by text
        }
    ]
    
    # Generate response
    with torch.no_grad():
        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=False, 
            max_new_tokens=max_new_tokens
        )
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response.strip())


def main():
    parser = argparse.ArgumentParser(description="Multi-agent VQA with MiniCPM-V models")
    parser.add_argument("--images", nargs='+', required=True, help="Paths to agent images")
    parser.add_argument("--prompt_file", required=True, help="Path to file containing prompt text")
    parser.add_argument("--output_file", required=True, help="Path to file for output text")
    parser.add_argument("--model", default="openbmb/MiniCPM-V-2_6", help="MiniCPM-V model name")
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
        print(f"ERROR in infer_minicpm.py: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)