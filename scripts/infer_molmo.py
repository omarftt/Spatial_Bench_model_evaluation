import argparse
import sys
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


SYSTEM_PROMPT = "You are an AI assistant performing an academic benchmark evaluation. The following question/proposition has 4 possible answers that are presented in alphabetical order. You must respond ONLY with the correct choice to the question with 'A', 'B', 'C', or 'D', where each letter corresponds to its respective answer choice and the text of the choice. Do NOT provide any explanation or reasoning, ONLY the selected choice in the specified format. The solution must be based only on the visual evidence in the two images. If multiple answers seem plausible, choose the most consistent with the given views."


def load_and_preprocess(path):
    """Open an image"""
    img = Image.open(path).convert("RGB")
    return img


def run_inference(image_paths, prompt_file, output_file, model_name, max_new_tokens):
    """Run inference on multiple images with the given prompt."""
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # Load processor and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Load and preprocess images
    images = [load_and_preprocess(img_path) for img_path in image_paths]
    
    # Build the full prompt with system prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    
    # Process inputs with Molmo's built-in dynamic preprocessing
    inputs = processor.process(
        images=images,
        text=full_prompt
    )
    
    # Move inputs to device and create batch
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # Generate with Molmo's generate_from_batch method
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            stop_strings="<|endoftext|>",
            do_sample=False
        ),
        tokenizer=processor.tokenizer
    )
    
    # Decode output (only get generated tokens)
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text.strip())


def main():
    parser = argparse.ArgumentParser(description="Multi-agent VQA with Molmo models")
    parser.add_argument("--images", nargs='+', required=True, help="Paths to agent images")
    parser.add_argument("--prompt_file", required=True, help="Path to file containing prompt text")
    parser.add_argument("--output_file", required=True, help="Path to file for output text")
    parser.add_argument("--model", default="allenai/Molmo-7B-D-0924", help="Molmo model name ")
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
        print(f"ERROR in infer_molmo.py: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)