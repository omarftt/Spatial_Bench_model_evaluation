import argparse
import sys
import os
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, Part

from prompts import SYSTEM_PROMPT


def load_images(image_paths):
    """Load images and create Parts for Vertex AI."""
    img_parts = []
    for i, img_path in enumerate(image_paths):
        img_parts.append(Part.from_text(f"Image {i+1}:"))
        img_parts.append(Part.from_data(
            data=open(img_path, 'rb').read(),
            mime_type="image/png" if img_path.endswith('.png') else "image/jpeg"
        ))
    return img_parts


def run_inference(image_paths, prompt_file, output_file, model_name, max_new_tokens):
    """Run inference on multiple images with the given prompt."""
    load_dotenv()

    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT not found in environment")

    vertexai.init(project=project_id, location='global')

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()

    img_parts = load_images(image_paths)
    content_parts = img_parts + [Part.from_text(prompt)]

    model = GenerativeModel(model_name, system_instruction=[SYSTEM_PROMPT])

    generation_config = {"temperature": 0.0}
    if max_new_tokens != 128:
        generation_config["max_output_tokens"] = max_new_tokens

    response = model.generate_content(
        content_parts,
        generation_config=generation_config
    )

    prediction = response.text.strip()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(prediction)


def main():
    parser = argparse.ArgumentParser(description="Multi-agent VQA with Gemini models on Vertex AI")
    parser.add_argument("--images", nargs='+', required=True, help="Paths to agent images")
    parser.add_argument("--prompt_file", required=True, help="Path to file containing prompt text")
    parser.add_argument("--output_file", required=True, help="Path to file for output text")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
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
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
