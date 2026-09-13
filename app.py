"""LLM-based OCR: extract structured fields from document images using Gemini.

Upload an image of an identity or government document (driving license,
passport, etc.) and get the key fields back as structured JSON, with a
model-reported confidence score per field.

Usage:
    export GOOGLE_API_KEY=your-key   # or put it in .env
    python app.py                    # launches the Gradio UI
    python app.py --image path.jpg   # one-shot CLI extraction
"""

import argparse
import json
import os
import re
import sys

import google.generativeai as genai
import gradio as gr
from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL_NAME = "gemini-1.5-flash"

PROMPT = (
    "You are an OCR engine for identity and government documents. "
    "Extract every clearly legible field from this document image (for example: "
    "name, date of birth, document number, country, issue/expiry dates). "
    "Respond with ONLY a JSON object mapping each field name to an object with "
    '"value" and "confidence" (0-100). Omit fields you cannot read.'
)


def get_model() -> genai.GenerativeModel:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("GOOGLE_API_KEY is not set. Copy .env.example to .env or export it.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from a model response that may be fenced in ```json blocks."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response:\n{text}")
    return json.loads(match.group(0))


def extract_fields(image: Image.Image) -> dict:
    model = get_model()
    response = model.generate_content([image, PROMPT])
    return parse_json_response(response.text)


def gradio_extract(image: Image.Image) -> str:
    if image is None:
        return "Upload a document image first."
    try:
        return json.dumps(extract_fields(image), indent=2)
    except Exception as exc:  # surface API/parse errors in the UI instead of crashing
        return f"Extraction failed: {exc}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LLM-based OCR") as demo:
        gr.Markdown("# LLM-based OCR\nExtract structured fields from document images with Gemini.")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Document image")
                submit = gr.Button("Extract", variant="primary")
            output = gr.Code(label="Extracted fields (JSON)", language="json")
        submit.click(fn=gradio_extract, inputs=img, outputs=output)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Extract from a single image file and print JSON")
    args = parser.parse_args()

    if args.image:
        result = extract_fields(Image.open(args.image))
        print(json.dumps(result, indent=2))
    else:
        build_ui().launch()


if __name__ == "__main__":
    main()
