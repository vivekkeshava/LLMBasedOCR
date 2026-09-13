# LLM-based OCR

Extract structured fields from document images — driving licenses, passports, and other
government IDs — using Google's Gemini vision model. Instead of classic OCR that returns a
wall of text, this returns clean JSON with a confidence score per field:

```json
{
  "name":            { "value": "JANE DOE",   "confidence": 97 },
  "country":         { "value": "USA",        "confidence": 99 },
  "document_number": { "value": "D1234567",   "confidence": 92 },
  "date_of_birth":   { "value": "1990-04-12", "confidence": 95 }
}
```

## How it works

1. The document image is sent to Gemini (`gemini-1.5-flash`) with a prompt that constrains
   the output to a JSON object of `field → { value, confidence }`.
2. The response is parsed defensively (models sometimes wrap JSON in code fences).
3. Results are shown in a Gradio UI, or printed to stdout in CLI mode.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your Google AI Studio API key
```

Get an API key at https://aistudio.google.com/apikey. The key is read from the
`GOOGLE_API_KEY` environment variable — never hardcode it.

## Usage

```bash
python app.py                     # launch the Gradio web UI
python app.py --image license.jpg # one-shot extraction, JSON to stdout
```

## Notes

- Works on any document type — the prompt asks the model to extract whatever fields are
  legible, so it isn't hardcoded to a specific ID layout.
- Confidence scores are model-reported estimates, not calibrated probabilities; treat them
  as a triage signal for manual review, not ground truth.
- Don't upload documents you don't have the right to process.

## License

MIT
