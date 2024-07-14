import google.generativeai as genai

GOOGLE_API_KEY='AIzaSyCKI8-1_Pc7VIPyvgEiN_w1Ijkamfl4cTA'

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')

from google.colab import files
from io import BytesIO
from PIL import Image
import json
import gradio as gr

uploaded = files.upload()
im = Image.open(BytesIO(next(iter(uploaded.values()))))

response1 = model.generate_content(
                                  [im,
                                  f'Extract the name, country information from the document in JSON format with the confidence percentage'])

print(response1.text)

string = response1.text
new_string = string.strip("```json ")
new_string.strip("```")

json_obj = json.loads(new_string)


def update_textbox(name):
  return json_obj.get('name'), json_obj.get('country')

with gr.Blocks(title="Title") as demo:
  with gr.Row():
    with gr.Column():

      output = gr.Textbox(label="Output")
      img = gr.Image(type="filepath")
      img = Image.open(BytesIO(next(iter(uploaded.values()))))
      submit = gr.Button("Extract")

      submit.click(fn=update_textbox, outputs=output)
demo.launch()

