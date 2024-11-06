import gradio as gr
import os
from PIL import Image

def process_images(folder_path):
  """Processes all images in the given folder path and returns a list of textboxes."""
  textboxes = []
  if os.path.isdir(folder_path):
      image_count = 0
      for filename in os.listdir(folder_path):
          if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
              image_count += 1
              if image_count % 5 == 0:
                  textboxes.append(gr.Textbox(label=f"Comments for image {filename}"))
  else:
      print(f"Invalid folder path: {folder_path}")
  return textboxes


iface = gr.Interface(
    fn=process_images,
    inputs=gr.Textbox(lines=1, placeholder="Enter folder path", label="Folder Path"),
    outputs=gr.Column(),
)

iface.launch()