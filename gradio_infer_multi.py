import gradio as gr
import clip
import torch
from generator import Generator

device = 'cuda:0'
weight = 'weights/MM-CelebA-HQ.pkl'
generator = Generator(device=device, path=weight)
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()

def infer(prompt, num_img):
    num_img = int(num_img)
    tokenized_text = clip.tokenize([prompt]*num_img).to(device)
    txt_fts = clip_model.encode_text(tokenized_text)
    txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)

    z = torch.randn((num_img, 512)).to(device)
    c = torch.randn((num_img, 1)).to(device) 
    img, _ = generator.generate(z=z, c=c, fts=txt_fts)
    to_show_img = generator.tensor_to_img(img, is_concat=False)

    return to_show_img

title = "Text to Face generation"
description = "Enter a prompt and submit."
examples = [
    ["Young man has black hair", 2],
    ["Old woman is smiling", 2],
    ["woman with bushy eyebrows is sad", 2],
    ["young man with red hair is sad", 2]
]

gr.Interface(
    infer,
    [
    gr.Textbox(label="Input Prompt"),
    gr.Number(label="Number of images", value=1),
    ],
    [gr.Gallery(label="Output Images")],
    title=title, 
    description=description, 
    examples=examples,
    allow_flagging="never"
).launch(share=True, debug=True)