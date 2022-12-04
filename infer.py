import argparse
import torch
import clip

from generator import Generator

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Text to Face')

    parser.add_argument('--prompt', type=str, default='informer', help='Input Prompt to generate face')
    parser.add_argument('--num_img', type=int, default=3, help='Number of image to be generated')
    parser.add_argument('--weight', type=str, default="weights/MM-CelebA-HQ.pkl", help='Generator weight')
    parser.add_argument('--save_path', type=str, default="generated.jpg", help='Save path')
    parser.add_argument('--device', type=str, default='cuda:0', help='options: [cuda:0, cpu')

    args = parser.parse_args()
    return args

@torch.no_grad()
def inference(prompt, num_img, generator, clip_model, save_path, device='cuda:0'):

    tokenized_text = clip.tokenize([prompt]*num_img).to(device)
    txt_fts = clip_model.encode_text(tokenized_text)
    txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)

    z = torch.randn((num_img, 512)).to(device)
    c = torch.randn((num_img, 1)).to(device) 
    img, _ = generator.generate(z=z, c=c, fts=txt_fts)
    to_show_img = generator.tensor_to_img(img)
    to_show_img.save(save_path)

def main():
    args = parse_args()

    generator = Generator(device=args.device, path=args.weight)
    clip_model, _ = clip.load("ViT-B/32", device=args.device)
    clip_model = clip_model.eval()

    inference(args.prompt, args.num_img, generator, clip_model, args.save_path, args.device)

if __name__ == '__main__':
    main()