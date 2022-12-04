import dnnlib
import legacy
import torch
from PIL import Image
import numpy as np

class Generator:
    def __init__(self, device, path):
        self.name = 'generator'
        self.model = self.load_model(device, path)
        self.device = device
        self.force_32 = False
        
    def load_model(self, device, path):
        with dnnlib.util.open_url(path) as f:
            network= legacy.load_network_pkl(f)
            self.G_ema = network['G_ema'].to(device)
            # self.D = network['D'].to(device)
            # self.G = network['G'].to(device)
            return self.G_ema
        
    def generate(self, z, c, fts, noise_mode='const', return_styles=True):
        return self.model(z, c, fts=fts, noise_mode=noise_mode, return_styles=return_styles, force_fp32=self.force_32)
    
    def generate_from_style(self, style, noise_mode='const'):
        ws = torch.randn(1, self.model.num_ws, 512)
        return self.model.synthesis(ws, fts=None, styles=style, noise_mode=noise_mode, force_fp32=self.force_32)
    
    def tensor_to_img(self, tensor, is_concat=True):
        img = torch.clamp((tensor + 1.) * 127.5, 0., 255.)
        img_list = img.permute(0, 2, 3, 1)
        img_list = [img for img in img_list]

        if not is_concat:
            img_list = [Image.fromarray(img.detach().cpu().numpy().astype(np.uint8)) for img in img_list]
            return img_list
        return Image.fromarray(torch.cat(img_list, dim=-2).detach().cpu().numpy().astype(np.uint8))