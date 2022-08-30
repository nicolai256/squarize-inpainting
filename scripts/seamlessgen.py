import argparse
import cv2
import time
import os
import shutil
from pathlib import Path


doc_path = os.path.expanduser('~\Documents')
visions_path = os.path.expanduser('~\Documents\\visions of chaos')

import subprocess



##generate texture with SD###

import random
parser = argparse.ArgumentParser()

parser.add_argument("--prompt",type=str,nargs="?",default="a painting of a virus monster playing guitar",help="the prompt to render")
parser.add_argument("--outdir",type=str,nargs="?",help="dir to write results to",default="outputs/txt2img-samples")
parser.add_argument("--skip_grid",action='store_true',help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",)
parser.add_argument("--skip_save",action='store_true',help="do not save individual samples. For speed measurements.",)
parser.add_argument("--ddim_steps",type=int,default=50,help="number of ddim sampling steps",)
parser.add_argument("--plms",action='store_true',help="use plms sampling",)
parser.add_argument("--laion400m",action='store_true',help="uses the LAION400M model",)
parser.add_argument("--fixed_code",action='store_true',help="if enabled, uses the same starting code across samples ",)
parser.add_argument("--ddim_eta",type=float,default=0.0,help="ddim eta (eta=0.0 corresponds to deterministic sampling",)
parser.add_argument("--n_iter",type=int,default=1,help="only works with 1 on seamless",)
parser.add_argument("--H",type=int,default=512,choices=["512"],help="image height, in pixel space",)
parser.add_argument("--W",type=int,default=512,choices=["512"],help="image width, in pixel space",)
parser.add_argument("--C",type=int,default=4,help="latent channels",)
parser.add_argument("--f",type=int,default=8,help="downsampling factor")
parser.add_argument("--n_samples",type=int,default=1,help="only works with 1 on seamless",)
parser.add_argument("--n_rows",type=int,default=0,help="rows in the grid (default: n_samples)",)
parser.add_argument("--scale",type=float,default=7.5,help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",)
parser.add_argument("--from-file",type=str,help="if specified, load prompts from this file",)
parser.add_argument("--config",type=str,default="configs/stable-diffusion/v1-inference.yaml",help="path to config which constructs model",)
parser.add_argument("--ckpt",type=str,default="models/ldm/stable-diffusion-v1/model.ckpt",help="path to checkpoint of model",)
parser.add_argument("--seed",type=int,default=[random.randint(1, 2703686851)],help="the seed (for reproducible sampling)",)
parser.add_argument("--precision",type=str,help="evaluate at this precision",choices=["full", "autocast"],default="autocast")
parser.add_argument("--mask", type=str, help="thickness of the mask for seamless inpainting",choices=["thinnest", "thin", "medium", "thick", "thickest"],default="medium")
#inpaint
parser.add_argument("--indir2",type=str,nargs="?",default="tmp360/tiled_image/",help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",)
parser.add_argument("--outdir2",type=str,nargs="?",default="tmp360/tiled_image/",help="dir to write results to",)
parser.add_argument("--steps2",type=int,default=50,help="number of ddim sampling steps",)
parser.add_argument("--indir3",type=str,nargs="?",default="tmp360/tiled_image2/",help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",)
parser.add_argument("--outdir3",type=str,nargs="?",default="outputs/txt2seamlessimg-samples/",help="dir to write results to",)
parser.add_argument("--steps3",type=int,default=50,help="number of ddim sampling steps",)
    
opt = parser.parse_args()
p = subprocess.Popen(['mkdir', 'tmp360'])
p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
thinnest = r'seamless/thinnest/1st_mask.png'
thin = r'seamless/thin/1st_mask.png'
medium = r'seamless/medium/1st_mask.png'
thick = r'seamless/thick/1st_mask.png'
thickest = r'seamless/thickest/1st_mask.png'

if opt.mask != thinnest:
		p = subprocess.Popen(['mkdir', 'tmp360'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
		print('temporary directories made')
		print('copying',opt.mask ,'mask to dir')
		shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thinnest/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
		print('thinnest mask copied')
elif opt.mask == thin:
		p = subprocess.Popen(['mkdir', 'tmp360'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
		print('temporary directories made')
		print('copying',opt.mask ,'mask to dir')
		shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thin/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
		print(opt.mask, 'mask copied')
elif opt.mask == medium:
		p = subprocess.Popen(['mkdir', 'tmp360'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
		print('temporary directories made')
		print('copying',opt.mask ,'mask to dir')
		shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/medium/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
elif opt.mask == thick:
		p = subprocess.Popen(['mkdir', 'tmp360'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
		print('temporary directories made')
		print('copying',opt.mask ,'mask to dir')
		shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thick/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
elif opt.mask == thickest:
		p = subprocess.Popen(['mkdir', 'tmp360'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
		print('temporary directories made')
		print('copying',opt.mask ,'mask to dir')
		shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thickest/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')

import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



def main():


    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"
	##choose the mask



    seed_everything(opt.seed)




    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"example.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'example.png'))
                    grid_count += 1

                toc = time.time()
#unable to call opt?
#parser = argparse.ArgumentParser()
#parser.add_argument("--mask", type=str, help="thickness of the mask for seamless inpainting", choices=["thinnest", "thin", "medium", "thick", "thickest"], default="medium")
#opt = parser.parse_args()
if __name__ == "__main__":
    main()

#	outpath = opt.outdir
#	sample_path = os.path.join(outpath, "samples")
output555= "outputs/txt2img-samples/samples/example.png"

##move opt.output to temp directory###
source = output555
destination = 'tmp360/original_image/example.png'
shutil.move(source, destination)

##tile the image
#p = subprocess.Popen(['mogrify', 'convert', '-virtual-pixel', 'tile', '-filter', 'point', '-set', 'option:distort:viewport', '1024x1024', '-distort', 'SRT', '0', '-path', r'tmp360/tiled2_image', r'tmp360/original_image/example.png'])
#print('image tiled')
#from PIL import Image # import pillow library (can install with "pip install pillow")
#im = Image.open('tmp360/tiled2_image/example.png')
#im = im.crop( (1, 0, 512, 512) ) # previously, image was 826 pixels wide, cropping to 825 pixels wide
#im.save('tmp360/tiled2_image/example.png') # saves the image
# im.show() # opens the image
subprocess.call([r'crop.bat'])
print('image center cropped')




##first pass of inpainting
import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":



    masks = sorted(glob.glob(os.path.join(opt.indir2, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir2, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath3 = os.path.join(opt.outdir2, os.path.split(image)[1])
                batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps2,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath3)



##move opt.output2 to temp directory###
outpath3 = 'tmp360/tiled_image2/example.png'
source2 = outpath3
destination2 ='tmp360/original_image2/'
shutil.move(source2, destination2)

##tile the image
p = subprocess.Popen(['mogrify', 'convert', '-virtual-pixel', 'tile', '-filter', 'point', '-set', 'option:distort:viewport', '1024x1024', '-distort', 'SRT', '0', r'tmp360/original_image2/', '-path', r'"tmp360/tiled_image2"', '*.png'])
p = subprocess.Popen(['mogrify' '-gravity' 'northeast' '-background' 'red' '-splice' '350x150' '-path' r'"tmp360/tiled_image2"' '*.png'])
p = subprocess.Popen(['mogrify',r'/tmp360/tiled_image2', '-gravity', 'center', '-crop', '512x512+0+0', '*.png'])

##mask 2
if opt.mask:
		if opt.mask == thinnest:
			src3 = 'seamless/thinnest/2nd_mask.png'
			trg3 = 'tmp360/tiled_image2/example_mask.png'
		if opt.mask == thin:
			src3 = 'seamless/thin/2nd_mask.png'
			trg3 = 'tmp360/tiled_image2/example_mask.png'
		if opt.mask == medium:
			src3 = 'seamless/medium/2nd_mask.png'
			trg3 = 'tmp360/tiled_image2/example_mask.png'
		if opt.mask == thick:
			src3 = 'seamless/thick/2nd_mask.png'
			trg3 = 'tmp360/tiled_image2/example_mask.png'
		if opt.mask == thickest:
			src3 = 'seamless/thickest/2nd_mask.png'
			trg3 = 'tmp360/tiled_image2/example_mask.png'
		shutil.copy2(src3, trg3)


##2nd pass of inpainting
import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":


    masks = sorted(glob.glob(os.path.join(opt.indir3, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir3, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath4 = os.path.join(opt.outdir3, os.path.split(image)[1])
                batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps2,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath4)