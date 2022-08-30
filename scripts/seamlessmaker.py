import argparse
import cv2
import time
import os
import shutil
from pathlib import Path


doc_path = os.path.expanduser('~\Documents')
visions_path = os.path.expanduser('~\Documents\\visions of chaos')

import subprocess
import random
parser = argparse.ArgumentParser()
#inpaint
parser.add_argument("--mask", type=str, help="thickness of the mask for seamless inpainting",choices=["thinnest", "thin", "medium", "thick", "thickest"],default="medium")
parser.add_argument("--input",type=str,nargs="?",default="tmp360/tiled_image/",help="input image",)
parser.add_argument("--indir2",type=str,nargs="?",default="tmp360/tiled_image/",help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",)
parser.add_argument("--outdir2",type=str,nargs="?",default="tmp360/tiled_image/",help="dir to write results to",)
parser.add_argument("--steps2",type=int,default=50,help="number of ddim sampling steps",)
parser.add_argument("--indir3",type=str,nargs="?",default="tmp360/tiled2_image2/",help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",)
parser.add_argument("--outdir3",type=str,nargs="?",default="outputs/txt2seamlessimg-samples/",help="dir to write results to",)
parser.add_argument("--steps3",type=int,default=50,help="number of ddim sampling steps",)
    





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
		
		opt = parser.parse_args()
		inputimg = opt.input
		destination = 'tmp360/original_image/example.png'
		shutil.copy(inputimg, destination)
		
		'''p = subprocess.Popen(['mkdir', 'tmp360'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])'''
#		masks = opt.mask
#		thinnest = r'seamless/thinnest/1st_mask.png'
#		thin = r'seamless/thin/1st_mask.png'
#		medium = r'seamless/medium/1st_mask.png'
#		thick = r'seamless/thick/1st_mask.png'
#		thickest = r'seamless/thickest/1st_mask.png'
#		
#		if masks == thinnest:
#				'''p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])'''
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/medium/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
#				print('thinnest mask copied')
#		elif masks == thin:
#				p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thin/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
#				print(opt.mask, 'mask copied')
#		elif masks == medium:
#				p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/medium/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
#		elif masks == thick:
#				p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thick/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
#		elif masks == thickest:
#				p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thickest/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
#		
#		#	outpath = opt.outdir
#		#	sample_path = os.path.join(outpath, "samples")
#		output555= "outputs/txt2img-samples/samples/example.png"
		
		"""##move opt.output to temp directory###
		source = output555
		destination = 'tmp360/original_image/example.png'
		shutil.move(source, destination)"""
		
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


if __name__ == "__main__":
		
		opt = parser.parse_args()
		inputimg = outpath3
		destination = 'tmp360/original_image2/example.png'
		shutil.copy(inputimg, destination)
		
		'''p = subprocess.Popen(['mkdir', 'tmp360'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
		p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#		p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])'''
#		masks = opt.mask
#		thinnest = r'seamless/thinnest/1st_mask.png'
#		thin = r'seamless/thin/1st_mask.png'
#		medium = r'seamless/medium/1st_mask.png'
#		thick = r'seamless/thick/1st_mask.png'
#		thickest = r'seamless/thickest/1st_mask.png'
#		
#		if masks == thinnest:
#				'''p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])'''
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/example_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
#				print('thinnest mask copied')
#		elif masks == thin:
#				p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thin/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
#				print(opt.mask, 'mask copied')
#		elif masks == medium:
#				p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/medium/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
#		elif masks == thick:
#				p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thick/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
#		elif masks == thickest:
#				p = subprocess.Popen(['mkdir', 'tmp360'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/original_image2'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled2_image'])
#				p = subprocess.Popen(['mkdir', 'tmp360/tiled_image2'])
#				print('temporary directories made')
#				print('copying',opt.mask ,'mask to dir')
#				shutil.copy('C:/deepdream-test/stable/stable-diffusion-2/seamless/thickest/1st_mask.png', 'C:/deepdream-test/stable/stable-diffusion-2/tmp360/tiled_image/example_mask.png')
		
		#	outpath = opt.outdir
		#	sample_path = os.path.join(outpath, "samples")
		#output555= "outputs/txt2img-samples/samples/example.png"
		
		"""##move opt.output to temp directory###
		source = output555
		destination = 'tmp360/original_image/example.png'
		shutil.move(source, destination)"""
		
		##tile the image
		#p = subprocess.Popen(['mogrify', 'convert', '-virtual-pixel', 'tile', '-filter', 'point', '-set', 'option:distort:viewport', '1024x1024', '-distort', 'SRT', '0', '-path', r'tmp360/tiled2_image', r'tmp360/original_image/example.png'])
		#print('image tiled')
		#from PIL import Image # import pillow library (can install with "pip install pillow")
		#im = Image.open('tmp360/tiled2_image/example.png')
		#im = im.crop( (1, 0, 512, 512) ) # previously, image was 826 pixels wide, cropping to 825 pixels wide
		#im.save('tmp360/tiled2_image/example.png') # saves the image
		# im.show() # opens the image
		subprocess.call([r'2ndpass.bat'])
		print('image center cropped')
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
		outpath4 = opt.outdir3
		base_count = len(os.listdir(outpath4))
		os.makedirs(opt.outdir2, exist_ok=True)
		with torch.no_grad():
		    with model.ema_scope():
		        for image, mask in tqdm(zip(images, masks)):
		            outpath4 = os.path.join(opt.outdir3, os.path.split(opt.outdir3)[1])
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
		            #Image.fromarray(inpainted.astype(np.uint8)).save(outpath4)
		            Image.fromarray(inpainted.astype(np.uint8)).save(os.path.join(outpath4, f"{base_count:05}.png"))
		            base_count += 1
