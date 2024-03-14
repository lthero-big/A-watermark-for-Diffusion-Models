import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import numpy as np
import torch
import io
import os
import argparse
from utils import set_random_seed, to_tensor, to_pil
from tqdm import tqdm
from typing import Union, Tuple, Optional
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler

# credit to:https://github.com/umd-huang-lab/WAVES
distortion_strength_paras = dict(
    rotation=(0, 90),
    resizedcrop=(1, 0.1),
    erasing=(0, 1),
    brightness=(1, 16),
    contrast=(1, 6),
    blurring=(0, 20),
    noise=(0, 0.5),
    compression=(100, 0),
    reversed=(0, 100),
    elastic=(0,100),
    horizontal_flip=(0,0),
    vertical_flip=(0,0),
    togray=(0,0),
    randomcrop=(1, 0),
    invert=(0,0)
)


def relative_strength_to_absolute(strength, distortion_type):
    assert 0 <= strength <= 1
    strength = (
        strength
        * (
            distortion_strength_paras[distortion_type][1]
            - distortion_strength_paras[distortion_type][0]
        )
        + distortion_strength_paras[distortion_type][0]
    )
    strength = max(strength, min(*distortion_strength_paras[distortion_type]))
    strength = min(strength, max(*distortion_strength_paras[distortion_type]))
    return strength


def apply_distortion(
    images,
    distortion_type,
    strength=None,
    distortion_seed=0,
    same_operation=False,
    relative_strength=True,
    return_image=True,
    image_str=""
):
    # Convert images to PIL images if they are tensors
    if not isinstance(images[0], Image.Image):
        images = to_pil(images)
    # Check if strength is relative and convert if needed
    if relative_strength:
        strength = relative_strength_to_absolute(strength, distortion_type)
    # Apply distortions
    distorted_images = []
    seed = distortion_seed
    for image in images:
        distorted_images.append(
            apply_single_distortion(
                image, distortion_type, strength, distortion_seed=seed,image_str=image_str
            )
        )
        # If not applying the same distortion, increment the seed
        if not same_operation:
            seed += 1
    # Convert to tensors if needed
    if not return_image:
        distorted_images = to_tensor(distorted_images)
    return distorted_images


def apply_single_distortion(image, distortion_type, strength=None, distortion_seed=0,image_str=""):
    # Accept a single image

    assert isinstance(image, Image.Image)
    # Set the random seed for the distortion if given
    set_random_seed(distortion_seed)
    # Assert distortion type is valid
    print("Distortion type:", distortion_type)
    print("Strength:", strength)
    print("Allowed range:", distortion_strength_paras[distortion_type])

    assert distortion_type in distortion_strength_paras.keys()
    # Assert strength is in the correct range
    if strength is not None:
        assert (
            min(*distortion_strength_paras[distortion_type])
            <= strength
            <= max(*distortion_strength_paras[distortion_type])
        )

    # Apply the distortion
    if distortion_type == "rotation":
        angle = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["rotation"])
        )
        distorted_image = F.rotate(image, angle)

    elif distortion_type == "resizedcrop":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["resizedcrop"])
        )
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=(scale, scale), ratio=(1, 1)
        )
        distorted_image = F.resized_crop(image, i, j, h, w, image.size)

    elif distortion_type == "erasing":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["erasing"])
        )
        image = to_tensor([image], norm_type=None)
        i, j, h, w, v = T.RandomErasing.get_params(
            image, scale=(scale, scale), ratio=(1, 1), value=[0]
        )
        distorted_image = F.erase(image, i, j, h, w, v)
        distorted_image = to_pil(distorted_image, norm_type=None)[0]

    elif distortion_type == "brightness":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["brightness"])
        )
        enhancer = ImageEnhance.Brightness(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "contrast":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["contrast"])
        )
        enhancer = ImageEnhance.Contrast(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "blurring":
        kernel_size = (
            int(strength)
            if strength is not None
            else random.uniform(*distortion_strength_paras["blurring"])
        )
        distorted_image = image.filter(ImageFilter.GaussianBlur(kernel_size))

    elif distortion_type == "noise":
        std = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["noise"])
        )
        image = to_tensor([image], norm_type=None)
        noise = torch.randn(image.size()) * std
        distorted_image = to_pil((image + noise).clamp(0, 1), norm_type=None)[0]

    elif distortion_type == "compression":
        quality = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["compression"])
        )
        quality = int(quality)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        distorted_image = Image.open(buffered)
    elif distortion_type == "reversed":
        steps = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["reversed"])
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        distorted_image=ddim_inversion(image_str, steps)
    elif distortion_type == "elastic": 
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["elastic"])
        )
        transform =v2.ElasticTransform(alpha=scale, sigma=0.02)
        distorted_image = transform(image)
    elif distortion_type == "togray": 
        distorted_image = F.rgb_to_grayscale(image)
    elif distortion_type=="horizontal_flip":
        distorted_image = F.hflip(image)
    elif distortion_type=="vertical_flip":
        distorted_image = F.vflip(image)
    elif distortion_type=="randomcrop":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["resizedcrop"])
        )
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=(scale, scale), ratio=(1, 1)
        )
        distorted_image = F.crop(image, i, j, h, w)
    elif distortion_type=="invert":
        distorted_image = F.invert(image)
    else:
        assert False

    return distorted_image


def process_images_in_directory(
    input_dir,
    output_dir_base,
    distortion_type,
    strength=None,
    distortion_seed=0,
    same_operation=False,
    relative_strength=True,
):
    print("input_dir",input_dir)
    # Create the output directory with the specified name
    if relative_strength:
        temp_strength = relative_strength_to_absolute(strength, distortion_type)
    output_dir = os.path.join(output_dir_base,f"{distortion_type}_{temp_strength}")
    # output_dir = f"{output_dir_base}_{distortion_type}_{temp_strength}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        print(filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read the image
            image = Image.open(input_path)

            # Apply distortion
            distorted_image = apply_distortion(
                [image],
                distortion_type,
                strength=strength,
                distortion_seed=distortion_seed,
                same_operation=same_operation,
                relative_strength=relative_strength,
                return_image=True,image_str=input_path
            )[0]

            # Save the distorted image
            distorted_image.save(output_path)




def load_image(imgname, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return T.ToTensor()(pil_img)[None, ...]  


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


@torch.no_grad()
def ddim_inversion(imgname, num_steps: int = 50) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base',
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype).to(device)
    vae = pipe.vae
    input_img = load_image(imgname).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)
    inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)
    pipe.scheduler = DDIMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
    image = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                    num_inference_steps=num_steps, latents=inv_latents)
    return Image.open(image)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply distortions to images in a directory.")
    parser.add_argument("--input_dir",required=True, default="" ,type=str, help="Directory containing the input images.")
    parser.add_argument("--output_dir_base",required=True, default="",type=str, help="Base directory for saving output images.")
    parser.add_argument("--distortion_type", type=str, choices=list(distortion_strength_paras.keys()), help="Type of distortion to apply.")
    parser.add_argument("--strength", type=float, default=0.5, help="Strength of the distortion (optional).")
    parser.add_argument("--distortion_seed", type=int, default=0, help="Seed for random distortion (optional).")
    parser.add_argument("--same_operation", action="store_true", help="Apply the same distortion to all images (optional).")
    parser.add_argument("--relative_strength", action="store_true", help="Use relative strength for distortion (optional).")

    args = parser.parse_args()

    process_images_in_directory(
        args.input_dir,
        args.output_dir_base,
        args.distortion_type,
        strength=args.strength,
        distortion_seed=args.distortion_seed,
        same_operation=args.same_operation,
        relative_strength=args.relative_strength,
    )
