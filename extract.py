import torch
from inverse_stable_diffusion_gs import InversableStableDiffusionPipeline
import argparse
from PIL import Image
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
from scipy.stats import norm
from diffusers import DPMSolverMultistepScheduler,StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np
from scipy.stats import norm
from datetime import datetime
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline
from diffusers.utils import load_image

from typing import Union, Tuple, Optional
from torchvision import transforms as tvt


# credit to: https://github.com/shaibagon/diffusers_ddim_inversion
# credit to: https://github.com/cccntu/efficient-prompt-to-prompt/blob/main/ddim-inversion.ipynb
# credit to: https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb
# Used for exactract_latents_DDIM_inversion

def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension

# Used for exactract_latents_DDIM_inversion
def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


@torch.no_grad()
def exactract_latents_DDIM_inversion(args) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    inverse_scheduler = DDIMInverseScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id,
                                                   scheduler=inverse_scheduler,
                                                #    safety_checker=None,
                                                   safety_checker = disabled_safety_checker,
                                                   torch_dtype=dtype)
    pipe.to(device)
    vae = pipe.vae

    input_img = load_image(args.single_image_path).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=args.num_inference_steps, latents=latents)
    return inv_latents.cpu()

#  credit to: https://github.com/YuxinWenRick/tree-ring-watermark 


def exactract_latents_DDIM_inversion_official(args):
    
    pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()
    generator = torch.manual_seed(0)

    

    # raw_image = load_image(args.single_image_path).convert("RGB").resize((768, 768))
    raw_image = load_image(args.single_image_path).convert("RGB").resize((512, 512))
    source_prompt = ""
    target_prompt = ""
    inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image, generator=generator).latents
    return inv_latents

def exactract_latents(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 
    if args.scheduler=="DPMs":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    elif args.scheduler=="DDIM":
        # not official but it works
        return exactract_latents_DDIM_inversion(args)
        
        # official's way doesn't work
        # return exactract_latents_DDIM_inversion_official(args)
    
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        variant='fp16',
        # revision='fp16',
        safety_checker = disabled_safety_checker
        )
    pipe = pipe.to(device)
    tester_prompt = ''
    # assume at the detection time, the original prompt is unknown，原论文中使用的就是空prompt
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    # 格式PIL.Image.Image
    orig_image = Image.open(args.single_image_path).convert("RGB")
    # 
    img_w = transform_img(orig_image).unsqueeze(0).to(torch.float16).to(device)
    # 
    image_latents = pipe.get_image_latents(img_w, sample=False)
    # 获得逆向后的噪声矩阵，这里与推理步数有关，越大会变好
    reversed_latents = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inference_steps,
        )

    return reversed_latents.cpu()



def recover_exactracted_message(reversed_latents, key, nonce, l=1):
    # 初始化Cipher
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    
    # 从reversed_latents中恢复m的二进制表示
    reconstructed_m_bits = []
    for z_s_T_value in np.nditer(reversed_latents):
        # 使用norm.ppf的逆操作恢复原始的y值
        y_reconstructed = norm.cdf(z_s_T_value) * 2**l
        reconstructed_m_bits.append(int(y_reconstructed))

    # 将二进制位转换为字节
    m_reconstructed_bytes = bytes(int(''.join(str(bit) for bit in reconstructed_m_bits[i:i+8]), 2) for i in range(0, len(reconstructed_m_bits), 8))

    # 解密m以恢复扩散过程前的数据s_d
    s_d_reconstructed = decryptor.update(m_reconstructed_bytes) + decryptor.finalize()
    
    # 使用vote机制
    # 解密得到的字节串转换为二进制表示
    bits_list = ['{:08b}'.format(byte) for byte in s_d_reconstructed]
    # 将二进制字符串合并为一个长字符串
    all_bits = ''.join(bits_list)
    # 分割为64个段，每段代表s_d中的一行
    segments = [all_bits[i:i+256] for i in range(0, len(all_bits), 256)]

    # 投票机制确定每个位
    reconstructed_message_bin = ''
    for i in range(256):
        # 计算每个位在所有行中为'1'的次数
        count_1 = sum(segment[i] == '1' for segment in segments)
        # 如果超过一半则该位为'1'，否则为'0'
        reconstructed_message_bin += '1' if count_1 > len(segments) / 2 else '0'

    return reconstructed_message_bin



def calculate_bit_accuracy(original_message_hex, extracted_message_bin):
    # Convert the original hex message to binary
    original_message_bin = bin(int(original_message_hex, 16))[2:].zfill(len(original_message_hex) * 4)
    # Ensure both binary strings are of the same length
    min_length = min(len(original_message_bin), len(extracted_message_bin))
    original_message_bin = original_message_bin[:min_length]
    extracted_message_bin = extracted_message_bin[:min_length]
    # Calculate bit accuracy
    matching_bits = sum(1 for x, y in zip(original_message_bin, extracted_message_bin) if x == y)
    bit_accuracy = matching_bits / min_length
    return original_message_bin,bit_accuracy

def get_result_for_one_image(args):
    # Process each image
    reversed_latents = exactract_latents(args)
    # 
    extracted_message_bin = recover_exactracted_message(reversed_latents, args.key, args.nonce, args.l)
    # 
    original_message_bin, bit_accuracy = calculate_bit_accuracy(args.original_message_hex, extracted_message_bin)
    # print(f"{os.path.basename(args.single_image_path)}\nOriginal Message: {original_message_bin} \nExtracted Message: {extracted_message_bin}\nBit Accuracy: {bit_accuracy}\n")
    print(f"{os.path.basename(args.single_image_path)}, Bit Accuracy,{bit_accuracy}\n")
    return original_message_bin,extracted_message_bin,bit_accuracy

# old version
# def process_directory(args):
#     # Get all image files in the directory
#     image_files = glob.glob(os.path.join(args.image_directory_path, "*.png")) + glob.glob(os.path.join(args.image_directory_path, "*.jpg"))

#     with open(args.image_directory_path+"/"+"result.txt", "a") as result_file:
#         result_file.write("=" * 40 +"Batch Info"+"=" * 40+"\n")
#         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         result_file.write(f"Time: {current_time}\n")
#         result_file.write(f"key_hex:{args.key_hex}\n")
#         result_file.write(f"nonce_hex:{args.nonce_hex} \n")
#         result_file.write(f"original_message_hex:{args.original_message_hex} \n")
#         result_file.write(f"num_inference_steps:{args.num_inference_steps}\n")
#         result_file.write(f"scheduler:{args.scheduler}\n")
#         result_file.write("=" * 40 +"Batch Start"+"=" * 40+"\n")
#         for image_path in tqdm(image_files):
#             args.single_image_path = image_path
#             try:
#                 # Process each image
#                 result=get_result_for_one_image(args)
#                 result_file.write(f"{os.path.basename(image_path)}, Bit Accuracy, {result[2]}\n")
#                 # result_file.write(f"{os.path.basename(image_path)}: Original Message: {result[0]} , Extracted Message: {result[1]}, Bit Accuracy: {result[2]}\n")
#             except Exception as e:
#                 print(f"Error processing {image_path}: {e}\n")
#                 result_file.write(f"Error processing {image_path}: {e}\n")
#         result_file.write("=" * 40 +"Batch End"+"=" * 40+"\n\n")
    
import os
import glob
from datetime import datetime
from tqdm import tqdm

def process_directory(args):
    # 是否递归子目录
    if args.is_traverse_subdirectories:
        for root, dirs, files in os.walk(args.image_directory_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                process_single_directory(dir_path, args)
    else:
        process_single_directory(args.image_directory_path, args)

def process_single_directory(dir_path, args):
    # 获取所有图像文件
    image_files = glob.glob(os.path.join(dir_path, "*.png")) + glob.glob(os.path.join(dir_path, "*.jpg"))
    if not image_files:
        return

    total_bit_accuracy = 0
    processed_images = 0
    result_file_path = os.path.join(dir_path, "result.txt")

    with open(result_file_path, "a") as result_file:
        write_batch_info(result_file, args)

        for image_path in tqdm(image_files):
            args.single_image_path = image_path
            try:
                # 处理每张图片
                result = get_result_for_one_image(args)
                result_file.write(f"{os.path.basename(image_path)}, Bit Accuracy, {result[2]}\n")
                total_bit_accuracy += float(result[2])
                processed_images += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}\n")
                result_file.write(f"Error processing {image_path}: {e}\n")
        
        if processed_images > 0:
            average_bit_accuracy = total_bit_accuracy / processed_images
            result_file.write(f"Average Bit Accuracy, {average_bit_accuracy}\n\n")
            result_file.write("=" * 40 + "Batch End" + "=" * 40 + "\n")
            # 如果是子目录的子目录（例如`batch_01_withmark_addnoise`下的`JPEG_QF_10`目录），更新父目录的result.txt
            parent_dir = os.path.dirname(dir_path)
            with open(os.path.join(parent_dir, "result.txt"), "a") as parent_result_file:
                write_batch_info(parent_result_file, args)
                parent_result_file.write(f"{os.path.basename(dir_path)}, Average Bit Accuracy, {average_bit_accuracy}\n")
                parent_result_file.write("=" * 40 + "Batch End" + "=" * 40 + "\n")

def write_batch_info(result_file, args):
    result_file.write("=" * 40 + "Batch Info" + "=" * 40 + "\n")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_file.write(f"Time,{str(current_time)}\n")
    result_file.write(f"key_hex,{args.key_hex}\n")
    result_file.write(f"nonce_hex,{args.nonce_hex}\n")
    result_file.write(f"original_message_hex,{args.original_message_hex}\n")
    result_file.write(f"num_inference_steps,{args.num_inference_steps}\n")
    result_file.write(f"scheduler,{args.scheduler}\n")
    result_file.write("=" * 40 + "Batch Start" + "=" * 40 + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract watermark from a image')
    # 模型选择
    # stabilityai/stable-diffusion-2-1 is for 768x768
    # stabilityai/stable-diffusion-2-1-base is for 512x512
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    # /home/dongli911/.wan/Project/AIGC/stablediffusion/outputs/txt2img-samples/n2t
    parser.add_argument('--image_directory_path', default="/home/dongli911/.wan/Project/AIGC/stablediffusion/outputs/txt2img-samples/n2t/gs_batch_01_withmark_addnoise", 
    help='The path of directory containing images to process')
    # /home/dongli911/.wan/Project/AIGC/stablediffusion/outputs/txt2img-samples/need2test/v2-1_512_00642-1_DDIM_50s.png
    parser.add_argument('--single_image_path', default="")
    # , required=True
    parser.add_argument('--key_hex', default="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7", help='Hexadecimal key used for encryption')
    # nonce_hex=14192f43863f6bad1bf54b7832697389
    parser.add_argument('--nonce_hex', default="05072fd1c2265f6f2e2a4080a2bfbdd8", help='Hexadecimal nonce used for encryption, if empty will use part of the key')
    parser.add_argument('--original_message_hex', default="6c746865726f0000000000000000000000000000000000000000000000000000", 
    help='Hexadecimal representation of the original message for accuracy calculation')
    parser.add_argument('--num_inference_steps', default=50, type=int, help='Number of inference steps for the model')
    parser.add_argument('--scheduler', default="DDIM", help="Choose a scheduler between 'DPMs' and 'DDIM' to inverse the image")
    parser.add_argument('--is_traverse_subdirectories', default=1, help="Whether to traverse subdirectories recursively")
    parser.add_argument('--l', default=1, type=int, help="The size of slide windows for m")
    args = parser.parse_args()

    # 将十六进制字符串转换为字节串
    args.key = bytes.fromhex(args.key_hex)
    if args.nonce_hex!="":
        # 使用参数的nonce_hex
        args.nonce = bytes.fromhex(args.nonce_hex)
    else:
        # 使用固定的nonce, 将nonce_hex转换为字节
        args.nonce = bytes.fromhex(args.key_hex[16:48])
    
    # 批处理
    if args.image_directory_path!="":
        process_directory(args)
    # 单次处理
    elif args.single_image_path!="":
        get_result_for_one_image(args)
    else:
        print("Please set the argument 'image_directory_path' or 'single_image_path'")


    
