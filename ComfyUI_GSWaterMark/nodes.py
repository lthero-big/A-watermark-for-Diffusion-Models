import torch
import os
import sys
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
import numpy as np
from scipy.stats import norm
import os
from datetime import datetime
import numpy as np
import torch
import comfy.model_management
import comfy.sample
import comfy.sampler_helpers
import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import latent_preview

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

MAX_RESOLUTION=8192

def choose_watermark_length(total_blocks_needed):
    if total_blocks_needed >= 1024 * 32:
        return 1024
    if total_blocks_needed >= 512 * 32:
        return 512
    if total_blocks_needed >= 256 * 32:
        return 256
    elif total_blocks_needed >= 128 * 32:
        return 128
    elif total_blocks_needed >= 64 * 32:
        return 64
    else:
        return 32

def gs_watermark_init_noise2(key_hex, nonce_hex, device, message, use_seed, randomSeed, width, height):
    if int(use_seed) == 1:
        rng = np.random.RandomState(seed=randomSeed)
    
    # Calculate initial noise vector dimensions
    width_blocks = width // 8
    height_blocks = height // 8
    total_blocks_needed = 4 * width_blocks * height_blocks  # Total blocks needed

    # Choose watermark length based on the total blocks needed
    watermark_length_bits = choose_watermark_length(total_blocks_needed)
    # print(watermark_length_bits)
    LengthOfMessage_bytes = watermark_length_bits // 8

    if message:
        message_bytes = str(message).encode()
        if len(message_bytes) < LengthOfMessage_bytes:
            padded_message = message_bytes + b'\x00' * (LengthOfMessage_bytes - len(message_bytes))
        else:
            padded_message = message_bytes[:LengthOfMessage_bytes]
        k = padded_message
    else:
        k = os.urandom(LengthOfMessage_bytes)

    # Calculate the number of repeats needed
    repeats = total_blocks_needed // watermark_length_bits  # Ensure we round down
    print("="*20)

    print(f"k {k}\nwatermark_length_bits {watermark_length_bits}\nrepeats {repeats}")

    s_d = k * repeats
    # print(s_d)
    print("="*20)

    if key_hex and nonce_hex:
        key = bytes.fromhex(key_hex)
        nonce = bytes.fromhex(nonce_hex)
    elif key_hex and not nonce_hex:
        key = bytes.fromhex(key_hex)
        nonce_hex = key_hex[16:48]
        nonce = bytes.fromhex(nonce_hex)
    else:
        key = os.urandom(32)
        nonce = os.urandom(16)

    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    m = encryptor.update(s_d) + encryptor.finalize()
    m_bits = ''.join(format(byte, '08b') for byte in m)

    l = 1
    index = 0
    Z_s_T_array = torch.zeros((4, height_blocks, width_blocks), dtype=torch.float32, device=device).cpu()

    for i in range(0, len(m_bits), l):
        window = m_bits[i:i + l]
        y = int(window, 2)

        if use_seed == 0:
            u = np.random.uniform(0, 1)
        else:
            u = rng.uniform(0, 1)
        z_s_T = norm.ppf((u + y) / 2**l)
        Z_s_T_array[index // (height_blocks * width_blocks), (index // width_blocks) % height_blocks, index % width_blocks] = z_s_T.item()
        index += 1

        if index >= 4 * height_blocks * width_blocks:
            break  # Ensure we don't exceed the array size
    else:
        print('Z_s_T_array is got normaly')
        print("="*20,f"Z_s_T_array.shape {Z_s_T_array.shape}","="*20)
    
    return Z_s_T_array

def gs_watermark_init_noise(key_hex, nonce_hex, device,message,use_seed,randomSeed,set64bit):
    if int(use_seed)==1:
        rng = np.random.RandomState(seed=randomSeed)  

    if int(set64bit)==1:
        LengthOfMessage_bytes=8
    else:
        LengthOfMessage_bytes=32

    if message:
        message_bytes = str(message).encode()
        if len(message_bytes) < LengthOfMessage_bytes:
            padded_message = message_bytes + b'\x00' * (LengthOfMessage_bytes - len(message_bytes))
        else:
            padded_message = message_bytes[:LengthOfMessage_bytes]
        k = padded_message
    else:
        k = os.urandom(LengthOfMessage_bytes)

    if int(set64bit)==1:
        k=k*4

    s_d = k * 64
    
    if key_hex and nonce_hex:
        key = bytes.fromhex(key_hex)
        nonce = bytes.fromhex(nonce_hex)
    elif key_hex and not nonce_hex:
        key = bytes.fromhex(key_hex)
        nonce_hex = key_hex[16:48]
        nonce = bytes.fromhex(nonce_hex)
    else:
        key = os.urandom(32)
        nonce = os.urandom(16)

    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    m = encryptor.update(s_d) + encryptor.finalize()
    m_bits = ''.join(format(byte, '08b') for byte in m)

    l = 1
    index = 0
    Z_s_T_array = torch.zeros((4, 64, 64), dtype=torch.float32, device=device).cpu()

    for i in range(0, len(m_bits), l):
        window = m_bits[i:i+l]
        y = int(window, 2)

        if use_seed==0:
            u = np.random.uniform(0, 1)
        else:
            u = rng.uniform(0, 1)
        z_s_T = norm.ppf((u + y) / 2**l)
        Z_s_T_array[index // (64*64), (index // 64) % 64, index % 64] = z_s_T.item()
        index += 1
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'info_data.txt', 'a') as f:
        f.write(f"Time: {current_time}\n")
        f.write(f'key: {key.hex()}\n')
        f.write(f'nonce: {nonce.hex()}\n')
        f.write(f'randomSeed: {randomSeed}\n')  
        f.write(f'set64bit: {set64bit}\n')
        f.write(f'message: {k.hex()}\n')
        f.write('----------------------\n')
    return Z_s_T_array


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, use_GS=False,GS_latent_noise=None):
    latent_image = latent["samples"]

    if use_GS:
        noise = GS_latent_noise["samples"]
    elif disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )


class GSKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_GS_noise": (["enable", "disable"], ),
                    "add_noise": (["disable","enable"], ),
                    "noise_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "GS_latent_noise": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "GSWatermark-lthero/sampling"

    def sample(self, model,add_GS_noise, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,GS_latent_noise,start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        
        use_GS = False
        if add_GS_noise == "enable":
            use_GS=True
        
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise, use_GS=use_GS,GS_latent_noise=GS_latent_noise)
       

class GSLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "use_seed": ("INT", {"default": 1, "min": 0, "max": 1}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
            "set64bit": ("INT", {"default": 1, "min": 0, "max": 1}),
            "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "key": ("STRING", {"default": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"}),
            "nonce": ("STRING", {"default": "05072fd1c2265f6f2e2a4080a2bfbdd8"}),
            "message": ("STRING", {"default": "lthero"}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }}
    RETURN_TYPES = ("LATENT","IMAGE")
    FUNCTION = "create_gs_latents"

    CATEGORY = "GSWatermark-lthero/latent/noise"

    
    def create_gs_latents(self, key,nonce,message, batch_size,use_seed,seed,width,height,set64bit):
        
        device = "cpu"
        # 512*512
        # Z_s_T_arrays = [gs_watermark_init_noise(key,nonce,device,message,use_seed,seed,set64bit) for _ in range(batch_size)]

        # any size
        Z_s_T_arrays = [gs_watermark_init_noise2(key,nonce,device,message,use_seed,seed,width=width,height=height) for _ in range(batch_size)]
        latent = torch.stack([Z_s_T_array.clone().detach().to(device).float() for Z_s_T_array in Z_s_T_arrays])

        return ({"samples": latent},latent[0])

  
NODE_CLASS_MAPPINGS = {
    "Lthero_GSLatent": GSLatent,
    "Lthero_GS_KSamplerAdvanced": GSKSamplerAdvanced,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lthero_GSLatent": "GS Latent Noise",
    "Lthero_GS_KSamplerAdvanced": "GS KSamplerAdvanced",
}
