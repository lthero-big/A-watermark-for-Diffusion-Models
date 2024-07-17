import modules.scripts as scripts
import modules.processing as processing
import gradio as gr
from modules.processing import process_images, slerp
from modules import devices, shared
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np
from scipy.stats import norm
import os
from datetime import datetime
import numpy as np
import modules.rng as rng
from modules.rng import ImageRNG

global_message = ""
global_key = ""
global_nonce = ""
global_use_treering = 0
global_use_randomSeed = 0
global_randomSeed=42
global_use_repeat=0


def init_gs_Z_s_T():
    global global_message, global_key, global_nonce, global_use_treering,global_randomSeed,global_use_randomSeed,global_use_repeat

    rng = np.random.RandomState(seed=global_randomSeed)  

    if int(global_use_repeat)==1:
        LengthOfMessage_bytes=8
    else:
        LengthOfMessage_bytes=32

    if global_message:
        message_bytes = str(global_message).encode()
        if len(message_bytes) < LengthOfMessage_bytes:
            padded_message = message_bytes + b'\x00' * (LengthOfMessage_bytes - len(message_bytes))
        else:
            padded_message = message_bytes[:LengthOfMessage_bytes]
        k = padded_message
    else:
        k = os.urandom(LengthOfMessage_bytes)

    if int(global_use_repeat)==1:
        k=k*4

    s_d = k * 64

    if global_key != "" and global_nonce != "":
        key = bytes.fromhex(global_key)
        nonce = bytes.fromhex(global_nonce)
    elif global_key != "" and global_nonce == "":
        key = bytes.fromhex(global_key)
        nonce_hex = global_key[16:48]
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
    Z_s_T_array = np.zeros((4, 64, 64))
    for i in range(0, len(m_bits), l):
        window = m_bits[i:i + l]
        y = int(window, 2)  
        if global_use_randomSeed==0:
            u = np.random.uniform(0, 1)
        else:
            u = rng.uniform(0, 1)
        z_s_T = norm.ppf((u + y) / 2 ** l)
        Z_s_T_array[index // (64 * 64), (index // 64) % 64, index % 64] = z_s_T
        index += 1

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'info_data.txt', 'a') as f:
        f.write(f"Time: {current_time}\n")
        f.write(f'key: {key.hex()}\n')
        f.write(f'nonce: {nonce.hex()}\n')
        f.write(f'randomSeed: {global_randomSeed}\n')  
        f.write(f'message: {k.hex()}\n')
        f.write('----------------------\n')

    return Z_s_T_array


def advanced_creator(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0,
                     p=None):
    global global_message, global_key, global_nonce,global_use_treering,global_randomSeed,global_use_randomSeed,global_use_repeat
    noise = torch.tensor(init_gs_Z_s_T()).float().to(shared.device)
    noise_with_new_dim = noise.unsqueeze(0)
    return noise_with_new_dim

def create_generator(seed):
    if shared.opts.randn_source == "NV":
        return rng_philox.Generator(seed)

    device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    generator = torch.Generator(device).manual_seed(int(seed))
    return generator

def randn_without_seed(shape, generator=None):
    if shared.opts.randn_source == "NV":
        return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)

    if shared.opts.randn_source == "CPU" or devices.device.type == 'mps':
        return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)

    return torch.randn(shape, device=devices.device, generator=generator)

class modified_ImageRNG:
    def __init__(self, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w

        self.generators = [create_generator(seed) for seed in seeds]

        self.is_first = True

    def first(self):
        global global_message, global_key, global_nonce,global_use_treering,global_randomSeed,global_use_randomSeed,global_use_repeat
        noise = torch.tensor(init_gs_Z_s_T()).float().to(shared.device)
        noise_with_new_dim = noise.unsqueeze(0)
        return noise_with_new_dim

    def next(self):
        if self.is_first:
            self.is_first = False
            return self.first()

        xs = []
        for generator in self.generators:
            x = randn_without_seed(self.shape, generator=generator)
            xs.append(x)

        return torch.stack(xs).to(shared.device)


def set_seed(seed=None):
    if seed is None or seed == -1:
        seed = np.random.randint(0, 2**32 - 1)
    return seed


class Script(scripts.Script):
    def title(self):
        return "GS_watermark_insert"

    def ui(self, is_img2img):
        global global_message, global_key, global_nonce,global_use_treering,global_randomSeed,global_use_randomSeed,global_use_repeat
        key_input = gr.Textbox(label='Input Key Here',value="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7")
        nonce_input = gr.Textbox(label='Input Nonce Here', value="05072fd1c2265f6f2e2a4080a2bfbdd8")
        message_input = gr.Textbox(label='Input Message Here',value="")
        use_repeat=gr.Textbox(label='1 means repeat message four times, 0 means not;Keep the length of message as 64bit and repeat it four times for eatch line.This can improve bit accuracy greatly',value="0")
        use_randomSeed_input = gr.Textbox(label='1 means use use_randomSeed, 0 means not',value="0")
        with gr.Row():
            seed_input = gr.Number(label="Seed",value="42")
            seed_button = gr.Button("Generate Random Seed")
        
        seed_button.click(fn=set_seed, inputs=None, outputs=seed_input)
        
        return [message_input, key_input, nonce_input,seed_input,use_randomSeed_input,use_repeat]

    def run(self, p, message, key, nonce,seed,use_randomSeed,use_repeat):
        real_creator = rng.ImageRNG
        print("===================run======================")
        try:
            rng.ImageRNG = modified_ImageRNG
            global global_message, global_key, global_nonce,global_randomSeed,global_use_randomSeed,global_use_repeat
            global_message = message
            global_key = key
            global_nonce = nonce
            global_randomSeed=int(set_seed(seed))
            global_use_randomSeed=int(use_randomSeed)
            global_use_repeat=int(use_repeat)
            print(global_key, global_nonce,global_message,global_randomSeed,global_use_randomSeed,global_use_repeat)
            return process_images(p)
        finally:
            rng.ImageRNG = modified_ImageRNG
