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

global_message = ""
global_key = ""
global_nonce = ""


def init_gs_Z_s_T():
    global global_message, global_key, global_nonce
    print("===================init_gs_Z_s_T======================")
    if global_message:
        # 将消息转换为字节串
        message_bytes = str(global_message).encode()
        # 确保编码后的消息为256位（32字节）
        if len(message_bytes) < 32:
            padded_message = message_bytes + b'\x00' * (32 - len(message_bytes))
        else:
            padded_message = message_bytes[:32]
        k = padded_message
    else:
        # 如果message为空，生成256bit的随机水印消息k
        k = os.urandom(32)

    # 扩散过程，复制64份
    s_d = k * 64

    # 使用ChaCha20加密
    # 默认使用传入参数的key_hex和nonce_hex
    if global_key != "" and global_nonce != "":
        # 将十六进制字符串转换为字节串
        key = bytes.fromhex(global_key)
        # 使用参数的nonce_hex
        nonce = bytes.fromhex(global_nonce)
    # nonce_hex可以省略，使用key_hex的中心16字节
    elif global_key != "" and global_nonce == "":
        # 将十六进制字符串转换为字节串
        key = bytes.fromhex(global_key)
        # 使用固定的nonce
        nonce_hex = global_key[16:48]
        # 将nonce_hex转换为字节
        nonce = bytes.fromhex(nonce_hex)
    else:
        key = os.urandom(32)
        nonce = os.urandom(16)

    # 加密
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    m = encryptor.update(s_d) + encryptor.finalize()
    # 将m转换为二进制形式,m服从均匀分布
    m_bits = ''.join(format(byte, '08b') for byte in m)

    # 窗口大小l，可以是除了1以外的其他值
    l = 1  # 例如，改变这里为需要的窗口大小

    index = 0
    Z_s_T_array = np.zeros((4, 64, 64))
    # 遍历m的二进制表示，根据窗口大小l进行切割
    for i in range(0, len(m_bits), l):
        window = m_bits[i:i + l]
        y = int(window, 2)  # 将窗口内的二进制序列转换为整数y

        # 生成随机u
        u = np.random.uniform(0, 1)
        # 计算z^s_T
        z_s_T = norm.ppf((u + y) / 2 ** l)
        Z_s_T_array[index // (64 * 64), (index // 64) % 64, index % 64] = z_s_T
        index += 1

    # 将数据写入文件
    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'info_data.txt', 'a') as f:
        f.write(f"Time: {current_time}\n")
        f.write(f'key: {key.hex()}\n')
        f.write(f'nonce: {nonce.hex()}\n')
        f.write(f'message: {k.hex()}\n')
        f.write('----------------------\n')

    return Z_s_T_array


def advanced_creator(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0,
                     p=None):
    global global_message, global_key, global_nonce
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
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""

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
        global global_message, global_key, global_nonce
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


#for newer webui >=1.6.0
import modules.rng as rng
from modules.rng import ImageRNG
class Script(scripts.Script):
    def title(self):
        return "GS_watermark_insert"

    def ui(self, is_img2img):
        global global_message, global_key, global_nonce
        key_input = gr.Textbox(label='Input Key Here',
                               value="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7")
        nonce_input = gr.Textbox(label='Input Nonce Here', value="05072fd1c2265f6f2e2a4080a2bfbdd8")
        message_input = gr.Textbox(label='Input Message Here',
                                   value="")
        return [message_input, key_input, nonce_input]

    def run(self, p, message, key, nonce):
        print("===================run======================")
        real_creator = rng.ImageRNG
        try:
            rng.ImageRNG = modified_ImageRNG
            global global_message, global_key, global_nonce
            global_message = message
            global_key = key
            global_nonce = nonce
            print(global_key, global_nonce,global_message)
            return process_images(p)
        finally:
            rng.ImageRNG = modified_ImageRNG