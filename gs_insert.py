from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np
from scipy.stats import norm
import os
from datetime import datetime

def gs_watermark_init_noise(opt,message=""):
    if message:
        # 将消息转换为字节串
        message_bytes = message.encode()
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
    if opt.key_hex!="" and opt.nonce_hex!="":
        # 将十六进制字符串转换为字节串
        key = bytes.fromhex(opt.key_hex)
        # 使用参数的nonce_hex
        nonce = bytes.fromhex(opt.nonce_hex)
    # nonce_hex可以省略，使用key_hex的中心16字节
    elif opt.key_hex!="" and opt.nonce_hex=="":
        # 将十六进制字符串转换为字节串
        key = bytes.fromhex(opt.key_hex)
        # 使用固定的nonce
        nonce_hex = opt.key_hex[16:48]
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
    # 初始化结果列表，用于存储每个窗口的处理结果
    results = []
    # 窗口大小l，可以是除了1以外的其他值
    l = 1  

    index=0
    Z_s_T_array = np.zeros((4, 64, 64))
    # 遍历m的二进制表示，根据窗口大小l进行切割
    for i in range(0, len(m_bits), l):
        window = m_bits[i:i+l]
        y = int(window, 2)  # 将窗口内的二进制序列转换为整数y
        # 生成随机u
        u = np.random.uniform(0, 1)
        # 计算z^s_T
        z_s_T = norm.ppf((u + y) / 2**l)
        Z_s_T_array[index // (64*64), (index // 64) % 64, index % 64] = z_s_T
        index+=1

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'info_data.txt', 'a') as f:
        f.write(f"Time: {current_time}\n")
        f.write(f'key: {key.hex()}\n')
        f.write(f'nonce: {nonce.hex()}\n')
        f.write(f'message: {k.hex()}\n')
        f.write('----------------------\n')
    return Z_s_T_array