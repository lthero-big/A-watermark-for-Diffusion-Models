# A-watermark-for-Diffusion-Models



> 将水印消息嵌入在初始噪声矩阵中，最大支持256bit水印消息
>
> 在水印图像无损失情况下，水印消息比特提取正确率约85%



## 使用教程

### 生成水印图像

1. 下载并确保原始的[Stable Diffusion项目](https://github.com/Stability-AI/stablediffusion)可以生成图像
2. 将本项目中**txt2img.py**放在Stable Diffusion项目的scripts目录下，**替换掉原txt2img.py**
3. 运行下面的命令，即可生成水印图像

```shell
python scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" \
--ckpt ../ckpt/v2-1_512-ema-pruned.ckpt \
--config ./configs/stable-diffusion/v2-inference.yaml \
--H 512 --W 512  \
--device cuda \
--n_samples 2 \
--key_hex "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7" \
--nonce_hex "" \
--message "lthero"
```



#### 参数解释

* --ckpt：SD的模型文件
* --config：与SD的模型文件配套的config文件
* --n_samples: 表示生成的批次，每批次固定生成3张
* --key_hex：密钥Key（32字节，256位）
  * 使用十六进制作为输入，用于将message进行加密（使用ChaCha20加密算法）
  * ChaCha20的核心是一个伪随机函数，它基于输入的密钥Key（256位），一个随机数（nonce，通常是64位或96位），和一个初始的计数器值**生成一个伪随机字节序列**。这个伪随机字节序列然后与明文或密文进行XOR操作，从而实现加密或解密。
* --nonce_hex：随机数nonce（16字节）
  * 用于将message进行加密
  * nonce_hex可以不输入，如果不输入，则nonce_hex默认使用key_hex中间16字节
* --message: 嵌入的水印消息，最大支持256bit（32字节），超过此长度会被截断，不足会被补充

> * key_hex和nonce_hex可以都不输入，此时会自动生成随机32字节的key_hex和随机16字节nonce_hex
> * message也可以留空，会自动生成256bit（32字节）的随机内容

以上参数都会被保存在**info_data.txt**中（在Stable Diffusion项目的根目录下）



------



### 提取水印消息

#### 方式1

1. 在extricate.py修改里面的参数
2. 再用命令`python extricate.py`运行extricate.py即可，

#### 方式2

```shell
python extricate.py --orig_image_path "path_to_image.png" \
--key_hex "xxxxxxxxxx" \
--original_message_hex "xxxxxxxxxxxxx" \
--num_inference_steps 100
```

#### 参数解释

* orig_image_path：水印图像路径
* key_hex：同上，被保留在info_data.txt中
* nonce_hex：同上，被保留在info_data.txt中
* original_message_hex：输入的消息会被转成十六进制，被保留在info_data.txt中
* num_inference_steps：推理步数，默认为100步，适当上调可以提高比特正确率



运行extricate.py后，会输出：

1. 原消息的二进制表示
2. 提取消息的二进制表示
3. 两者的位比特正确率

```shell
Recovered message :
01101100011001000110000011100111010
 
Original message:
01101100011101000110100001100101011
 
Bit accuracy:  0.8515625
```







------

## ChaCha20加密相关

ChaCha20是一种流加密算法，由Daniel J. Bernstein设计。它是Salsa20算法的改进版，提供了高速、高安全性的加密能力。ChaCha20在多个安全协议和标准中被采用，包括用于TLS和SSH。下面是ChaCha20加密和解密的基本方法介绍：

### 加密和解密过程

ChaCha20的核心是一个伪随机函数，它基于输入的密钥Key（256位），一个随机数（nonce，通常是64位或96位），和一个初始的计数器值**生成一个伪随机字节序列**。这个伪随机字节序列然后与明文或密文进行XOR操作，从而实现加密或解密。

1. **初始化状态**：ChaCha20的状态是一个16个32位字（words）的数组，它包括：
   - 常数值（4个字）
   - 密钥**Key**（8个字，256位）
   - 计数器（1个字，用于确保每个块的唯一性）
   - 随机数**nonce**（3个字，96位，或2个字，64位）

2. **伪随机字节序列生成**：状态数组经过多轮（通常是20轮）的混合和置换操作，产生一个伪随机输出。每轮包括一系列的四字操作，如四字加法、异或和位移。

3. **加密/解密**：生成的**伪随机字节序列**与**明文或密文进行XOR操作**，输出加密后的密文或解密后的明文。

### 加密和解密方法

ChaCha20的加密和解密操作是相同的，因为它们都是基于XOR操作。这意味着加密和解密使用相同的函数，关键是确保使用相同的密钥、随机数（nonce）和计数器值。

```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def chacha20_encrypt_decrypt(key, nonce, data):
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor_decryptor = cipher.encryptor()  # 对于解密，也可以调用 cipher.decryptor()
    
    return encryptor_decryptor.update(data) + encryptor_decryptor.finalize()
```

**注意**：在实际应用中，确保密钥(Key)和随机数(nonce)的安全性至关重要，因为它们直接影响加密的安全性。随机数(nonce)不应重复使用同一密钥，以防止重放攻击和其他安全漏洞。



-----------

