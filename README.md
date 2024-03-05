# A-watermark-for-Diffusion-Models

> [!NOTE]
> 在水印图像无损失情况下，水印消息比特提取**正确率约90%**
>
> 无需训练，仅对初始噪声矩阵进行修改（**初始噪声矩阵仍保持高斯分布**）

 

- [x] 支持Stable Diffusion v1-4 , v2-0 ,v2-1 :tada:
- [x] 支持**命令行SD**和**可视化SD-webui** :+1:

-----------




## 【命令行】使用教程

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
--scheduler "DPMs"
```

 



#### 参数解释

* --ckpt：Stable Diffusion的[模型文件](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main)
* --config：Stable Diffusion[配套的config文件](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion)
* --n_samples: 表示生成的批次，每批次固定生成3张
* --key_hex：密钥Key（32字节）
  * 使用**十六进制作为输入**，用于将message进行加密（使用**ChaCha20加密算法**）
* --nonce_hex：随机数nonce（16字节）
  * 使用**十六进制作为输入**，用于将message进行加密
  * nonce_hex可以不输入，如果不输入，则nonce_hex**默认使用key_hex中间16字节**
* --message: 嵌入的水印消息，最大支持256bit（32字节），超过此长度会被截断，不足会补充
* --scheduler: 选择采样器，有"DPMs"和"DDIM"两种选择，原论文使用DDIM，但DDIM还没测试成功

 

> [!important]
>
> * key_hex和nonce_hex可以**都不输入**，则自动生成随机32字节的key_hex和随机16字节nonce_hex
> * message也可以留空，会自动生成256bit（32字节）的随机内容
> * 以上参数都会被保存在**info_data.txt**中（在Stable Diffusion项目的根目录下）



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

* orig_image_path：水印图像的路径
* key_hex：使用**十六进制作为输入**，被保留在info_data.txt中
* nonce_hex：使用**十六进制作为输入**，被保留在info_data.txt中
* original_message_hex：输入的消息会**被转成十六进制**，被保留在info_data.txt中
* num_inference_steps：推理步数，默认为100步，适当上调可以提高比特正确率

> [!caution]
>
> original_message_hex一定要输入十六进制的格式，严格按info_data.txt中输入即可

 

运行extricate.py后，会输出如下内容

```shell
原消息的二进制表示
01101100011001000110000011100111010
 
提取消息的二进制表示
01101100011101000110100001100101011
 
两者的位比特正确率
Bit accuracy:  0.8515625
```



## 【可视化】使用教程

> 可视化是基于[Stable Diffusion-WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)项目，本项目以脚本的形式实现功能

### 脚本安装

1. 把本项目`scripts`目录下的GS_watermark_insert.py放在Stable Diffusion-WebUI的`scripts`目录下面
2. 随后，**重启webui**，在**txt2img和img2img**的最下方脚本中找到“**GS_watermark_insert**”

![image-20240303215705020](https://cdn.lthero.cn/post_images/course/ML/image-20240303215705020.png)

### 脚本提供的三个参数

* Key, Nonce, Message
* Key需要**32字节十六进制输入**，Nonce需要**16字节十六进制输入**
  * 可以仅填写Key，将Nonce留空
  * Key和Nonce都可以留空
* Message内容不超过32字节（可以输入字符串）

> [!important]
>
> 可在Stable Diffusion-WebUI的根目录下，找到info_data.txt记录着Key，Nonce，Message

### 图像生成

* 填写完成脚本提供的三个参数后，按正常的习惯生成图像即可，图像会自带水印
* 水印的提取方式同上述的命令行教程



------



## 附录

### ChaCha20加密相关

ChaCha20的核心是一个伪随机函数，它基于输入的密钥Key（256位），一个随机数（nonce，通常是64位或96位），和一个初始的计数器值**生成一个伪随机字节序列**。这个伪随机字节序列然后与明文或密文进行XOR操作，从而实现加密或解密。

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

