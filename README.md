# A watermark for Diffusion Models



<p align="center"><a href="./README_en.md">English</a> | 中文</p>

 

> [!NOTE]
> This is an **unofficial** implementation of the Paper by Kejiang Chen et.al. on **Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models**

 

## 特性

- [x] 在**水印图像**无损失情况下，水印消息提取**正确率100%** :tada:
- [x] 对于多种不同的高强度失真攻击，拥有极好的鲁棒性；如**JPEG压缩QF=10，平均正确率90%** :+1:
- [x] 支持Stable Diffusion不同版本：v1-4 , v2-0 ,v2-1 :tada:
- [x] 支持**命令行SD**和**可视化SD-webui** :+1:
- [x] 无需额外训练，仅对初始噪声矩阵进行修改，对图像质量几乎无影响 :sparkles:
- [x] 即插即用，插件化使用方式 :heavy_check_mark:

-----------
## 生成图像展示

<div align=center>
<img src="images/without_gs_watermark.png" width="256" height="256"><img src="images/with_gs_watermark.png" width="256" height="256"/>
</div>
<p align="center">左侧是无水印图像，右侧是有水印图像</p>

<div align=center>
<img src="images/Attacks.png"/>
    <p align="center">受失真攻击后的图像</p>
</div>
<div align=center>
<img src="images/Results.png"/>
    <p align="center">比特正确率结果</p>
</div>



## 【命令行】使用教程

### 生成水印图像

1. 下载并确保原始的[Stable Diffusion项目](https://github.com/Stability-AI/stablediffusion)可以生成图像
1. 将本项目中的gs_insert.py放在Stable Diffusion目录下的scripts目录中
1. 对txt2img.py添加一部分代码

在txt2img.py代码中`parse_args()`添加下面的代码

```python
parser.add_argument(
    "--message",
    type=str,
    default="",
    help="watermark message",
)
parser.add_argument(
    "--key_hex",
    type=str,
    default="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7",
    help="key_hex",
)
parser.add_argument(
    "--nonce_hex",
    type=str,
    default="05072fd1c2265f6f2e2a4080a2bfbdd8",
    help="nonce_hex",
)
```

最终效果如下

```python
parser.add_argument(
    "--bf16",
    action='store_true',
    help="Use bfloat16",
)
# 在这里添加新代码
parser.add_argument(
    "--message",
    type=str,
    default="",
    help="watermark message",
)
parser.add_argument(
    "--key_hex",
    type=str,
    default="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7",
    help="key_hex",
)
parser.add_argument(
    "--nonce_hex",
    type=str,
    default="05072fd1c2265f6f2e2a4080a2bfbdd8",
    help="nonce_hex",
)
# 以下为原代码
opt = parser.parse_args()
return opt
```



在txt2img.py代码中`for n in trange(opt.n_iter, *desc*="Sampling"):`的首行添加下面的代码

```python
from gs_insert import gs_watermark_init_noise
Z_s_T_arrays = [gs_watermark_init_noise(opt,opt.message) for _ in range(opt.n_samples)]
start_code = torch.stack([torch.tensor(Z_s_T_array).float() for Z_s_T_array in Z_s_T_arrays]).to(device)
```

最终效果为

```python
for n in trange(opt.n_iter, desc="Sampling"):
    # 在这里添加了三行
    from gs_insert import gs_watermark_init_noise
    Z_s_T_arrays = [gs_watermark_init_noise(opt,opt.message) for _ in range(opt.n_samples)]
    start_code = torch.stack([torch.tensor(Z_s_T_array).float() for Z_s_T_array in Z_s_T_arrays]).to(device)
    # 下面的保持原代码
    for prompts in tqdm(data, desc="data"):
        uc = None
        if opt.scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
```



3. 运行下面的命令，即可生成水印图像

```shell
python scripts/txt2img.py \
--prompt "a professional photograph of an astronaut riding a horse" \
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

* --ckpt：Stable Diffusion的[模型文件](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main)

* --config：Stable Diffusion[配套的config文件](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion)

* --n_samples: 表示生成的批次，每批次固定生成3张

* --key_hex：密钥Key（32字节）
  * 使用**十六进制作为输入**，用于将message进行加密（使用**ChaCha20加密算法**）
  
* --nonce_hex：随机数nonce（16字节）
  * 使用**十六进制作为输入**，用于将message进行加密
  * nonce_hex可以不输入，nonce_hex**默认使用key_hex中间16字节**
  
* --message: 嵌入的水印消息，最大支持256bit（32字节），超过此长度会被截断，不足会补充

  

 

> [!important]
>
> * key_hex和nonce_hex可以**都不输入**，则自动生成随机32字节的key_hex和随机16字节nonce_hex
> * message也可以留空，会自动生成256bit（32字节）的随机内容
> * 以上参数都会被保存在**info_data.txt**中（在Stable Diffusion项目的根目录下）
>
> 
>
> * 如果是第一次运行，key_hex和nonce_hex**都不输入**，从而让代码自动生成；或使用下面的代码生成key_hex和nonce_hex

```python
import os
key = os.urandom(32)
nonce = os.urandom(16)
print(key.hex())
print(nonce.hex())
```





------



## 【WebUI】使用教程

> 基于[Stable Diffusion-WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)项目，本项目以脚本的形式实现嵌入水印的功能，操作简单

### 脚本安装

1. 把本项目`scripts`目录下的`GS_watermark_insert.py`文件放在Stable Diffusion-WebUI的`scripts`目录下面
2. 随后，**重启webui**，在**txt2img和img2img**栏目最下方的脚本选项中可以找到“**GS_watermark_insert**”

<div align=center>
<img src="images/image-script_in_webui.png" >
</div>
<p align="center">在webui中的GS_watermark_insert脚本</p>

### 脚本提供的三个参数

* Key：需要输入**十六进制形式的32字节内容**
* Nonce：需要输入**十六进制形式的16字节内容**
* Message：内容不**超过32字节**（可以输入字符串）
  * 可以仅填写Key，将Nonce留空，会自动选择Nonce

* Key和Nonce都可以留空，此时**会自动生成Key和Nonce**

> [!important]
>
> 可在Stable Diffusion-WebUI的根目录下，可以找到info_data.txt，其记录着Key，Nonce，Message

### 图像生成

* 填写完成脚本提供的三个参数后，按正常的流程生成图像即可，生成的图像会带有水印



---



## 提取水印消息

#### 方式1

1. 在`extricate.py`修改里面的参数
2. 再用命令`python extricate.py`运行extricate.py即可，

#### 方式2

在命令中传入参数

```shell
python extricate.py 
--single_image_path "path to image"
--image_directory_path "directory toimage" \
--key_hex "xxxxxxxxxx" \
--original_message_hex "xxxxxxxxxxxxx" \
--num_inference_steps 50
--scheduler "DDIM"
--is_traverse_subdirectories 0
```

#### 参数解释

* single_image_path： **单张处理**，输入**单张待检测图像**的路径，如"/xxx/images/001.png"
* image_directory_path：**批量处理**，待检测图像的**目录路径**，如"/xxx/images"
  * 两种方式每次只能选择一种，另一种留空；
  * **如果都不为空，仅按目录路径处理**
  
* key_hex：需要输入**十六进制形式的32字节内容**
  * key_hex被**保留在info_data.txt**中

* nonce_hex：需要输入**十六进制形式的16字节内容**
  * nonce_hex被**保留在info_data.txt**中

* original_message_hex：当你生成图像时，**原始的消息被被自动转成十六进制**，并被**保留在info_data.txt**中
* num_inference_steps：逆向推理步数，默认为**50步**；
  * 不建议继续上调，如解码速度慢，可以适当下降到20步

* scheduler: 选择采样器，有"DPMs"和"DDIM"两种选择，默认使用DDIM
* is_traverse_subdirectories: 是否对子目录进行递归提取
  * 设置为0，则仅对目录下的所有图像处理
  * 设置为1，则仅对目录下的所有子目录中的图像处理（包含子目录的子目录等）


> [!caution]
>
> original_message_hex一定要输入十六进制的格式，严格按info_data.txt中输入即可

 

运行extricate.py后，会输出**图像名**与**Bit正确率**

```shell
v2-1_512_00098-3367722000JPEG_QF_75.jpg
Bit accuracy:  1.0
```

> [!note]
>
> 如果使用**批量处理方式**，会在输入的目录中产生一个result.txt文件，记录每张图像的结果
>
> 如果使用**递归处理方式**，image_directory_path下的**每个子目录会有result.txt文件**，并且image_directory_path下会有result.txt记录着**每个子目录中平均Bit正确率**







-----------

