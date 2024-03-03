# A-watermark-for-Diffusion-Models



> 将水印消息嵌入在初始噪声矩阵中，最大支持256bit水印消息
>
> 在水印图像无损失情况下，水印消息比特提取正确率约85%



## 使用教程

### 生成水印图像

1. 下载并确保原始的Stable Diffusion项目可以运行
2. 将txt2img.py放在Stable Diffusion目录的scripts下，**替换掉原txt2img.py**
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



### 参数解释

* --ckpt：SD的模型文件
* --config：与SD的模型文件配套的config文件
* --n_samples: 表示生成的批次，每批次固定生成3张
* --key_hex：32字节，使用十六进制作为输入，用于将message进行加密（使用ChaCha20加密算法）
* --nonce_hex：16字节，用于将message进行加密，增强加密性
  * nonce_hex可以不输入，如果不输入，则nonce_hex默认使用key_hex中间16字节
* --message: 想要嵌入的消息，最大支持256bit（32字节），超过此长度会被截断

> * key_hex和nonce_hex可以都不输入，此时会自动生成随机32字节的key_hex和随机16字节nonce_hex
> * message也可以留空，会自动生成256bit（32字节）的随机内容

以上参数都会被保存在info_data.txt中



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

* orig_image_path 图像路径
* key_hex：同上，被保留在info_data.txt中
* nonce_hex：同上，被保留在info_data.txt中
* original_message_hex：输入的消息会被转成十六进制，被保留在info_data.txt中
* num_inference_steps：推理步数，默认为100步，适当上调可以提高比特正确率





