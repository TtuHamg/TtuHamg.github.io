---
title: Alchemy-Experience
date: 2023-12-23 13:48:32
categories: Diffusion Models
aside: false
description: 在闲暇之际，记录炼丹经验。
---
> 由于笔者刚步入生成领域，对于大模型的炼丹方法经验不足。在炼丹闲暇之余，将实操过程中的一些经验总结于此。

## 显卡
提到大模型，就不得不提最重要的工具——显卡。市面上比较常见的显卡包括2080Ti，3090，4090乃至A100，A800。不同显卡的差距在哪里，同一张卡的不同版本之间有什么区别，以下是笔者查到的相关数据：

|                          | A100 80G/40G PCIe |A800 80G/40G PCIe |4090 24G | 3090 24G | 2080Ti 12G |
| ------------------------ | ----------------- |----------------- |-------- | -------- | ---------- |
| FP32 Cuda Core(TFLOPS)   | 20                |19.5 |73       | 35       | 13         |
| BF16 Tensor Core(TFLOPS) | 312               |312 |330      | 71       | 26.8       |
| TF32 Tensor Core(TFLOPS) | 156               |156 |83       | 35       | 13.4       |
| 参考报价                  |108000        |87000 |12999    | 11999    |              
Notes:

1. TFLOPS（Floating-point operations per second）：每秒万亿次浮点运算。

2. BF16与FP16区别：前者用8bit 表示指数，7bit 表示小数；后者用5bit 表示指数，10bit 表示小数。

3. 不论CNN还是Transformer，绝大多数的浮点计算量都集中在矩阵乘法上面，而这部分的负载恰好能用 tensor core 运行。因此，尽管看上去3090的FP32 cuda core比A100大，但tensor core上A100远超于3090。

4. PCIe（peripheral component interconnect express）版指显卡插在主板的PCIe卡槽上，而SXM版是英伟达公司设计出来的，它的出现主要是为高性能计算和数据中心提高更强的计算能力和传输速度。SXM规格的一般用在英伟达的DGX服务器中，通过主板上集成的NVSwitch实现NVLink的连接，不需要通过主板上的PCIe进行通信，**其传输、通信速率快于PCIe版，在算力方面SXM版本是PCIe版本两倍**。

5. A800相比于A100，主要是将NVLink的传输速率由A100的600GB/s降至了400GB/s，其他参数与A100基本一致。这样做是为了遵守美国限制规则（限制出口芯片的I/O带宽传输速率大于或等于600 Gbyte/s）。

## 模型
对于大模型本身，从零训练一个生成模型是一件非常费卡的事情。根据hugging face社区给出的Stable Diffusion的训练条件，对于高校而言，实在是望尘莫及。

### Stable Diffusion

#### v1版本

根据Compvis提供的训练细节，v1版本分为了v1.1-v1.5五个迭代版本。硬件条件：32x8xA100 40G PCle。优化器：AdamW。梯度累加：2。Batch：32x8x2x4=2048。学习率：10000 steps warm up到0.0001，之后保持不变。v1训练了150000小时（24天）。

- v1.1在laion2B-en数据集上以256\*256分辨率训练237,000 steps，随后在laion-high-resolution数据集上以512\*512分辨率训练194,000 steps。前者是laion5B数据集的一个子集。laion5B数据集是从网页数据Common Crawl中筛选出来的图像-文本对数据集，它包含5.85B的图像-文本对，其中文本为英文的数据量为2.32B，这就是laion2B-en数据集。后者是从laion5B数据集中分辨率在1024*1024以上的170,000,000个文本对。

- 在v1.1的基础上，v1.2在laion-improved-aesthetics数据集上以512*512分辨率训练515,000 steps。该数据集为laion2B-en的子集，要求审美评分大于5，水印概率小于0.5。

- 在v1.2的基础上，v1.3同样在laion-improved-aesthetics数据集训练195,000 steps，并对文本条件进行10% dropout以提升classifier-free guidance采样。

- 在v1.2的基础上，v1.4在laion-aesthetics v2 5+数据集训练225,000 steps，并对文本条件进行10% dropout以提升classifier-free guidance采样。

## 炼丹经验
1. 如果服务器只有2080Ti，即使卡再多，也别妄想跑大模型。虽然batch_size或者显存的问题能够通过梯度累计（[Gradient Accumulations](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/train_util.py#L190)）或者多级多卡DDP训练解决。但是算力永远是过不去的坎。同样8卡相同batch_size训练[DiT-S/2](https://github.com/facebookresearch/DiT)，达到相同steps，2080Ti需要10天而A100只需要1天。

2. 自己写模型代码时，对于需要两个矩阵相乘的情况，千万不要初始化为0，不然可能会造成梯度一直为0无法更新。

3. 对于DDP多机多卡训练，pytorch推荐torch启动，而非torch.distributed.launch。
```python
# 指定服务器数量、每台服务器的显卡数量、当前服务器的编号、rank=0的服务器ip地址、指定端口号
torchrun --nnodes=2 --nproc_per_node=4  --node_rank=0 --master_addr='10.214.241.20' --master_port=2345 

dist.init_process_group("nccl")

rank = dist.get_rank() # global rank
device = rank % torch.cuda.device_count() #local rank
torch.cuda.set_device(device)
seed = args.global_seed * dist.get_world_size() + rank
torch.manual_seed(seed)

model = DDP(model.to(device), device_ids=[device])
```