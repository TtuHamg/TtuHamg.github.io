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