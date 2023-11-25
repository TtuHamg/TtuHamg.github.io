---
title: DMs Sampler之DDIM
date: 2023-11-25 19:29:56
updated: 2023-10-21 01:13:50
categories: Diffusion Models
mathjax: true
aside: false
description: 本文是Diffusion Model中DDPM采样方法学习。笔者将从直观加噪角度、VAE角度和Bayes角度解释DDPM采样方法。
---

在DDPM的去噪过程中，为得到$t-1$时刻的后验概率分布，使用了Bayes公式：
$$
p(x_{t-1}|x_{t},x_{0})=\frac{p(x_{t}|p_{t-1},x_{0})p(x_{t-1}|x_{0})}{p(x_{t}|x_{0})} \tag 1
$$
观察公式可以发现，第一项$p(x_{t}|p_{t-1},x_{0})$要求知道$x_{t-1}$才能计算，即需要前向过程一步步采样。**我们能否调过正项的一步步过程？如果能，$p(x_{t}|x_{t-1},x_{0})$怎么求？**

实际上，不一定要用Bayes公式求$p(x_{t-1}|x_{t},x_{0})$，可以通过以下方式（恒成立，且不涉及$p(x_{t}|x_{t-1})$）：
$$
\begin{align}
    \int p(x_{t-1}|x_{t},x_{0})\, p(x_{t}|x_{0}) dx_{t}&=\, p(x_{t-1}|x_{0}) \tag 2 \\
\end{align}
$$
由于DDPM中求出了$p(x_{t-1}|x_{t},x_{0})$为正态分布，此处我们同样可以假设$p(x_{t-1}|x_{t},x_{0}) \sim \mathcal{N}(x_{t-1};\kappa_{t}x_{t}+\lambda_{t}x_{0};\sigma_{t}^2I)$。

> DDPM中得到$p(x_{t-1}|x_{t},x_{0})$是正态分布是由Markov假设使得$p(x_{t}|x_{t-1},x_{0})=p(x_{t}|x_{t-1})$。但是DDIM没有要求Markov过程，是更一般的假设。

> 问：没有Markov过程，为什么还能假设$p(x_{t-1}|x_{t},x_{0})$是正态分布？\
答：不排除有有其他分布的解，但是假设是正态分布，通过其待定系数法一定有解，因为DDPM的$p(x_{t-1}|x_{t},x_{0})$就是正态分布。再强调一遍：**DDIM并没有假设Markov过程**。

$p(x_{t-1}|x_{t},x_{0})\, p(x_{t}|x_{0})$可认为是$(x_{t},x_{t-1})$的联合分布，通过对x_{t}积分从而求得边缘分布$p(x_{t-1}|x_{0})$。公式（2）实际存在三个未知数$\kappa_{t}, \lambda_{t}, \sigma_{t}$，我们便是需要通过公式（2）待定系数求得未知数。为了求值方便，我们假设DDIM的前向过程与DDPM一致，不同在于每一步加噪是从$x_0$直接得到t时刻图像，同时并不要求Markov过程。



