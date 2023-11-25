---
title: DMs Sampler之DDPM
date: 2023-10-23 23:26:19
updated: 2023-10-21 01:13:50
categories: Diffusion Models
mathjax: true
aside: false
description: 本文是Diffusion Model中DDPM采样方法学习。笔者将从直观加噪角度、VAE角度和Bayes角度解释DDPM采样方法。
---

<!--more-->

DDPM采样方法由Jonathan Ho在《Denoising Diffusion Probabilistic Models》中提出。

## 直观加噪

### 加噪过程

对于一张图像$x_0$，我们可以对其添加噪声。在t时刻，添加噪声的过程如下：
$$
\begin{align}
x_t=\alpha_t x_{t-1}+\beta_t \varepsilon_t, \quad\varepsilon_t\sim\mathcal{N}(0,I) \tag 1\\
\alpha_t^2+\beta_t^2=1, \quad \alpha,\beta>0,\beta \rightarrow0 \tag 2\\
\end{align}
$$
在输入$x_{t-1}$和噪声$\varepsilon_t$进行$\alpha_t$和$\beta_t$的加权。至于权重系数为什么不是$\alpha_t+\beta_t=1$的原因是$\alpha_t^2+\beta_t^2=1$便于后续推导。$\beta\rightarrow0$是因为我们希望添加很小的噪声。我们可以将公式（1）继续展开，过程如下：
$$
\begin{align}
x_t = \,& \alpha_t x_{t-1}+\beta_t \varepsilon_t \notag\\
    = \,& \alpha_t(\alpha_{t-1}x_{t-2}+\beta_{t-1}\varepsilon_{t-1})+\beta_t \varepsilon_t \notag\\
    = \,& \alpha_t(\alpha_{t-1}(\alpha_{t-2}x_{t-3}+\beta_{t-2} \varepsilon_{t-2})+\beta_{t-1}\varepsilon_{t-1})+\beta_t \varepsilon_t\notag\\
    = \,& \dots \notag\\
    = \,& (\alpha_t\cdots\alpha_1) x_0 + \underbrace{(\alpha_t\cdots\alpha_2)\beta_1 \varepsilon_1 + (\alpha_t\cdots\alpha_3)\beta_2 \varepsilon_2 + \cdots + \alpha_t\beta_{t-1} \varepsilon_{t-1} + \beta_t \varepsilon_t}_{\text{多个相互独立的正态噪声之和}} \tag{3}
\end{align}
$$
对于公式（3）的标准正态分布叠加，显然均值为0，方差推导如下：
$$
\begin{aligned}
\Sigma = \,& (\alpha_t\cdots\alpha_2)^2\beta_1^2 + (\alpha_t\cdots\alpha_3)^2\beta_2^2 + \cdots + \alpha_t^2\beta_{t-1}^2 + \beta_t^2 \quad \quad \text{利用公式2}\\
       = \,& (\alpha_t\cdots\alpha_2)^2(1-\alpha_1^2) + (\alpha_t\cdots\alpha_3)^2(1-\alpha_2^2) + \cdots + \alpha_t^2(1-\alpha_{t-1}^2) + (1-\alpha_t^2) \\
       = \,& 1- (\alpha_t\cdots\alpha_1)^2
\end{aligned}
$$
可以发现，多个相互独立的标准正态噪声叠加的结果为$\bar \varepsilon_t \sim \mathcal{N}(0,1- (\alpha_t\cdots\alpha_1)^2I)$，因此公式（3）可继续化简为：
$$
x_t = \underbrace{(\alpha_t\cdots\alpha_1)}_{\text{记为}\bar\alpha_t} x_0 + \underbrace{\sqrt{1 - (\alpha_t\cdots\alpha_1)^2}}_{\text{记为}\bar{\beta}_t} \bar\varepsilon_t, \quad \bar \varepsilon_t\sim\mathcal{N}(0,I) \tag{4}
$$
当$t=T\rightarrow \infty$时，$\alpha_t\cdots\alpha_1\rightarrow 0$。最终$x_T$完全变成噪声。

### 去噪过程

现在我们已经有了扩散过程的各个图像$x_T,x_{T-1}\cdots,x_{0}$。在逆向过程中，我们希望去除噪声，在$x_t$时恢复$x_{t-1}$图像。一个很自然的想法是用神经网络来拟合$x_{t-1}$。假设模型为$F_\theta(x_{t})$。没有任何依据预测$x_{t-1}$是很困难训练模型的，显然我们还需要时间$t$的输入。判断模型输出和$x_{t-1}$差异的最简单损失函数（欧式距离）为
$$\left\Vert x_{t-1}-F_\theta(x_t,t)\right\Vert^2 \tag 5$$
利用公式（1）我们可以得到
$$x_{t-1}=\frac{1}{\alpha_t}(x_t-\beta_t\varepsilon_t) \tag 6$$
仔细观察公式（6）的$x_{t-1}$表达式，我们发现模型不一定要预测$x_{t-1}$图像，而是可以预测从t-1到t添加的噪声。再利用公式（6）求得$x_{t-1}$。借着这个思路，模型可以改写成$\epsilon_\theta(x_t,t)$。于是，损失函数公式（5）变为
$$
\begin{align}
\left\Vert x_{t-1}-F_\theta(x_t,t)\right\Vert^2= & \left\Vert \frac{1}{\alpha_t}(x_t-\beta_t\varepsilon_t) - \frac{1}{\alpha_t}(x_t-\beta_t\epsilon_\theta(x_t,t)) \right\Vert^2 \notag \\
\propto &\left\Vert \varepsilon_t - \epsilon_\theta(x_t,t) \right\Vert^2 \notag \\
\propto &\left\Vert \varepsilon_t - \epsilon_\theta(\alpha_t x_{t-1} + \beta_t\varepsilon_t,t) \right\Vert^2 \notag \\
\propto &\left\Vert \varepsilon_t - \epsilon_\theta(\alpha_t (\bar \alpha_{t-1} x_0 + \bar \beta_{t-1} \bar \varepsilon_{t-1}) + \beta_t\varepsilon_t,t) \right\Vert^2 \notag \\
\propto &\left\Vert \varepsilon_t - \epsilon_\theta(\bar \alpha_{t} x_0 + \alpha_t \bar \beta_{t-1} \bar \varepsilon_{t-1} + \beta_t\varepsilon_t,t) \right\Vert^2 \tag7 \\
\end{align}
$$
仔细看上述公式，是先选择对$x_{t}$展开，再对展开的$x_{t-1}$展开，原因是：直接对$x_t$展开得到的$(\bar \alpha_{t} x_0 + \bar \beta_{t} \bar \varepsilon_{t}$，其中$\bar \varepsilon_{t}$和公式（7）中的$\varepsilon_t$不是**相互独立的，不能完全独立的采样**$\varepsilon_t$，**同时在降低方差中，不能使用正态分布叠加性质**。

### 降低方差

在公式（7）中，我们需要采样4个随机变量：

- 从训练集中采样一个$x_0$。
- 从标准正态分布中采样$\bar \varepsilon_{t-1}$和$\varepsilon_t$。
- 从1-T中采样一个时间t。
  **要采样的随机变量越多，就越难对损失函数做出准确的估计，即估计的损失值波动过大**。我们可以用以下方式消除一个随机变量（笔者感觉技巧性挺强的）。

由于$\bar \varepsilon_{t-1}$和$\varepsilon_t$相互独立，利用正态分布的叠加性可以得到：
$$
\alpha_t\bar{\beta}_{t-1}\bar{\varepsilon}_{t-1} + \beta_t \varepsilon_t \sim \bar{\beta}_t{\varepsilon}\,|\,\varepsilon \sim \mathcal{N}(0,I) \tag 8
$$
同时，我们构造另一个随机变量：
$$
\beta_t \bar \varepsilon_{t-1} - \alpha_t\bar{\beta}_{t-1} \varepsilon_t \sim \bar\beta_t\omega\,|\, \omega\sim \mathcal{N}(0,I) \tag  9
$$

> 数学知识：协方差描述两个变量的变化是同向的还是反向的。如果两个变量独立，他们的相关系数一定为0，并且如果两个变量的相关系数不为0，两个独立变量一定不独立。如果两个变量的相关系数为0（变量不相关），两个不一定独立。因为相关系数只能描述线性相关性。虽然两个变量不相关，但是他们之间依然可能有非线性的关系。

可以发现，$\bar{\beta}_t{\varepsilon}$（设为X）和$\bar\beta_t\omega$（设为Y）相互独立。证明两随机变量独立的等价条件是协方差矩阵为零。验证如下（不用关心系数，因为均值都是0，方差都一样）(懒得写系数了，带系数的话，最终协方差也是0)：
$$
\begin{align}
& \, Cov[\bar\beta_t\varepsilon \cdot \bar\beta_t\omega^T] \notag \\
    = & \, \mathbb{E}(\bar\beta_t\varepsilon - 0 )(\bar\beta_t\omega^T - 0 )\notag \\
    = & \, \mathbb{E}[(\alpha_t\bar{\beta}_{t-1}\bar{\varepsilon}_{t-1} + \beta_t \varepsilon_t)(\beta_t \bar \varepsilon_{t-1}^T - \alpha_t\bar{\beta}_{t-1} \varepsilon_t^T)] \notag \\
    \rightarrow & \, \mathbb{E}[\bar \varepsilon_{t-1} \bar \varepsilon_{t-1}^T - \bar \varepsilon_{t-1}\varepsilon_t^T + \varepsilon_t \bar \varepsilon_{t-1}^T -  \varepsilon_t \varepsilon_t^T]\notag \\ 
    \rightarrow & \, \mathbb{E}[\bar \varepsilon_{t-1} \bar \varepsilon_{t-1}^T] - \mathbb{E}[\bar \varepsilon_{t-1}\varepsilon_t^T] + \mathbb{E}[\varepsilon_t \bar \varepsilon_{t-1}^T] - \mathbb{E}[\varepsilon_t \varepsilon_t^T] \notag \\
    \rightarrow & \, \underbrace{\mathbb{D}[\bar \varepsilon_{t-1}] - \mathbb{D}[\varepsilon_{t}]}_{相减为0} +\underbrace{\mathbb{E}^2[\bar \varepsilon_{t-1}]}_{均值为0} - \underbrace{\mathbb{E}^2[\varepsilon_{t}]}_{均值为0} \notag \\
    = & \, 0 \notag
\end{align}
$$

联立公式（8）（9），对$\bar \varepsilon_{t-1}$进行消元，得到$\varepsilon_t$如下：
$$
\varepsilon_t = \frac{(\beta_t \varepsilon - \alpha_t\bar\beta_t-1 \omega)\bar\beta_t}{\beta_t^2 + \alpha_t^2\bar\beta_{t-1}^2} = \frac{\beta_t \varepsilon - \alpha_t\bar\beta_{t-1} \omega}{\bar\beta_t} \tag{10}
$$
将公式（8）（10）带入公式（7）中可以得到（为了方便起见，我们抛去公式（7）的系数，将$\propto$改成$=$）：
$$
\begin{align} 
&\,\mathbb{E}_{\bar\varepsilon_{t-1}, \varepsilon_t\sim \mathcal{N}(0,I)}\left\Vert \varepsilon_t - \epsilon_\theta(\bar \alpha_{t} x_0 + \alpha_t \bar \beta_{t-1} \bar \varepsilon_{t-1} + \beta_t\varepsilon_t,t) \right\Vert^2 \notag\\ 
=&\,\mathbb{E}_{\omega, \varepsilon \sim \mathcal{N}(0, I)} \left\Vert \frac{\beta_t \varepsilon - \alpha_t\bar\beta_{t-1} \omega}{\bar\beta_t}  - \epsilon_\theta(\bar\alpha_t x_0 + \bar \beta_t \varepsilon, t)\right\Vert^2 \notag \\
=&\,\mathbb{E}_{\omega, \varepsilon \sim \mathcal{N}(0, I)}  \left\Vert \frac{\beta_t}{\bar \beta_t}\varepsilon - \epsilon_\theta(\bar \alpha_t x_0 + \bar \beta_t\varepsilon ,t) - \frac{\alpha_t \bar \beta_{t-1}}{\bar \beta_t}\omega \right\Vert^2 \tag{11}
\end{align}
$$

> $E(X^2)=D(X)+E^2(X)$。如果X和Y独立，那么$E[XY]=E[X]\cdot E[Y]$

我们关注与$\omega$相关项，可以发现对其平方求期望结果为常数，一次项求期望为0。（$\varepsilon^2$求期望不是0！！）。因此，公式（11）可以改写成：
$$
\begin{align}
    &\,\mathbb{E}_{\bar\varepsilon_{t-1}, \varepsilon_t\sim \mathcal{N}(0,I)}\left\Vert  \frac{\beta_t}{\bar \beta_t}\varepsilon - \epsilon_\theta(\bar \alpha_t x_0 + \bar \beta_t\varepsilon ,t) \right\Vert^2 \tag{12}\\ 
\end{align}
$$
可以发现公式（7）的两个高斯随机变量变成了公式（12）的一个随机变量。可以发现，在上述推导过程中，笔者完全没有用原论文中的条件后验概率。

训练完$\epsilon_\theta(x_t,t)$后，我们一步步采样得到每时刻噪声并逐渐恢复图像。对于该生成过程，可以发现：它像串联式自回归生成，显然其生成速度为瓶颈。如果了解过PixelRNN/PixelCNN等自回归生成模型的可以知道，PixelCNN/PixelRNN通过上一时刻（位置）的像素决定该时刻（位置）的像素，最终的生成效果跟这个顺序紧密相关。这种按照位置顺序依次输出像素的方法**充分依赖经验设计（Inductive Bias）**。DDPM与该类方法的不同之处在于，重新定义了一个自回归方向，对于所有的像素来说则都是平权的、无偏的。DDPM减少对经验设计的依赖，从而提升了效果。

### 超参数选择

在超参数的设计上，设置
$$
\begin{align}
T=1000 \notag  \\
\alpha_t=\sqrt{1-\frac{0.02t}{T}} \notag \\
\end{align}
$$
选择单调递减的$\alpha_t$和较大的T原因如下：

- 我们知道欧氏距离并不是图像真实度的一个很好的度量，除非是输入和输出两张图片非常接近时，用欧式距离才会得到比较清晰的结果。
- 从扩散角度看，当$t$比较小时，$x_t$还比较接近真实图像，为了缩小$x_t$和$x_{t-1}$的差距，以便更适用欧氏距离公式（5），因此要用较大的$\alpha_t$；当$t$比较大时，$x_t$已经比较接近纯噪声了，噪声用欧式距离无妨，所以可以稍微增大$x_{t-1}$与$x_t$的差距，即可以用较小的$\alpha_t$。
- 从去噪角度看，我们选择较大的$T$才能使得去噪尽可能的彻底，使得输出与输入图像尽可能接近，用欧式距离衡量更为合适。

## VAE角度

我们先回顾下VAE(Variational Autoencoder)方法：
VAE由编码过程和生成过程组成，约定$x \rightarrow z$为编码过程，$z \rightarrow x$为生成过程。$p(z|x)$为编码分布，$p(z)$为先验分布，$q(x|z)$为生成分布。我们希望$p(x,z)$与$q(x,z)$尽可能接近，选择使用KL散度进行衡量：
$$
\begin{align}
KL(p(x,z)|q(x,z)) = &\, \int p(z|x)\tilde{p}(x)ln\frac{p(z|x)\tilde{p}(x)}{q(z,x)}dxdz \notag \\
                    = &\, \int \tilde{p}(x)\left( \int p(z|x)ln\frac{p(z|x)\tilde{p}(x)}{q(x,z)}dz \right)dx \notag \\
                    = &\, \int \tilde{p}(x)\left( ln \, \tilde{p}(x)\int p(z|x)dz + \int p(z|x)ln\frac{p(z|x)}{q(x,z)}dz  \right)dx \notag \\
                    = &\, \int \underbrace{\tilde{p}(x)ln\,\tilde{p(x)}dx}_{常数}  + \mathbb{E}_{x \sim \tilde{p}(x)}\int p(z|x)ln\frac{p(z|x)}{q(x|z)q(z)}dz \notag \\
                    \Leftrightarrow &\, \mathbb{E}_{x \sim \tilde{p}(x)} \left[ \int -p(z|x)ln\,q(x|z)dz + \int p(z|x)ln\frac{p(z|x)}{q(z)}dz  \right] \notag \\
                    = &\, \mathbb{E}_{x \sim \tilde{p}(x)}\left[  -\underbrace{\mathbb{E}_{z \sim p(z|x)} ln(q(x|z))}_{生成过程} + \underbrace{\mathbb{E}_{z \sim p(z|x)}KL(p(z|x)|q(z))}_{编码过程}\right] \notag \\
\end{align}
$$

> 上式中第一项可以这么理解：它是在得到采样一个$p(z_1|x_1)$求$ln(q(\hat x|z_1))$的期望，实际上是为了让$\hat x$与$x_1$接近，在代码层面是求$x_1$与$\hat x$的mse损失，在公式层面是让$q(\hat x|z_1)$接近于1（更确定是$x_1$），那么-log就会最小，是损失函数的优化目标。

对于扩散模型，可以当成多层VAE模型，可以直接写出其优化目标：
$$
\begin{align}
KL(p(x_0,x_1,\dots,x_T)|q(x_T,x_{T-1},\dots,x_0)) = &\, \int p(x_0,x_1,\dots,x_T)ln\frac{p(x_0,x_1,\dots,x_T)}{q(x_T,x_{T-1},\dots,x_0)}dx_0dx_1\dots dx_T \notag \\
                                                  = &\, \int p(x_T|x_{T-1})p(x_{T-1}|x_{T-2})\cdots p(x_1|x_0)\tilde{p}(x_0) ln \frac{ p(x_T|x_{T-1})p(x_{T-1}|x_{t-2})\cdots p(x_1|x_0)\tilde{p}(x_0)}{ q(x_0|x_{1})\cdots q(x_{T-1}|x_{T})q(x_T)}dx_0dx_1\dots dx_T \notag \\
                                                  = &\, \int p(x_{T}|x_{T-1})p(x_{T-1}|x_{T-2})\cdots p(x_1|x_0)\tilde{p}(x_0)lnp(x_{T}|x_{T-1})p(x_{T-1}|x_{T-2})\cdots p(x_1|x_0)\tilde{p}(x_0)dx_0dx_1\dots dx_T \notag \\
                                                  &\,-\, \int p(x_{T}|x_{T-1})p(x_{T-1}|x_{T-2})\cdots p(x_1|x_0)\tilde{p}(x_0)ln q(x_0|x_{1})\cdots q(x_{T-1}|x_{T})q(x_T)dx_0dx_1\dots dx_T\notag \\
\end{align}
$$

对于扩散模型，第一项为常数，不考虑进入优化目标。我们对第二项继续分析，其中$q(x_T)$为标准正态分布。那么我们只要关心ln中的每一项$q(x_t|x_{t-1})$：
$$
\begin{align}
& \int p(x_{T}|x_{T-1})p(x_{T-1}|x_{T-2})\cdots p(x_1|x_0)\tilde{p}(x_0)ln q(x_{t-1}|x_{t})dx_0dx_1 \notag \\
= \,& \int \underbrace{p(x_{T}|x_{T-1})dx_{T}}_{1} \, \underbrace{p(x_{T-1}|x_{T-2})dx_{T-1}}_{1}\cdots p(x_t|x_{t-1})p(x_{t-1}|x_{t-2})\cdots p(x_{1}|x_{0})\tilde{p}(x_0)ln q(x_{t-1}|x_{t})dx_{t}\cdots dx_{0} \notag \\
= \,& \int p(x_{t}|x_{t-1})ln q(x_{t-1}|x_{t})dx_{t}dx_{t-1}\,p(x_{t-1},x_{t-2}\cdots x_{1}|x_{0})dx_{t-2}\cdots dx_{1}\,\tilde{p}(x_{0})dx_{0} \qquad \qquad 由于\tilde{p}(x_{0})未知，不积分掉\notag \\
= \, &\int \underbrace{p(x_{t}|x_{t-1})}_{x_{t}=\alpha_tx_{t-1}+\beta_t\varepsilon_{t}}\quad\underbrace{p(x_{t-1}|x_{0})}_{x_{t-1=\bar \alpha_{t-1}x_{0}+\bar\beta_{t-1}\bar \varepsilon_{t-1}}} \quad \tilde{p}(x_{0}) \quad \underbrace{ln \, q(x_{t-1}|x_{t})}_{\frac{1}{2\sigma_t^2}\left \Vert x_{t-1} - F_\theta(x_t,t) \right \Vert^2} \quad dx_{t}dx_{t-1}dx_{0} \qquad \qquad 又x_{t-1}=\frac{1}{\alpha_t}(x_t-\beta_t\varepsilon_t)\notag \\

= \,& \mathbb{E}_{\varepsilon_t,\bar \varepsilon_{t-1}\sim \mathcal{N}(0,I),x_0 \sim \bar p(x_0)}\frac{\alpha_t^2}{\beta_t^2}\left \Vert \varepsilon_t - \epsilon_\theta(\bar \alpha_t x_{0}+\alpha_t \bar\beta_{t-1}\bar \varepsilon_{t-1}+\beta_t\varepsilon_t,t) \right \Vert^2 \tag{13}
\end{align}
$$

公式（13）与公式（7）思想一致，后续同样通过降低方差方式进行操作。

值得注意的是，公式（13）的推导过程中，$q(x_{t-1}|x_{t}) \sim \mathcal{N}(x_{t-1};F_\theta(x_t,t),\sigma_t^2I)$，对于$\sigma_t^2$的选择，理论上，对于不同的数据集$\tilde{p}(x_0)$，对应不同的$\sigma_t^2$。
$$
\begin{align}
q(x_{t-1}|x_{t},x_{0}) = &\, \frac{p(x_t|x_{t-1},x_{0})p(x_{t-1}|x_{0})}{p(x_{t}|x_{0})} \notag \\
                       = &\, \frac{p(x_t|x_{t-1})p(x_{t-1}|x_{0})}{p(x_{t}|x_{0})} \qquad \qquad Markov过程\notag \\
                       关注指数，求均值和方差：&\,  \frac{\left \Vert x_t - \alpha_t x_{t-1} \right \Vert^2}{2\beta_t^2} + \frac{\left \Vert x_{t-1} - \bar\alpha_{t-1} x_0 \right \Vert^2}{2\bar\beta_{t-1}^2} - \frac{\left \Vert x_{t} - \bar\alpha_{t} x_0 \right \Vert^2}{2\bar\beta_{t}^2} \notag \\
                       = &\, \mathcal{N}(x_{t-1};\frac{\alpha_t \bar \beta_{t-1}^2}{\bar \beta^2_{t}}x_t+\frac{\bar \alpha_{t-1}\beta_t^2}{\bar \beta^2_{t}}x_{0},\frac{\bar \beta_{t-1}^2\beta_t^2}{\bar \beta^2_t}I)\qquad \qquad 关注x_{t-1}是二次和一次系数。 \notag \\
\end{align}
$$

- 情况一：若只有一个样本，不是一般的样本为0：

$$
\begin{align}
q(x_{t-1}|x_{t}) = &\, q(x_{t-1}|x_{t},x_{0})=\mathcal{N}(x_{t-1};\frac{\alpha_t \bar \beta_{t-1}^2}{\bar \beta^2_{t}}x_t,\frac{\bar \beta_{t-1}^2\beta_t^2}{\bar \beta^2_t}I) \notag \\
\sigma_t^2 = &\, \frac{\bar \beta_{t-1}^2\beta_t^2}{\bar \beta^2_t} \notag \\
\end{align}
$$

- 情况二：若$\tilde{p}(x_0)=\mathcal{N}(x_0;0,I)$。由于$x_{t}=\bar \alpha_t x_0 +\bar \beta_t \bar \varepsilon_t$，根据正态分布叠加行，$x_t$也是标准正态分布：

$$
\begin{align}
q(x_{t-1}|x_{t}) = &\, \frac{p(x_{t}|x_{t-1})p(x_{t-1})}{p(x_{t})} \notag \\
                 观察指数：&\, \frac{\left \Vert x_t - \alpha_t x_{t-1} \right \Vert^2}{2\beta_t^2} + \frac{\left \Vert x_{t-1} \right \Vert^2}{2} - \frac{\left \Vert x_{t} \right \Vert^2}{2} \notag \\
                 = &\, \frac{x_t^2+\alpha_t^2x^2_{t-1}-2\alpha_t x_t x_{t-1} + \beta_t^2 x_{t-1}^2 - \beta_t^2 x_t^2}{2\beta_t^2} \notag \\
                 = &\,\frac{\alpha_t^2 x_t^2 -2\alpha_t x_t x_{t-1} + x_{t-1}^2}{2\beta_t^2} \notag \\
                 = &\, \frac{\left \Vert x_{t-1}-\alpha_t x_{t-1} \right \Vert^2}{2\beta_t^2} \notag \\
                 \sim &\, \mathcal{N}(x_{t-1};\alpha_t x_{t-1},\beta_t^2I) \notag \\
                 \sigma_t^2 = &\, \beta_t^2 \notag \\
\end{align}
$$




参考：

1. [生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼](https://kexue.fm/archives/9119)

2. [生成扩散模型漫谈（二）：DDPM = 自回归式VAE](https://kexue.fm/archives/9152)

3. [生成扩散模型漫谈（三）：DDPM = 贝叶斯 + 去噪](https://kexue.fm/archives/9164)