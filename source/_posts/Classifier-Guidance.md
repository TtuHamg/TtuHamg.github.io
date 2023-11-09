---
title: DMs Sampler之Classifier Guidance
date: 2023-10-20 00:13:50
updated: 2023-10-21 01:13:50
categories: Diffusion Models
mathjax: true
aside: false
description: 本文是Diffusion Model中Classifier Guidance采样方法学习。

---

<!--more-->

classifier guidance采样方法由Prafulla Dhariwal在《Diffusion Models Beat GANs on Image Synthesis》中提出，其核心在于**使用分类器梯度信息指导模型以提高采样质量**。

我们希望训练好的DMs能够生成某一类别的图像。DDPM的做法是生成多张图像（1000张），其中存在50张该类别的图像。该方式存在效率太低问题。我们希望生成FID低（多样性高）的同时，同样能够可控生成。一种**直觉**的做法是使用**条件引导(y)**进行图像生成，即$q(x_{t-1}|x_{t},y)$。

在DDPM的推导中，我们用UNet模型求得后验概率$q(x_{t-1}|x_{t})$对前向扩散的$q(x_t|x_{t-1})$，实际上UNet模型是**预测前向过程加的噪音**。我们对$q(x_{t-1}|x_{t})$加上条件y，即$\hat q(x_{t-1}|x_{t},y)$，利用贝叶斯公式进行变换得到下式:
$$\hat q(x_{t-1}|x_t,y)=\frac{\hat q(x_{t-1}|x_t)\hat q(y|x_{t-1},x_{t} )}{\hat q(y|x_t)} \tag 1$$
公式（1）由$\hat q(x_{t-1}|y)=\frac{\hat q(y|x_{t-1})\hat q(x_{t-1})}{\hat q(y)})$加上条件$|x_t$得到。

我们的目标需要给出公式（1）的显示表达。其中**分母$\hat q(y|x_{t})$在去噪过程中$x_t$和$y$均已知，因此是个常量**。

此外，我们需要有/假设一些其他的已知条件：$\hat q(x_{t-1}|x_t,y)$的扩散过程和DDPM的正向过程保持一致，如下公式（2）。这样做的好处是能够用以训练好的DDPM模型，如同DDIM的思想，同时也是为了后续更好的推导。我们同样保持前向过程的Markov性质，如下公式4。
$$\hat q(x_t|x_{t-1},y)=q(x_t|x_{t-1}) \tag 2$$
$$\hat q(x_0)=q(x_0) \tag 3$$
$$\hat q(x_{1:T}|x_0,y)=\prod_{t=1}^{T}\hat q(x_t|x_{t-1},y) \tag 4$$

对于分子的第一项$\hat q(x_{t-1}|x_t)$进行贝叶斯公式变换得到下式，请注意$\hat q(x_{t}|x_{t-1})$和$\hat q(x_{t}|x_{t-1},y)$不一样。
$$\hat q(x_{t-1}|x_t)=\frac{\hat q(x_{t}|x_{t-1})\hat q(x_{t-1})}{\hat q(x_{t})} \tag 5$$
对于公式（5）分子第一项$\hat q(x_t|x_{t-1})$，我们通过全概率公式引入条件y，使得能够利用的已知条件（2），其过程如下
$$
\begin{aligned}
\hat q(x_t|x_{t-1}) = & \int_y \hat q(x_t,y|x_{t-1})dy \qquad \qquad \qquad &写出条件概率，再加|x_{t-1}\\ 
                    = & \int_y \hat q(x_t|y,x_{t-1})\hat q(y|x_{t-1})dy &利用公式（2）\\
                    = & \int_y q(x_t|x_{t-1})\hat q(y|x_{t-1})dy \\
                    = & q(x_t|x_{t-1})\int_y \hat q(y|x_{t-1})dy &积分为1\\
                    = & q(x_t|x_{t-1})
\end{aligned}
$$
$$\Rightarrow \hat q(x_t|x_{t-1})=q(x_t|x_{t-1}) \tag 6$$

对于公式（5）分子第二项$\hat q(x_{t-1})$和分母第二项$\hat q(x_{t})$，求解过程如下:
$$
\begin{aligned}
\hat q(x_t) = & \int_{x_{0:t-1}} \hat q(x_{0:t})dx_{0:t-1} \\
            = & \int_{x_{0:t-1}} \hat q(x_0)\hat q(x_{1:t}|x_0)dx_{0:t-1} \qquad \qquad  &利用公式（3）\\
            = & \int_{x_{0:t-1}} q(x_0)\hat q(x_{1:t}|x_0)dx_{0:t-1}
\end{aligned}
$$

对于$\hat q(x_{1:t}|x_0)$，同样通过引入全概率公式引入条件y，使用能够利用的已知条件（2）求解：
$$
\begin{aligned}
\hat q(x_{1:t}|x_0) = & \int_y \hat q(x_{1:t},y|x_0)dy \qquad \qquad \qquad &写出条件概率，再加|x_{0}\\
                    = & \int_y \hat q(x_{1:t}|y,x_0)\hat q(y|x_0)dy &利用公式（4）\\
                    = & \int_y \prod_1^t\hat q(x_t|y,x_{t-1})\hat q(y|x_0)dy &利用公式（2）\\
                    = & \int_y \prod_1^tq(x_t|x_{t-1})\hat q(y|x_0)dy \\
                    = & \prod_1^tq(x_t|x_{t-1})\int_y \hat q(y|x_0)dy &积分为1 \\
                    = & \prod_1^tq(x_t|x_{t-1}) \\
                    = & q(x_{1:t}|x_0) 
\end{aligned}
$$
$$\Rightarrow \hat q(x_{1:t}|x_0) = q(x_{1:t}|x_0) \tag 7$$
将公式（7）带入$\hat q(x_t)$的推导中继续推导：
$$
\begin{aligned}
\hat q(x_t) = & \int_{x_{0:t-1}} q(x_0)\hat q(x_{1:t}|x_0)dx_{0:t-1} \\
            = & \int_{x_{0:t-1}} q(x_0)q(x_{1:t}|x_0)dx_{0:t-1} \quad \quad  利用公式（7）\\
            = & \int_{x_{0:t-1}} q(x_{0:t})dx_{0:t-1}\\
            = & q(x_t)
\end{aligned}
$$
$$\Rightarrow \hat q(x_t)=q(x_t) \tag 8$$

现在我们可以利用公式（6）和公式（8）带入求解公式（5)：
$$
\begin{aligned}
\hat q(x_{t-1}|x_t) = & \frac{\hat q(x_{t}|x_{t-1})\hat q(x_{t-1})}{\hat q(x_{t})} \\
                    = & \frac{ q(x_{t}|x_{t-1}) q(x_{t-1})}{ q(x_{t})} 
\end{aligned}
$$
$$\Rightarrow \hat q(x_{t-1}|x_t) = \frac{ q(x_{t}|x_{t-1}) q(x_{t-1})}{ q(x_{t})} \tag 9 $$

对于公式（1）的第二项$\hat q(y|x_{t-1},x_{t})$，使用贝叶斯公式，再加$|x_{t-1}$，求解如下：
$$
\begin{aligned}
\hat q(y|x_{t-1},x_{t}) = & \frac{\hat q(x_{t}|y,x_{t-1})\hat q(y|x_{t-1})}{\hat q(x_t|x_{t-1})} \quad \quad 利用公式（2）（6）\\
                        = & q(x_t|x_{t-1}) \frac{\hat q(y|x_{t-1})}{q(x_t|x_{t-1})} \\
                        = & \hat q(y|x_{t-1})
\end{aligned}
$$
$$\Rightarrow\hat q(y|x_{t-1},x_{t}) = \hat q(y|x_{t-1}) \tag {10}$$
观察是公式（10）可以发现，是**对$x_{t-1}$的分类结果**。

结合公式（6）和公式（10），我们现在可以写出公式（1）的显示表达式：
$$
\hat q(x_{t-1}|x_t,y)=\mathbb{Z} 
\space q(x_{t-1}|x_{t}) q(y|x_{t-1}) \tag {11}
$$
其中$\mathbb{Z}$为归一化因子，确保概率积分为1。观察公式（11），我们可以发现，**可以通过分类器的梯度信息指导模型生成某一类别的图像**，其中分类器是与DDPM用同一数据集训练好的。因此，我们可以用训练好的DDPM模型$P_\theta (x_{t-1}|x_t)$和提前训练的分类器模型$P_\psi(y|x_{t-1})$进行去。唯一的问题在于，在t时刻的去噪过程中，我们已知y和$x_{t}$，但是$q(y|x_{t-1})$的**条件是$x_{t-1}$**。接下来我们需要解决该问题。

在DDPM中，$P_\theta(x_{t-1}|x_t)\sim \mathcal{N}(\mu(x_t,t),\Sigma(x_t,t))$。我们对公式（11）取log，第一项为：
$$logP_\theta(x_{t-1}|x_{t})=-\frac{1}{2}(x_{t-1}-\mu)^T\Sigma^{-1}(x_{t-1}-\mu)+C_1 \tag{12}$$

在DDPM中，我们假设方差$\Sigma$很小，其概率分布呈现尖峰状，大部分会落在$\mu$附近。我们考虑在$\mu$进行**一阶Tayler展开**：
$$
logP_\psi(y|x_{t-1})= logP_\psi(y|x_{t-1})|_{x_{t-1}=\mu}+(x_{t-1}-\mu)\nabla_{x_{t-1}}logP_\psi(y|x_{t-1})|_{x_{t-1}=\mu}+C_2 
$$
上述公式中，$C_2$代表其余高阶小量，第一项为由于指定了$x_{t-1}=\mu$，因此为常数，将其写入$C_2$中，同时为了方便观察，我们将$\nabla_{x_{t-1}}logP_\psi(y|x_{t-1})|_{x_{t-1}=\mu}$用$g$表示。**值得注意的是，$g$是一个常数，跟$x_{t-1}$无关**。于是，可以得到：
$$
logP_\psi(y|x_{t-1})= (x_{t-1}-\mu)g+C_2 \tag {13}
$$
将公式（13）（12）带入公式（11）中，可以得到：
$$
\begin{aligned}
log\hat q(x_{t-1}|x_t,y) = & -\frac{1}{2}(x_{t-1}-\mu)^T\Sigma^{-1}(x_{t-1}-\mu) + (x_{t-1}-\mu)g+C_3 \\
                         = & -\frac{1}{2}(x_{t-1}-\mu-\Sigma g)^T\Sigma^{-1}(x_{t-1}-\mu-\Sigma g)+\frac{1}{2}g^T\Sigma g+C_3 \\
\end{aligned}
$$
$$(验证(只关注g):-\frac{1}{2}(-x_{t-1}^T\Sigma^{-1}\Sigma g+\mu^T\Sigma^{-1}\Sigma g-g^T\Sigma^T\Sigma^{-1}x_{t-1}+g^T\Sigma^T\Sigma^{-1}\mu+g^T\Sigma^T\Sigma^{-1}\Sigma g))+\frac{1}{2}g^T\Sigma g$$
由于$g^T\Sigma g$不含有$x_{t-1}$，因此该项为常数，可以得到：
$$
log\hat q(x_{t-1}|x_t,y) = -\frac{1}{2}(x_{t-1}-\mu-\Sigma g)^T\Sigma^{-1}(x_{t-1}-\mu-\Sigma g)+C_4 \\
$$
$$log\hat q(x_{t-1}|x_t,y) = \sim \mathcal{N}(\mu+\Sigma g,\Sigma) $$
$$x_{t-1}=\mu+\Sigma g+\Sigma \epsilon \tag{14}$$
由公式（14）可以看到，**分类器梯度引入了去噪过程中**。

>接下来部分，笔者理解不深，仅简单写下。(下面公式中是从t+1时刻预测t时刻的噪声，需要区别上文中t时刻预测t-1时刻噪声)

上述过程完成了使用DDPM方法训练得到的模型用classifier guidance方法进行采样。但是DDIM方法训练得到的模型中，$\Sigma$为0，在公司（14）中无法引入梯度信息。**作者从score角度出发**，给出DDIM方法训练得到的模型用classifier guidance方法进行采样。

在DDPM中，我们有：
$$
\begin{aligned}
P(x_t|x_0) & \sim \mathcal{N}(\sqrt{\bar{\alpha_t}}x_0,(1-\bar{\alpha_t})I) \\
logP(x_t|x_0) & = -\frac{1}{2}\frac{(x_t-\sqrt{\bar{\alpha_t}}x_0)^2 }{1-\bar{\alpha_t}} \\
\nabla_{x_t}logP(x_t|x_0) & = -\frac{x_t-\sqrt{\bar{\alpha_t}}}{1-\bar{\alpha_t}} \\
又有：x_t & = \sqrt{\bar{\alpha_t}}x_0+(1-\sqrt{\bar{\alpha_t}})\epsilon \\
\nabla_{x_t}logP(x_t|x_0) & = -\frac{\epsilon}{\sqrt{1-\bar{\alpha_t}}} \\
\end{aligned}
$$
所以，模型建模对象为
$$
\nabla_{x_t}logP_\theta(x_t) = -\frac{\epsilon_\theta}{\sqrt{1-\bar{\alpha_t}}} \tag{15}
$$
对于Classifer Guidance，我们有：
$$
\begin{aligned}
\nabla_{x_t}logP_\theta(x_t)P_\psi(y|x_t) = & \nabla_{x_t}logP_\theta(x_t)+ \nabla_{x_t}logP_\psi(y|x_t) \\
-\frac{\hat \epsilon_\theta(x_t)}{\sqrt{1-\bar{\alpha_t}}} = & -\frac{\epsilon_\theta(x_t)}{\sqrt{1-\bar{\alpha_t}}}+\nabla_{x_t}logP_\psi(y|x_t) \\
\end{aligned}
$$

$$
\hat \epsilon_\theta(x_t) = \epsilon_\theta(x_t)-\sqrt{1-\bar{\alpha_t}}\nabla_{x_t}logP_\psi(y|x_t) \tag{16} \\
$$

公式（16）中，等式左边为classifier guidance预测的noise，等式右边第一项为DDIM预测的noise，第二项不再像DDPM classifier guidance中用$\Sigma$，而是用$\sqrt{1-\bar{\alpha_t}}$表示。

在[代码实现](https://github.com/openai/guided-diffusion)中，分类器梯度指导如下：指定生成图像类别，计算损失函数并反向传播到x_in

```python
def cond_fn(x, t, y=None):
    assert y is not None
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
```

```python
eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
    x, self._scale_timesteps(t), **model_kwargs
)
```