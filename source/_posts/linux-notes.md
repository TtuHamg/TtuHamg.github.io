---
title: Linux Notes
date: 2023-10-30 20:32:17
mathjax: false
aside: false
description: 本文是笔者在使用服务器过程中的常用命令。

---

<!-- more -->

#### 服务器硬盘

```python
df -h #查看服务器磁盘大小
du -h #查询当前目录下，所有文件夹的大小
du -sh #查询当前目录下所有子目录总大小

df -h <目录名> #查看该目录挂载情况
```

#### 显卡

```python
nvidia-smi -L #查看显卡型号
```

#### Git
git clone项目超时
```git
git clone https://gitclone.com/github.com/facebookresearch/DiTgit  
```

#### Tmux
```python
tmux attach -t <name> #进入name窗口
tmux new -s <name> #创建name窗口
ctrl + D #exit/kill当前窗口
ctrl + B, :, set -g mouse on #开启鼠标模式
```
