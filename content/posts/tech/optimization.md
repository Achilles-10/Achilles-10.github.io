---
title: "深度学习面试题：优化函数"
date: 2023-08-27T23:49:30+08:00
lastmod: 2023-08-27T23:49:30+08:00
author: ["Achilles"]
# keywords: 
# - 
categories: # 没有分类界面可以不填写
- 
tags: ["面试","学习笔记"] # 标签
description: "深度学习算法岗优化函数相关常见面试题"
weight:
slug: ""
draft: true # 是否为草稿
comments: true # 本页面是否显示评论
# reward: true # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true #顶部显示路径
math: true
cover:
    image: "" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## 优化算法

# 优化算法

1. **SGD**: `torch.optim.SGD(param, lr:float, momontum:float, weight_decay:float, nesterov:bool)`
    * **SGD**: $$ \omega^*=\omega-lr\cdot\text{g} $$

    * **SGD+momontum**: 记录历史梯度值，减轻卡在局部最小值的危险
      $$
      v_t = v_{t-1}\cdot m+\text{g}_t\cdot(1-m)\\\\
      \omega^*=\omega-lr\cdot v_t
      $$

2. **Nesterov**: 同 `SGD(nesterov:bool)`

    * Nesterov 先用当前的梯度更新得到临时参数，在用临时参数计算梯度更新当前梯度。

3. **Adagrad**: `torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)`

    * 加入梯度平方，自适应缩小学习率，但可能过早过量减小学习率
      $$
      r_t = r_{t-1}+\text{g}^2\\\\
      \omega^*=\omega-\frac{lr}{\sqrt{r+\epsilon}}\cdot\text{g}
      $$
      

4. **RMSProp**: `torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)`
    * 加入梯度平方的衰减项，只记录最近的梯度
      $$
      r_t = r_{t-1}\cdot\alpha+ \text{g}^2\cdot(1-\alpha)\\\\
      \omega^*=\omega-\frac{lr}{\sqrt{r+\epsilon}}\cdot\text{g}
      $$
      

5. **Adam**: `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)`

    * 更新一阶梯度和梯度平方，自适应减少学习率
      $$
      m_{t}=\frac{m_{t-1}\cdot\beta_1+\text{g}\cdot(1-\beta_1)}{1-\beta_1^t}\\\\
      v_{t}=\frac{v_{t-1}\cdot\beta_2+\text{g}^2\cdot(1-\beta_2)}{1-\beta_2^t}\\\\
      \omega^*=\omega-\frac{lr}{\sqrt{v+\epsilon}}\cdot\text{m}
      $$
      

