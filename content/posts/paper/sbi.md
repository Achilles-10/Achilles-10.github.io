---
title: "Detecting Deepfakes with Self-Blended Images"
date: 2023-03-25T20:37:10+08:00
lastmod: 2023-03-25T20:37:10+08:00
author: ["Achilles"]
# keywords: 
# - 
categories: # 没有分类界面可以不填写
tags: ["论文阅读","人脸伪造检测", "CVPR"] # 标签
description: "采用自换脸生成负样本，检测伪造边界"
weight:
slug: ""
draft: false # 是否为草稿
comments: true # 本页面是否显示评论
# reward: true # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true #顶部显示路径
math: true # 是否开启数学公式
cover:
    image: "posts/paper/sbi/sbi.jpg" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

[[paper]](https://arxiv.org/abs/2204.08376) [[code]](https://github.com/mapooon/SelfBlendedImages)

## 动机与介绍

已有方法对跨域数据集和高压缩高曝光数据的检测能力大幅下降(泛化性差)；

难以识别的fake样本通常包含更一般伪造痕迹，故要学习更通用和鲁棒的面部伪造表征；

定义了四种常见的伪影(artifacts)：
![artifacts](artifacts.png)

## 主要贡献

1. 提出了source-target generator (STG) and mask generator (MG)来学习更一般鲁棒的人脸伪造表征
2. 通过自换脸而非寻找最接近的landmark换脸，降低了计算成本
3. 在cross-dataset和cross-maniputation测试中都取得了SOTA

![overview](overview.png)

## 方法

学习伪造人脸与背景的不一致分为下列三个模块

1. **Source-Target Generator(STG):** 
   * 对source和target进行数据增强以产生不一致，并且对source进行resize和translate以再现边界混合和landmarks不匹配；
   * 首先对Target和Source之一做图像增强 (**color**：RGB channels, hue, saturation, value, brightness, and contrast；**frequency**：downsample or sharpen)；
   * 然后对source进行裁剪：$H_r=u_hH,\quad W_r=u_wW$,其中$\ u_h和u_w$是一组均分分布中的随机值，再对裁剪后的图像zero-padded 或者 center-cropped还原回初始大小；
   * 最后对source做变形(translate)：traslate vector$\ t=[t_h,t_w]$,$\ t_h=v_hH,t_w=v_wW$，$v_h和v_w$是一组均分分布中的随机值。

2. **Mask Generator:** 生成变形的灰度mask图

   * 计算面部landmarks的convex hull来初始化mask，然后对mask变形(elastic deformation)，在用两个不同参数的高斯滤波器(gaussian filter)对mask进行平滑处理。最后在{0.25, 0.5, 0.75, 1, 1, 1}中选取混合指数(blending ration)；
  
3. **Blending:** 用Mask来混合source和target图得到SBI

    $$I_{SB}=I_s\odot M+I_t\odot(1-M)$$

    ![sample](sample.png)

**Train with SBIs**: 将target而非原图作为”REAL“，使得模型集中在伪造痕迹上

## 实验

1. 实现细节
   * **预处理**：Dlib和RetinaFace裁帧，面部区域裁剪：4~20%(训练),12.5%(推理)；
   * **Source-Target Augmentation**：RGBShift, HueSaturationValue, RandomBrightnessContrast, Downscale, and Sharpen
   * **推理策略**：如果在一帧中检测到两个或多个人脸，则将分类器应用于所有人脸，并将最高的虚假置信度用作该帧的预测置信度。
2. 实验设定：**各类baseline**
3. **跨数据集评估**
    ![cross-dataset](tab1.png)
4. **跨操作评估**
    ![cross-manipulation](tab2.png)
5. **定量分析**
    ![BI](tab3.png)
    ![I2G](tab4.png)
6. **消融实验**
    ![ablation](tab5.png)
    ![ablation2](tab6.png)
    ![ablation3](tab7.png)
7. **定性分析**

## 局限性
缺乏时序信息、无法解决GAN生成的伪造图像
