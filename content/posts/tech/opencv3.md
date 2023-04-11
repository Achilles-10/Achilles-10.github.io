---
title: "OpenCV-Python学习笔记(3)：几何变换"
date: 2023-04-11T22:14:19+08:00
lastmod: 2023-04-11T22:14:19+08:00
author: ["Achilles"]
# keywords: 
# - 
categories: # 没有分类界面可以不填写
- 
tags: ["OpenCV","学习笔记"] # 标签
description: ""
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
math: true
cover:
    image: "" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## 变换

OpenCV提供了`cv2.warpAffine`和`cv2.warpPerspective`两个转换函数，`cv2.warpAffine`采用`2x3`的转换矩阵，`cv2.warpPerspective`采用`3x3`转换矩阵。

### 缩放

使用`cv2.resize`实现图像的缩放，可以指定缩放尺寸或缩放比例，以及插值方法。首选的插值方法是用于缩小的 `cv2.INTER_AREA `和用于缩放的 `cv2.INTER_CUBIC`（慢）和 `cv2.INTER_LINEAR`。`cv2.INTER_LINEAR`是默认的缩放插值方法。可以用一下两种方法实现：

```python
import numpy as np
import cv2
img = cv2.imread('face.png')
res = cv2.resize(img, None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
# OR
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
```

### 平移

如果在(x,y)方向上的平移量为$(t_x,t_y)$，则可以得到转换矩阵**M**:
$$
M=\begin{bmatrix} 1 & 0 & t_x \\\\ 0 & 1 & t_y \end{bmatrix}
$$
将其转换为`np.float32`的numpy数组并传入`cv2.warpAffine`函数，以平移`(100,50)`为例：

```
rows,cols,_ = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))
```

<div align=center><img src="shift.png" style="zoom:120%"/></div>

> `cv2.warpAffine`的第三个参数是输出图像的大小，形式为`(width,height)`

### 旋转

图像旋转角度为$\theta$是通过以下变换矩阵实现的：
$$
M = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix}
$$
OpenCV提供了可缩放的旋转和可调整的旋转中心，修改后的变换矩阵为：
$$
\begin{bmatrix} \alpha & \beta & (1- \alpha ) \cdot center.x - \beta \cdot center.y \\\\ - \beta & \alpha & \beta \cdot center.x + (1- \alpha ) \cdot center.y \end{bmatrix}
$$
其中：
$$
\alpha=scale\cdot\cos\theta,\\\\\beta=scale\cdot\sin\theta
$$
为了得到该变换矩阵，OpenCV提供了`cv2.getRotationMatrix2D`函数，以将图像相对于中心旋转逆时针90度缩放比例为1：

```python
rows,cols,_ = img.shape
# cols-1 和 rows-1 是坐标限制
M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))
```

<div align=center><img src="rotate.png" style="zoom:120%"/></div>

### 仿射变换（Affine Transformation）

在仿射转换中，原始图像中的所有并行线仍将在输出图像中平行。为了得到转换矩阵，需要从输入图像中的三个点及其在输出图像中的对应位置。通过`cv2.getAffineTransform`函数创建一个2x3的矩阵，并传递给`cv2.warpAffine`。

```python
rows,cols,ch = img.shape
pts1 = np.float32([[100,100],[100,400],[400,100]])
pts2 = np.float32([[50,50],[100,400],[350,50]])
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
```

<div align=center><img src="affine.png" style="zoom:120%"/></div>

### 透视变换（Perspective Transformation）

透视转换需要一个3x3转换矩阵。即使在转换后，直线也将保持直线。需要在输入图像上有四个点，在输出图像中需要对应的四个点，其中三个点不共线。可通过`cv2.getPersperctiveTransform`得到变换矩阵，并传递给`cv2.warpPerspective`。

```python
rows,cols,ch = img.shape
pts1 = np.float32([[40,100],[400,100],[0,400],[360,400]])
pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(cols,rows))
```

<div align=center><img src="perspective.png" style="zoom:120%"/></div>
