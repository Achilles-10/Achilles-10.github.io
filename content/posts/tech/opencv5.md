---
title: "OpenCV-Python学习笔记(5)：形态学变换与图像梯度"
date: 2023-05-29T21:36:45+08:00
lastmod: 2023-05-29T21:36:45+08:00
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
    image: "posts/tech/opencv5/sobel.png" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## 形态学转换

形态变换是基于图像形状的简单操作，两种基本形态学算子是`侵蚀`和`膨胀`，也包括变体形式`开运算`和`闭运算`等。

### 侵蚀(Erosion)

侵蚀前景物体的边界，内核在2D卷积时，只有当内核下所有像素都为`1`时才为`1`，否则被侵蚀变成`0`。根据内核大小，边界附近的像素都会被丢弃，因此能减小前景对象（白色区域）的大小，有助于去除小的白色噪声。如下例。

```python
img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
plt.imshow(np.hstack((img,erosion)),'gray')
```

<div align=center><img src="erosion.png"/></div>

### 扩张、膨胀(Dilation)

与侵蚀相反，当内核下至少一个像素为`1`时，则为`1`，因此能增加图像中前景对象（白色区域）的大小。

```python
dilation = cv2.dilate(img,kernel,iterations = 1)
plt.imshow(np.hstack((img,dilation)),'gray')
```

<div align=center><img src="dialation.png"/></div>

### 开运算(Opening)

开运算是先侵蚀再扩张，有助于消除盐噪声，如下例。

```python
opening = cv2.morphologyEx(salt, cv2.MORPH_OPEN, kernel)
plt.imshow(np.hstack((salt,opening)),'gray')
```

<div align=center><img src="open.png"/></div>

### 闭运算(Closing)

闭运算是先扩张再侵蚀，有助于去除前景对象内部的小黑点（椒噪声），如下例。

```python
closing = cv2.morphologyEx(pepper, cv2.MORPH_CLOSE, kernel)
plt.imshow(np.hstack((pepper,closing)),'gray')
```

<div align=center><img src="close.png"/></div>

### 形态学梯度

图像扩张和侵蚀的差，结果类似于图像的轮廓线。

```python
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) # dilation-erosion
plt.imshow(np.hstack((img,gradient)),'gray')
```

<div align=center><img src="gradient.png"/></div>

### Top Hat

输入图像与开运算的差。

```python
kernel_tophat = np.ones((9,9),np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_tophat) # img-opening
plt.imshow(np.hstack((img,tophat)),'gray')
```

<div align=center><img src="tophat.png"/></div>

### Black Hat

输入图像与闭运算的差。

```python
kernel_blackhat = np.ones((9,9),np.uint8)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_blackhat) # img-closing
plt.imshow(np.hstack((img,blackhat)),'gray')
```

<div align=center><img src="blackhat.png"/></div>

### 结构元素

可以通过`cv.getStructuringElement()`函数来创建椭圆形、圆形的内核，只需传递内核形状和大小即可。

```python
>>> cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) #矩形
array([[1, 1, 1, 1, 1],
	  [1, 1, 1, 1, 1],
	  [1, 1, 1, 1, 1],
	  [1, 1, 1, 1, 1],
	  [1, 1, 1, 1, 1]], dtype=uint8)
>>> cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #椭圆
array([[0, 0, 1, 0, 0],
	  [1, 1, 1, 1, 1],
	  [1, 1, 1, 1, 1],
	  [1, 1, 1, 1, 1],
	  [0, 0, 1, 0, 0]], dtype=uint8)
>>> cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #十字形
array([[0, 0, 1, 0, 0],
	  [0, 0, 1, 0, 0],
	  [1, 1, 1, 1, 1],
	  [0, 0, 1, 0, 0],
	  [0, 0, 1, 0, 0]], dtype=uint8)
```

## 图像梯度

OpenCV提供三种类型的梯度滤波器或高通滤波器用于查找图像梯度和边缘，即Sobel，Scharr和Laplacian。

### Sobel和Scharr算子

Sobel算子是高斯平滑加微分运算的联合运算，它对噪声更具鲁棒性。可以通过`xorder`和`yorder`来指定导数方向，通过`ksize`指定内核大小，当`ksize=-1`时，使用`Scharr滤波器`，效果比`Sobel滤波器`更好。

### Laplacian算子

计算由关系$\Delta src=\frac{\partial src}{\partial^2 x^2}+\frac{\partial^2 src}{\partial y^2}$给出的图像的拉普拉斯图。每一阶导数是由Sobel算子计算。如果`ksize=1`，则使用以下内核进行滤波：
$$
kernel=\begin{bmatrix}0&1&0\\\\1&-4&1\\\\0&1&0\end{bmatrix}
$$

### 代码

以下实例在一个图标中展示了所有算子，内核都是`5x5`大小，深度设置为`1`来得到`np.int8`类型的结果图像。

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('sudoku.png',0)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
plt.figure(figsize=(8,8))
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
```

<div align=center><img src="sobel.png"/></div>

