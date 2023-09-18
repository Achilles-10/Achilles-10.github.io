---
title: "深度学习面试题：评价指标"
date: 2023-09-16T15:11:51+08:00
lastmod: 2023-09-16T15:11:51+08:00
author: ["Achilles"]
# keywords: 
# - 
categories: # 没有分类界面可以不填写
- 
tags: ["面试","学习笔记"] # 标签
description: "深度学习算法岗评价指标相关常见面试题"
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
    image: "posts/tech/eval/confusion.png" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## 1. 说说机器学习评价指标

### 混淆矩阵

<div align=center><img src="confusion.png" style="zoom:50%;" /></div>

### 准确率（Accuracy, ACC）

所有分类正确的样本占全部样本的比例。
$$
\text{ACC} = \frac{TP+TN}{TP+FN+TN+FP}
$$

### 精确率（Precision）

表示预测为正例的样本中真正为正例的比例，较高的 Precision 意味着模型预测为正例的样本中有很少的错误，衡量**误检**。
$$
\text{Precision} = \frac{TP}{TP+FP}
$$

### 召回率（Recall）

表示所有正例中被正确预测为正例的比例，较高的 Recall 意味着模型对于正例的检测能力较强，衡量**漏检**。
$$
\text{Recall} = \frac{TP}{TP+FN}
$$

### F1-Score

F1 分数是 Precision 和 Recall 的调和平均值，用于综合考虑模型的精确性和召回率。
$$
\text{F1\ Score}=\frac{2}{\frac{1}{\text{Precision}}+\frac{1}{\text{Recall}}}=\frac{2\cdot \text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}}
$$

### AUC 与 ROC

AUC 衡量二分类模型在不同阈值下的分类能力，表示 ROC 曲线下的面积。适用于处理不同阈值下分类性能变化较大和类别不平衡的情况。ROC 曲线**横坐标为 FPR**，**纵坐标为 TPR**。

<div align=center><img src="roc.png" style="zoom:50%;" /></div>

### Precision-Recall 曲线

<div align=center><img src="pr.png" style="zoom:70%;" /></div>

PR 曲线的横坐标是精确率 P，纵坐标是召回率 R，曲线上的点就对应 F1-score。

### IoU 和 mIoU

$$
\text{IoU}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}
$$

mIoU 一般都是基于类进行计算的，将每一类的IOU计算之后累加，再进行平均，得到的就是mIOU。

### AP 和 mAP

AP（Average Precision）和 mAP（mean Average Precision）常用于目标检测任务中。AP 就是每一类的Precision 的平均值。而 mAP 是所有类的AP的平均值。

## 2. AUC 是什么？是否对正负样本比例敏感

AUC 定义：AUC 值为 ROC 曲线所覆盖的区域面积，AUC 越大，分类器分类效果越好。

AUC 还有另一个意义：分别随机从正负样本集中抽取一个正样本，一个负样本，正样本的预测值大于负样本的概率。
$$
\text{AUC} = \frac{\sum(pred_{pos}>pred_{neg})}{Num_{pos}*Num_{neg}}
$$
AUC对正负样本比例**不敏感**

## 3. 讲讲分类，回归，推荐，搜索的评价指标

* 分类：Acc, Precision, Recall, F1-score, AUC

* 回归：MSE, RMSE, MAE, R Squared

* 推荐任务：

  * 离线评估 offline evaluation

    评分预测，预测用户对物品的评分，用 MAE 和 RMSE

    对于 Top N 模型：对排名进行评估，用 Precision、Recall 和 F1

  * 在线评估 online evaluation

    A/B Test，用 CTR（用户点击率）和 CR（用户转化率）。

    * **CTR (点击率, Click-Through Rate)**

      CTR 是用于衡量推荐内容或广告的点击效果。它表示了推荐或展示多少次内容之后，用户点击了该内容的次数。
      $$
      CTR=\frac{点击次数}{展示次数}
      $$

    * **CR (转化率, Conversion Rate)**

      CR 用于衡量用户从点击到完成某个目标行为（如购买、注册等）的转化效果。
      $$
      CR=\frac{完成目标行为的用户数}{点击数}
      $$

## 4. A/B Test 的原理

也称为拆分测试或两样本假设测试，用于比较两个（或更多）版本的网页或应用程序，以确定哪一个版本的表现更好。

将用户划分为 A，B 两组，A 实验组用户，接受所设计的推荐算法推荐的商品，B 对照组用户，接受基线方法推荐的商品。通过对比两组用户的行为来评估推荐算法的性能。

A/B 测试的考虑因素：

- **持续时间与样本大小**：为了得到可靠的结果，测试需要运行足够长的时间，并有足够大的样本大小。
- **单变量测试**：通常，一次 A/B 测试只会测试一个变化（例如，更改按钮的颜色或文本），这样可以确定是哪个具体变化导致了结果的变动。
- **外部因素**：确保外部事件（例如，节日、促销活动或新闻事件）不会对 A/B 测试结果产生影响。







