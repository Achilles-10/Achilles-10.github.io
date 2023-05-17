---
title: "使用pathlib优雅操作路径"
date: 2023-05-17T12:07:49+08:00
lastmod: 2023-05-17T12:07:49+08:00
author: ["Achilles"]
# keywords: 
# - 
categories: # 没有分类界面可以不填写
- 
tags: ["学习笔记","python库"] # 标签
description: "结合glob，使用pathlib库操作路径"
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
    image: "posts/tech/pathlib/cover.jpg" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

pathlib在功能和易用性上已经超越os库，比如以下这个获取上层目录和上上层目录的例子，pathlib的链式调用比os的嵌套调用更加灵活方便。

* os方法

  ```python
  import os
  # 获取上层目录
  os.path.dirname(os.getcwd())
  # 获取上上层目录
  os.path.dirname(os.path.dirname(os.getcwd()))
  ```

* pathlib方法

  ```python
  from pathlib import Path
  # 获取上层目录
  Path.cwd().parent
  # 获取上上层目录
  Path.cwd().parent.parent
  ```

### glob基础使用

基础语法，glob默认不匹配隐藏文件

|       通配符       |                描述                |    示例    |       匹配       |    不匹配    |
| :----------------: | :--------------------------------: | :--------: | :--------------: | :----------: |
|        `*`         |    匹配0个或多个字符，包含空串     |  `glob*`   | `glob`,`glob123` |    `glo`     |
|        `?`         |            匹配1个字符             |   `?lob`   |  `glob`,`flob`   |    `lob`     |
|      `[abc]`       |   匹配括号内字符集合中的单个字符   | `[gf]lob`  |  `glob`,`flob`   | `lob`,`hlob` |
|      `[a-z]`       |   匹配括号内字符范围中的单个字符   | `[a-z]lob` |  `alob`,`zlob`   | `lob`,`1lob` |
| `[^abc]`或`[!abc]` | 匹配不在括号内字符集合中的单个字符 | `[^cb]at`  |      `aat`       |  `at`,`cat`  |
| `[^a-z]`或`[!a-z]` | 匹配不在括号内字符范围中的单个字符 | `[^a-z]at` |      `1at`       |  `at`,`cat`  |

> 在 bash 命令行中`[!abc]`需要转义成`[\!abc]`

扩展语法

|      通配符       |                             描述                             |       示例       |               匹配                |    不匹配    |
| :---------------: | :----------------------------------------------------------: | :--------------: | :-------------------------------: | :----------: |
|    `{x,y,...}`    |      Brace Expansion，展开花括号内容，支持展开嵌套括号       | `a.{png,jp{e}g}` |     `a.png`,`a.jpg`,`a.jpeg`      |              |
|       `**`        | globstar，匹配所有文件和任意层目录，若`**`后面紧接着`/`则只匹配目录，不含隐藏目录 |     `src/**`     | `src/a.py`,`src/b/c.txt`,`src/b/` | `src/.hide/` |
| `?(pattern-list)` |                     匹配0次或1次给定模式                     |                  |                                   |              |
| `*(pattern-list)` |                   匹配0次或多次给定的模式                    |  `a.*(txt\|py)`   |      `a.`, `a.txt`, `a.txtpy`       |     `a`      |
| `+(pattern-list)` |                   匹配1次或多次给定的模式                    |                  |                                   |              |
| `@(pattern-list)` |                         匹配给定模式                         |                  |                                   |              |
| `!(pattern-list)` |                       匹配非给定的模式                       |                  |                                   |              |

> `pattern-list`是一组以`|`为分隔符的模式集合

### pathlib基础使用

Path为pathlib的主类，首先导入主类：

```python
from pathlib import Path
```

* 列出子目录：

  ```python
  >>> p=Path('./Research/')
  >>> [x for x in p.iterdir() if x.is_dir()]
  [PosixPath('Research/Two_Stream'), PosixPath('Research/pytorch_wavelets'), PosixPath('Research/FDFL')]
  ```

* 列出当前目录树下所有`.py`文件

  ```python
  >>> p=Path('./DFDC/')
  >>> list(p.glob('**/*.py'))
  [PosixPath('DFDC/processing/__init__.py'), PosixPath('DFDC/processing/crop_face.py'), PosixPath('DFDC/processing/make_dataset.py')]
  ```

* 在目录树中移动，用`'/'`进行路径拼接

  ```python
  >>> p=Path('.')
  >>> q = p/'DFDC'/'processing'
  >>> q
  PosixPath('DFDC/processing')
  ```

  使用`joinpath()`。

  ```python
  >>> p=Path.cwd()
  >>> p.joinpath('pathlib')
  PosixPath('/media/sda/zhy/pathlib')
  ```

  按照分隔符将文件路径分割

  ```python
  >>> q.parts
  ('DFDC', 'processing')
  ```

* 查询路径属性

  ```python
  >>> q.exists()
  True
  >>> q.is_dir()
  Trueq
  >>> q.is_file()
  False
  ```

* 创建目录和文件

  ```python
  p = Path('./pathlib/')
  # parents默认为False，若父目录不存在抛出异常
  # exist_ok默认为False，若目录已存在抛出异常
  p.mkdir(parents=True, exist_ok=True)
  
  p = Path('./pathlib/test.txt')
  # touch创建文件，父目录必须存在否则抛出异常
  p.touch(exist_ok=True)
  ```

* 获取文件/目录信息

  ```python
  >>> p = Path('DFDC/processing/__init__.py')
  # 获取文件/目录名
  >>> p.name
  '__init__.py'
  # 获取不包含后缀的文件名
  >>> p.stem
  '__init__'
  # 获取文件后缀名
  >>> p.suffix
  '.py'
  >>> p = Path.cwd()
  # 获取上层目录路径
  >>> p.parent
  PosixPath('/media/sda')
  # 获取所有上层目录路径
  >>> [path for path in p.parents]
  [PosixPath('/media/sda'), PosixPath('/media'), PosixPath('/')]
  # 获取文件/目录属性
  >>> p.stat()
  os.stat_result(st_mode=16895, st_ino=171704321, st_dev=2048, st_nlink=16, st_uid=1018, st_gid=1019, st_size=4096, st_atime=1684207391, st_mtime=1684207367, st_ctime=1684207367)
  ```

* 重命名/移动文件

  重命名文件时，当新命名的文件重复时，会抛出异常。

  ```python
  >>> p = Path('pathlib/test.txt')
  # 重命名
  >>> new_name = p.with_name('test_new.txt')
  >>> p.rename(new_name)
  PosixPath('pathlib/test_new.txt')
  # 修改后缀
  >>> new_suffix = new_name.with_suffix('.json')
  >>> new_name.rename(new_suffix)
  PosixPath('pathlib/test_new.json')
  ```

  移动文件，当新路径下文件已存在时，无法创建。

  ```python
  >>> p = Path('pathlib/test_new.json')
  >>> p.rename('test.json')
  PosixPath('test.json')
  ```

  `replace()`与`rename()`用法基本相同，但是当新命名的文件重复时，`replace()`不会抛出异常而是直接覆盖旧文件。

* 删除文件/目录

  删除文件，`missing_ok=True`设置文件不存在时不会抛出异常。

  ```python
  >>> p = Path('test.json')
  >>> p.unlink(missing_ok=True)
  ```

  删除目录，目录必须为空，否则抛出异常。

  ```python
  >>> p=Path('pathlib')
  >>> p.rmdir()
  ```