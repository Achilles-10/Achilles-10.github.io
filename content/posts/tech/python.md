---
title: "深度学习面试题：Python"
date: 2023-08-08T15:38:01+08:00
lastmod: 2023-08-08T15:38:01+08:00
author: ["Achilles"]
# keywords: 
# - 
categories: # 没有分类界面可以不填写
- 
tags: ["面试","学习笔记"] # 标签
description: "深度学习算法岗 Python 语言相关常见面试题"
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
    image: "posts/tech/python/cover.png" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## 1. Python 深拷贝与浅拷贝

**浅拷贝（copy）**：拷贝父对象，不会拷贝对象内部的子对象；*类似指针和引用*。

<div align=center><img src="copy.png" style="zoom:40%;" /></div>

**深拷贝（deepcopy）**：copy 模块的 deepcopy 方法，完全拷贝父对象及其子对象。

<div align=center><img src="deepcopy.png" style="zoom:40%;" /></div>

## 2. Python 多线程能用多个 cpu 吗？

不能。Python 解释器使用了 **GIL(Global Interpreter Lock)**，在任意时刻只允许单个 python 线程运行。

GIL 为全局解释器锁，其功能是在 CPython 解释器中执行的每一个 Python 线程都会先锁住自己，以阻止别的线程执行。

利用 multiprocessing 库可以在多个 CPU 上并行运行。在多进程编程中，每个进程都有自己独立的 Python 解释器和内存空间，因此不存在 GIL 的限制。

### 2.1 什么是Python的GIL?

GIL 是 Python 的全局解释器锁，是 CPython 解释器的实现特性。它是一个互斥锁，阻止多个线程同时执行 Python 程序，这意味着即使在多核处理器上，Python代码也无法同时在多个核心上运行。

### 2.2 GIL对Python的性能有何影响?

GIL 导致 Python 在多线程环境下，无法充分利用多核处理器。即使在多核处理器上，Python 的多线程代码无法实现真正的并行运行，这对于计算密集型任务来说，可能会成为性能瓶颈。

### 2.3 为什么Python要设计GIL?

GIL的存在可以简化 Python 对象模型的实现，防止数据竞争和状态冲突，使得对象管理变得更加简单。此外，GIL也使得 CPython 的实现变得更加简单，对于某些依赖于 C 扩展的 Python 程序来说，这是非常重要的。

### 2.4 如何避免GIL对性能的影响?

对于 IO 密集型任务，由于大部分时间都在等待 IO，所以 GIL 的影响并不大。对于计算密集型任务，可以使用多进程模块 multiprocessing，因为每个进程都有自己的解释器和内存空间，因此可以避免 GIL 的影响。

## 3. Python 垃圾回收机制

[[参考]](https://blog.51cto.com/u_14666251/4674779)

Python 垃圾回收机制：以**引用计数器**为主，**标记清除**和**分代回收**为辅。

* **引用计数**：每个对象内部都维护了一个值，记录此对象被引用的次数，若次数为 0，则 Python 垃圾回收机制会自动清除此对象；
* **标记-清除（Mark-Sweep）**：遍历所有对象，如果有对象引用它，则标记为可达的（reachable）；再次遍历对象，如果某个对象没有被标记为可达，则将其回收。*解决循环引用问题*
* **分代回收**：`对象存在时间越长，越可能不是垃圾，应该越少去收集`。根据对象存活时间划分为不同集合（代），Python 将内存分为了 3 代。
* 当代码中主动执行 `gc.collect()` 命令时，Python 解释器就会进行垃圾回收。

## 4. Python 里的生成器是什么

生成器（Generator）是一种特殊的迭代器，生成器函数通过 yield 来生成一个值，并暂停执行保存当前状态。可以有效节省内存空间，执行过程可以被暂停和恢复。

实例如下：

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1
# 使用生成器函数创建生成器对象
generator = countdown(4)
# 通过迭代生成器对象获取值
for value in generator:
    print(value)
# 输出结果为
# 4
# 3
# 2
# 1
```

## 5. 迭代器和生成器的区别

[[参考]](https://pythonhowto.readthedocs.io/zh_CN/latest/iterator.html)

**迭代器**是一种对象，有 `__iter__` 和 `__next__` 方法，每次调用 `__next__` 方法都返回可迭代对象中的下一个元素，当没有更多元素可返回时，会引发 StopIteration 异常（防止无限循环）。Python 中迭代器是一个惰性序列。

用 `iter()` 内建方法可以把 list、dict、str 等可迭代对象转换成迭代器。

```python
list0 = [0, 1, 2]
iter0 = iter(list0)
print(type(iter0))

# <class 'list_iterator'>
```

**生成器**是一种特殊的迭代器。可将生成列表和字典的推到表达式的中括号换成小括号，就得到了生成器表达式。可以借助 `next()` 方法来获取下一个元素。

```python
list_generator = (x * x for x in range(5000))
for i in list_generator:
    print(i)
```

区别：

<div align=center><img src="iter.png" style="zoom:50%;" /></div>

## 6. 装饰器

[[参考]](https://www.runoob.com/w3cnote/python-func-decorators.html)

装饰器本质上是一个函数，它接受一个函数作为输入，并返回一个新的函数作为输出，这个新函数通常会在原函数的基础上添加一些额外的功能或行为。

装饰器的作用是实现代码的重用和功能的动态扩展，它可以在不修改原函数代码的情况下，对函数进行功能增强、日志记录、性能统计等操作。

以下是一个装饰器示例，用于记录函数的执行时间：

```python
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 执行时间：{execution_time} 秒")
        return result
    return wrapper

@timer
def my_function():
    time.sleep(2)
    print("执行完成")

my_function()

# 执行完成
# 函数 my_function 执行时间：2.0054352283477783 秒
```

## 7. Python 有哪些数据类型

* 数字类型（Numeric Types）：整数 `int`，浮点数 `float`，复数 `complex` 等；
* 字符串类型（String Type）：由字符组成的序列 `str`，用于表示文本信息；
* 列表类型（List Type）：有序可变的集合，可以包含任意类型的元素，使用方括号 `[]` 表示；
* 元组类型（Tuple Type）：有序不可变的集合，可以包含任意类型的元素，使用圆括号 `()` 表示；
* 集合类型（Set Type）：无序的可变集合，不允许重复元素，使用花括号 `{}` 表示；
* 字典类型（Dictionary Type）：无序的键值对集合，用于存储具有唯一键的值，使用花括号 `{}` 表示；
* 布尔类型（Boolean Type）：表示真或假的值，包括 `True` 和 `False` 两个取值；
* 空值类型（None Type）：表示空对象或缺失值的特殊类型，只有一个取值 `None`。

## 8. Python 中列表 List 的 del，remove 和 pop 的用法和区别

* del，是一个关键字，用于删除整个列表或列表中指定位置的元素
  * 删除指定位置的元素：del 关键字加上列表名和索引，例如 `del my_list[index]`；
  * 删除整个列表：del 关键字加上列表名，例如 `del my_list`。
* remove，是列表的方法，用于删除列表中第一个匹配的指定元素
  * 参数为要删除的元素，例如 `my_list.remove(element)`，将删除列表中第一个匹配元素 element；若 element 不在列表中则抛出 `ValueError`。
* pop，是列表方法，用于删除并返回指定指定位置的元素
  * 参数为要删除元素的索引，默认为列表最后一个元素，例如 my_list.pop(0)，将删除列表的第一个元素并返回该元素的值。

## 9. Python yeild 和 return 的区别

* return 用于返回最终结果并终止函数执行，yield 用于定义生成器函数，可以通过多次调用来逐步产生结果；
* return 只能返回一次值，yield 可以多次返回值并保留函数状态，使函数可从上次 yield 语句的位置继续执行。

## 10. Python set 的底层实现

set 底层是基于哈希表实现的，发生哈希冲突时使用开放地址法或链表解决冲突。有以下特点：

* 元素的添加、删除和查找平均时间复杂度为 $O(1)$；
* set 中元素使无序的，每次遍历的顺序可能不同；
* set 中的元素不重复，重复添加的元素只会保留一个；
* set 不支持索引操作，因为元素的存储位置不是固定的。

## 11. Python 字典和 set() 的区别

* 存储结构：字典是键值对的集合；set 是元素的集合，没有键值对；
* 唯一性：字典中键是唯一的；set 中元素是唯一的；
* 访问方式：字典通过键来访问值，`my_dict[key]`；set 只能通过遍历或 `in` 来判断某元素是否存在于集合中；
* 可变性：字典是可变数据类型，可以增删改；set 也是可变的，可以添加删除元素；
* 有序性：字典和 set 都是无序的。

## 12. 怎么对字典的值进行排序

* 转化为元素 (key,val) 的形式用 sorted() 排序；
* 设置 sorted 中 key 的参数的值

```python
# 方法一
z = zip(d.values(),d.keys())
#x = zip(d.itervalues(),d.iterkeys())  迭代类型，省空间
s = sorted(z)

# 方法二
a = d.items()
s = sorted(a,key= lambda x:x[1])
```

## 13. `__init__`、`__new__` 和 `__call__` 的区别

`__new__` 方法用于对象的创建，`__init__` 方法用于对象的初始化。 `__new__` 方法在对象创建前被调用，`__init__` 方法在对象创建后被调用。 `__call__` 方法用于将对象作为函数进行调用，可以赋予对象函数的行为。

## 14. Python 的 lambda 函数

lambda 函数是一种匿名函数，由关键字 lambda 后面跟上参数列表，跟上一个冒号和表达式作为函数的返回值。

> lambda arguments: expression

```python
# 求两数之和
add = lambda x, y: x + y
# 对列表排序
sorted_numbers = sorted(my_list, key=lambda x:x)
# 作为 map 函数的参数
squared_numbers = map(lambda x:x**2, my_list)
```

* 为什么要使用 lambda 函数？

  匿名函数可以在任何需要的地方使用，但是这个函数只能使用一次，因此 lambda 函数也称为丢弃函数，它可以与其他预定义函数（如filter(),map()等）一起使用。相对于我们定义的可重复使用的函数来说，这个函数更加**简单便捷**。

## 15. Python 内存管理

Python 使用**引用计数**和**垃圾回收**两种方式来管理内存。

* 引用计数：当一个对象的引用计数变为0时，Python 会立即回收该对象的内存。
* 垃圾回收：Python 使用垃圾回收机制来处理循环引用的特殊情况。垃圾回收机制会周期性地扫描所有的对象，找出不再被引用的对象，并回收它们的内存空间。

## 16. Python 在内存上做了哪些优化

* 引用计数
* 垃圾回收：Python 使用**标记-清除**算法（Mark and Sweep）和**分代回收**算法（Generational Garbage Collection）来回收不再使用的对象，释放内存空间。
* 内存池：Python通过**内存池管理器**（Memory Pool Manager）来分配内存。内存池预先在内存中申请一定数量的，大小相等的内存块留作备用，当有新的内存需求时，就先从内存池中分配内存给这个需求，不够之后再申请新的内存。这样做最显著的优势就是能够减少内存碎片，提升效率。
* 内存共享：对于一些不可变对象（如小整数、字符串等），Python会进行内存共享。多个变量引用相同的不可变对象时，它们可以共享相同的内存空间，从而节约内存。
* 迭代器和生成器：Python中的迭代器和生成器可以节省大量内存，特别是处理大型数据集时。它们按需生成数据，而不是一次性将所有数据加载到内存中。
* 字符串驻留机制：对于一些短字符串，Python会将其驻留（intern）在内存中，即多个变量引用相同的字符串对象。这种机制可以减少相同字符串的内存使用。

## 17. Python 中类方法（class method）和静态方法（static method）的区别

* **装饰器不同**：类方法使用 `@classmethod`，静态方法使用 `@staticmethod`；
* **参数不同**：类方法接收类 `cls` 作为隐式第一个参数；静态方法不接收隐式的第一个参数，类似普通函数；
* **访问类属性和方法的方式不同**：类方法可以访问和修改类属性，也可以调用其他类方法和静态方法；静态方法不能访问类属性，也不能调用其他类方法或静态方法，只能访问静态方法中定义的局部变量；
* **使用场景不同**：类方法通常用于执行与类相关的操作，例如创建类的实例或修改类属性。静态方法通常用于执行与类无关的操作，它们在类的命名空间中定义，但与类的状态无关。
* 类方法和静态方法均可通过类名或实例对象（*类方法不推荐*）来调用。

```python
class MyClass(object):
    # 实例方法
    def instance_method(self): # 需要传入 self
        print('instance method called', self)
    # 类方法
    @classmethod
    def class_method(cls):
        print('class method called', cls)
    # 静态方法
    @staticmethod
    def static_method():
        print('static method called')
```

## 18. Python 多线程怎么实现

可以用以下模块：

* _thread
* threading
* Queue
* multiprocessing

## 19. 点积和矩阵相乘的区别

**点积**：又叫内积，数量积，由两个维度相同的的向量相乘求和，得到一个标量。$a\cdot b=\sum_{i}^{n}{a_1b_1+a_2b_2+\dots+a_nb_n}$

矩阵乘法：得到仍然是一个矩阵

## 20. Python 中错误和异常处理

Python 中会发生两种类型错误：

* **语法错误**：如果未遵循正确的语言语法，则会引发语法错误，这种错误编译器会指出；
* **逻辑错误（异常）**：在运行时中，通过语法测试后发生错误的情况称为异常或逻辑类型。比如除数除以0，运行时就会抛出一个抛出一个 ZeroDivisionError 异常，叫处理异常。**处理方式为：通过 `try..except..finally` 代码块来处理捕获异常并手动处理。**

## 21. Python 的传参是传值还是传址

在Python中，函数参数传递方式是"传对象引用"，也可以称为"传对象的地址"，即“传址”。根据对象的类型（可变对象和不可变对象），在函数内部对参数对象的操作可能会有不同的效果。

## 22. 什么是猴子补丁

猴子补丁（Monkey Patching）是指在运行时动态修改已有的代码，通常是在不修改原始代码的情况下添加、替换或修改现有的功能。

例如，很多代码用到 import json，后来发现ujson性能更高， 如果觉得把每个文件的import json 改成 import ujson as json成本较高，或者说想测试一下用ujson替换json是否符合预期，只需要在入口加上：

```python
import json 
import ujson
def monkey_patch_json(): 
    json.__name__ = 'ujson' 
    json.dumps = ujson.dumps 
    json.loads = ujson.loads 
monkey_patch_json()
```

## 23. CPython 退出时是否释放所有内存分配

[[参考]](https://docs.python.org/zh-cn/3.8/faq/design.html#why-isn-t-all-memory-freed-when-cpython-exits)

当 Python 退出时，从全局命名空间或 Python 模块引用的对象并不总是被释放。 如果存在循环引用，则可能发生这种情况，C库分配的某些内存也是不可能释放的（例如像 Purify 这样的工具）。 但是，Python在退出时清理内存并尝试销毁每个对象。

## 24. Python 中 is 和 == 有什么区别

`is` 比较两个对象的 id 值是否相等，即是否指向同一个内存地址；`==` 比较两个对象的值是否相等，默认调用对象的 `__eq__()` 方法。

## 25. gbk 和 utf8 的区别

GBK编码专门用来解决中文编码的，是双字节的。

UTF－8 编码是用以解决国际上字符的一种多字节编码，它对英文使用8位（即一个字节），中文使用24位（三个字节）来编码。

## 26. 遍历字典可以用什么方法

```python
my_dict = {'name': 'Jack', 'age': 26, 'address': 'Downtown', 'phone': '1234567890'}
for i in my_dict: # 遍历字典中的键
    print(i)
for key in my_dict.keys():   # 遍历字典中的键
       print(key)
for value in my_dict.values():  # 遍历字典中的值
    print(value)
for item in my_dict.items():  # 遍历字典中的元素
    print(item)
```

## 27. 反转列表的方法

* reversed() 迭代器
* sorted() 指定 reverse
* [::-1]

## 28. Python 元组中等元组转为字典

```python
my_tuple = ((1, '2'), (3, '4'), (5, '6'))
my_dict = dict((y,x) for x,y in my_tuple)
```

## 29. range 在 Python2 和 Python3 里的区别

py2 中，range 得到一个列表；

py3 中，range 得到一个生成器；

## 30. `__init__.py` 文件的作用和意义

[[参考]](https://blog.csdn.net/wokaowokaowokao12345/article/details/128934877)

这个文件定义了包的属性和方法，可以只是一个空文件，但是必须存在。一个文件夹根目录下存在`__init__.py`那就会认为该文件夹是Python包，否则那这个文件夹就是一个普通的文件夹。

## 31. Python 列表去重

```python
my_list = list(set(my_list))
```
