---
title: "剑指offer复习笔记(1)"
date: 2023-04-05T22:38:12+08:00
lastmod: 2023-04-05T22:38:12+08:00
author: ["Achilles"]
# keywords: 
# - 
categories: # 没有分类界面可以不填写
- 
tags: ["算法题","学习笔记"] # 标签
description: "leetcode 剑指offer 前25题提示清单"
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
    image: "posts/algo/offer1/cover.jpg" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## 1. [剑指 Offer 03. 数组中重复的数字(简单)](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

<div align=center><img src="03.png" /></div>

* 哈希表：用哈希表（Set）记录遍历到的数字，若找到重复的数字则返回。

* **原地交换**：数组元素的**索引**和**值**是一对多的关系。因此，可遍历数组并通过交换操作，使元素的索引与值一一对应（即$nums[i]=i$）。

  **算法流程**

  1. 遍历数组，索引初始值i=0;
     1. 若`nums[i]=i`：说明该数字已在对应的索引处，无需交换，跳过；
     2. 若`nums[nums[i]]=nums[i]`：说明索引nums[i]处和索引i处的值均为nums[i]，即找到一组重复，返回nums[i]；
     3. 否则交换nums[nums[i]]与nums[i]；
  2. 若遍历完未返回，返回-1。

  **复杂度**：时间O(N)，空间O(1)

  **代码**：

  ```python
  class Solution:
      def findRepeatNumber(self, nums: List[int]) -> int:
          n=len(nums)
          i=0
          while i<n:
              if nums[i]==i:
                  i+=1
                  continue
              if nums[nums[i]]==nums[i]:
                  return nums[i]
              nums[nums[i]],nums[i]=nums[i],nums[nums[i]]
  ```

  > Python中a,b=c,d的原理是暂存元组(c,d)，然后按左右顺序赋值，在此处需要先给nums[nums[i]]赋值。

## 2. [剑指 Offer 04. 二维数组中的查找(中等)](https://leetcode.cn/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

<div align=center><img src="04.png" /></div>

* 暴力法：时间复杂度O(MN)，未利用到数组的排序信息

* **二分法**：时间复杂度O(M+N)

  如下图，考虑将二维数组旋转45°，类似一棵二叉搜索树：故我们可以从数组的左下角(n-1,0)或右上角(0,m-1)开始，运用二分的思想进行搜索。

  <div align=center><img src="04_1.png" style="zoom: 50%;" /></div>

  ```python
  class Solution:
      def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
          i,j=len(matrix)-1,0
          while 0<=i and j<len(matrix[0]):
              if target==matrix[i][j]: return True
              elif target>matrix[i][j]: j+=1
              else: i-=1
          return False
  ```

## 3. [剑指 Offer 05. 替换空格(简单)](https://leetcode.cn/problems/ti-huan-kong-ge-lcof/)

<div align=center><img src="05.png" /></div>

* 库函数:`return s.replace(' ','%20')`

* 遍历添加：初始化一个新的str，时间空间复杂度O(N)

* **原地修改**(python str是不可修改，无法实现)：

  1. 遍历得到空格数`cnt`
  2. 修改s长度为`len+2*cnt`
  3. 倒序遍历，i指向原字符串末尾，j指向新字符串末尾，当`i=j`时跳出（左方已没有空格）；
     1. `s[i]=' '`：s[j-2:j]='%20'，j-=2
     2. `s[i]!=' '`：s[j]=s[i]

  ```C++
  string replaceSpace(string s) {
      int count = 0, len = s.size();
      for (char c : s) {
          if (c == ' ') count++;
      }
      s.resize(len + 2 * count);
      for(int i = len - 1, j = s.size() - 1; i < j; i--, j--) {
          if (s[i] != ' ')
              s[j] = s[i];
          else {
              s[j - 2] = '%';
              s[j - 1] = '2';
              s[j] = '0';
              j -= 2;
          }
      }
      return s;
  }

## 4. [剑指 Offer 06. 从尾到头打印链表(简单)](https://leetcode.cn/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

<div align=center><img src="06.png" /></div>

* 辅助栈：遍历链表，将各节点入栈，返回倒序列表。时间空间复杂度O(N)

* 递归：时间空间复杂度O(N)

  ```python
  class Solution:
      def reversePrint(self, head: ListNode) -> List[int]:
          return self.reversePrint(head.next) + [head.val] if head else []
  ```

## 5. [剑指 Offer 07. 重建二叉树(中等)](https://leetcode.cn/problems/zhong-jian-er-cha-shu-lcof/)

<div align=center><img src="07.png" /></div>

* **递归**：

  preorder=[root,L,R], inorder=[L,root,R]

  找到root在inorder中的下标，构建root的左右子树

  ```python
  class Solution:
      def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
          if not preorder:
              return None
          root=TreeNode(preorder[0])
          rootidx=inorder.index(preorder[0])
          root.left=self.buildTree(preorder[1:rootidx+1],inorder[:rootidx])
          root.right=self.buildTree(preorder[rootidx+1:],inorder[rootidx+1:])
          return root
  ```

* 迭代：待续

## 6. [剑指 Offer 09. 用两个栈实现队列(简单)](https://leetcode.cn/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

<div align=center><img src="09.png" /></div>

* 双栈：将一个栈当作输入栈，用于将数据入队；另一个栈当作输出栈，用于数据出队。每次出队时，若输出栈不为空，则直接从输出栈弹出；否则现将所有数据从输入栈弹出并压入输出栈，再从输出栈弹出。
  ```python
  class CQueue:
      def __init__(self):
          self.stack1, self.stack2=[],[]
      def appendTail(self, value: int) -> None:
          self.stack1.append(value)
      def deleteHead(self) -> int:
          if self.stack2: return self.stack2.pop()   
          if not self.stack1: return -1
          while self.stack1:
              self.stack2.append(self.stack1.pop())
          return self.stack2.pop()
  ```

## 7. [剑指 Offer 10- I. 斐波那契数列(简单)](https://leetcode.cn/problems/fei-bo-na-qi-shu-lie-lcof/)

<div align=center><img src="10.png" /></div>

* 动态规划：

  ```python
  class Solution:
      def fib(self, n: int) -> int:
          a,b=0,1
          while n:
              a,b=b,a+b
              n-=1
          return a%(10**9+7)
  ```

* **矩阵快速幂**：时间复杂度O(log n)，空间复杂度O(1)

$$
\left[\begin{matrix}
    1&1 \\\\
    1&0
\end{matrix}
\right]
\left[\begin{matrix}
    F(n) \\\\
    F(n-1)
\end{matrix}
\right]
= \left[
\begin{matrix}
    F(n)+F(n-1)\\\\
    F(n)
\end{matrix}
\right]
=\left[\begin{matrix}
    F(n+1) \\\\
    F(n)
\end{matrix}\right]
$$

$$
  \left[\begin{matrix}
     F(n+1) \\\\
     F(n)
    \end{matrix}\right]
    =
    \left[\begin{matrix}
     1&1 \\\\
     1&0
    \end{matrix}
    \right]^n
    \left[\begin{matrix}
     F(1) \\\\
     F(0)
    \end{matrix}\right]
$$

  令：
$$
  M=\left[\begin{matrix}
     1&1 \\\\
     1&0
    \end{matrix}
    \right]
$$
  关键在于快速计算矩阵M的n次幂。

  ```python
  class Solution:
      def fib(self, n: int) -> int:
          MOD = 10 ** 9 + 7
          if n < 2:
              return n
          def multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
              c = [[0, 0], [0, 0]]
              for i in range(2):
                  for j in range(2):
                      c[i][j] = (a[i][0] * b[0][j] + a[i][1] * b[1][j]) % MOD
              return c
          def matrix_pow(a: List[List[int]], n: int) -> List[List[int]]:
              ret = [[1, 0], [0, 1]]
              while n > 0:
                  if n & 1:
                      ret = multiply(ret, a)
                  n >>= 1
                  a = multiply(a, a)
              return ret
          res = matrix_pow([[1, 1], [1, 0]], n - 1)
          return res[0][0]
  ```


## 8. [剑指 Offer 11. 旋转数组的最小数字(简单)](https://leetcode.cn/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

<div align=center><img src="11.png" /></div>

* **二分法**：

  如下图，考虑数组最后一个元素x，最小值右侧的元素一定小于等于x，最小值左侧的元素一定大于等于x。

  可以分为三种情况：

  * nums[mid]>x：left=mid+1
  * nums[mid]<x：right=mid
  * nums[mid]=x：此时无法判断nums[mid]在最小值左侧还是右侧，但**可以确定nums[right]有nums[mid]这个替代值，故可以忽略右端点**。right-=1

  <div align=center><img src="11_1.png" /></div>

  ```python
  class Solution:
      def minArray(self, numbers: List[int]) -> int:
          n=len(numbers)
          l,r=0,n-1
          while l<r:
              mid = l+(r-l)//2
              if numbers[mid]>numbers[r]:
                  l=mid+1
              elif numbers[mid]<numbers[r]:
                  r=mid
              else: r-=1
          return numbers[l]
  ```

## 9. [剑指 Offer 12. 矩阵中的路径(中等)](https://leetcode.cn/problems/ju-zhen-zhong-de-lu-jing-lcof/)

<div align=center><img src="12.png" /></div>

* **回溯**(DFS)：时间复杂度$O(MN3^L)$，空间复杂度O(MN)

  用backtracking(i,j,idx)表示从位置(i,j)出发能否匹配字符串word[idx:]，执行步骤如下：

  * 若board\[i][j]!=word[idx]，不匹配返回False
  * 若当前字符匹配且到了字符串末尾，返回True
  * 否则，遍历当前相邻位置

  ```python
  class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m,n=len(board),len(board[0])
        vis=set()
        directions=[(1,0),(-1,0),(0,1),(0,-1)]
        def backtracking(i,j,idx):
            if board[i][j]!=word[idx]:
                return False
            if idx==len(word)-1:
                return True
            vis.add((i,j))
            for di,dj in directions:
                ii,jj=i+di,j+dj
                if 0<=ii<m and 0<=jj<n and (ii,jj) not in vis:
                    if backtracking(ii,jj,idx+1):
                        return True
            vis.remove((i,j))
            return False
        for i in range(m):
            for j in range(n):
                if backtracking(i,j,0):
                    return True
        return False
  ```

## 10. [剑指 Offer 14- I. 剪绳子(中等)](https://leetcode.cn/problems/jian-sheng-zi-lcof/)

<div align=center><img src="14.png" /></div>

* **动态规划**：时间空间复杂度O(n)

  dp[1]=1,dp[2]=1,状态转移方程：
  $$ dp[i]=max(2\*dp[i-2],3\*dp[i-3],2\*(i-2),3\*(i-3)) $$

  ```python
  class Solution:
      def cuttingRope(self, n: int) -> int:
          dp=[0]*(n+1)
          dp[1]=dp[2]=1
          for i in range(3,n+1):
              dp[i]=max(2*dp[i-2],3*dp[i-3],2*(i-2),3*(i-3))
          return dp[n]
  ```

* 数学推导（贪心）:

  ```python
  class Solution:
      def cuttingRope(self, n: int) -> int:
          if n<4:
              return n-1
          a,b=n//3,n%3
          if b==1: return int(math.pow(3,a-1)*4)
          elif b==2: return int(math.pow(3,a)*2)
          return int(math.pow(3,a))
  ```

## 11. [剑指 Offer 15. 二进制中1的个数(简单)](https://leetcode.cn/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

<div align=center><img src="15.png" /></div>

* 循环遍历

* **位运算优化**：时间复杂度O(log n)。每次将n与n-1做与操作，可以将n的最低位的1变为0。例如`6(110)&5(101)=4(100)`。

  ```python
  class Solution:
      def hammingWeight(self, n: int) -> int:
          ans=0
          while n:
              n&=(n-1)
              ans+=1
          return ans
  ```

## 12. [剑指 Offer 16. 数值的整数次方(中等)](https://leetcode.cn/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

<div align=center><img src="16.png" style="zoom:50%;" /></div>

* **快速幂乘法**：递归和迭代

  ```python
  class Solution:
      def myPow(self, x: float, n: int) -> float:
        	# 迭代
          def quickMul(n):
              ans=1.0
              xx=x
              while n:
                  if n&1: ans*=xx
                  xx*=xx
                  n>>=1
              return ans
          return quickMul(n) if n>=0 else 1/quickMul(-n)
  ```

  ```python
  class Solution:
      def myPow(self, x: float, n: int) -> float:
          def quickMul(n):
              if n==0:
                  return 1.0
              y=quickMul(n//2)
              return y*y if n&1==0 else y*y*x
          return quickMul(n) if n>=0 else 1/quickMul(-n)
  ```

## 13. [剑指 Offer 17. 打印从1到最大的n位数(简单)](https://leetcode.cn/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

<div align=center><img src="17.png" style="zoom: 50%;" /></div>

* **DFS**：用字符串来正确表示大数（本题不需要），依次遍历长度1~n的数，第一位只能为1~9，其他位为0~9。

  ```python
  class Solution:
      def printNumbers(self, n: int) -> List[int]:
          ans=[]
          def dfs(k,n,s):
              if k==n:
                  ans.append(int(s))
                  return
              for i in range(10):
                  dfs(k+1,n,s+str(i))
          for i in range(1,n+1):
              for j in range(1,10):
                  dfs(1,i,str(j))
          return ans
  ```

## 15. [剑指 Offer 18. 删除链表的节点(简单)](https://leetcode.cn/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

<div align=center><img src="18.png" style="zoom: 50%;" /></div>

* 前驱结点遍历：考虑要删除节点为头结点的特殊情况。

  ```python
  class Solution:
      def deleteNode(self, head: ListNode, val: int) -> ListNode:
          if head.val==val: return head.next
          pre,p=head,head.next
          while p and p.val!=val:
              pre,p=p,p.next
          if p: pre.next = p.next
          return head
  ```

## 16. [剑指 Offer 19. 正则表达式匹配(困难)](https://leetcode.cn/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

<div align=center><img src="19.png" style="zoom:67%;" /></div>

* **动态规划**：

  dp\[i][j]表示s\[:i]与p\[:j]是否匹配，考虑以下情况：

  * `p[j]='*'`: 若`s[i]`与`p[j-1]`不匹配，则`p[j-1]`匹配零次即为`dp[i][j]=dp[i][j-2]`；否则`p[j-1]`匹配1次或多次，即`dp[i][j]=dp[i-1][j] or dp[i][j-2]`
  * `p[j]!='*'`：若`s[i]`与`p[j]`匹配，则`dp[i][j]=dp[i-1[j-1]`
  * `s[i]`与`p[j]`匹配时满足，`s[i]=p[j] or p[j]=='.'`
  * 初始化时，`dp[0][0]=True`，考虑s为空数组，只有p的偶数位为`*`时能够匹配

  ```python
  class Solution:
      def isMatch(self, s: str, p: str) -> bool:
          m,n=len(s),len(p)
          dp = [[False for _ in range(n+1)]for _ in range(m+1)]
          dp[0][0]=True
          for j in range(2,n+1,2):
              if p[j-1]=='*':
                  dp[0][j]=dp[0][j-2]
          for i in range(1,m+1):
              for j in range(1,n+1):
                  if p[j-1]=='*':
                      if s[i-1]==p[j-2] or p[j-2]=='.':
                          dp[i][j]=dp[i-1][j] or dp[i][j-2]
                      else:
                          dp[i][j]=dp[i][j-2]
                  else:
                      if s[i-1]==p[j-1] or p[j-1]=='.':
                          dp[i][j]=dp[i-1][j-1]
          return dp[-1][-1]
  ```

## 17. [剑指 Offer 20. 表示数值的字符串(中等)](https://leetcode.cn/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

<div align=center><img src="20.png" style="zoom:50%;" /></div>

* 模拟

* **有限状态机**：定义状态->画状态转移图->编写代码

  * 字符类型：空格` `，数字`1-9`，正负号`+-`，小数点`.`，幂符号`eE`。

  * 状态定义：

    1. 开始的空格
    2. 幂符号前的正负号
    3. 小数点前的数字
    4. 小数点，小数点后的数字
    5. 当小数点前为空格时，小数点和小数点后的数字
    6. 幂符号
    7. 幂符号后的正负号
    8. 幂符号后的数字
    9. 结尾的空格

    状态转移图：

    <div align=center><img src="20_1.png" style="zoom:67%;" /></div>

    ```python
    class Solution:
        def isNumber(self, s: str) -> bool:
            states = [
                {' ':0,'s':1,'d':2,'.':4}, # 0. start with 'blank'
                {'d':2,'.':4},             # 1. 'sign' before 'e'
                {'d':2,'.':3,'e':5,' ':8}, # 2. 'digit' before 'dot'
                {'d':3,'e':5,' ':8},       # 3. 'digit' after 'dot'
                {'d':3},                   # 4. 'digit' after 'dot' (‘blank’ before 'dot')
                {'s':6,'d':7},             # 5. 'e'
                {'d':7},                   # 6. 'sign' after 'e'
                {'d':7,' ':8},             # 7. 'digit' after 'e'
                {' ':8}                    # 8. end with 'blank'
            ]
            p = 0                          # start with state 0
            for c in s:
                if '0'<=c<='9': t = 'd'    # digit
                elif c in "+-": t = 's'    # sign
                elif c in "eE": t = 'e'    # e or E
                elif c in ". ": t =  c     # dot, blank
                else: t = '?'              # unknown
                if t not in states[p]: return False
                p = states[p][t]
            return p in (2, 3, 7, 8)
    ```

## 18. [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面(简单)](https://leetcode.cn/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

<div align=center><img src="21.png" style="zoom:50%;" /></div>

* 双指针交换

  ```python
  class Solution:
      def exchange(self, nums: List[int]) -> List[int]:
          i,j=0,len(nums)-1
          while i<j:
              while i<j and nums[i]%2: i+=1
              while i<j and nums[j]%2==0: j-=1
              nums[i],nums[j]=nums[j],nums[i]
          return nums
  ```

## 19. [剑指 Offer 22. 链表中倒数第k个节点(简单)](https://leetcode.cn/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

<div align=center><img src="22.png" style="zoom: 50%;" /></div>

* **双指针**：让快指针先走k个节点

  ```python
  class Solution:
      def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
          former, latter = head, head
          for _ in range(k):
              former = former.next
          while former:
              former, latter = former.next, latter.next
          return latter
  ```

## 20. [剑指 Offer 24. 反转链表(简单)](https://leetcode.cn/problems/fan-zhuan-lian-biao-lcof/)

<div align=center><img src="24.png" style="zoom:50%;" /></div>

* **迭代（双指针）**：

  ```python
  class Solution:
      def reverseList(self, head: ListNode) -> ListNode:
          pre,cur=None,head
          while cur:
              cur.next,pre,cur=pre,cur,cur.next
          return pre
  ```

* **递归**：

  ```python
  class Solution:
      def reverseList(self, head: ListNode) -> ListNode:
          def recur(pre,cur):
              if not cur: return pre
              res = recur(cur,cur.next)
              cur.next=pre
              return res
          return recur(None,head)
  ```

## 21. [剑指 Offer 25. 合并两个排序的链表(简单)](https://leetcode.cn/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

<div align=center><img src="25.png" style="zoom:50%;" /></div>

* **递归**：

  ```python
  class Solution:
      def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
          if not l1 or not l2:
              return l1 or l2
          if l1.val<l2.val:
              l1.next = self.mergeTwoLists(l1.next,l2)
              return l1
          else:
              l2.next = self.mergeTwoLists(l1,l2.next)
              return l2
  ```

## 22. [剑指 Offer 26. 树的子结构(中等)](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

<div align=center><img src="26.png" style="zoom: 50%;" /></div>

* 先序遍历：

  分为两步，先序遍历树*A*中的每个节点$n_A$；判断以$n_A$为根节点的子树是否包含树B。

  same函数判断以$n_A$为根节点的子树是否包含树B，若B为空，则表示树B匹配完成，返回True；若A为空，则说明越过A，返回False；若A和B值不同，则不匹配，返回False。

  ```python
  class Solution:
      def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
          def same(A,B):
              if not B: return True
              if not A: return False
              return A.val==B.val and same(A.left,B.left) and same(A.right,B.right)
          return bool(A and B) and (same(A,B) or self.isSubStructure(A.left,B) or self.isSubStructure(A.right,B))
  ```

## 23. [剑指 Offer 27. 二叉树的镜像(简单)](https://leetcode.cn/problems/er-cha-shu-de-jing-xiang-lcof/)

<div align=center><img src="27.png" style="zoom:50%;" /></div>

* 递归：注意同时赋值或者临时保存子树

  ```python
  class Solution:
      def mirrorTree(self, root: TreeNode) -> TreeNode:
          if not root:
              return None
          root.left,root.right = self.mirrorTree(root.right),self.mirrorTree(root.left)
          return root
  ```

* 迭代：用栈保存节点

  ```python
  class Solution:
      def mirrorTree(self, root: TreeNode) -> TreeNode:
          if not root:
              return None
          stack=[root]
          while stack:
              node = stack.pop()
              if node.left: stack.append(node.left)
              if node.right: stack.append(node.right)
              node.left,node.right=node.right,node.left
          return root
  ```

## 24. [剑指 Offer 28. 对称的二叉树(简单)](https://leetcode.cn/problems/dui-cheng-de-er-cha-shu-lcof/)

<div align=center><img src="28.png" style="zoom:50%;" /></div>

* 递归：

  ```python
  class Solution:
      def isSymmetric(self, root: TreeNode) -> bool:
          def recur(L,R):
              if not L and not R: return True
              if not L or not R: return False
              return L.val==R.val and recur(L.left,R.right) and recur(L.right,R.left)
          return not root or recur(root.left,root.right)
  ```

* 迭代：

  ```python
  class Solution:
      def isSymmetric(self, root: TreeNode) -> bool:
          if not root or not (root.left or root.right):
              return True
          queue=[root.left,root.right]
          while queue:
              l=queue.pop(0)
              r=queue.pop(0)
              if not l and not r:
                  continue
              if not l or not r or l.val!=r.val:
                  return False
              queue.append(l.left)
              queue.append(r.right)
              queue.append(l.right)
              queue.append(r.left)
          return True
  ```

## 25. [剑指 Offer 29. 顺时针打印矩阵(简单)](https://leetcode.cn/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

<div align=center><img src="29.png" style="zoom:50%;" /></div>

* **设置边界**：设置top，bottom，left，right，遍历

  ```python
  class Solution:
      def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
          if not matrix: return []
          top,bottom,left,right,ans=0,len(matrix)-1,0,len(matrix[0])-1,[]
          while True:
              for j in range(left,right+1): ans.append(matrix[top][j])
              top+=1
              if top>bottom: break
              for i in range(top,bottom+1): ans.append(matrix[i][right])
              right-=1
              if right<left: break
              for j in range(right,left-1,-1): ans.append(matrix[bottom][j])
              bottom-=1
              if top>bottom: break
              for i in range(bottom,top-1,-1): ans.append(matrix[i][left])
              left+=1
              if right<left: break
          return ans
  ```

