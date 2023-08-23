---
title: "LeetCode 热题 100 : 51~100"
date: 2023-08-21T16:27:41+08:00
lastmod: 2023-08-21T16:27:41+08:00
author: ["Achilles"]
# keywords: 
# - 
categories: # 没有分类界面可以不填写
- 
tags: ["算法题","学习笔记"] # 标签
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
    image: "posts/algo/hot2/cover.png" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## 图论

### 51. [岛屿数量（中等）](https://leetcode.cn/problems/number-of-islands/)

<div align=center><img src="51.png" style="zoom:50%;" /></div>

* DFS

  当面试时不能修改原数组时需要使用标记。

  ```python
  def numIslands(self, grid: List[List[str]]) -> int:
      directions=[(1,0),(0,1),(-1,0),(0,-1)]
      m,n,ans=len(grid),len(grid[0]),0
      vis = set()
      def dfs(i,j):
        vis.add((i,j))
        for di,dj in directions:
            ii,jj = i+di,j+dj
            if 0<=ii<m and 0<=jj<n and grid[ii][jj]=='1' and (ii,jj) not in vis:
                dfs(ii,jj)
      for i in range(m):
          for j in range(n):
              if grid[i][j]=='1' and (i,j) not in vis:
                  dfs(i,j)
                  ans+=1
      return ans
  ```

### 52.  [腐烂的橘子（中等）](https://leetcode.cn/problems/rotting-oranges/)

<div align=center><img src="52.png" style="zoom:50%;" /></div>

* BFS

  ```python
  def orangesRotting(self, grid: List[List[int]]) -> int:
      m,n=len(grid),len(grid[0])
      direction=[(1,0),(0,1),(-1,0),(0,-1)]
      ans,remain,stack=0,0,[]
      for i in range(m):
          for j in range(n):
              if grid[i][j]==2: stack.append((i,j))
              elif grid[i][j]==1: remain+=1
      while stack and remain:
          for _ in range(len(stack)):
              i,j=stack.pop(0)
              for di,dj in direction:
                  ii,jj=i+di,j+dj
                  if 0<=ii<m and 0<=jj<n and grid[ii][jj]==1:
                      grid[ii][jj]=2
                      remain-=1
                      stack.append((ii,jj))
          ans+=1
      return ans if remain==0 else -1
  ```

### 53. ⭐️ [课程表（中等）](https://leetcode.cn/problems/course-schedule/)

<div align=center><img src="53.png" style="zoom:50%;" /></div>

* DFS

  定义节点三个状态，0：待搜索；1：正在搜索；2：完成搜索。如果在 DFS 过程中遇到了状态为 1 的节点，说明遇到了环，无法完成所有课程的学习。

  ```python
  def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
      edges = defaultdict(list)
      for cour,pre in prerequisites:
          edges[pre].append(cour)
      visited = [0]*numCourses
      valid=True
      def dfs(u):
          nonlocal valid
          visited[u]=1 # 正在搜索
          for v in edges[u]:
              if visited[v]==0: # 当前节点未被搜索
                  dfs(v)
                  if not valid: return
              elif visited[v]==1: # 遇到环
                  valid=False
                  return
          visited[u]=2 # 完成搜索
      for i in range(numCourses):
          if valid and not visited[i]:
              dfs(i)
      return valid
  ```

* BFS

  若一个课程节点的入度为 0，则表示该课程没有先修课程或先修课程已经学完，可以学习当前课程，去掉该节点的所有出边，则表示它的相邻节点少了一门先修课程。维护一个队列，不断地将入度为 0 的课程节点加入，直到答案中包含所有的节点（得到了一种拓扑排序）或者不存在没有入边的节点（图中包含环）。
  
  ```python
  def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
      edges = defaultdict(list)
      indeg = [0]*numCourses
      for cour,pre in prerequisites:
          edges[pre].append(cour)
          indeg[cour]+=1 # 入度
      q = collections.deque([u for u in range(numCourses) if indeg[u]==0]) # 将入度为 0 的节点加入队列
      visited=0 # 遍历的节点数
      while q:
          u=q.popleft()
          visited+=1
          for v in edges[u]:
              indeg[v]-=1
              if indeg[v]==0: # 入度为 0，加入队列
                  q.append(v)
      return visited==numCourses 
  ```
  

### 54. [实现 Trie (前缀树)（中等）](https://leetcode.cn/problems/implement-trie-prefix-tree/)

<div align=center><img src="54.png" style="zoom:50%;" /></div>

* 设计

  将 Trie 视作一个节点

  ```python
  class Trie:
      def __init__(self):
          self.child = [None]*26
          self.isEnd = False
      def insert(self, word: str) -> None:
          node = self
          for c in word:
              idx = ord(c)-ord('a')
              if not node.child[idx]:
                  node.child[idx]=Trie()
              node=node.child[idx]
          node.isEnd=True
      def searchPrefix(self, word):
          node=self
          for c in word:
              idx = ord(c)-ord('a')
              if not node.child[idx]:
                  return None
              node = node.child[idx]
          return node
      def search(self, word: str) -> bool:
          node = self.searchPrefix(word)
          return node is not None and node.isEnd
      def startsWith(self, prefix: str) -> bool:
          node = self.searchPrefix(prefix)
          return node is not None
  ```

## 回溯

### 55. [全排列（中等）](https://leetcode.cn/problems/permutations/)

<div align=center><img src="55.png" style="zoom:50%;" /></div>

* 回溯

  回溯到 idx 处时，依次将后面未排列的数交换到 idx 处。

  ```python
  def permute(self, nums: List[int]) -> List[List[int]]:
      ans=[]
      n=len(nums)
      def backtracking(idx):
          if idx==n: 
              ans.append(nums[:])
          for i in range(idx,n):
              nums[idx],nums[i]=nums[i],nums[idx]
              backtracking(idx+1)
              nums[idx],nums[i]=nums[i],nums[idx]
      backtracking(0)
      return ans
  ```

### 56. [子集（中等）](https://leetcode.cn/problems/subsets/)

<div align=center><img src="56.png" style="zoom:50%;" /></div>

* 回溯

  ```python
  def subsets(self, nums: List[int]) -> List[List[int]]:
      ans = []
      def dfs(path,idx):
          if idx==len(nums):
              ans.append(path[:])
              return
          dfs(path+[nums[idx]],idx+1)
          dfs(path,idx+1)
      dfs([],0)
      return ans
  ```

### 57. [电话号码的字母组合（中等）](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

<div align=center><img src="57.png" style="zoom:50%;" /></div>

* 回溯

  ```python
  def letterCombinations(self, digits: str) -> List[str]:
      maps = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
      if not digits: return []
      ans,n=[],len(digits)
      def dfs(idx,path):
          if idx==n:
              ans.append(''.join(path))
              return
          for c in maps[digits[idx]]:
              path.append(c)
              dfs(idx+1,path)
              path.pop()
      dfs(0,[])
      return ans
  ```

  

### 58. [组合总和（中等）](https://leetcode.cn/problems/combination-sum/)

<div align=center><img src="58.png" style="zoom:50%;" /></div>

* 回溯+剪枝

  ```python
  def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
      candidates.sort(reverse=True)
      ans,n=[],len(candidates)
      def backtracking(idx,path,s):
          if s==target:
              ans.append(path[:])
              return
          if s>target: return
          for i in range(idx,n):
              path.append(candidates[i])
              backtracking(i,path,s+candidates[i])
              path.pop()
      backtracking(0,[],0)
      return ans
  ```

### 59. [括号生成（中等）](https://leetcode.cn/problems/generate-parentheses/)

<div align=center><img src="59.png" style="zoom:50%;" /></div>

* 回溯

  定义回溯函数 backtracking(path,left,right)，其中 left 和 right 表示还能添加的左括号和右括号的数量。每次添加左括号时，left=left-1, right=right+1；添加右括号时，right=right-1。

  ```python
  def generateParenthesis(self, n: int) -> List[str]:
      ans = []
      def backtracking(path,left,right):
          if len(path)==2*n:
              ans.append(''.join(path))
              return
          if left>0:
              path.append('(')
              backtracking(i+1,path,left-1,right+1)
              path.pop()
          if right>0:
              path.append(')')
              backtracking(i+1,path,left,right-1)
              path.pop()
      backtracking([],n,0)
      return ans
  ```


### 60. ⭐️ [单词搜索（中等）](https://leetcode.cn/problems/word-search/)

<div align=center><img src="60.png" style="zoom:50%;" /></div>

* 回溯

  设置标志，注意返回条件

  ```python
  def exist(self, board: List[List[str]], word: str) -> bool:
      directions = [(1,0),(0,1),(-1,0),(0,-1)]
      m,n = len(board),len(board[0])
      length = len(word)
      seen = set()
      def dfs(i,j,cnt):
          if board[i][j]!=word[cnt]: return False
          if cnt==length-1: return True
          flag=False
          seen.add((i,j))
          for di,dj in directions:
              ii,jj=i+di,j+dj
              if (ii,jj) not in seen and 0<=ii<m and 0<=jj<n:
                  if dfs(ii,jj,cnt+1):
                      flag = True
                      break
          seen.remove((i,j))
          return flag
      for i in range(m):
          for j in range(n):
              if dfs(i,j,0):
                  return True
      return False
  ```

### 61. [分割回文串（中等）](https://leetcode.cn/problems/palindrome-partitioning/)

<div align=center><img src="61.png" style="zoom:50%;" /></div>

* 回溯

  ```python
  def partition(self, s: str) -> List[List[str]]:
      def check(i,j):
          while i<j:
              if s[i]!=s[j]: return False
              i+=1
              j-=1
          return True
      n=len(s)
      ans=[]
      def dfs(idx,path):
          if idx==n:
              ans.append(path[:])
              return
          for j in range(idx,n):
              if check(idx,j):
                  path.append(s[idx:j+1])
                  dfs(j+1,path)
                  path.pop()
      dfs(0,[])
      return ans
  ```

### 62. ⭐️ [N 皇后（困难）](https://leetcode.cn/problems/n-queens/)

<div align=center><img src="62.png" style="zoom:50%;" /></div>

* 回溯

  按行生成合法的棋盘

  ```python
  def solveNQueens(self, n: int) -> List[List[str]]:
      ans=[]
      grid = [['.' for _ in range(n)]for _ in range(n)]
      def check(i,j,grid):
          for ii in range(n):
              for jj in range(n):
                  if grid[ii][jj]=='Q' and (ii==i or jj==j or ii+jj==i+j or ii-jj==i-j):
                          return False
          return True
      def backtracking(i):
          if i==n:
              ans.append([''.join(tmp) for tmp in grid])
              return
          for j in range(n):
              if check(i,j,grid):
                  grid[i][j]='Q'
                  backtracking(i+1)
                  grid[i][j]='.'
      backtracking(0)
      return ans
  ```

## 二分查找

### 63. [搜索插入位置（简单）](https://leetcode.cn/problems/search-insert-position/)

<div align=center><img src="63.png" style="zoom:50%;" /></div>

* 二分查找

  ```python
  def searchInsert(self, nums: List[int], target: int) -> int:
      n=len(nums)
      i,j=0,n-1
      while i<=j:
          mid=i+(j-i)//2
          if target==nums[mid]: return mid
          elif target>nums[mid]: i=mid+1
          else: j=mid-1
      return i
  ```

### extra. [二分查找（简单）](https://leetcode.cn/problems/binary-search/)

<div align=center><img src="105.png" style="zoom:50%;" /></div>

* 二分查找

  ```python
  def search(self, nums: List[int], target: int) -> int:
      n=len(nums)
      left,right=0,n-1
      while left<=right:
          mid = left+(right-left)//2
          if nums[mid]==target: return mid
          elif nums[mid]>target: right=mid-1
          else: left=mid+1
      return -1
  ```

### 64. [搜索二维矩阵（中等）](https://leetcode.cn/problems/search-a-2d-matrix/)

<div align=center><img src="64.png" style="zoom:50%;" /></div>

* 二分查找

  从右上角出发

  ```python
  def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
      m,n=len(matrix),len(matrix[0])
      i,j=0,n-1
      while i<m and 0<=j:
          if target==matrix[i][j]: return True
          elif target>matrix[i][j]: i+=1
          else: j-=1
      return False
  ```



### 65. ⭐️ [在排序数组中查找元素的第一个和最后一个位置（中等）](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

<div align=center><img src="65.png" style="zoom:50%;" /></div>

* 二分查找

  设置一个 lower 标志，控制二分查找是查找第一个**大于等于** target 值还是第一个**大于** target 值的下标。

  ```python
  def searchRange(self, nums: List[int], target: int) -> List[int]:
      def bisearch(nums,target,lower):
          left,right,ans=0,len(nums)-1,len(nums)
          while left<=right:
              mid = left+(right-left)//2
              # 若 lower = True，即使 mid=target，也继续向左搜索。否则，向右搜索第一个大于 target 的下标
              if nums[mid]>target or (lower and nums[mid]>=target): 
                  right=mid-1
                  ans=mid
              else: left=mid+1
          return ans
      n=len(nums)
      l = bisearch(nums,target,True)
      if l>=n or nums[l]!=target: return [-1,-1] # 判断 target 在数组中是否存在
      r = bisearch(nums,target,False)
      return [l,r-1]
  ```

### 66. ⭐️ [搜索旋转排序数组（中等）](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

<div align=center><img src="66.png" style="zoom:50%;" /></div>

* 二分查找

  判断 mid 左右是否有序。
  $$
  \begin{cases}
  left\leq mid: 左侧有序
  \begin{cases}
  left\leq tar<mid: tar\ 在\ mid\ 左侧\\\\
  tar\ 在\ mid\ 右侧
  \end{cases}\\\\
  右侧有序
  \begin{cases}
  mid<tar\leq right: tar\ 在\ mid\ 右侧\\\\
  tar\ 在\ mid\ 左侧
  \end{cases}
  \end{cases}
  $$

  ```python
  def search(self, nums: List[int], target: int) -> int:
      n=len(nums)
      left,right=0,n-1
      while left<=right:
          mid = (left+right)//2
          if target==nums[mid]: return mid
          if nums[left]<=nums[mid]:
              if nums[left]<=target<nums[mid]: right=mid-1
              else: left=mid+1
          else:
              if nums[mid]<target<=nums[right]: left=mid+1
              else: right=mid-1
      return -1
  
  # 标准库写法
  def search(self, nums: List[int], target: int) -> int:
      n=len(nums)
      left,right = 0,n-1
      while left<right:
          mid = left+(right-left)//2 # 防溢出
          if nums[mid]<target:
              left = mid+1
          else:
              right=mid
      return left if nums[left]==target else -1 # 判断是否合法
  ```

### 67. [寻找旋转排序数组中的最小值 II（困难）](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/)

<div align=center><img src="67.png" style="zoom:50%;" /></div>

* 二分查找

  每次将 mid 与 right 比较：

  * 若相等，则说明 right 有可替代的元素，舍弃，right-=1
  * 若 mid>right，则说明最小值在 mid 右侧，left=left+1
  * 若 mid<right，则说明最小值在 mid 左侧或是 mid

  <div align=center><img src="671.png" style="zoom:50%;" /></div>

  ```python
  def findMin(self, nums: List[int]) -> int:
      n=len(nums)
      l,r=0,n-1
      while l<r:
          mid = l+(r-l)//2
          if nums[mid]==nums[r]: r-=1
          elif nums[mid]>nums[r]: l=mid+1
          else: r=mid
      return nums[l]
  ```

### 68. ⭐️ [寻找两个正序数组的中位数（困难）](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

<div align=center><img src="68.png" style="zoom:50%;" /></div>

* 归并：时间复杂度 $O(m+n)$

* ⭐️ 二分查找

  将寻找中位数转换为在两个数组中寻找第 k 小的元素。

  每次循环比较两个数组的第 k//2 个元素，加入 nums1[k//2]<=nums2[k//2]，则说明 nums1 的前 k//2 个元素都不可能为中位数，将其从数组中移除，并将 k 减去相应的值，反之亦然。当 k=1 时，只需去两个数组中最小的值即可；当其中一个数组为空时，直接取另一个数组中对应的值。

  ```python
  def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
      def getKthElement(k):
          index1, index2 = 0, 0
          while True:
              # 特殊情况
              if index1 == m: return nums2[index2 + k - 1]
              if index2 == n: return nums1[index1 + k - 1]
              if k == 1: return min(nums1[index1], nums2[index2])
              # 正常情况
              newIndex1 = min(index1 + k // 2 - 1, m - 1)
              newIndex2 = min(index2 + k // 2 - 1, n - 1)
              pivot1, pivot2 = nums1[newIndex1], nums2[newIndex2]
              if pivot1 <= pivot2:
                  k -= newIndex1 - index1 + 1
                  index1 = newIndex1 + 1
              else:
                  k -= newIndex2 - index2 + 1
                  index2 = newIndex2 + 1
      m, n = len(nums1), len(nums2)
      totalLength = m + n
      if totalLength % 2 == 1:
          return getKthElement((totalLength + 1) // 2)
      else:
          return (getKthElement(totalLength//2) + getKthElement(totalLength//2+1)) / 2
  ```

* 划分数组：见[题解](https://leetcode.cn/problems/median-of-two-sorted-arrays/solution/xun-zhao-liang-ge-you-xu-shu-zu-de-zhong-wei-s-114/)

## 栈

### 69. [有效的括号（简单）](https://leetcode.cn/problems/valid-parentheses/)

<div align=center><img src="69.png" style="zoom:50%;" /></div>

* 栈

  若栈顶与遍历到的右括号不匹配，则返回 False；遍历结束检查栈内是否有多余的左括号。

  ```python
  def isValid(self, s: str) -> bool:
      com = ['()','{}','[]']
      n,stack=len(s),['?']
      for c in s:
          if c in ['(','{','[']: stack.append(c)
          else:
              tmp=stack.pop()
              if tmp+c not in com: return False
      return len(stack)==1
  ```

* 匹配

  循环匹配括号，判断字符串是否完全匹配。

  ```python
  def isValid(self, s: str) -> bool:
      while '()' in s or '{}' in s or '[]' in s:
          s=s.replace('()','')
          s=s.replace('{}','')
          s=s.replace('[]','')
      if not s: return True
      return False
  ```


### 70. ⭐️ [最小栈（中等）](https://leetcode.cn/problems/min-stack/)

<div align=center><img src="70.png" style="zoom:50%;" /></div>

* 辅助栈

  维护一个辅助栈，栈顶始终是当前栈的最小值。当入栈元素小于辅助栈栈顶元素时，将该元素入栈辅助栈，否则辅助栈栈顶元素大小不变。

  <div align=center><img src="70.gif" style="zoom:60%;" /></div>

  ```python
  class MinStack:
      def __init__(self):
          self.stack=[]
          self.min_stack=[inf]
      def push(self, val: int) -> None:
          self.stack.append(val)
          if val<self.min_stack[-1]:
              self.min_stack.append(val)
          else:
              self.min_stack.append(self.min_stack[-1])
      def pop(self) -> None:
          self.stack.pop()
          self.min_stack.pop()
      def top(self) -> int:
          return self.stack[-1]
      def getMin(self) -> int:
          return self.min_stack[-1]
  ```


### 71. [字符串解码（中等）](https://leetcode.cn/problems/decode-string/)

<div align=center><img src="71.png" style="zoom:50%;" /></div>

* 栈+模拟

  ```python
  def decodeString(self, s: str) -> str:
      stack=[]
      i=0
      while i<len(s):
          if 0<=ord(s[i])-ord('0')<=9:
              num=0
              while 0<=ord(s[i])-ord('0')<=9:
                  num=num*10+int(s[i])
                  i+=1
              stack.append(num)
              continue
          elif s[i]==']':
              tmp = []
              while stack and stack[-1]!='[':
                  c = stack.pop()
                  tmp.append(c)
              stack.pop() # 弹出左括号
              num = stack.pop()
              tmp = ''.join(tmp[::-1]*num)
              stack.append(tmp)
          else: stack.append(s[i]) # 左括号和字符入栈
          i+=1
      return ''.join(stack)
  ```

### 72. [每日温度（中等）](https://leetcode.cn/problems/daily-temperatures/)

<div align=center><img src="72.png" style="zoom:50%;" /></div>

* 单调栈

  维护一个单调栈，存储温度下标，保证栈内下标对应的温度呈单调下降。每当遇到比栈顶对应温度高的温度时，将栈顶出栈，对应下标的 answer[idx] = i-idx。

  ```python
  def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
      n=len(temperatures)
      answer,stack = [0]*n,[]
      for i,temp in enumerate(temperatures):
          while stack and temp>temperatures[stack[-1]]:
              idx = stack.pop()
              answer[idx]=i-idx
          stack.append(i)
      return answer
  ```


### 73. ⭐️ [柱状图中最大的矩形（困难）](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

<div align=center><img src="73.png" style="zoom:50%;" /></div>

* 暴力解法

* 单调栈

  遍历每一柱子，找到左右两侧最近的高度小于 h 的柱子，这两根柱子之间的高度均不小于 h，则以当前柱子高度为高度的矩形面积为 (right-left-1)*h。维护一个单调栈使得栈中元素的高度非递减。

  ```python
  def largestRectangleArea(self, heights: List[int]) -> int:
      n=len(heights)
      left,right=[-1]*n,[n]*n
      stack=[(-1,0)]
      for i,h in enumerate(heights):
          while stack and h<stack[-1][1]:
              ii,hh=stack.pop()
              right[ii]=i
          left[i]=stack[-1][0]
          stack.append((i,h))
      ans = max((right[i]-left[i]-1)*heights[i] for i in range(n))
      return ans
  ```

### extra. ⭐️ [最大矩形（困难）](https://leetcode.cn/problems/maximal-rectangle/)

<div align=center><img src="731.png" style="zoom:50%;" /></div>

* 前缀和+单调栈

  把本题转换为上一题的解法。

  ```python
  def maximalRectangle(self, matrix: List[List[str]]) -> int:
      if not matrix: return 0
      m,n=len(matrix),len(matrix[0])
      def get_heights(heights,n): # 单调栈求解单层最大矩形
          left,right=[-1]*n,[n]*n
          stack=[(-1,0)]
          for i,h in enumerate(heights):
              while stack and h<stack[-1][1]:
                  ii,hh=stack.pop()
                  right[ii]=i
              left[i]=stack[-1][0]
              stack.append((i,h))
          ans = max((right[i]-left[i]-1)*heights[i] for i in range(n))
          return ans
        
      for j in range(n):
          matrix[0][j]=1 if matrix[0][j]=='1' else 0
      for i in range(1,m): # 前缀和
          for j in range(n):
                  matrix[i][j]=matrix[i-1][j]+1 if matrix[i][j]=='1' else 0
      ans = max(get_heights(matrix[i],n) for i in range(m))
      return ans
  ```

## 堆

### 74. [数组中的第K个最大元素（中等）](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

<div align=center><img src="74.png" style="zoom:50%;" /></div>

* 堆排序：$O(n\log\ k)$

  维护一个最小堆，每当遍历到的数大于堆顶元素，则 pushpop 更新堆，保证堆元素为已遍历数组的最大的 K 个元素。

  ```python
  import heapq
  def findKthLargest(self, nums: List[int], k: int) -> int:
      n=len(nums)
      heap = []
      for i in range(k): heap.append(nums[i])
      heapq.heapify(heap)
      for i in range(k,n):
          if nums[i]>heap[0]: heapq.heappushpop(heap,nums[i])
      return heap[0]
  ```

* 基于快排的划分：$O(n)$

  每次选取一个哨兵进行快排，返回哨兵在排序后的下标，若下标等于 `n-k`，则说明哨兵刚好是第 K 大的元素。否则，则对第 K 大元素所在的区间继续进行快排。

  ```python
  import heapq
  def findKthLargest(self, nums: List[int], k: int) -> int:
      def partition(l,r):
          flag = nums[l]
          idx = l+1
          for p in range(idx,r+1):
              if nums[p]<flag:
                  nums[idx],nums[p]=nums[p],nums[idx]
                  idx+=1
          nums[idx-1],nums[l]=nums[l],nums[idx-1]
          return idx-1
      n=len(nums)
      i,j=0,n-1
      while True:
          idx = partition(i,j)
          if idx==n-k: return nums[idx]
          elif idx>n-k: j=idx-1
          else: i=idx+1
  ```

* 冒泡排序：$O(nk)$

  进行 K 次冒泡操作。

### 75. [前 K 个高频元素（中等）](https://leetcode.cn/problems/top-k-frequent-elements/)

<div align=center><img src="75.png" style="zoom:50%;" /></div>

* 堆

  ```python
  def topKFrequent(self, nums: List[int], k: int) -> List[int]:
      cnt = defaultdict(int)
      for num in nums: cnt[num]+=1
      heap = []
      for key,val in cnt.items():
          heapq.heappush(heap,(val,key))
          if len(heap)>k: heapq.heappop(heap)
      ans = [key for val,key in heap]
      return ans
  ```

* Counter

  ```python
  def topKFrequent(self, nums: List[int], k: int) -> List[int]:
      cnt = Counter(nums)
      common = cnt.most_common(k)
      return [x[0] for x in common]
  ```

### 76. [数据流的中位数（困难）](https://leetcode.cn/problems/find-median-from-data-stream/)

<div align=center><img src="76.png" style="zoom:50%;" /></div>

* 堆

  将数据流划分为大小两部分，分别用两个堆存储。

  当 left 为空或者 num 小于等于 max(left) 时，将 num 放入 left；若放入 left 后 left 数组长度大于 right 数组长度加一；

  反之亦然。

  ```python
  import heapq
  class MedianFinder:
      def __init__(self):
          self.left = []
          self.right = []
      def addNum(self, num: int) -> None:
          if not self.left or num<=-self.left[0]:
              heapq.heappush(self.left,-num)
              if len(self.right)+1<len(self.left):
                  heapq.heappush(self.right,-heapq.heappop(self.left))
          else:
              heapq.heappush(self.right,num)
              if len(self.right)>len(self.left):
                  heapq.heappush(self.left,-heapq.heappop(self.right))
      def findMedian(self) -> float:
          if len(self.right)==len(self.left):
              return (self.right[0]-self.left[0])/2
          else: return -self.left[0]
  ```

## 贪心算法

### 77. [买卖股票的最佳时机（简单）](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

<div align=center><img src="77.png" style="zoom:50%;" /></div>

* 贪心

  ```python
  def maxProfit(self, prices: List[int]) -> int:
      buy,profit = inf,0
      for price in prices:
          buy=min(buy,price)
          profit=max(profit,price-buy)
      return profit
  ```


### extra. [买卖股票的最佳时机 II（中等）](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

<div align=center><img src="e77.png" style="zoom:50%;" /></div>

* 贪心

  ```python
  def maxProfit(self, prices: List[int]) -> int:
      ans=0
      for i in range(len(prices)-1):
          if prices[i+1]>prices[i]:
              ans+=prices[i+1]-prices[i] # 能赚钱就赚
      return ans
  ```

  

### 78. [跳跃游戏（中等）](https://leetcode.cn/problems/jump-game/)

<div align=center><img src="78.png" style="zoom:50%;" /></div>

* 贪心

  记录当前能够到达的最远位置，当遍历到超过最远位置时，返回 False。

  ```python
  def canJump(self, nums: List[int]) -> bool:
      far = 0
      for i in range(len(nums)):
          if i>far:
              return False
          far = max(far,i+nums[i])
          if far>=len(nums)-1:
              return True
  ```

### 79. ⭐️ [跳跃游戏 II（中等）](https://leetcode.cn/problems/jump-game-ii/)

<div align=center><img src="79.png" style="zoom:50%;" /></div>

* 贪心

  记录最远位置，每当达到最远位置就 cnt+1

  ```python
  def jump(self, nums: List[int]) -> int:
      ans,n=0,len(nums)
      maxi,end=0,0 # 分别记录下一步最远位置和当前最远位置
      for i in range(n-1):
          maxi=max(maxi,i+nums[i])
          if i==end: # 达到当前最远位置
              ans+=1
              end=maxi
      return ans
  ```

### 80. [划分字母区间（中等）](https://leetcode.cn/problems/partition-labels/)

<div align=center><img src="80.png" style="zoom:50%;" /></div>

* 贪心

  记录每个字母最右边出现的位置。当前遍历位置超出当前区间内字母最右出现位置时得到一个划分区间。

  ```python
  def partitionLabels(self, s: str) -> List[int]:
      tab,ans={},[]
      for i,c in enumerate(s):
          tab[c]=i
      start,far=0,tab[s[0]]
      for i in range(1,len(s)):
          if i>far:
              ans.append(far-start+1)
              start=i
              far=tab[s[i]]
          else:
              far = max(far,tab[s[i]])
      ans.append(len(s)-start)
      return ans
  ```

## 动态规划

### 81. [爬楼梯（简单）](https://leetcode.cn/problems/climbing-stairs/)

<div align=center><img src="81.png" style="zoom:50%;" /></div>

* 动态规划

  ```python
  def climbStairs(self, n: int) -> int:
      a,b=1,2
      while n>1:
          a,b=b,a+b
          n-=1
      return a
  ```

### 82. [杨辉三角（简单）](https://leetcode.cn/problems/pascals-triangle/)

<div align=center><img src="82.png" style="zoom:50%;" /></div>

* 动态规划

  ```python
  def generate(self, numRows: int) -> List[List[int]]:
      if numRows<3:
          return [[1]] if numRows==1 else [[1],[1,1]]
      ans=[[1],[1,1]]
      for i in range(2,numRows):
          tmp=[1]*(i+1)
          for j in range(1,i):
              tmp[j]=ans[i-1][j]+ans[i-1][j-1]
          ans.append(tmp[:])
      return ans
  ```

### 83. [打家劫舍（中等）](https://leetcode.cn/problems/house-robber/)

<div align=center><img src="83.png" style="zoom:50%;" /></div>

* 动态规划

  ```python
  def rob(self, nums: List[int]) -> int:
      n=len(nums)
      if n<3:
          return max(nums)
      dp=[nums[0],max(nums[:2])]
      for i in range(2,n):
          dp.append(max(dp[-1],dp[-2]+nums[i]))
      return dp[-1]
  ```

### extra. [打家劫舍 II（中等）](https://leetcode.cn/problems/house-robber-ii/)

<div align=center><img src="e83.png" style="zoom:50%;" /></div>

* 动态规划

  分别计算打劫第一间房和不打劫第一间房两种情况

  ```python
  def rob(self, nums: List[int]) -> int:
      def rob_range(nums,l,r):
          a,b = nums[l],max(nums[l:l+2])
          for i in range(l+2,r+1):
              a,b = b,max(b,a+nums[i])
          return b
      if len(nums)<4:
          return max(nums)
      return max(rob_range(nums,0,len(nums)-2),rob_range(nums,1,len(nums)-1))
  ```

### extra. [打家劫舍 III（中等）](https://leetcode.cn/problems/house-robber-iii/)

<div align=center><img src="e83_.png" style="zoom:50%;" /></div>

* DFS

  ```python
  def rob(self, root: Optional[TreeNode]) -> int:
      def dfs(node): 
          if not node: return 0,0
          l,notl = dfs(node.left)
          r,notr = dfs(node.right)
          return node.val+notl+notr,max(l,notl)+max(r,notr) # 返回偷或不偷该点的最大值
      return max(dfs(root))
  ```


### 84. [完全平方数（中等）](https://leetcode.cn/problems/perfect-squares/)

<div align=center><img src="84.png" style="zoom:50%;" /></div>

* 动态规划

  dp[i]=min(dp[i-j]+dp[j])，其中 j 是小于 i 的完全平方数

  ```python
  def numSquares(self, n: int) -> int:
      squres = [i*i for i in range(1,101)]
      dp=[inf]*(n+1)
      for i in range(1,n+1):
          if i in squres: dp[i]=1
          else:
              for j in squres:
                  if j>=i: break
                  dp[i]=min(dp[i],dp[i-j]+dp[j])
      return dp[-1]
  ```

### 85. [零钱兑换（中等）](https://leetcode.cn/problems/coin-change/)

<div align=center><img src="85.png" style="zoom:50%;" /></div>

* 动态规划

  dp[i] 表示 凑成金额 i 所需的最小硬币数。遍历硬币，当遍历到硬币 coin 时，有 dp[j]=min(dp[j],dp[j-coin]+1)

  ```python
  def coinChange(self, coins: List[int], amount: int) -> int:
      n=len(coins)
      dp = [0]+[inf]*amount
      for coin in coins:
          for j in range(coin,amount+1):
              dp[j]=min(dp[j],dp[j-coin]+1)
      return dp[-1] if dp[-1]!=inf else -1
  ```


### 86. ⭐️ [单词拆分（中等）](https://leetcode.cn/problems/word-break/)

<div align=center><img src="86.png" style="zoom:50%;" /></div>

* 动态规划

  dp[i] 表示以下标 i 结尾的字符串能否成功拆分；dp[i]=dp[j] && check(s[j:i])，check 表示子串是否出现在单词字典中。

  ```python
  def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    n=len(s)
    dp = [True]+[False]*n
    for i in range(n):
        for j in range(i+1):
            if s[j:i+1] in wordDict:
                dp[i+1]|=dp[j]
    return dp[-1]
  ```

* 记忆化回溯

  定义回溯函数 backtracking(s) 表示 s 能否用单词字典拼接而成。遍历区间 [0,n-1]，若 s[:i+1] 在单词字典中，则 res = backtracking(s[i+1:]) or res。

  ```python
  def wordBreak(self, s: str, wordDict: List[str]) -> bool:
      @functools.cache
      def backtracking(s):
          if not s: return True
          res = False
          for i in range(len(s)):
              if s[:i+1] in wordDict:
                  res |= backtracking(s[i+1:])
          return res
      return backtracking(s)
  ```

### 87. ⭐️ [最长递增子序列（中等）](https://leetcode.cn/problems/longest-increasing-subsequence/)

<div align=center><img src="87.png" style="zoom:50%;" /></div>

* 动态规划

  定义 dp[i] 表示以 nums[i] 结尾的最长递增子序列长度，则有 $\text{dp[i]=max(dp[j])+1}$，其中 $0\leq j<i,\ nums[i]>nums[j]$。

  ```python
  def lengthOfLIS(self, nums: List[int]) -> int:
      n=len(nums)
      dp = [1]*n
      for i in range(1,n):
          for j in range(i):
              if nums[i]>nums[j]: dp[i]=max(dp[i],dp[j]+1)
      return max(dp)
  ```

* 动态规划+二分查找

  维护一个 tails 数组，tails[i] 表示长度为 i+1 的递增子序列尾部元素的值，可以证明，tails 数组为**严格递增**的数组。

  设 res 为 tails 当前长度，当遍历到 nums[k] 时，通过二分法找到 nums[k] 在 tails 数组里的大小分界点，会有以下两种情况：

  1. 区间里存在 tails[i] > nums[k]：更新 tails[i] 为 nums[k]；
  2. 区间里不存在 tails[i] > nums[k]：意味着 nums[k] 大于 tails 所有元素，nums[k] 可以接在当前最长递增子序列尾部，则 res+1；

  ```python
  def lengthOfLIS(self, nums: List[int]) -> int:
      res, tails = 0, [0]*n
      for num in nums:
          i,j=0,res
          while i<j:
              m=(i+j)//2
              if num>tails[m]: i=m+1
              else: j=m
          tails[i]=num
          if j==res: res+=1            
      return res
  ```

### 88. [乘积最大子数组（中等）](https://leetcode.cn/problems/maximum-product-subarray/)

<div align=center><img src="88.png" style="zoom:50%;" /></div>

* 动态规划

  记录遍历到 i 处时的最大最小值。

  ```python
  def maxProduct(self, nums: List[int]) -> int:
      ans=maxn=minn=nums[0]
      for i in range(1,len(nums)):
          ma,mi= maxn*nums[i],minn*nums[i]
          maxn = max(ma,mi,nums[i])
          minn = min(ma,mi,nums[i])
          ans=max(ans,maxn)
      return ans
  ```


### 89. [分割等和子集（中等）](https://leetcode.cn/problems/partition-equal-subset-sum/)

<div align=center><img src="89.png" style="zoom:50%;" /></div>

* 动态规划

  转换为 0-1 背包问题，判断数组能否有子集的和为 s/2。使用滚动数组优化空间。

  ```python
  def canPartition(self, nums: List[int]) -> bool:
      s,n=sum(nums),len(nums)
      if s%2: return False
      target = s//2
      dp = [True]+[False]*target
      for i in range(n):
          for j in range(target,nums[i]-1,-1):
              dp[j]|=dp[j-nums[i]]
          if dp[-1]: return True
      return False
  ```

### 90. ⭐️ [最长有效括号（困难）](https://leetcode.cn/problems/longest-valid-parentheses/)

<div align=center><img src="90.png" style="zoom:50%;" /></div>

* 动态规划

  dp[i] 表示以下标 i 结尾的序列最长括号长度。由于左括号结尾的括号长度均为0，只需考虑右括号。

  1. 若 s[i-1]=='('，则组成形如 "()" 的括号，dp[i]=dp[i-2]+2;
  2. 若 s[i-1]==')' 且 s[i-dp[i-1]-1]=='('，则组成形如 "(...)" 的括号，dp[i]=dp[i-dp[i-1]-2]+dp[i-1]+2；

  ```python
  def longestValidParentheses(self, s: str) -> int:
      ans,stack=0,[]
      dp=[0]*(len(s)+1)
      for i in range(1,len(s)):
          if s[i]==')':
              if s[i-1]=='(':
                  dp[i]=dp[i-2]+2
              elif s[i-1]==')':
                  if i-dp[i-1]-1>=0 and s[i-dp[i-1]-1]=='(':
                      dp[i]=dp[i-dp[i-1]-2]+dp[i-1]+2
      return max(dp)
  ```

* 栈

  维护一个栈，保持栈底元素为`最后一个没有被匹配的右括号`，初始为 -1。每当遇到左括号时，将下标入栈；每当遇到右括号时，将栈顶元素弹出，若弹出后栈为空，说明当前的右括号为没有被匹配的右括号，将其入栈；否则此时组成一个括号，长度为 i-stack[-1]；

  ```python
  def longestValidParentheses(self, s: str) -> int:
      ans,stack=0,[-1]
      for i,c in enumerate(s):
          if c=='(': stack.append(i)
          else:
              stack.pop()
              if not stack: stack.append(i)
              else: ans=max(ans,i-stack[-1])
      return ans
  ```

* 双遍历+计数

  统计当前左括号和右括号数量，当相等时更新答案；右括号大于左括号时重置答案；左右遍历一次。

## 多维动态规划

### 91. [不同路径（中等）](https://leetcode.cn/problems/unique-paths/)

<div align=center><img src="91.png" style="zoom:50%;" /></div>

* 动态规划

  ```python
  def uniquePaths(self, m: int, n: int) -> int:
      dp = [[1 for _ in range(n)]for _ in range(m)]
      for i in range(1,m):
          for j in range(1,n):
              dp[i][j]=dp[i-1][j]+dp[i][j-1]
      return dp[-1][-1]
  ```

* 数学

  一共移动 m+n-2 次，其中 m-1 次向下移动，计算 $C_{m+n-2}^{m-1}$ 即可。

  ```python
  def uniquePaths(self, m: int, n: int) -> int:
      return math.comb(m + n - 2, m - 1)
  ```

### 92. [最小路径和（中等）](https://leetcode.cn/problems/minimum-path-sum/)

<div align=center><img src="92.png" style="zoom:50%;" /></div>

* 动态规划

  不用额外空间。

  ```python
  def minPathSum(self, grid: List[List[int]]) -> int:
      for i in range(len(grid)):
          for j in range(len(grid[0])):
              if i==j==0: continue
              if i==0: grid[i][j]+=grid[i][j-1]
              elif j==0: grid[i][j]+=grid[i-1][j]
              else: grid[i][j] += min(grid[i-1][j],grid[i][j-1])
      return grid[-1][-1]
  ```

### 93. [最长回文子串（中等）](https://leetcode.cn/problems/longest-palindromic-substring/)

<div align=center><img src="93.png" style="zoom:50%;" /></div>

* 动态规划

  初始化 dp\[i][i]=1，若 s[i]=s[j]，则dp\[i][j] = dp\[i+1][j-1]+2；记录最大长度和起始索引。

  注意，应该倒序遍历行，因为转移方程依赖于 dp\[i+1]。

  ```python
  def longestPalindrome(self, s: str) -> str:
      n=len(s)
      dp = [[0 for _ in range(n)]for _ in range(n)]
      for i in range(n): dp[i][i]=1
      max_l,start = 1,0
      for i in range(n-1,-1,-1):
          for j in range(i+1,n):
              if s[i]==s[j]:
                  if j-i<3: dp[i][j]=j-i+1
                  elif dp[i+1][j-1]!=0: dp[i][j]=dp[i+1][j-1]+2
                  if dp[i][j]>max_l:
                      max_l=dp[i][j]
                      start = i
      return s[start:start+max_l]

### 94. [最长公共子序列（中等）](https://leetcode.cn/problems/longest-common-subsequence/)

<div align=center><img src="94.png" style="zoom:50%;" /></div>

* 动态规划

  定义 dp\[i][j] 为 text1[:i] 与 text[:j] 的最长公共子序列长度：

  * 当 text[i]=text[j] 时，dp\[i][j]=dp\[i-1][j-1]+1
  * 否则，dp\[i][j]=max(dp\[i-1][j],dp\[i][j-1])

  ```python
  def longestCommonSubsequence(self, text1: str, text2: str) -> int:
      m,n=len(text1),len(text2)
      dp = [[0 for _ in range(n+1)]for _ in range(m+1)]
      for i in range(1,m+1):
          for j in range(1,n+1):
              if text1[i-1]==text2[j-1]: dp[i][j]=dp[i-1][j-1]+1
              else: dp[i][j]=max(dp[i-1][j],dp[i][j-1])
      return dp[-1][-1]
  ```

### 95. [编辑距离（困难）](https://leetcode.cn/problems/edit-distance/)

<div align=center><img src="95.png" style="zoom:50%;" /></div>

* 动态规划

  定义 dp\[i][j] 表示将 word1\[:i] 转换成 word2[:j] 的最小操作数：

  * 当 word1\[i]=word2[j] 时，dp\[i][j]=dp\[i-1][j-1]
  * 否则，dp\[i][j] 等于 dp\[i-1]\[j-1]（修改word1）,dp\[i-1][j]（添加word1）,dp\[i][j-1]（删除word1） 三者中的最小值 +1

  ```python
  def minDistance(self, word1: str, word2: str) -> int:
      n1,n2=len(word1),len(word2)
      dp = [[0 for _ in range(n2+1)]for _ in range(n1+1)]
      # 初始化
      for i in range(1,n1+1): dp[i][0]=i
      for j in range(1,n2+1): dp[0][j]=j
      for i in range(1,n1+1):
          for j in range(1,n2+1):
              if word1[i-1]==word2[j-1]: dp[i][j]=dp[i-1][j-1]
              else: dp[i][j]=min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])+1
      return dp[-1][-1]
  ```

## 技巧

### 96. [只出现一次的数字（简单）](https://leetcode.cn/problems/single-number/)

<div align=center><img src="96.png" style="zoom:50%;" /></div>

* 异或

  ```python
  ### 调库
  from functools import reduce
  class Solution:
      def singleNumber(self, nums: List[int]) -> int:
          return reduce(lambda x,y:x^y,nums)
  ```

### 97. [多数元素（简单）](https://leetcode.cn/problems/majority-element/)

<div align=center><img src="97.png" style="zoom:50%;" /></div>

* 投票

  ```python
  def majorityElement(self, nums: List[int]) -> int:
      ans,cnt=nums[0],1
      for i in range(1,len(nums)):
          if nums[i]==ans: cnt+=1
          else:
              cnt-=1
              if cnt==0:
                  ans,cnt = nums[i],1
      return ans
  ```

* 哈希表计数

* 排序：排序后下标 $\lfloor\frac{n}{2}\rfloor$ 处的元素为众数

* 分治法

  若 a 是 nums 的众数，将 nums 分为两部分，则 a 至少是其中一部分的众数。将数组分成左右两部分，分别求出左半部分的众数 `a1` 以及右半部分的众数 `a2`。若 `a1=a2`，则合并后的众数不变；否则，比较两个众数在整个区间内出现的次数。

  ```python
  def majorityElement(self, nums: List[int]) -> int:
      def majority_element_rec(lo, hi) -> int:
          if lo == hi: 
              return nums[lo] # 单位长度
          mid = (hi - lo) // 2 + lo
          left = majority_element_rec(lo, mid)
          right = majority_element_rec(mid + 1, hi)
          if left == right: 
              return left # 左右半区众数相等
          left_count = sum(1 for i in range(lo, hi + 1) if nums[i] == left)
          right_count = sum(1 for i in range(lo, hi + 1) if nums[i] == right)
          return left if left_count > right_count else right
      return majority_element_rec(0, len(nums) - 1)
  ```

### 98. ⭐️ [颜色分类（中等）](https://leetcode.cn/problems/sort-colors/)

<div align=center><img src="98.png" style="zoom:50%;" /></div>

* 单指针+两次遍历

  第一次遍历移动 0，第二次遍历移动 1

* ⭐️ 双指针

  设置两个指针 p0 和 p1，分别指向 0 和 1 需要放置的位置。

  当遇到 1 时，将 1 交换至 p1 处，p1 右移一位；

  当遇到 0 时，将 0 交换至 p0 处，但原本 p0 处可能为 1，此时还需将 1 移动至 p1 处。

  ```python
  def sortColors(self, nums: List[int]) -> None:
      n=len(nums)
      p0=p1=0
      for i in range(n):
          if nums[i]==1:
              nums[i],nums[p1]=nums[p1],nums[i]
              p1+=1
          elif nums[i]==0:
              nums[i],nums[p0]=nums[p0],nums[i]
              if nums[i]==1:
                  nums[i],nums[p1]=nums[p1],nums[i]
              p0+=1
              p1+=1
  ```

### 99. ⭐️ [下一个排列（中等）](https://leetcode.cn/problems/next-permutation/)

<div align=center><img src="99.png" style="zoom:50%;" /></div>

* 双指针

  倒序遍历，找到最后一个满足 nums[i]<nums[i+1] 的升序对，令 left=i；倒序遍历，找到最后一个 nums[j]>nums[i]，令 right=j，交换 nums[left] 和 nums[right]。此时 nums[left+1:] 为非升序排列，只需将其倒序变为非降序排列即可。

  ```python
  def nextPermutation(self, nums: List[int]) -> None:
      i=len(nums)-2
      while i>=0 and nums[i]>=nums[i+1]: # 找最后一个升序的位置
          i-=1
      if i>=0:
          j=len(nums)-1
          while nums[j]<=nums[i]: # 找最后一个大于 nums[i] 的 nums[j]
              j-=1
          nums[i],nums[j]=nums[j],nums[i]
      l,r=i+1,len(nums)-1
      while l<r: # 将 nums[l:] 变为非降序排列
          nums[l],nums[r]=nums[r],nums[l]
          l+=1
          r-=1
  ```

### 100. ⭐️ [寻找重复数（中等）](https://leetcode.cn/problems/find-the-duplicate-number/)

<div align=center><img src="100.png" style="zoom:50%;" /></div>

* 排序

  ```python
  def findDuplicate(self, nums: List[int]) -> int:
      nums.sort()
      for i in range(1,len(nums)):
          if nums[i]==nums[i-1]:
              return nums[i]
  ```

* 双指针

  将该问题转换为找到链表的环入口。考虑以下两种情况：

  1. 数组中没有重复数。如数组 `[1,3,4,2]`，建立数组下标 `n` 和 `nums[n]` 的映射关系，有：`0->1,1->3,2->4,3->2`，由此可以产生一个类似链表的序列：`0->1->3->2->4->null`；

  2. 数组中有重复数。如数组 `[1,3,4,2,2]`，映射关系：`0->1,1->3,2->4,3->2,4->2`，链表序列：`0->1->3->2->4->2->4->2->...`，该序列如下图所示：

     <div align=center><img src="1001.png" style="zoom:70%;" /></div>

  因此，当数组中有重复数时，会产生一个多对一的映射，这样形成的链表就有环。综上：

  1. 数组中有一个重复的整数 <==> 链表中存在环
  2. 找到数组中的重复整数 <==> 找到链表的环入口

  ```python
  def findDuplicate(self, nums: List[int]) -> int:
      slow=fast=0
      slow,fast = nums[slow],nums[nums[fast]]
      while slow!=fast:
          slow,fast = nums[slow],nums[nums[fast]]
      p=0
      while p!=slow:
          p,slow=nums[p],nums[slow]
      return p
  ```

* 二分查找

  取 `[left,right]` 区间中间的数 `mid`，统计小于等于 `mid` 的数的个数 `cnt`，若 `cnt` 严格大于 `mid`，根据抽屉原理，重复的数在区间 `[left,mid]` 中；否则，在区间 `[mid+1,right]` 中。

  ```python
  def findDuplicate(self, nums: List[int]) -> int:
      left,right=1,len(nums)-1
      while left<right:
          cnt=0
          mid = left+(right-left)//2
          for num in nums: # 统计小于 mid 的个数
              if num<=mid: cnt+=1
          if cnt>mid: right=mid
          else: left=mid+1
      return left
  ```

* 原地交换

  当数组有重复元素时，数组元素的索引和值是多对一的关系。

  遍历数组，第一次遇到数字 x 时，将其交换至索引 x 处；而当第二次遇到数字 x 时，有 nums[x]=x ，即 nums[nums[x]]=nums[x]，此时得到重复数字。

  ```python
  def findDuplicate(self, nums: List[int]) -> int:
      i=0
      while i<len(nums):
          if nums[i]==i:
              i+=1
              continue
          if nums[i]==nums[nums[i]]: return nums[i]
          nums[nums[i]],nums[i]=nums[i],nums[nums[i]]
  ```

