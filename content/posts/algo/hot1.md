---
title: "LeetCode 热题 100 与 CodeTop"
date: 2023-07-23T16:11:27+08:00
lastmod: 2023-07-23T16:11:27+08:00
author: ["Achilles"]
# keywords: 
# - 
categories: # 没有分类界面可以不填写
- 
tags: [] # 标签
description: ""
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

[TOC]



## 哈希

### 1. [两数之和（简单）](https://leetcode.cn/problems/two-sum/)

<div align=center><img src="1.png" style="zoom:50%;" /></div>

* 哈希

  ```python
  def twoSum(self, nums: List[int], target: int) -> List[int]:
      tab = {}
      for i in range(len(nums)):
          if target-nums[i] in tab:
              return [i,tab[target-nums[i]]]
          tab[nums[i]]=i
  ```

### 2. [字母异位词分组（中等）](https://leetcode.cn/problems/group-anagrams/)

<div align=center><img src="2.png" style="zoom:50%;" /></div>

* 排序+哈希

  ```python
  def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
      tab = defaultdict(list)
      for s in strs:
          key = ''.join(sorted(s)) # list需要转换为字符串才能进行哈希
          tab[key].append(s)
      return list(tab.values())
  ```

* 计数+哈希

  ```python
  def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
      mp = defaultdict(list)
      for st in strs:
          counts = [0] * 26
          for ch in st:
              counts[ord(ch) - ord("a")] += 1  # 需要将 list 转换成 tuple 才能进行哈希
          mp[tuple(counts)].append(st)
      return list(mp.values())
  ```

### 3. [最长连续序列（中等）](https://leetcode.cn/problems/longest-consecutive-sequence/)

<div align=center><img src="3.png" style="zoom:50%;" /></div>

* 哈希

  当 num-1 在集合中时，跳过。

  当 num-1 不在集合中，说明 num 为一个序列的起点，依次判断 cur+1 是否在集合中。

  ```python
  def longestConsecutive(self, nums: List[int]) -> int:
      ans=0
      nums_set = set(nums)
      for num in nums:
          if num-1 not in nums_set:
              cur,long = num,1
              while cur+1 in nums_set:
                  cur = cur+1
                  long += 1
              ans = max(ans,long)
      return ans
  ```

* 哈希+动规

  记录序列端点的最大长度，每次遍历到 num，得到 num-1 和 num+1 序列的长度 l 和 r，则连续长度为 l+r+1，更新哈希表。

  ```python
  def longestConsecutive(self, nums: List[int]) -> int:
      tab = dict()
      ans = 0
      for num in nums:
          if num not in tab:
              l = tab.get(num-1,0)
              r = tab.get(num+1,0)
              cur = l+r+1
              ans = max(ans,cur)
              tab[num]=cur
              tab[num-l]=cur
              tab[num+r]=cur
      return ans
  ```

## 双指针

### 4. [移动零（简单）](https://leetcode.cn/problems/move-zeroes/)

<div align=center><img src="4.png" style="zoom:50%;" /></div>

* 双指针

  将把 0 移动到末尾转换为把非零数字移动到数组开头，这样可以保证非零数字的相对顺序。

  设置左右指针 l 和 r，l 指向非零数字需要放置的地方，遍历 r，每当遇到 0 就和 l 处交换。

  ```python
  def moveZeroes(self, nums: List[int]) -> None:
      """
      Do not return anything, modify nums in-place instead.
      """
      l,r,n=0,0,len(nums)
      while r<n:
          if nums[r]!=0:
              nums[l],nums[r]=nums[r],nums[l]
              l+=1
          r+=1
  ```

### 5. [盛最多水的容器（中等）](https://leetcode.cn/problems/container-with-most-water/)

<div align=center><img src="5.png" style="zoom:50%;" /></div>

* 双指针

  每次移动较小的指针，才有可能使存储的水量变多。

  ```python
  def maxArea(self, height: List[int]) -> int:
      n=len(height)
      ans,i,j=0,0,n-1
      while i<j:
          water = min(height[i],height[j])*(j-i)
          ans = max(water,ans)
          if height[i]<=height[j]: i+=1
          else: j-=1
      return ans
  ```

### 6. [三数之和（中等）](https://leetcode.cn/problems/3sum/)

<div align=center><img src="6.png" style="zoom:50%;" /></div>

* 双指针+排序

  排序，遍历到相同的数时跳过以达到去重的目的。

  ```python
  def threeSum(self, nums: List[int]) -> List[List[int]]:
      n=len(nums)
      ans=[]
      nums.sort()
      for i in range(n-2):
          if i!=0 and nums[i]==nums[i-1]: continue
          if nums[i]>0: break
          l,r=i+1,n-1
          while l<r:
              if nums[l]+nums[r]==-nums[i]:
                  ans.append([nums[i],nums[l],nums[r]])
                  while l<r and nums[l+1]==nums[l]: l+=1
                  while l<r and nums[r-1]==nums[r]: r-=1
                  l+=1
                  r-=1
              elif nums[l]+nums[r]>-nums[i]: r-=1
              else: l+=1
      return ans
  ```




### extra. [合并两个有序数组（简单）](https://leetcode.cn/problems/merge-sorted-array/)

<div align=center><img src="102.png" style="zoom:50%;" /></div>

* 逆向双指针

  ```python
  def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
      p1,p2=m-1,n-1
      idx = m+n-1
      while p2>=0:
          if p1<0 or nums1[p1]<nums2[p2]:
              nums1[idx]=nums2[p2]
              p2-=1
          else:
              nums1[idx]=nums1[p1]
              p1-=1
          idx-=1
  ```

## 滑动窗口

### 8. [无重复字符的最长子串（中等）](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

<div align=center><img src="8.png" style="zoom:50%;" /></div>

* 滑动窗口+哈希

  哈希表存储窗口内的子串，遇到重复则更新左指针。

  ```python
  def lengthOfLongestSubstring(self, s: str) -> int:
      seen=set()
      n=len(s)
      rp,ans=-1,0
      for i in range(n):
          if i!=0: seen.remove(s[i-1])
          while rp+1<n and s[rp+1] not in seen:
              seen.add(s[rp+1])
              rp+=1
          ans = max(ans,rp-i+1)
      return ans
   ####################################################
  def lengthOfLongestSubstring(self, s: str) -> int:
      n,ans=len(s),0
      i,j=0,0
      tab = dict()
      while j<n:
          if s[j] in tab: i = max(i,tab[s[j]]+1) # 更新左指针
          ans = max(j-i+1,ans)
          tab[s[j]]=j
          j+=1
      return ans
  ```

## 子串

## 普通数组

### 13. [最大子数组和（中等）](https://leetcode.cn/problems/maximum-subarray/)

<div align=center><img src="13.png" style="zoom:50%;" /></div>

* 动态规划

  定义 dp[i] 表示取 nums[i] 时的最大子数组和，则当 dp[i-1] 大于零时，加上 nums[i]；否则，取 nums[i]。

  注意初始值不能为零，考虑负值。

  ```python
  def maxSubArray(self, nums: List[int]) -> int:
      n=len(nums)
      s,ans=-inf,-inf
      for i in range(n):
          if s>=0: s+=nums[i]
          else: s=nums[i]
          ans=max(ans,s)
      return ans
  ```

  

## 矩阵

## 链表

### 23. [反转链表（简单）](https://leetcode.cn/problems/reverse-linked-list/)

<div align=center><img src="23.png" style="zoom:50%;" /></div>

* 指针

  一行赋值时注意先更新 p.next，若先更新 p，则 p.next 会改变。

  ```python
  def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
      if not head or not head.next:
          return head
      pre,p = None,head
      while p:
          pre,p.next,p = p,pre,p.next # 注意先更新 p.next
      return pre
  ```

### 25. [环形链表（简单）](https://leetcode.cn/problems/linked-list-cycle/)

<div align=center><img src="25.png" style="zoom:50%;" /></div>

* 快慢指针

  ```python
  def hasCycle(self, head: Optional[ListNode]) -> bool:
      if not head or not head.next:
          return False
      dummy = ListNode()
      dummy.next=head
      slow=fast=dummy
      while fast and fast.next:
          fast=fast.next.next
          slow=slow.next
          if slow==fast: return True
      return False
  ```

  

### 27. [合并两个有序链表（简单）](https://leetcode.cn/problems/merge-two-sorted-lists/)

<div align=center><img src="27.png" style="zoom:50%;" /></div>

* 递归

  ```python
  def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
      if not list1: return list2
      if not list2: return list1
      if list1.val < list2.val:
          list1.next = self.mergeTwoLists(list1.next,list2)
          return list1
      else:
          list2.next = self.mergeTwoLists(list1,list2.next)
          return list2
  ```

  

### 32. [K 个一组翻转链表（困难）](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

<div align=center><img src="32.png" style="zoom:50%;" /></div>

* 模拟

  循环，每次设置四个指针 `pre, head, tail, nxt`，分别指向待翻转区间前的节点，待翻转区间头结点，待翻转区间尾节点，带翻转区间后的节点。

  翻转待翻转区间，设置 `pre.next=head,tail.next=nxt`。

  ```python
  def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
      def reverse(head,tail):
          pre, p, tmp = None, head, head
          while pre != tail:
              pre,p.next,p = p,pre,p.next
          return pre,tmp
      pre = tail = dummy = ListNode(next=head)
      while True:
          for i in range(k):
              if not tail.next:
                  return dummy.next
              tail = tail.next
          nxt = tail.next
          head,tail = reverse(head,tail)
          pre.next = head
          tail.next = nxt
          pre,head = tail,tmp
  ```

  





### 36. ⭐️ [LRU 缓存（中等）](https://leetcode.cn/problems/lru-cache/)

<div align=center><img src="36.png" style="zoom:50%;" /></div>

* 双向链表+哈希

  双向链表，表头为最久未使用节点，表尾为最新节点。

  每次 `get`，将访问节点移至表尾；每次 `put`，若 key 存在，则取出对应值并将该节点移至表尾，否则，则新建一个节点添加至表尾，此时若链表长度超过容量，则删除表头节点，并在哈希表中删除对应元素。

  在表头表尾各用一个空节点表示 head，tail。

  ```python
  class ListNode:
      def __init__(self,key=0,val=0,pre=None,net=None):
          self.key=key
          self.val=val
          self.pre=pre
          self.next=net
  
  class LRUCache:
      def __init__(self, capacity: int):
          self.capacity = capacity
          self.num = 0
          self.head = ListNode()
          self.tail = ListNode()
          self.head.next = self.tail
          self.tail.pre = self.head
          self.tab = dict()
      
      def delnode(self,node):
          node.pre.next = node.next
          node.next.pre = node.pre
  
      def delhead(self):
          node = self.head.next
          self.delnode(node)
          return node
  
      def addtotail(self,node):
          self.tail.pre.next = node
          node.pre = self.tail.pre
          node.next = self.tail
          self.tail.pre = node
  
      def movetotail(self,node):
          self.delnode(node)
          self.addtotail(node)
  
      def get(self, key: int) -> int:
          if key not in self.tab: return -1
          node = self.tab[key]
          self.movetotail(node)
          return node.val
  
      def put(self, key: int, value: int) -> None:
          if key in self.tab:
              node = self.tab[key]
              node.val = value
              self.movetotail(node)
          else:
              node = ListNode(key,value)
              self.addtotail(node)
              self.tab[key]=node
              self.num+=1
              if self.num>self.capacity:
                  node = self.delhead()
                  self.tab.pop(node.key)
                  self.num-=1
  ```

* OrderedDict

  直接使用封装好的 OrderedDict 定义 LRU。

  ```python
  class LRUCache(collections.OrderedDict): #继承 OrderedDict
      def __init__(self, capacity: int):
          super().__init__()
          self.capacity = capacity
  
      def get(self, key: int) -> int:
          if key not in self: return -1
          self.move_to_end(key)
          return self[key]
  
      def put(self, key: int, value: int) -> None:
          if key in self:
              self.move_to_end(key)
          self[key] = value
          if len(self) > self.capacity:
              self.popitem(last=False) # last=False 删除第一个(最早添加)节点
  ```

## 二叉树

### 42. [二叉树的层序遍历（中等）](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

<div align=center><img src="42.png" style="zoom:50%;" /></div>

* BFS

  ```python
  def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
      if not root: return []
      queue, ans = [root], []
      while queue:
          tmp, n = [], len(queue)
          for _ in range(n):
              node = queue.pop(0)
              tmp.append(node.val)
              if node.left: queue.append(node.left)
              if node.right: queue.append(node.right)
          ans.append(tmp)
      return ans
  ```

### extra. [二叉树的锯齿形层序遍历（中等）](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/)

<div align=center><img src="103.png" style="zoom:50%;" /></div>

* BFS

  增加一个标志变量，判断正序还是倒序添加每一层的遍历结果

  ```
  def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
          if not root: return []
          queue, ans = [root], []
          flag=True
          while queue:
              tmp, n = [], len(queue)
              for _ in range(n):
                  node = queue.pop(0)
                  tmp.append(node.val)
                  if node.left: queue.append(node.left)
                  if node.right: queue.append(node.right)
              if flag:
                  ans.append(tmp)
              else:
                  ans.append(tmp[::-1])
              flag=not flag
          return ans
  ```

  

### 50. ⭐️ [二叉树的最近公共祖先（中等）](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

<div align=center><img src="50.png" style="zoom:50%;" /></div>

* DFS

  若 root 是 p,q 的最近公共祖先，则为以下几种情况：

  1. p,q 在 root 的子树中且分列 root 的异侧；
  2. p=root 且 q 为 root 的子树；
  3. q=root 且 p 为 root 的子树

  考虑通过递归对二叉树进行先序遍历，每当遇到 p 或 q 时返回。假设左右子树递归的返回值分别为 left 和 right，返回值有以下几种情况：

  1. left 和 right 同时为空。表示 root 的左右子树都不包含 p 和 q，返回 None；
  2. left 和 right 均不为空。表示 p,q 在 root 的子树中且分列 root 的异侧，root 为最近公共祖先，返回 root；
  3. left 为空，right 不为空。表示 p,q 都不在 root 的左子树中，返回right。具体分为两种情况：
     1. p,q 都在 root 的右子树，此时 right 指向最近公共祖先；
     2. p,q 其一在 root 的右子树，此时 right 指向在右子树中的节点
  4. 同 3.

  ```python
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
      if not root or root.val==p.val or root.val==q.val:
          return root
      left = self.lowestCommonAncestor(root.left,p,q)
      right = self.lowestCommonAncestor(root.right,p,q)
      if not left: return right
      if not right: return left
      return root
  ```

## 图论

### 52. [岛屿数量（中等）](https://leetcode.cn/problems/number-of-islands/)

<div align=center><img src="52.png" style="zoom:50%;" /></div>

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

  

## 回溯

## 二分查找

### 67. ⭐️ [搜索旋转排序数组（中等）](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

<div align=center><img src="67.png" style="zoom:50%;" /></div>

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
  ```

## 栈

### 70. [有效的括号（简单）](https://leetcode.cn/problems/valid-parentheses/)

<div align=center><img src="70.png" style="zoom:50%;" /></div>

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

  

## 堆

### 75. [数组中的第K个最大元素（中等）](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

<div align=center><img src="75.png" style="zoom:50%;" /></div>

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

## 贪心算法

### 78. [买卖股票的最佳时机（简单）](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

<div align=center><img src="78.png" style="zoom:50%;" /></div>

* 贪心

  ```python
  def maxProfit(self, prices: List[int]) -> int:
      buy,profit = inf,0
      for price in prices:
          buy=min(buy,price)
          profit=max(profit,price-buy)
      return profit
  ```

  

## 动态规划

## 多维动态规划

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

## 技巧



## 补充

### 101. ⭐️ [手撕排序（中等）](https://leetcode.cn/problems/sort-an-array/)

<div align=center><img src="101.png" style="zoom:50%;" /></div>

* 快速排序：不稳定排序，时间复杂度$O(n\log\ n)$，最坏时间复杂度$O(n^2)$（有序数组或重复数组）

  ```python
  def sortArray(nums):
      def partition(arr,left,right):
          pivot = arr[left]
          idx = left+1
          for i in range(idx,right+1):
              if arr[i]<pivot:
                  arr[i],arr[idx]=arr[idx],arr[i]
                  idx+=1
          nums[left],nums[idx-1]=nums[idx-1],nums[left]
          return idx-1
      # 双指针
      def partition2(arr,left,right):
          pivot = arr[left]
          l,r = left+1,right
          while l<=r:
              while nums[l]<=pivot and l<=r: l+=1
              while nums[r]>pivot and l<=r: r-=1
              if l<=r:
                  nums[l],nums[r]=nums[r],nums[l]
                  l+=1
                  r-=1
          nums[left],nums[r]=nums[r],nums[left]
          return r
      def quicksort(arr,left,right):
          if left<right:
              idx = partition(arr,left,right)
              quicksort(arr,left,idx-1)
              quicksort(arr,idx+1,right)
      n=len(nums)
      quicksort(nums,0,n-1)
      return nums
  ```

  优化方法：选择更合适的 pivot（随机选择、三数取中）；当数据量小时，使用插入排序。

  * 选择**更合适的 pivot**

  ```python
  # 随机选择 pivot
  def random_partition(nums,left,right):
      p_idx = random.randint(left,right)
      nums[left],nums[p_idx]=nums[p_idx],nums[left]
      return partition(nums,left,right)
  # 三数取中选择 pivot，解决数组基本有序的情况
  def tree_partition(nums,left,right):
      mid = left+(right-left)//2
      if nums[left]<=nums[mid]:
          p_idx = mid if nums[right]>=nums[mid] else left if nums[right]<=nums[left] else right
      else:
          p_idx = left if nums[right]>=nums[left] else mid if nums[right]<=nums[mid] else right
      nums[left],nums[p_idx]=nums[p_idx],nums[left]
      return partition(nums,left,right)
  ```

  * 当数据量小于等于 20 时，使用**插入排序**

  ```python
  # 选择排序
  def insertsort(arr,left,right):
      for i in range(left+1,right+1):
          j=i-1
          cur=nums[i]
          while j>=0 and nums[j]>cur:
              nums[j],nums[j+1]=nums[j+1],nums[j]
              j-=1
  def quicksort(arr,left,right):
      if right-left<=20:
          insertsort(arr,left,right)
          return
      if left<right:
          idx = tree_partition(arr,left,right)
          quicksort(arr,left,idx-1)
          quicksort(arr,idx+1,right)
  ```

  * **三路快排**，开辟一块区域存储和 pivot 相等的元素。

  ```python
  def sortArray(self, nums: List[int]) -> List[int]:
      if len(nums) <= 1:
          return nums
      # 随机取数 避免因为pivot区分度不强造成的算法退化
      pivot = random.choice(nums)
      # O(n)划分
      left  = self.sortArray([x for x in nums if x < pivot])
      right = self.sortArray([x for x in nums if x > pivot])
      # 相同值保留 避免因为大量相同元素造成的算法退化
      mid  = [x for x in nums if x == pivot]
      return left + mid + right 
  ```

* 堆排序：$O(n\log\ n)$

  先建堆，对最后一个非叶子节点 `n//2-1` 开始向前遍历；排序时，每次将未排序的最后一个结点与根节点交换，然后对根节点建堆。

  ```python
  # 大顶堆 -> 升序排列
  def heapify(self,arr,n,i):
      largest = i
      l,r=2*i+1,2*i+2
      if l<n and arr[l]>arr[largest]: largest=l
      if r<n and arr[r]>arr[largest]: largest=r
      if largest!=i:
          arr[i],arr[largest]=arr[largest],arr[i]
          self.heapify(arr,n,largest)
  
  def sortArray(self, nums: List[int]) -> List[int]:
      n=len(nums)
      for i in range(n//2-1,-1,-1):
          self.heapify(nums,n,i)
      for i in range(n-1,0,-1):
          nums[i],nums[0]=nums[0],nums[i]
          self.heapify(nums,i,0)
      return nums
  ```

* 归并排序：$O(n\log\ n)$

  ```python
  def merge_sort(self, nums, l, r):
      if l == r:
          return
      mid = (l + r) // 2
      self.merge_sort(nums, l, mid)
      self.merge_sort(nums, mid + 1, r)
      tmp = []
      i, j = l, mid + 1
      while i <= mid or j <= r:
          if i > mid or (j <= r and nums[j] < nums[i]):
              tmp.append(nums[j])
              j += 1
          else:
              tmp.append(nums[i])
              i += 1
      nums[l: r + 1] = tmp
  
  def sortArray(self, nums: List[int]) -> List[int]:
      self.merge_sort(nums, 0, len(nums) - 1)
      return nums
  ```



