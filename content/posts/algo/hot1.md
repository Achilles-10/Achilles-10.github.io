---
title: "LeetCode 热题 100 与 CodeTop"
date: 2023-07-23T16:11:27+08:00
lastmod: 2023-07-23T16:11:27+08:00
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
    image: "posts/algo/hot1/cover.png" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## 持续更新。。。

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

### 7. ⭐️ [接雨水（困难）](https://leetcode.cn/problems/trapping-rain-water/)

<div align=center><img src="7.png" style="zoom:50%;" /></div>

* 动态规划

  维护两个数组 left 和 right，分别表示 i 处左右最高柱子的高度。i 处接的雨水为 min(left[i],right[i])-height[i]。

  ```python
  def trap(self, height: List[int]) -> int:
      n=len(height)
      left,right=[height[0]]+[0]*(n-1),[0]*(n-1)+[height[-1]]
      for i in range(1,n):
          left[i]=max(left[i-1],height[i])
          right[n-i-1]=max(right[n-i],height[n-i-1])
      return sum(min(left[i],right[i])-height[i] for i in range(n))
  ```

* 双指针

  使用双指针优化动态规划方法的空间复杂度。维护双指针 left 和 right 以及 两个变量 leftmax 和 rightmax 标志左右两侧柱子最高的高度。

  ```python
  def trap(self, height: List[int]) -> int:
      n,ans=len(height),0
      left,right=0,n-1
      leftmax,rightmax=0,0
      while left<right:
          leftmax=max(leftmax,height[left])
          rightmax=max(rightmax,height[right])
          if height[left]<height[right]:
              ans+=leftmax-height[left]
              left+=1
          else:
              ans+=rightmax-height[right]
              right-=1
      return ans
  ```

* 单调栈

  维护一个单调栈，存储下标，保证下标对应的柱子高度递减。

  当遍历到下标 i 时，若下标 i 的柱子的高度大于栈顶元素 top 的高度，由于 height[top-1]>height[top]，此时得到一个可以接雨水的区域。重复这个操作直至栈空或 height[top]>=height[i]。将 i 入栈。

  ```python
  def trap(self, height: List[int]) -> int:
      n,ans,stack=len(height),0,[]
      for i, hi in enumerate(height):
          while stack and hi>height[stack[-1]]:
              top = stack.pop()
              if not stack: break
              w = i-stack[-1]-1
              h = min(hi,height[stack[-1]])-height[top]
              ans += w*h
          stack.append(i)
      return ans
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

### 11. ⭐️ [滑动窗口最大值（困难）](https://leetcode.cn/problems/sliding-window-maximum/)

<div align=center><img src="11.png" style="zoom:50%;" /></div>

* ⭐️ 堆

  维护一个大根堆，存储元素为 (-nums[i], i)。当窗口右端点遍历到 j 处时，将窗口右边的数加入到堆中，若堆顶元素 i<j-k+1，说明堆顶元素已不在窗口内，将其从堆中删除。

  存储下标很重要，用于判断堆顶元素是否在窗口内。

  ```python
  import heapq
  class Solution:
      def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
          heap,j=[],0
          ans,n=[],len(nums)
          for i in range(k):
              heapq.heappush(heap,(-nums[i],i))
          ans.append(-heap[0][0])
          for j in range(k,n):
              heapq.heappush(heap,(-nums[j],j))
              while heap[0][1]<=j-k:
                  heapq.heappop(heap)
              ans.append(-heap[0][0])
          return ans
  ```

* 单调双端队列

  维护一个单调双端队列，队列里的元素单调递减。队列里存储下标，每次移动窗口时，先判断队尾元素与新元素的大小关系，保持队列单调递减，将新元素下标加入到队列；判断队首元素下标是否在窗口外，如果在则出队。

  存储下标很重要，用于判断队首元素是否在窗口内。

  > 用 list 模拟双端队列效率较低，尽量用 collections.deque

  ```python
  class Solution:
      def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
          n = len(nums)
          q = collections.deque()
          for i in range(k):
              while q and nums[i] >= nums[q[-1]]:
                  q.pop()
              q.append(i)
          ans = [nums[q[0]]]
          for i in range(k, n):
              while q and nums[i] >= nums[q[-1]]:
                  q.pop()
              q.append(i)
              while q[0] <= i - k:
                  q.popleft()
              ans.append(nums[q[0]])
          return ans
  ```

  

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


### 14. [合并区间（中等）](https://leetcode.cn/problems/merge-intervals/)

<div align=center><img src="14.png" style="zoom:50%;" /></div>

* 排序

  ```python
  def merge(self, intervals: List[List[int]]) -> List[List[int]]:
      intervals.sort()
      ans=[]
      for interval in intervals:
          if not ans or interval[0]>ans[-1][1]: # 不重叠区间
              ans.append(interval)
          else: # 重叠区间，更新右端点
              ans[-1][1]=max(ans[-1][1],interval[1])
      return ans
  ```

  

## 矩阵

### 19. [螺旋矩阵（中等）](https://leetcode.cn/problems/spiral-matrix/)

<div align=center><img src="19.png" style="zoom:50%;" /></div>

* 模拟

  设置上下左右四个边界，模拟螺旋过程依次输出，遇到越界则停止。

  ```python
  def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
      m,n=len(matrix),len(matrix[0])
      top,bottom,left,right=0,m-1,0,n-1
      ans=[]
      while True:
          for j in range(left,right+1): ans.append(matrix[top][j]) # 从左到右
          top+=1
          if top>bottom: break
          for i in range(top,bottom+1): ans.append(matrix[i][right]) # 从上到下
          right-=1
          if left>right: break
          for j in range(right,left-1,-1): ans.append(matrix[bottom][j]) # 从右到左
          bottom-=1
          if top>bottom: break
          for i in range(bottom, top-1,-1): ans.append(matrix[i][left]) # 从下到上
          left+=1
          if left>right: break
      return ans
  ```

  

## 链表

### 22. [相交链表（简单）](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

<div align=center><img src="22.png" style="zoom:50%;" /></div>

* 双指针

  双指针遍历，当遇到表尾时，跳转至另一表头，直至双指针相遇。

  ```python
  def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
      pa,pb=headA,headB
      while pa!=pb:
          pa = pa.next if pa else headB
          pb = pb.next if pb else headA
      return pa
  ```

* 哈希表

  先遍历 headA，将每一个节点加入到哈希表中；遍历 headB，判断节点是否在哈希表中。

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

### extra. ⭐️ [反转链表 II（中等）](https://leetcode.cn/problems/reverse-linked-list-ii/)

<div align=center><img src="104.png" style="zoom:50%;" /></div>

* 穿针引线

  先定位 left 和 right，然后设置前驱和后继指针 pre 和 tail，对 left 和 right 区间反转链表。这个方法的弊端是当 left 和 right 的跨度特别大时，需要遍历链表两次。

  ```python
  def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
      def reverse(head,tail):
          pre,p=None,head
          while pre!=tail:
              pre,p.next,p=p,pre,p.next
          return tail,head
      pre=dummy=ListNode(next=head)
      tail=head
      for i in range(left-1):
          head=head.next
          pre=pre.next
      for j in range(right-1):
          tail=tail.next
      nxt=tail.next
      head,tail=reverse(head,tail)
      pre.next,tail.next=head,nxt
      return dummy.next
  ```

* ⭐️ **头插法**（一次穿针引线）

  <div align=center><img src="1042.png" style="zoom:40%;" /></div>

  在需要反转的区间里，每遍历到一个节点，让这个新节点来到反转部分的起始位置。下面的图展示了整个流程。具体来说，需要三个指针：`pre`，`curr` 和 `next`。

  * pre：指向 left 的前一个节点
  * curr：指向反转区域的第一个节点 left
  * next：指向curr的下一个节点

  具体步骤如下：

  1. 设置 curr 的下一个节点为 next
  2. curr.next 指向 next.next
  3. next.next 指向 pre.next
  4. pre.next 指向 next

  <div align=center><img src="1043.png" style="zoom:30%;" /></div>

  ```python
  def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
      pre=dummy=ListNode(next=head)
      for _ in range(left-1):
          pre=pre.next
      curr = pre.next
      for _ in range(right-left):
          nxt = curr.next
          curr.next = nxt.next
          nxt.next = pre.next
          pre.next = nxt
      return dummy.next
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


### 26. [环形链表 II（中等）](https://leetcode.cn/problems/linked-list-cycle-ii/)

<div align=center><img src="26.png" style="zoom:45%;" /></div>

* 快慢指针

  设置快慢指针，假设链表存在环，则快慢指针会在环内相遇，如下图：

  <div align=center><img src="261.png" style="zoom:30%;" /></div>

  此时有 $2(a+b)=a+(n+1)(b+c)+b\Rightarrow a=n(b+c)+c$，即头结点到入环点的距离为相遇点到入环点的距离加上 n 圈的环长。因此，再设置一个指针从头结点出发，则一定能和慢指针在入环点相遇。

  ```python
  def detectCycle(self, head: ListNode) -> ListNode:
      slow,fast=head,head
      while fast and fast.next:
          slow,fast=slow.next,fast.next.next
          if slow==fast: break
      # 这个判断条件不能为 if slow!=fast，需考虑只有一个节点且没有环的情况
      if not fast or not fast.next: return None
      p=head
      while p!=slow:
          p,slow=p.next,slow.next
      return slow
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


### 29. [删除链表的倒数第 N 个结点（中等）](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

<div align=center><img src="29.png" style="zoom:50%;" /></div>

* 快慢指针

  设置快指针先遍历 n 次，然后用慢指针定位待删除节点的前节点。

  ```python
  def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
      slow = dummy = ListNode(next=head) # 使用 dummy 定位待删除节点的前节点
      fast=head
      for _ in range(n):
          fast=fast.next
      while fast:
          slow,fast=slow.next,fast.next
      slow.next=slow.next.next
      return dummy.next
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

  

### 34. ⭐️ [排序链表（中等）](https://leetcode.cn/problems/sort-list/)

<div align=center><img src="34.png" style="zoom:50%;" /></div>

* 归并排序（递归）

  通过快慢指针将链表一分为二，进行归并排序

  ```python
  class Solution:
  		def merge(self,list1,list2):
          if not list1 or not list2: return list1 or list2
          if list1.val<=list2.val:
              list1.next = self.merge(list1.next,list2)
              return list1
          else:
              list2.next = self.merge(list1,list2.next)
              return list2
      def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
          if not head or not head.next: return head
          slow=fast=head
          while fast.next and fast.next.next:
              fast=fast.next.next
              slow=slow.next
          mid,slow.next = slow.next,None # 划分链表
          return self.merge(self.sortList(head),self.sortList(p))
  ```

* ⭐️ 迭代

  若面试中要求常数级空间复杂度，需要迭代。

  令子链表长度依次为 1, 2, 4, ... ，循环归并。

  <div align=center><img src="341.png" style="zoom:50%;" /></div>

  ```python
  class Solution:
      def merge(self,list1,list2):
          p = dummy = ListNode(0)
          while list1 and list2:
              if list1.val<=list2.val:
                  p.next = list1
                  list1=list1.next
              else:
                  p.next = list2
                  list2=list2.next
              p=p.next
          if not list1: p.next = list2
          if not list2: p.next = list1
          return dummy.next
  
      def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
          if not head or not head.next: return head
          n,tmp=0,head
          while tmp:
              tmp=tmp.next
              n+=1
          dummy = ListNode(next=head)
          k = 1 # 当前合并子链表长度
          while k<=n:
              pre,cur = dummy,dummy.next
              while cur:
                  left = cur # 左子链表头
                  for i in range(1,k):
                      if cur.next:
                          cur = cur.next
                      else:
                          break
                  if not cur.next: # 左子链表长度不足 k，不用再归并
                      pre.next = left
                      break
                  right = cur.next # 右子链表头
                  cur.next = None # 断开左子链表
                  cur = right
                  for i in range(1,k):
                      if cur.next:
                          cur = cur.next
                      else:
                          break
                  nxt = None
                  if cur.next:
                      nxt = cur.next # 记录后续待合并链表表头
                      cur.next = None # 断开右子链表
                  cur = nxt
                  pre.next = self.merge(left,right) # 将合并好的链表接在已合并好链表的表尾
                  while pre.next:
                      pre = pre.next
              k*=2
          return dummy.next
  ```

  

### 35. ⭐️ [合并 K 个升序链表（困难）](https://leetcode.cn/problems/merge-k-sorted-lists/)

<div align=center><img src="35.png" style="zoom:50%;" /></div>

* 分治合并

  两两合并链表。

  ```python
  def merge(self, list1, list2): # 迭代合并
          p1,p2=list1,list2
          tmp=dummy=ListNode()
          while p1 and p2:
              if p1.val<p2.val:
                  tmp.next=p1
                  p1=p1.next
              else:
                  tmp.next=p2
                  p2=p2.next
              tmp=tmp.next
          if not p1: tmp.next=p2
          if not p2: tmp.next=p1
          return dummy.next
  def merge_(self, list1, list2): # 递归合并
      if not list1 or not list2: return list1 or list2
      if list1.val<list2.val:
          list1.next = self.merge_(list1.next,list2)
          return list1
      else:
          list2.next = self.merge_(list1,list2.next)
          return list2
  def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
      if not lists:
          return None
      n=len(lists)
      if n==1: return lists[0]
      mid = n//2
      return self.merge(self.mergeKLists(lists[:mid]),self.mergeKLists(lists[mid:]))
  ```

* ⭐️ 堆合并（K个指针）

  维护当前每个链表没有被合并的元素的最前面一个，每次在这些元素里面选取值最小的元素合并到答案中。使用小顶堆来维护这些元素。

  ```python
  def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
      if not lists: return None
      p=dummy=ListNode()
      heap=[]
      for i in range(len(lists)):
          if lists[i]:
              heapq.heappush(heap,(lists[i].val,i))
              lists[i]=lists[i].next
      while heap:
          val,idx = heapq.heappop(heap)
          p.next = ListNode(val)
          if lists[idx]:
              heapq.heappush(heap,(lists[idx].val,idx))
              lists[idx]=lists[idx].next
          p=p.next
      return dummy.next
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

### 37. ⭐️ [二叉树的中序遍历（简单）](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

<div align=center><img src="37.png" style="zoom:50%;" /></div>

* 递归

  ```python
  def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
      def dfs(node,ans):
          if not node:
              return
          dfs(node.left,ans)
          ans.append(node.val)
          dfs(node.right,ans)
      ans=[]
      dfs(root,ans)
      return ans
  ```

* ⭐️ 迭代

  结合指针和栈，每当节点不为空时，则将节点入栈，遍历左子树；当节点为空时，则说明左子树遍历完毕，此时栈顶为当前左子树的根节点，将根节点值加入到 ans 中，遍历右子树。

  ```python
  def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
      if not root: return []
      ans,stack=[],[]
      cur=root
      while cur or stack:
          if cur:
              stack.append(cur)
              cur=cur.left
          else:
              cur=stack.pop()
              ans.append(cur.val)
              cur=cur.right
      return ans
  ```

* ⭐️ 前序遍历和后序遍历的迭代解法

  ```python
  # 前序遍历
  def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
      if not root: return []
      stack,ans=[],[]
      cur=root
      while cur or stack:
          if cur:
              ans.append(cur.val) # 先处理根节点
              stack.append(cur)
              cur = cur.left
          else:
              cur = stack.pop()
              cur = cur.right
      return ans
  ```

  后序遍历有两种方法，一种是按照后序遍历的顺序访问节点，一种是将 左-右-中 的后序遍历转换为 中-右-左 的前序遍历，在输出时将答案翻转。

  ```python
  # 后序遍历顺序访问节点
  def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
      if not root: return []
      stack, ans =[], []
      cur,pre = root,None # 设置一个 pre 指针，指向上一个加入 ans 数组的节点
      while cur or stack:
          while cur:
              stack.append(cur)
              cur = cur.left
          cur = stack.pop()
          if not cur.right or cur.right==pre: # 若右节点为 pre，说明右子树已经遍历完毕
              ans.append(cur.val)
              pre = cur
              cur = None
          else:
              stack.append(cur)
              cur = cur.right
      return ans
  ```

  ```python
  # 转换为 中-右-左 的前序遍历
  def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
      if not root:
          return []
      stack, ans =[], []
      cur = root
      while cur or stack:
          if cur:
              ans.append(cur.val)
              stack.append(cur)
              cur = cur.right
          else:
              cur = stack.pop()
              cur = cur.left
      return ans[::-1]
  ```

  

  



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


### 46. [二叉树的右视图（中等）](https://leetcode.cn/problems/binary-tree-right-side-view/)

<div align=center><img src="46.png" style="zoom:50%;" /></div>

* 层序遍历

  将每层最后一个节点的值加入到 ans

  ```python
  def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
      if not root:
          return []
      queue,ans = [root],[]
      while queue:
          for _ in range(len(queue)):
              node = queue.pop(0)
              if node.left: queue.append(node.left)
              if node.right: queue.append(node.right)
          ans.append(node.val)
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

### 51. [二叉树中的最大路径和（困难）](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

<div align=center><img src="51.png" style="zoom:50%;" /></div>

* DFS

  如下列二叉树，最大路径和可能为以下三种情况：

  1. p+a+b
  2. p+a+c
  3. a+b+c

  ```
        p
       /
      a
     / \
    b   c
  ```

  因此当递归到节点 a 时，求出以 a 为根节点时的最大路径和，更新 ans 为 上述第三种情况和当前 ans 的最大值，返回时返回 a+max(b,c)。

  ```python
  def maxPathSum(self, root: Optional[TreeNode]) -> int:
      self.ans=-float('inf')
      def dfs(node):
          if not node: return 0
          l=max(dfs(node.left),0)
          r=max(dfs(node.right),0)
          self.ans=max(self.ans,node.val+l+r)
          return node.val+max(l,r)
      dfs(root)
      return self.ans
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

### 56. [全排列（中等）](https://leetcode.cn/problems/permutations/)

<div align=center><img src="56.png" style="zoom:50%;" /></div>

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

### 60. [括号生成（中等）](https://leetcode.cn/problems/generate-parentheses/)

<div align=center><img src="60.png" style="zoom:50%;" /></div>

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

  

## 二分查找

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

### 66. ⭐️ [在排序数组中查找元素的第一个和最后一个位置（中等）](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

<div align=center><img src="66.png" style="zoom:50%;" /></div>

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

### 69. ⭐️ [寻找两个正序数组的中位数（困难）](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

<div align=center><img src="69.png" style="zoom:50%;" /></div>

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

### 99. ⭐️ [下一个排列](https://leetcode.cn/problems/next-permutation/)

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

### 102. [字符串相加（简单）](https://leetcode.cn/problems/add-strings/)

<div align=center><img src="e102.png" style="zoom:50%;" /></div>

* 模拟

  设置两个指针与进位标志，模拟竖式加法。

  ```python
  def addStrings(self, num1: str, num2: str) -> str:
      p1, p2, c = len(num1)-1,len(num2)-1, 0
      ans=""
      while p1>=0 or p2>=0 or c:
          v1 = int(num1[p1]) if p1>=0 else 0
          v2 = int(num2[p2]) if p2>=0 else 0
          s = (v1+v2+c)%10
          c = (v1+v2+c)//10 # 后更新 c
          ans = str(s)+ans
          p1-=1
          p2-=1
      return ans
  ```

### 103. [重排链表（中等）](https://leetcode.cn/problems/reorder-list/)

<div align=center><img src="e103.png" style="zoom:50%;" /></div>

* 线性表

  使用一个列表按序存储链表，空间复杂度为 $O(n)$

* 寻找链表中点+反转链表+合并链表

  先寻找链表中点，然后对链表后半部分进行反转，最后合并。

  ```python
  def reorderList(self, head: Optional[ListNode]) -> None:
      def midNode(head):
          slow,fast=head,head
          while fast.next and fast.next.next:
              slow,fast=slow.next,fast.next.next
          return slow
  
      def reverse(head):
          pre,p=None,head
          while p:
              pre,p.next,p = p,pre,p.next
          return pre
  
      def merge(l1,l2):
          while l1 and l2:
              t1,t2=l1.next,l2.next
              l1.next,l2.next=l2,t1
              l1,l2=t1,t2
  
      mid = midNode(head)
      rhead,mid.next = mid.next,None
      rhead = reverse(rhead)
      merge(head,rhead)
  ```

### 104. [用栈实现队列（简单）](https://leetcode.cn/problems/implement-queue-using-stacks/)

<div align=center><img src="e104.png" style="zoom:50%;" /></div>

* 栈

  维护一个输入栈和一个输出栈，每次输入时直接将元素压入输入栈；每次输出时，若输出栈不会空，直接弹出输出栈栈顶元素，否则将输出栈全部元素一次弹出压入输出栈，再弹出输出栈栈顶元素。

  ```python
  class MyQueue:
      def __init__(self):
          self.in_stack = []
          self.out_stack = []
  
      def push(self, x: int) -> None:
          self.in_stack.append(x)
  
      def pop(self) -> int:
          if self.out_stack:
              return self.out_stack.pop()
          while self.in_stack:
              self.out_stack.append(self.in_stack.pop())
          return self.out_stack.pop()
  
      def peek(self) -> int:
          if self.out_stack:
              return self.out_stack[-1]
          while self.in_stack:
              self.out_stack.append(self.in_stack.pop())
          return self.out_stack[-1]
  
      def empty(self) -> bool:
          return not self.in_stack and not self.out_stack
  ```

### 105. [删除排序链表中的重复元素 II（中等）](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/)

<div align=center><img src="e105.png" style="zoom:50%;" /></div>

* 遍历

  ```python
  def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
      pre = dummy = ListNode(next=head)
      cur = head
      while cur:
          if cur.next and cur.val==cur.next.val: # 遇到重复节点
              while cur.next and cur.val == cur.next.val:
                  cur = cur.next
              cur=pre.next=cur.next
          else:
              pre,cur=pre.next,cur.next
      return dummy.next
  ```

### 106. [复原 IP 地址（中等）](https://leetcode.cn/problems/restore-ip-addresses/)

<div align=center><img src="e106.png" style="zoom:50%;" /></div>

* 回溯

  定义回溯函数 backtracing(path,idx,num) ，path 表示当前已经分割好的整数，idx 表示当前遍历到的下标，num 表示已经添加了多少 `.`。每次循环判断 s[idx:idx+i] 是否为合法整数，如果是则加入 path，进入 backtracking(path,idx+i,num+1)。

  ```python
  def restoreIpAddresses(self, s: str) -> List[str]:
          def legal(s):
              if not s or (s[0]=='0' and len(s)>1) or int(s)>255:
                  return False
              return True
          def backtracking(path,idx,num):
              if num==3 and legal(s[idx:]): # 已经添加了三个.并且剩下部分是合法整数
                  ans.append('.'.join(path+[s[idx:]]))
                  return
              if idx>=n: return # 遍历到结尾
              for i in range(1,4):
                  sub = s[idx:idx+i]
                  if legal(sub):
                      path.append(sub)
                      backtracking(path,idx+i,num+1)
                      path.pop()
          ans,n=[],len(s)
          if n<4 or n>12: return [] # 剪枝
          backtracking([],0,0)
          return ans
  ```

### 107. ⭐️ [x 的平方根（简单）](https://leetcode.cn/problems/sqrtx/)

<div align=center><img src="e107.png" style="zoom:50%;" /></div>

* 二分查找

  找到满足 $k^2\leq x$ 最大的k

  ```python
  class Solution:
      def mySqrt(self, x: int) -> int:
          l,r,ans=0,x,-1
          while l<=r:
              mid = l+(r-l)//2
              if mid*mid<=x:
                  ans = mid
                  l=mid+1
              else: r=mid-1
          return ans
  ```

* 牛顿迭代法

  选取初始点 $x_i$，不断通过 $x_i$ 处的切线来逼近零点。

  选取函数上一点 $(x_i,x_i^2-C)$，则此处的切线为 $y=2x_i(x-x_i)+x_i^2-C$，则该切线与 x 轴的交点即为逼近的下一个点 $x_{i+1}$。令 $y=0$，有 $x=\frac{x_i^2-C}{2x_i}=\frac{1}{2}(x_i-\frac{C}{x_i})$。

  <div align=center><img src="e1071.png" style="zoom:50%;" /></div>

  ```python
  def mySqrt(self, x: int) -> int:
      if x<=1:
          return x
      k=x
      while k*k>x:
          k=(k+x/k)//2
      return int(k)
  ```

  



