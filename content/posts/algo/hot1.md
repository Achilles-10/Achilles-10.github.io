---
title: "LeetCode 热题 100 : 0~50"
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

### 9. [找到字符串中所有字母异位词（中等）](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

<div align=center><img src="9.png" style="zoom:50%;" /></div>

* 滑动窗口+哈希表

  ```python
  def findAnagrams(self, s: str, p: str) -> List[int]:
      m,n,ans=len(s),len(p),[]
      if m<n: return []
      cntp = Counter(p)
      cnts = Counter(s[:n])
      if cntp==cnts: ans.append(0)
      for i in range(1,m-n+1):
          cnts[s[i-1]]-=1
          cnts[s[i+n-1]]+=1
          if cntp==cnts: ans.append(i)
      return ans
  ```

* 滑动窗口+数组

  使用长度为 26 的数组代替哈希表

  ```python
  def findAnagrams(self, s: str, p: str) -> List[int]:
      m,n,ans=len(s),len(p),[]
      if m<n: return []
      cntp,cnts = [0]*26,[0]*26
      for i in range(n):
          cnts[ord(s[i])-ord('a')]+=1
          cntp[ord(p[i])-ord('a')]+=1
      if cnts==cntp: ans.append(0)
      for i in range(1,m-n+1):
          cnts[ord(s[i-1])-ord('a')]-=1
          cnts[ord(s[i+n-1])-ord('a')]+=1
          if cnts==cntp: ans.append(i)
      return ans
  ```

## 子串

### 10. ⭐️ [和为 K 的子数组（中等）](https://leetcode.cn/problems/subarray-sum-equals-k/)

<div align=center><img src="10.png" style="zoom:50%;" /></div>

* ⭐️ 哈希表+前缀和

  用哈希表记录前缀和出现的次数，若当前 s-k 在哈表中出现过，则说明可以构成 dict[s-k] 个和为 k 的子数组。

  ```python
  def subarraySum(self, nums: List[int], k: int) -> int:
      s=ans=0
      my_dict = defaultdict(int)
      my_dict[0]=1
      for num in nums:
          s += num
          ans += my_dict[s-k]
          my_dict[s]+=1
      return ans
  ```

  

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

### 12. [最小覆盖子串（困难）](https://leetcode.cn/problems/minimum-window-substring/)

<div align=center><img src="12.png" style="zoom:50%;" /></div>

* 滑动窗口+哈希表

  维护一个哈希表，存储所需字符的数量。

  不断右移窗口右端点，直至窗口内包含了 t 的所有元素；不断右移窗口左端点，将不必要的元素排除，记录最佳答案；不断重复，直至右端点超出了 s 的范围。

  额外使用一个 cnt 变量来记录所需元素的总数量，当 cnt=0 时即可开始收缩窗口。

  ```python
  def minWindow(self, s: str, t: str) -> str:
      m,n=len(s),len(t)
      if n>m: return ""
      tab = Counter(t)
      cnt,i = n,0
      ans,start=float('inf'),-1
      for j,c in enumerate(s):
          if tab[c]>0:
              cnt-=1
          tab[c]-=1
          if cnt==0: # 满足所需字符
              while s[i] not in tab or tab[s[i]]<0: # 排除不必要元素
                  if tab[s[i]]<0:
                      tab[s[i]]+=1
                  i+=1
              if j-i+1<ans: ans,start=j-i+1,start # 最佳答案
              tab[s[i]]+=1
              cnt+=1
              i+=1
      return s[start:start+ans] if start!=-1 else ""
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


### 15. ⭐️ [轮转数组（中等）](https://leetcode.cn/problems/rotate-array/)

<div align=center><img src="15.png" style="zoom:50%;" /></div>

* 使用额外数组

  ```python
  def rotate(self, nums: List[int], k: int) -> None:
      """
      Do not return anything, modify nums in-place instead.
      """
      n,k=len(nums),k%n
      if k==0: return
      nums[:]=nums[n-k:]+nums[:n-k]
  ```

* ⭐️ 数组翻转

  当我们将数组的元素向右移动 $k$ 次后，尾部 $k\bmod n$ 个元素会移动至数组头部，其余元素向后移动 $k\bmod n$ 个位置。因此可依次翻转全部数组、翻转 $[0,k\bmod n-1]$ 区间，翻转 $[k\bmod n,n-1]$ 区间。

  <div align=center><img src="151.png" style="zoom:50%;" /></div>

  ```python
  def rotate(self, nums: List[int], k: int) -> None:
      """
      Do not return anything, modify nums in-place instead.
      """
      def reverse(i,j):
          while i<j:
              nums[i],nums[j]=nums[j],nums[i]
              i,j=i+1,j-1
      n=len(nums)
      reverse(0,n-1)
      reverse(0,k%n-1)
      reverse(k%n,n-1)
  ```

* ⭐️ 循环交换

  位置为 i 的元素将会出现在位置 (i+k)%n，每次处理时保存新位置的元素并将其交换至下一个位置，直到遍历完全部元素，遍历循环次数为 gcd(n,k)。

  ```python
  def rotate(self, nums: List[int], k: int) -> None:
      """
      Do not return anything, modify nums in-place instead.
      """
      def gcd(a,b):
          return gcd(b%a,a) if a else b
      n=len(nums)
      k=k%n
      cnt = gcd(k,n)
      for start in range(cnt):
          cur = (start + k) % n
          prev = nums[start]
          while start!=cur:
              nums[cur],prev=prev,nums[cur]
              cur = (cur + k)%n
          nums[cur],prev=prev,nums[cur]
  ```


### 16. [除自身以外数组的乘积（中等）](https://leetcode.cn/problems/product-of-array-except-self/)

<div align=center><img src="16.png" style="zoom:50%;" /></div>

* 额外数组

  使用两个额外的数组 left 和 right 分别存储当前元素左侧和右侧元素的乘积。

  ```python
  def productExceptSelf(self, nums: List[int]) -> List[int]:
      n,ans=len(nums),[]
      left,right=[1]*n,[1]*n
      for i in range(n-1):
          left[i+1]=left[i]*nums[i]
          right[n-2-i]=right[n-1-i]*nums[n-1-i]
      for i in range(n):
          ans.append(left[i]*right[i])
      return ans
  ```

* 常数空间

  使用除法操作；同时记录下零的数量和索引。

  ```python
  def productExceptSelf(self, nums: List[int]) -> List[int]:
      n=len(nums)
      ans,mul,zeros,zero_idx=[0]*n,1,0,0
      for i in range(n):
          if nums[i]==0:
              zeros,zero_idx=zeros+1,i
              if zeros==2: return [0]*n
          else:
              mul*=nums[i]
      if zeros==1: ans[zero_idx]=mul
      else:
          for i in range(n):
              ans[i]=mul//nums[i]
      return ans
  ```

### 17. ⭐️ [缺失的第一个正数（困难）](https://leetcode.cn/problems/first-missing-positive/)

<div align=center><img src="17.png" style="zoom:50%;" /></div>

* 标记数组

  首先将所有负数转换为 n+1；遍历数组，将属于 $[1,N]$ 的元素对应位置标记为负数；寻找第一个不为负数的位置，即为缺失的第一个正数。

  <div align=center><img src="171.png" style="zoom:50%;" /></div>

  ```python
  def firstMissingPositive(self, nums: List[int]) -> int:
      n = len(nums)
      for i in range(n):
          if nums[i]<=0: nums[i]=n+1
      for i in range(n):
          num=abs(nums[i]) # 若已被标记为负数，取其绝对值进行判断
          if num<n+1: nums[num-1]=-abs(nums[num-1])
      for i in range(n):
          if nums[i]>0: return i+1
      return n+1
  ```

* 置换

  通过置换使 $nums[i]=i+1$。当 $1\leq nums[i]\leq n\ 且\ nums[i]-1\neq i$ 时，进行交换。为防止死循环，还需要保证进行置换的两个值不相等，即 $nums[i]\neq nums[nums[i]-1]$。

  ```python
  def firstMissingPositive(self, nums: List[int]) -> int:
      n = len(nums)
      for i in range(n):
          while 1<=nums[i]<=n and nums[i]!=nums[nums[i]-1]: # 为防止死循环，判断条件不能为 i!=nums[i]-1
              nums[nums[i]-1],nums[i]=nums[i],nums[nums[i]-1]
      for i in range(n):
          if nums[i]-1!=i:
              return i+1
      return n+1
  ```


## 矩阵

### 18. ⭐️ [矩阵置零（中等）](https://leetcode.cn/problems/set-matrix-zeroes/)

<div align=center><img src="18.png" style="zoom:50%;" /></div>

* 使用标记数组

  记录下置零的行和列，O(m+n) 空间复杂度。

  ```python
  def setZeroes(self, matrix: List[List[int]]) -> None:
      m,n=len(matrix),len(matrix[0])
      rows,cols=[0]*m,[0]*n
      for i in range(m):
          for j in range(n):
              if matrix[i][j]==0:
                  rows[i]=cols[j]=1
      for i in range(m):
          for j in range(n):
              if rows[i] or cols[j]:
                  matrix[i][j]=0
  ```

* ⭐️ 标记常量

  使用矩阵第一行和第一列代替标记数组，但这会导致第一行和第一列被修改。使用两个标记变量记录第一行和第一列是否包含零。常数空间复杂度。

  ```python
  def setZeroes(self, matrix: List[List[int]]) -> None:
      m,n=len(matrix),len(matrix[0])
      flag_r0 = any(matrix[0][j]==0 for j in range(n))
      flag_c0 = any(matrix[i][0]==0 for i in range(m))
      for i in range(1,m):
          for j in range(1,n):
              if matrix[i][j]==0:
                  matrix[i][0]=matrix[0][j]=0
      for i in range(1,m):
          for j in range(1,n):
              if matrix[i][0]==0 or matrix[0][j]==0:
                  matrix[i][j]=0
      if flag_r0:
          for j in range(n):
              matrix[0][j]=0
      if flag_c0:
          for i in range(m):
              matrix[i][0]=0
  ```

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


### 20. [旋转图像（中等）](https://leetcode.cn/problems/rotate-image/)

<div align=center><img src="20.png" style="zoom:50%;" /></div>

* 原地交换

  找到对应四个点的坐标：$(i,j)\rightarrow(j,n-1-i)\rightarrow(n-1-i,n-1-j)\rightarrow(n-1-j,i)\rightarrow(i,j)$

  ```python
  def rotate(self, matrix: List[List[int]]) -> None:
      """
      Do not return anything, modify matrix in-place instead.
      """
      n=len(matrix)
      cnt = (n+1)//2
      for i in range(cnt):
          for j in range(i,n-1-i):
              matrix[i][j],matrix[j][n-1-i],matrix[n-1-i][n-1-j],matrix[n-1-j][i]=matrix[n-1-j][i],matrix[i][j],matrix[j][n-1-i],matrix[n-1-i][n-1-j]
  ```

### 21. [搜索二维矩阵 II（中等）](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

<div align=center><img src="22.png" style="zoom:50%;" /></div>

* 二分查找

  ```python
  def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
      m,n=len(matrix),len(matrix[0])
      i,j=0,n-1
      while i<m and j>=0:
          if target==matrix[i][j]:
              return True
          elif target>matrix[i][j]:
              i+=1
          else:
              j-=1
      return False
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

### 24. [回文链表（简单）](https://leetcode.cn/problems/palindrome-linked-list/)

<div align=center><img src="24.png" style="zoom:50%;" /></div>

* 反转链表

  找中点+反转链表+遍历比较

  ```python
  def isPalindrome(self, head: Optional[ListNode]) -> bool:
      slow = fast = head
      while fast and fast.next:
          slow = slow.next
          fast = fast.next.next
      pre,p = None,slow
      while p:
          pre,p.next,p = p,pre,p.next
      p = head
      while p and pre:
          if p.val != pre.val: return False
          p,pre=p.next,pre.next
      return True
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

### 28. [两数相加（中等）](https://leetcode.cn/problems/add-two-numbers/)

<div align=center><img src="28.png" style="zoom:50%;" /></div>

* 模拟

  ```python
  def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
      dummy = ListNode()
      p1,p2,p=l1,l2,dummy
      carry = 0
      while p1 or p2 or carry:
          int1 = p1.val if p1 else 0
          int2 = p2.val if p2 else 0
          s = int1+int2+carry
          val,carry = s%10,s//10
          p.next = ListNode(val=val)
          p=p.next
          p1=p1.next if p1 else p1
          p2=p2.next if p2 else p2
      return dummy.next
  ```

### extra. ⭐️ [两数相加 II（中等）](https://leetcode.cn/problems/add-two-numbers-ii/)

<div align=center><img src="281.png" style="zoom:50%;" /></div>

* 栈

  若不反转链表，则用栈存储链表的值。返回时注意链表头。

  ```python
  def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
      s1,s2=[],[]
      while l1:
          s1.append(l1.val)
          l1=l1.next
      while l2:
          s2.append(l2.val)
          l2=l2.next
      dummy=ListNode()
      carry = 0
      while s1 or s2 or carry:
          int1 = s1.pop() if s1 else 0
          int2 = s2.pop() if s2 else 0
          s = int1+int2+carry
          carry,val = s//10,s%10 # divmod(s,10)
          dummy.next = ListNode(val,dummy.next)
      return dummy.next
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


### 30. [两两交换链表中的节点（中等）](https://leetcode.cn/problems/swap-nodes-in-pairs/)

<div align=center><img src="30.png" style="zoom:50%;" /></div>

* 迭代

  ```python
  def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
      pre = dummy = ListNode(next=head)
      while pre.next and pre.next.next:
          s,f = pre.next,pre.next.next
          pre.next,s.next,f.next=f,f.next,s
          pre = s
      return dummy.next
  ```

* 递归

  ```python
  def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
      if not head or not head.next:
          return head
      pre,cur = head,head.next
      pre.next = self.swapPairs(cur.next)
      cur.next = pre
      return cur
  ```

### 31. [K 个一组翻转链表（困难）](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

<div align=center><img src="31.png" style="zoom:50%;" /></div>

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


### 32. [复制带随机指针的链表（中等）](https://leetcode.cn/problems/copy-list-with-random-pointer/)

<div align=center><img src="32.png" style="zoom:50%;" /></div>

* 节点拆分

  三次遍历。第一次遍历，在每个节点的后面复制一个新节点；第二次遍历设置新节点的随机指针为前指针的随机指针的新节点；第三次遍历连接复制的新节点。

  <div align=center><img src="321.jpg" style="zoom:50%;" /></div>

  ```python
  def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
      dummy = Node(x=-1,next=head)
      p=head
      while p:
          new_node = Node(x=p.val,next=p.next) # 新建节点
          p.next,p=new_node,p.next
      p=head
      while p:
          p.next.random = p.random.next if p.random else None # 设置新节点的随机指针
          p=p.next.next
      p=dummy
      while p and p.next:
          p.next,p=p.next.next,p.next.next # 连接新节点
      return dummy.next
  ```

* 哈希表

  使用哈希表来存储节点对应的复制节点

  ```python
  def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
      if not head: return None
      p,tab = head,{}
      while p:
          tab[p]=Node(x=p.val)
          p=p.next
      p=head
      while p:
          tab[p].next = tab.get(p.next,None)
          tab[p].random = tab.get(p.random,None)
          p=p.next
      return tab[head]
  ```

### 33. ⭐️ [排序链表（中等）](https://leetcode.cn/problems/sort-list/)

<div align=center><img src="33.png" style="zoom:50%;" /></div>

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

  <div align=center><img src="331.png" style="zoom:50%;" /></div>

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


### 34. ⭐️ [合并 K 个升序链表（困难）](https://leetcode.cn/problems/merge-k-sorted-lists/)

<div align=center><img src="34.png" style="zoom:50%;" /></div>

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

### 35. ⭐️ [LRU 缓存（中等）](https://leetcode.cn/problems/lru-cache/)

<div align=center><img src="35.png" style="zoom:50%;" /></div>

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

### 36. ⭐️ [二叉树的中序遍历（简单）](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

<div align=center><img src="36.png" style="zoom:50%;" /></div>

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


### 37. [二叉树的最大深度（简单）](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

<div align=center><img src="37.png" style="zoom:50%;" /></div>

* 递归

  ```python
  def maxDepth(self, root: Optional[TreeNode]) -> int:
      if not root:
          return 0
      l=self.maxDepth(root.left)
      r=self.maxDepth(root.right)
      return max(l,r)+1
  ```

### 38. [翻转二叉树（简单）](https://leetcode.cn/problems/invert-binary-tree/)

<div align=center><img src="38.png" style="zoom:50%;" /></div>

* 递归

  ```python
  def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
      if not root: return root
      root.left,root.right = self.invertTree(root.right),self.invertTree(root.left) # 同时赋值
      return root
  ```

  

### 39. [对称二叉树（简单）](https://leetcode.cn/problems/symmetric-tree/)

<div align=center><img src="39.png" style="zoom:50%;" /></div>

* 递归

  ```python
  def isSymmetric(self, root: Optional[TreeNode]) -> bool:
      def sym(a,b):
          if not a and not b:
              return True
          if not a or not b:
              return False
          return a.val == b.val and sym(a.left,b.right) and sym(a.right,b.left)
      return sym(root.left,root.right)
  ```

* 迭代

  ```python
  def isSymmetric(self, root: Optional[TreeNode]) -> bool:
      queue = [root.left,root.right]
      while queue:
          l=queue.pop(0)
          r=queue.pop(0)
          if not l and not r: continue
          if not l or not r: return False
          if l.val!=r.val: return False
          queue.append(l.left)
          queue.append(r.right)
          queue.append(l.right)
          queue.append(r.left)
      return True
  ```

### extra. [平衡二叉树（简单）](https://leetcode.cn/problems/balanced-binary-tree/)

<div align=center><img src="e43.png" style="zoom:50%;" /></div>

* 自底向上

  ```python
  def isBalanced(self, root: Optional[TreeNode]) -> bool:
      def helper(node):
          if not node:
              return 0
          l = helper(node.left)
          r = helper(node.right)
          if l==-1 or r==-1 or abs(l-r)>1: # -1 表示不平衡，提前终止
              return -1
          return max(l,r)+1
      return helper(root)>=0
  ```

* 自顶向下

  ```python
  def isBalanced(self, root: TreeNode) -> bool:
      def height(root: TreeNode) -> int:
          if not root:
              return 0
          return max(height(root.left), height(root.right)) + 1
      if not root:
          return True
      return abs(height(root.left) - height(root.right)) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)
  ```

### 40. [二叉树的直径（简单）](https://leetcode.cn/problems/diameter-of-binary-tree/)

<div align=center><img src="40.png" style="zoom:50%;" /></div>

* DFS

  ```python
  def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
      def dfs(node):
          if not node: return 0
          nonlocal ans
          l=dfs(node.left)
          r=dfs(node.right)
          ans=max(ans,l+r)
          return max(l,r)+1
      ans=0
      dfs(root)
      return ans
  ```

### 41. [二叉树的层序遍历（中等）](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

<div align=center><img src="41.png" style="zoom:50%;" /></div>

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

### 42. [将有序数组转换为二叉搜索树（简单）](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

<div align=center><img src="42.png" style="zoom:50%;" /></div>

* 递归

  ```python
  def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
      if not nums: return None
      mid=len(nums)//2
      root = TreeNode(nums[mid])
      root.left = self.sortedArrayToBST(nums[:mid])
      root.right = self.sortedArrayToBST(nums[mid+1:])
      return root
  ```

### 43. [验证二叉搜索树（中等）](https://leetcode.cn/problems/validate-binary-search-tree/)

<div align=center><img src="43.png" style="zoom:50%;" /></div>

* DFS

  设置一个上下界，每当节点元素越界时，则不是二叉搜索树，递归到下一层时更新上下界为当前节点元素值。

  ```python
  def isValidBST(self, root: Optional[TreeNode]) -> bool:
      def dfs(node,low,high):
          if not node:
              return True
          if node.val <=low or node.val >=high:
              return False
          return dfs(node.left,low,node.val) and dfs(node.right,node.val,high)
      return dfs(root,-inf,inf)
  ```


### 44. [二叉搜索树中第K小的元素（中等）](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)

<div align=center><img src="44.png" style="zoom:50%;" /></div>

* 中序遍历（递归）

  ```python
  def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
      self.cnt,self.ans=0,-1
      def dfs(node):
          if not node or self.ans!=-1:
              return
          dfs(node.left)
          self.cnt+=1
          if self.cnt==k:
              self.ans=node.val
          dfs(node.right)
      dfs(root)
      return self.ans
  ```

* 中序遍历（遍历）

  ```python
  def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
      stack = []
      while stack or root:
          while root:
              stack.append(root)
              root=root.left
          root=stack.pop()
          k-=1
          if k==0: return root.val
          root=root.right
  ```

* 记录子树节点数

  > 如果你需要频繁地查找第 k 小的值，你将如何优化算法？

  记录下以每个结点为根结点的子树的结点数，使用类似二分查找的方法搜索

  ```python
  class Bst:
      def __init__(self,root):
          self.root=root
          self._node_num = {}
          self._count_node_num(root)
          
      def _count_node_num(self,node):
          if not node:
              return 0
          self._node_num[node] = 1+self._count_node_num(node.left)+self._count_node_num(node.right)
          return self._node_num[node]
  
      def _get_node_num(self,node):
          return self._node_num[node] if node else 0
  
      def kth_smallest(self,k):
          node = self.root
          while node:
              left = self._get_node_num(node.left)
              if left==k-1:
                  return node.val
              elif left<k-1:
                  node = node.right
                  k-=left+1
              else:
                  node = node.left
      
  class Solution:
      def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
          bst = Bst(root)
          return bst.kth_smallest(k)
  ```

### 45. [二叉树的右视图（中等）](https://leetcode.cn/problems/binary-tree-right-side-view/)

<div align=center><img src="45.png" style="zoom:50%;" /></div>

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

### 46. ⭐️ [二叉树展开为链表（中等）](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)

<div align=center><img src="46.png" style="zoom:50%;" /></div>

* 前序遍历+原地展开

  类似迭代前序遍历，先将右节点压栈，遍历左节点。

  ```python
  def flatten(self, root: Optional[TreeNode]) -> None:
      if not root: return
      stack = []
      pre,cur=None,root
      while stack or cur:
          if cur:
              if cur.right: stack.append(cur.right)
              cur.right,cur.left = cur.left, None
              pre,cur = cur,cur.right
          else:
              cur = stack.pop()
              pre.right = cur
  ```

* ⭐️ 寻找前驱节点

  空间复杂度为 O(1)

  展开的关键操作是将左子树的最后一个访问节点的右节点设置为当前节点的右节点，于是寻找每个节点左子树的最后一个节点。

  ```python
  def flatten(self, root: Optional[TreeNode]) -> None:
      """
      Do not return anything, modify root in-place instead.
      """
      if not root: return
      cur = root
      while cur:
          if cur.left:
              pre=cur.left
              while pre.right:
                  pre = pre.right
              pre.right = cur.right
              cur.right,cur.left = cur.left,None
          cur = cur.right
  ```

### 47. [从前序与中序遍历序列构造二叉树（中等）](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

<div align=center><img src="47.png" style="zoom:50%;" /></div>

* 递归

  preorder[0] 是根节点的值，找到根节点在 inorder 的下标，划分左右子树递归处理。

  ```python
  def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
      if not preorder:
          return None
      root_val = preorder[0]
      root = TreeNode(root_val)
      idx = inorder.index(root_val)
      root.left = self.buildTree(preorder[1:idx+1],inorder[:idx])
      root.right = self.buildTree(preorder[idx+1:],inorder[idx+1:])
      return root
  ```

* 迭代

### 48. ⭐️ [路径总和 III（中等）](https://leetcode.cn/problems/path-sum-iii/)

<div align=center><img src="48.png" style="zoom:50%;" /></div>

* 前缀和+哈希

  使用一个字典存储前缀和，若当前 s-targetSum 在前缀和字典中存在，则说明到当前节点有路径存在。

  ```python
  def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
      def dfs(node,s):
          if not node: return 0
          res=0
          s+=node.val
          res+=presum[s-targetSum] #到当前节点满足要求的路径数量
          presum[s]+=1 # 更新前缀和
          res+=dfs(node.left,s)
          res+=dfs(node.right,s)
          presum[s]-=1
          return res
      presum = defaultdict(int)
      presum[0]=1
      return dfs(root,0)
  ```

### extra. [路径总和（简单 ）](https://leetcode.cn/problems/path-sum/)

<div align=center><img src="e491.png" style="zoom:50%;" /></div>

* 递归

  ```python
  def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
      def dfs(node,target):
          if not node:
              return False
          if not node.left and not node.right:
              return node.val==target
          l=dfs(node.left,target-node.val)
          r=dfs(node.right,target-node.val)
          return l or r
      return dfs(root,targetSum)
  ```

### extra. [路径总和 II（中等）](https://leetcode.cn/problems/path-sum-ii/)

<div align=center><img src="e49.png" style="zoom:50%;" /></div>

* 递归

  ```python
  def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
      ans,path=[],[]
      def dfs(node,s):
          if not node: return
          s+=node.val
          path.append(node.val)
          if s==target and not node.left and not node.right:
              ans.append(path[:])
          dfs(node.left,s)
          dfs(node.right,s)
          path.pop()
      dfs(root,0)
      return ans
  ```


### 49. ⭐️ [二叉树的最近公共祖先（中等）](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

<div align=center><img src="49.png" style="zoom:50%;" /></div>

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

### 50. [二叉树中的最大路径和（困难）](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

<div align=center><img src="50.png" style="zoom:50%;" /></div>

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