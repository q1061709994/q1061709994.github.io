---
icon: article
title: 刷题笔记
author: huan
date: 2022-01-02
category: 算法笔记
tag: 
    - 数据结构与算法
star: true
---

## 链表

#### [剑指 Offer 22. 链表中倒数第k个节点 ](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

#### **[剑指 Offer II 024. 反转链表](https://leetcode-cn.com/problems/UHnkqh/)**

#### [1290. 二进制链表转整数](https://leetcode-cn.com/problems/convert-binary-number-in-a-linked-list-to-integer/) 

#### [1669. 合并两个链表](https://leetcode-cn.com/problems/merge-in-between-linked-lists/)

#### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

建立一个新链表，相同的数字只取第一个不一样的

#### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

虚拟头节点+暴力

#### [剑指 Offer II 023. 两个链表的第一个重合节点](https://leetcode-cn.com/problems/3u1WK4/)

遍历完一个链表后遍历另一个链表a + c + b等于b + c + a

#### [876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

双指针或者遍历两次

#### [1721. 交换链表中的节点](https://leetcode-cn.com/problems/swapping-nodes-in-a-linked-list/)

双指针

#### [剑指 Offer II 027. 回文链表](https://leetcode-cn.com/problems/aMhZSa/)

数组+双指针

#### [面试题 02.04. 分割链表](https://leetcode-cn.com/problems/partition-list-lcci/)

双指针

#### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

#### [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

双指针+虚拟头节点(当需要特判)

#### [445. 两数相加 II](https://leetcode-cn.com/problems/add-two-numbers-ii/)

反转后相加

#### [237. 删除链表中的节点](https://leetcode-cn.com/problems/delete-node-in-a-linked-list/)

#### [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

快慢指针

#### [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

快慢指针

#### [82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

双指针

#### [203. 移除链表元素](https://leetcode-cn.com/problems/remove-linked-list-elements/)

#### [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

#### [143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

双指针+寻找中点+链表逆序+链表合并

#### [817. 链表组件](https://leetcode-cn.com/problems/linked-list-components/)

#### [328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)

#### [面试题 02.01. 移除重复节点](https://leetcode-cn.com/problems/remove-duplicate-node-lcci/)

#### [86. 分隔链表](https://leetcode-cn.com/problems/partition-list/)

相对顺序不能发生变化

#### [23. 合并K个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

~~~
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) return null;
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        PriorityQueue<ListNode> pq = new PriorityQueue<>((o1, o2) -> o1.val - o2.val);
        for (ListNode list : lists) {
            if (list == null) continue;
            pq.add(list);
        }
        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            cur = cur.next = node;
            if (node.next != null) pq.add(node.next);
        }
        return dummy.next;
    }
}
~~~



#### [剑指 Offer II 029. 排序的循环链表](https://leetcode.cn/problems/4ueAj6/)

~~~
class Solution {
    public Node insert(Node head, int insertVal) {
        Node node = new Node(insertVal);
        if (head == null) {
            node.next = node;
            return node;
        }
        Node p = head, q = head.next;
        while (q != head) {
            if (p.val <= node.val && node.val <= q.val) break;
            if (p.val > q.val) {
                if (p.val < node.val || q.val > node.val) {
                    break;
                }
            }
            p = p.next;
            q = q.next;
        }
        p.next = node;
        node.next = q;
        return head;
    }
}
~~~

#### [61. 旋转链表](https://leetcode.cn/problems/rotate-list/)

~~~
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode a = dummy;
        int num = 0;
        while (a.next != null) {
            num ++ ;
            a = a.next;
        }
        k %= num;
        ListNode b = dummy;
        for (int i = 0; i < num - k; i ++ ) b = b.next;
        a.next = dummy.next;
        dummy.next = b.next;
        b.next = null;
        return dummy.next;
    }
}
~~~

#### [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

**解法：模拟**

~~~
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head; 
        for (ListNode p = dummy;; ) {
            ListNode q = p;
            // 对于链表中的每一个点, 先遍历其后的k个点, 如果遍历完为空, 说明不够k个点
            for (int i = 0; i < k && q != null; i ++ ) q = q.next;
            if (q == null) break;
            //System.out.println(q.val);
            ListNode a = p.next, b = a.next;
            //开始翻转对k个节点进行内部翻转, 需要翻转k-1次
            for (int i = 0; i < k - 1; i ++ ) {
                ListNode c = b.next;
                b.next = a;
                a = b;
                b = c;
            }
            ListNode c = p.next;
            p.next = a;
            c.next = b;
            // 更新链表到下一个节点
            p = c;
        }
        return dummy.next;
    }
}
~~~

#### [725. 分隔链表](https://leetcode.cn/problems/split-linked-list-in-parts/)

**解法：模拟**

~~~
class Solution {
    public ListNode[] splitListToParts(ListNode head, int k) {
        ListNode[] res = new ListNode[k];
        ListNode cur = head;
        int n = 0;
        while (cur != null) {
            n ++ ;
            cur = cur.next;
        }
        int a = n / k, b = n % k;
        ListNode p = head, pre = null;
        for (int i = 0; i < k; i ++ ) {
            res[i] = p;
            int t = a + (b -- > 0 ? 1 : 0);
            for (int j = 0; j < t; j ++ ) {
                pre = p;
                p = p.next;
            }
            if (pre != null) pre.next = null;
        }
        return res;
    }
}
~~~

#### [138. 复制带随机指针的链表](https://leetcode.cn/problems/copy-list-with-random-pointer/)

**解法：使用哈希进行快速拷贝**

~~~
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Node cur = head;
        Map<Node, Node> map = new HashMap<>();
        //存放旧节点对应的新节点
        while (cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        //再次遍历老链表，进行新链表的连接
        while (cur != null) {
            Node next = cur.next;
            Node random = cur.random;
            map.get(cur).next = map.get(next);
            map.get(cur).random = map.get(random);
            cur = cur.next;
        }
        return map.get(head);
    }
}
~~~

**解法2：不使用额外空间**

~~~
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return head;
        Node cur = head;
		//复制节点进行连接 1->2  ==  1->1`->2->2`
        while (cur != null) {
            Node next = cur.next;
            cur.next = new Node(cur.val);
            cur = cur.next.next = next;
        }
        //连接random
        cur = head;
        while (cur != null) {
            if (cur.random != null) cur.next.random = cur.random.next;
            else cur.next.random = null;
            cur = cur.next.next;
        }
        //分离新链表
        //需要将原链表恢复
        Node res = head.next;
        cur = head;
        while (cur.next != null) {
            Node temp = cur.next;
            cur.next = cur.next.next;
            cur = temp;
        }
        return res;
    }
}
~~~

#### [147. 对链表进行插入排序](https://leetcode.cn/problems/insertion-sort-list/)

~~~
class Solution {
    public ListNode insertionSortList(ListNode head) {
        ListNode dummy = new ListNode(-1);
        for (var p = head; p != null; ) {
            ListNode cur = dummy, next = p.next;
            while (cur.next != null && cur.next.val <= p.val) cur = cur.next;
            p.next = cur.next;
            cur.next = p;
            p = next;
        }
        return dummy.next;
    }
}
~~~



## 单调栈

常见模型：找到每个数左边离他最近比它大/小的数

#### [6080. 使数组按非递减顺序排列](https://leetcode.cn/problems/steps-to-make-array-non-decreasing/)

~~~
//https://leetcode.cn/problems/steps-to-make-array-non-decreasing/solution/by-newhar-6k75/
class Solution {
    public int totalSteps(int[] nums) {
    	// 单调栈
        // 1. 每个元素一定时被左侧第一个更大的元素消除的
        // 2. 设 x 消除 y，也就是 [x] .... [y]，那么
        //    中间的 .... 一定先被消除，再 +1 次消除（x 消除 y）
        // 3. 那么，x 被消除所需轮数就是 [....] 中的最大消除轮数 + 1
        int n = nums.length;
        int res = 0;
        int[] f = new int[n];
        ArrayDeque<Integer> s = new ArrayDeque<>();
        for (int i = 0; i < nums.length; i ++ ) {
            int cur = 0;
            while (!s.isEmpty() && nums[s.peek()] <= nums[i]) {
                cur = Math.max(cur, f[s.peek()]);
                s.pop();
            }
            if (!s.isEmpty()) {
                f[i] = cur + 1;
                res = Math.max(res, f[i]);
            }
            s.push(i);
        }
        return res;
    }
}
~~~

#### [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)

~~~
class Solution {
    public int[] dailyTemperatures(int[] tem) {
        int n = tem.length;
        int[] ans = new int[n];
        Stack<Integer> s = new Stack<>();
        for (int i = 0; i < n; i ++ ) {
            while (s.size() > 0 && tem[s.peek()] < tem[i]) {
                ans[s.peek()] = i - s.peek();
                s.pop();
            }
            s.push(i);
        }
        return ans;
    }
}
~~~

#### [907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/)

~~~
class Solution {
    public int sumSubarrayMins(int[] arr) {
        int mod = (int)1e9 + 7;
        int n = arr.length;
        ArrayDeque<Integer> s = new ArrayDeque<>();
        int[] l = new int[n], r = new int[n];
        long ans = 0;
        for (int i = 0; i < n; i ++ ) {
			while (!s.isEmpty() && arr[s.peek()] > arr[i]) s.pop();
            if (s.isEmpty()) l[i] = -1;
            else l[i] = s.peek();
			s.push(i);
		}
        s = new ArrayDeque<>(); 
		for (int i = n - 1; i >= 0; i -- ) {
			while (!s.isEmpty() && arr[s.peek()] >= arr[i]) s.pop();
            if (s.isEmpty()) r[i] = n;
            else r[i] = s.peek();
			s.push(i);
		}
        for (int i = 0; i < n; i ++ ) {
            ans += (long) (i - l[i]) * (r[i] - i) * arr[i];
            ans %= mod;
        }
        return (int)(ans + mod) % mod;
    }
}
~~~

**解法：单调栈&三次遍历**

~~~
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n, res, mod = len(arr), 0, 10 ** 9 + 7
        left, s = [-1] * n, deque()
        for i, x in enumerate(arr):
            while s and x < arr[s[-1]]: s.pop()
            if s: left[i] = s[-1]
            s.append(i)
        right, s = [n] * n, deque()
        for i in range(n - 1, -1, -1):
            # 避免因相同数字，重复统计子数组
            while s and arr[i] <= arr[s[-1]]: s.pop()
            if s: right[i] = s[-1]
            s.append(i)
        for i, (x, l, r) in enumerate(zip(arr, left, right)):
            res += x * (i - l) * (r - i)
        return res % mod
~~~

**解法：单调栈&两次遍历**

~~~
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n, res, mod = len(arr), 0, 10 ** 9 + 7
        left, right, s = [-1] * n, [n] * n, deque()
        for i, x in enumerate(arr):
            while s and x <= arr[s[-1]]: 
                right[s.pop()] = i
            if s: left[i] = s[-1]
            s.append(i)
        for i, (x, l, r) in enumerate(zip(arr, left, right)):
            res += x * (i - l) * (r - i)
        return res % mod
~~~

**解法：单调栈&一次遍历**

~~~
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
    	# 栈顶下面的元素正好也是栈顶的左边界
        arr.append(-1)
        res, mod = 0, 10 ** 9 + 7
        s = [-1]
        for r, x in enumerate(arr):
            while len(s) > 1 and x <= arr[s[-1]]: 
                i = s.pop()
                res += arr[i] * (i - s[-1]) * (r - i)
            s.append(r)
        return res % mod
~~~



#### [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

~~~
class Solution {
    public boolean verifyPostorder(int[] pos) {
        ArrayDeque<Integer> q = new ArrayDeque<>();
        int pre = Integer.MAX_VALUE;
        for (int i = pos.length - 1; i >= 0; i -- ) {
            if (pos[i] > pre) return false;
            //当pos[i]小于栈顶元素时，表示要进入左子树了
            while (!q.isEmpty() && q.peek() > pos[i]) pre = q.pop();
            q.push(pos[i]);
        }
        return true;
    }
}
~~~

#### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

**解法1：单调栈**

~~~
class Solution {
    public int trap(int[] h) {
        int ans = 0, n = h.length;
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i ++ ) {
            while (!q.isEmpty() && h[q.peek()] < h[i]) {
                int cur = q.pop();
                if (q.isEmpty()) break;
                //考虑每个位置左边和右边 第一个 比自身高的矩形条，以及三个矩形条构成的 U 型，相当于对水的面积按 行 进行拆解。
                //h[q.peek()]为左边第一个比cur高的矩形，h[i]为右边第一个比cur高的矩形
                ans += (i - q.peek() - 1) * (Math.min(h[i], h[q.peek()]) - h[cur]);
            }
            q.push(i);
        }
        return ans;
    }
}
~~~

**解法2：双指针**

~~~
class Solution {
    public int trap(int[] h) {
        int n = h.length;
        int l = 0, r = n - 1, res = 0;
        //记录l, r遍历过的最大值
        int lMax = 0, rMax = 0;
        while (l < r) {
            lMax = Math.max(h[l], lMax);
            rMax = Math.max(h[r], rMax);
            //若lMax较小，则i号柱子的体积由lMax决定，反之
            if (lMax <= rMax) {
                res += lMax - h[l];
                l ++ ;
            } else {
                res += rMax - h[r];
                r -- ;
            }
        }
        return res;
    }
}
~~~

#### [402. 移掉 K 位数字](https://leetcode.cn/problems/remove-k-digits/)

**解法：单调栈**

~~~
class Solution {
    //尽可能让最高位小，最高位相同的情况下尽可能让次高位小
    public String removeKdigits(String num, int k) {
        StringBuilder res = new StringBuilder();
        for (char c : num.toCharArray()) {
            while (k > 0 && res.length() > 0 && res.charAt(res.length() - 1) > c) {
                k -- ;
                res.delete(res.length() - 1, res.length());
            }
            res.append(c);
        }
        while (k -- > 0) res.delete(res.length() - 1, res.length());
        k = 0;
        //删除前导0
        while (k < res.length() && res.charAt(k) == '0') k ++ ;
        return k == res.length() ? "0" : res.substring(k, res.length());
    }
}
~~~

#### [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

**解法1：枚举左右边界**

O（N * N）超时

~~~
class Solution {
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int res = 0;
        for (int l = 0; l < n; l ++ ) {
            int h = 100010;
            for (int r = l; r < n; r ++ ) {
                h = Math.min(h, heights[r]);
                res = Math.max(res, h * (r - l + 1));
            }
        }
        return res;
    }
}
~~~

**解法2：枚举高度**

O（N * N）超时

**解法3：单调栈**

~~~
class Solution {
    //找到每个柱形条左边和右边最近的比自己低的矩形条，然后用宽度乘上当前柱形条的高度
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int res = 0;
        ArrayDeque<Integer> s = new ArrayDeque<>();
        for (int i = 0; i < n; i ++ ) {
            while (!s.isEmpty() && heights[s.peek()] > heights[i]) {
            	//矩形的高度为h[cur], 宽度由cur两边最近的比h[cur]低的决定
                int cur = s.pop();
                if (s.isEmpty()) 
                    res = Math.max(res, heights[cur] * i);
                else
                    res = Math.max(res, heights[cur] * (i - s.peek() - 1));
            }
            s.push(i);
        }
        //处理栈中其他的数据      
        while (!s.isEmpty()) {
            int cur = s.pop();
            if (s.isEmpty()) 
                res = Math.max(res, heights[cur] * n);
            else
                res = Math.max(res, heights[cur] * (n - s.peek() - 1));
        }
        return res;
    }
}
~~~

#### [768. 最多能完成排序的块 II](https://leetcode.cn/problems/max-chunks-to-make-sorted-ii/)

~~~
class Solution {
    public int maxChunksToSorted(int[] arr) {
        //维护每一个块的最大值
        ArrayDeque<Integer> s = new ArrayDeque<>();
        for (int i = 0; i < arr.length; i ++ ) {
            if (s.isEmpty() || arr[i] >= arr[s.peek()]) s.push(i);
            else {
                int x = s.pop();
                //当前数融入之前块的过程
                //直到遇到一个块，使得该块的最大值小于或等于这个新添加的数，
                //或者当前数已经融合了所有块
                while (!s.isEmpty() && arr[s.peek()] > arr[i])
                    s.pop();
                s.push(x);
            }
        }
        return s.size();
    }
}
~~~

#### [1124. 表现良好的最长时间段](https://leetcode.cn/problems/longest-well-performing-interval/)

**解法：单调栈&前缀和**

~~~
class Solution {
    public int longestWPI(int[] h) {
        int n = h.length;
        for (int i = 0; i < n; i ++ ) 
            if (h[i] > 8) h[i] = 1;
            else h[i] = -1;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i ++ ) 
            s[i + 1] = s[i] + h[i];
        int res = 0;
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i <= n; i ++ ) 
            if (stack.isEmpty() || s[stack.peek()] > s[i])
                stack.push(i);
        for (int i = n; i >= 0; i -- ) {
            while (!stack.isEmpty() && s[stack.peek()] < s[i]) {
                res = Math.max(res, i - stack.peek());
                stack.pop();
            }
        }
        return res;
    }
}
~~~

#### [962. 最大宽度坡](https://leetcode.cn/problems/maximum-width-ramp/)

~~~
class Solution {
    public int maxWidthRamp(int[] nums) {
        int res = 0;
        ArrayDeque<Integer> s = new ArrayDeque<>();
        for (int i = 0; i < nums.length; i ++ ) 
            if (s.isEmpty() || nums[s.peek()] > nums[i])
                s.push(i);
        for (int i = nums.length - 1; i >= 0; i -- ) 
            while (!s.isEmpty() && nums[s.peek()] <= nums[i])
                res = Math.max(res, i - s.pop());
        return res;
    }
}
~~~

#### [503. 下一个更大元素 II](https://leetcode.cn/problems/next-greater-element-ii/)

~~~
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        var res = new int[n];
        var s = new ArrayDeque<Integer>();
        Arrays.fill(res, -1);
        for (int i = 0; i < n * 2; i ++ ) {
            int x = nums[i % n];
            while (!s.isEmpty() && nums[s.peek()] < x) {
                res[s.peek()] = x;
                s.pop();
            }
            if (i < n) s.push(i);
        }
        return res;
    }
}
~~~

#### [2289. 使数组按非递减顺序排列](https://leetcode.cn/problems/steps-to-make-array-non-decreasing/)

~~~
class Solution {
    public int totalSteps(int[] nums) {
        var res = 0;
        var s = new ArrayDeque<Pair<Integer, Integer>>();
        for (int i = 0; i < nums.length; i ++ ) {
            int cur = 0;
            while (!s.isEmpty() && nums[s.peek().getKey()] <= nums[i]) 
                cur = Math.max(cur, s.pop().getValue());

            if (!s.isEmpty()) {
                s.push(new Pair<>(i, cur + 1));
                res = Math.max(res, cur + 1);
            }
            else 
                s.push(new Pair<>(i, 0));
        }
        return res;
    }
}
~~~

#### [901. 股票价格跨度](https://leetcode.cn/problems/online-stock-span/)

~~~
class StockSpanner {
    ArrayDeque<Pair<Integer, Integer>> stk;
    int cur;
    public StockSpanner() {
        stk = new ArrayDeque<>();
    }
    
    public int next(int price) {
        while (!stk.isEmpty() && stk.peek().getValue() <= price) stk.pop();
        int pre = stk.isEmpty() ? -1 : stk.peek().getKey();
        var res = cur - pre;
        stk.push(new Pair(cur ++ , price));
        return res;
    }
}
~~~

#### [6227. 下一个更大元素 IV](https://leetcode.cn/problems/next-greater-element-iv/)

**解法：单调栈**

~~~
class Solution {
    public int[] secondGreaterElement(int[] nums) {
        var n = nums.length;
        var res = new int[n];
        Arrays.fill(res, -1);
        var s1 = new ArrayDeque<Integer>();
        var s2 = new ArrayDeque<Integer>();
        var tem = new ArrayDeque<Integer>();
        for (int i = 0; i < n; i ++ ) {
            while (!s2.isEmpty() && nums[s2.peek()] < nums[i])
                res[s2.pop()] = nums[i];
            while (!s1.isEmpty() && nums[s1.peek()] < nums[i]) 
                tem.push(s1.pop());
            while (!tem.isEmpty()) s2.push(tem.pop());
            s1.push(i);
        }
        return res;
    }
}
~~~

**解法：单调栈&堆**

~~~
class Solution {
    public int[] secondGreaterElement(int[] nums) {
        var n = nums.length;
        var res = new int[n];
        Arrays.fill(res, -1);
        var s1 = new ArrayDeque<Integer>();
        var q = new PriorityQueue<Integer>((a, b) -> nums[a] - nums[b]);
        for (int i = 0; i < n; i ++ ) {
            while (!q.isEmpty() && nums[q.peek()] < nums[i]) 
                res[q.poll()] = nums[i];  
            while (!s1.isEmpty() && nums[s1.peek()] < nums[i]) 
                q.add(s1.pop());
            s1.push(i);
        }
        return res;
    }
}
~~~

## 单调队列

常见模型：找到滑动窗口中离他最近比它大/小的数

#### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

~~~
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        ArrayDeque<Integer> q = new ArrayDeque<>();
        if (nums == null || nums.length < 2) return nums;
        int[] res = new int[nums.length - k + 1];
        int idx = 0;
        for (int i = 0; i < nums.length; i ++ ) {
            if (!q.isEmpty() && i - k + 1 > q.getFirst()) q.pollFirst();
            while (!q.isEmpty() && nums[q.getLast()] <= nums[i]) q.pollLast();
            q.add(i);
            if (i >= k - 1) res[idx ++ ] = nums[q.getFirst()];
        }
        return res;
    }
}
~~~

#### [862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)

~~~
class Solution {
    public int shortestSubarray(int[] nums, int k) {
        int ans = Integer.MAX_VALUE;
        int n = nums.length;
        long[] s = new long[n + 1];
        for (int i = 0; i < n; i ++ ) s[i + 1] = s[i] + nums[i];
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i <= n; i ++ ) {
            while (!q.isEmpty() && s[q.getFirst()] + k <= s[i]) {
                ans = Math.min(ans, i - q.getFirst());
                q.pollFirst();
            }
            //删除大于等于s[i]的数，原因不是最优
            while (!q.isEmpty() && s[q.getLast()] >= s[i]) {
                q.pollLast();
            }
            q.add(i);
        }
        if (ans == Integer.MAX_VALUE) return -1;
        return ans;
    }
}
~~~

~~~
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        s = [0] * (n + 1)
        for i in range(n):
            s[i + 1] = s[i] + nums[i]
        q, res = deque(), inf
        for i in range(n + 1):
            while q and s[q[0]] + k <= s[i]:
                res = min(res, i - q.popleft())
            while q and s[q[-1]] >= s[i]:
                q.pop()
            q.append(i)
        return res if res < inf else -1
~~~



#### [1438. 绝对差不超过限制的最长连续子数组](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

~~~
class Solution {
    public int longestSubarray(int[] nums, int limit) {
        int n = nums.length;
        int ans = 0;
        ArrayDeque<Integer> q1 = new ArrayDeque<>(), q2 = new ArrayDeque<>();
        for (int i = 0, j = 0; i < n; i ++ ) {
            while (!q1.isEmpty() && nums[q1.getLast()] > nums[i]) q1.pollLast();
            while (!q2.isEmpty() && nums[q2.getLast()] < nums[i]) q2.pollLast();
            q1.add(i);
            q2.add(i);
            while (!q1.isEmpty() && !q2.isEmpty() && nums[q2.getFirst()] - nums[q1.getFirst()] > limit) {
                if (q1.getFirst() == j) q1.pollFirst();
                if (q2.getFirst() == j) q2.pollFirst();
                j ++ ;
            }
            ans = Math.max(ans, i - j + 1);
        }
        return ans;
    }
}
~~~



## 二分查找

10.30

#### [704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

#### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

#### [35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

#### [162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

#### [69. Sqrt(x)](https://leetcode-cn.com/problems/sqrtx/)

#### [278. 第一个错误的版本](https://leetcode-cn.com/problems/first-bad-version/)

溢出的两种解决方式1、转long在转回int	方式2、l + (l - r) / 2

#### [AcWing14.不修改数组找出重复的数字](https://www.acwing.com/problem/content/description/15/)

抽屉原理+分治

抽屉原理：n+1 个苹果放在 n 个抽屉里，那么至少有一个抽屉中会放两个苹果

#### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

转换为142题求带环链表的入口

#### [74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)

#### [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

~~~
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int n = matrix.length, m = matrix[0].length;
        int i = 0, j = m - 1;
        while (i < n && j >= 0) {
            int t = matrix[i][j];
            if (t == target) return true;
            else if (t > target) j -- ;//如果t > target去掉一列
            else i ++ ;//如果t < target去掉一行
        }
        return false;
    }
}
~~~

#### [2271. 毯子覆盖的最多白色砖块数](https://leetcode.cn/problems/maximum-white-tiles-covered-by-a-carpet/)

~~~
class Solution {
    public int maximumWhiteTiles(int[][] tiles, int carpetLen) {
        int n = tiles.length;
        Arrays.sort(tiles, (a, b) -> a[0] - b[0]);
        int[] s = new int[n + 1];
        //前缀和
        for (int i = 0; i < n; i ++ ) {
            int len = tiles[i][1] - tiles[i][0] + 1;
            s[i + 1] = s[i] + len;
        }
        
        //枚举每个区间的左边界
        //查找毯子的右边界
        int ans = 0;
        for (int i = 0; i < n; i ++ ) {
            int tar = tiles[i][0] + carpetLen - 1;
            int pos = find(tiles, i, n - 1, tar);
            int cnt = s[pos] - s[i];
            //区分右边界落在区间内还是区间外
            if (tar <= tiles[pos][1]) {
            	//右边界再区间内的情况
                cnt += tar - tiles[pos][0] + 1;
            } else {
            	//右边界再区间外的情况
                cnt += s[pos + 1] - s[pos];
            }
            ans = Math.max(ans, cnt);
        }
        return ans;
    }
    public int find(int[][] tiles, int l, int r, int tar) {
        while (l <= r) {
            int mid = l + r >> 1;
            if (tar < tiles[mid][0]) r = mid - 1;
            else l = mid + 1;
        }
        return r;
    }
}
~~~

#### [668. 乘法表中第k小的数](https://leetcode.cn/problems/kth-smallest-number-in-multiplication-table/)

~~~
class Solution {
    public int findKthNumber(int m, int n, int k) {
        int l = 0,r = m * n;
        while(l < r){
            int mid = l + r >> 1;
            if(chk(mid, m, n, k)) r = mid;
            else l = mid + 1;
        }
        return r;
    }
    public boolean chk(int mid, int m, int n, int k) {
        int cnt = 0;
        //统计每行每列小于等于mid的数
        for(int i = 1; i <= m; i++ ) {
            cnt += Math.min(mid / i, n);
        }
        return cnt >= k;
    }
}
~~~

#### [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode.cn/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

~~~
class Solution {
    public int search(int[] nums, int target) {
        if (nums.length == 0) return 0;
        int l = 0, r = nums.length - 1;
        //二分找出左边界
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] >= target) r = mid;
            else l = mid + 1;
        }
        //答案不存在返回0
        if (nums[l] != target) return 0;
        //二分找出右边界
        int i = l, j = nums.length - 1;
        while (i < j) {
            int mid = i + j + 1 >> 1;
            if (nums[mid] <= target) i = mid;
            else j = mid - 1;
        }
        return j - l + 1;
    }
}
~~~

#### [6096. 咒语和药水的成功对数](https://leetcode.cn/problems/successful-pairs-of-spells-and-potions/)

~~~
class Solution {
    public int[] successfulPairs(int[] s, int[] p, long success) {
        Arrays.sort(p);
        int n = s.length, m = p.length;
        int[] ans = new int[n];
        for (int i = 0; i < n; i ++ ) {
            int t = s[i];
            long k = t * 1l * p[m - 1];
            if (k < success) ans[i] = 0;
            else {
                int l = 0, r = m - 1;
                while (l < r) {
                    int mid = l + r >> 1;
                    long q = t * 1l * p[mid];
                    if (q >= success) r = mid;
                    else l = mid + 1;
                }
                ans[i] = m - l;
            }
        }
        return ans;
    }
}
~~~

#### [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)

~~~
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int n = nums.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i++ ) s[i + 1] = s[i] + nums[i];
        int ans = Integer.MAX_VALUE;
        //s[j] - s[i] >= target -> s[j] >= s[i] + target
        for (int i = 0; i <= n; i ++ ) {
            int l = i, r = n;
            while (l < r) {
                int mid = l + r >> 1;
                if (s[mid] >= s[i] + target) r = mid;
                else l = mid + 1;
            }
            if (s[l] >= s[i] + target) ans = Math.min(ans, r - i);
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }
}
~~~

#### [1760. 袋子里最少数目的球](https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/)

~~~
class Solution {
    public int minimumSize(int[] nums, int m) {
        int l = 1, r = 0;
        for (int i : nums) r = Math.max(r, i);
        while (l < r) {
            int mid = l + r >> 1;
            int t = 0;
            for (int i : nums) {
                t += i / mid;
                if (i % mid == 0) t -- ;
            }
            if (t <= m) r = mid;
            else l = mid + 1;
        }
        return l;
    }
}
~~~

#### [658. 找到 K 个最接近的元素](https://leetcode.cn/problems/find-k-closest-elements/)

~~~
class Solution {
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        List<Integer> ans = new ArrayList<>();
        int l = 0, r = arr.length;
        //二分找到第一个大于等于接近x的数
        while (l < r) {
            int mid = l + r >> 1;
            if (arr[mid] >= x) r = mid;
            else l = mid + 1;
        }
        int pl = l - 1, pr = l;
        //双指针寻找结果
        while (k -- > 0) {
            if (pl == -1) pr ++ ;
            else if (pr == arr.length) pl -- ;
            else {
                if (x - arr[pl] <= arr[pr] - x) pl -- ;
                else pr ++ ;
            }
        } 
        for (int i = pl + 1; i < pr; i ++ ) ans.add(arr[i]);
        return ans;
    }
}
~~~

#### [1894. 找到需要补充粉笔的学生编号](https://leetcode.cn/problems/find-the-student-that-will-replace-the-chalk/)

~~~
class Solution {
    public int chalkReplacer(int[] chalk, int k) {
        int n = chalk.length;
        long[] s = new long[n + 1];
        for (int i = 0; i < n; i ++ ) s[i + 1] = s[i] + chalk[i];
        k %= s[n];
        int l = 0, r = n;
        while (l < r) {
            int mid = l + r >> 1;
            if (s[mid] > k) r = mid;
            else l = mid + 1;
        }
        return l - 1;
    }
}
~~~

#### [719. 找出第 K 小的数对距离](https://leetcode.cn/problems/find-k-th-smallest-pair-distance/)

~~~
class Solution {
    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int l = 0, r = nums[n - 1] - nums[0];
        while (l < r) {
            int mid = l + r >> 1;
            //给定一个候选值 mid，我们可以通过双指针算法，
            //在线性时间内求出小于等于 mid 的数对有多少个
            int t = 0;
            /*for (int i = 0, j = 0; i < n; i ++ ) {
                while (j < n && nums[j] - nums[i] <= mid) j ++ ;
                t += j - i;
            }*/
            for (int i = 1, j = 0; i < n; i ++ ) {
                while (j < i && nums[i] - nums[j] > mid) j ++ ;
                t += i - j;
            }
            if (t >= k) r = mid;
            else l = mid + 1;
        }
        return l;
    }
}
~~~

#### [287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)

~~~
class Solution {
    public int findDuplicate(int[] nums) {
        Arrays.sort(nums);
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = l + r + 1 >> 1;
            if (mid < nums[mid]) l = mid;
            else r = mid - 1;
        }
        return l + 1;
    }
}
~~~

#### [719. 找出第 K 小的数对距离](https://leetcode.cn/problems/find-k-th-smallest-pair-distance/)

~~~
class Solution {
    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int l = 0, r = nums[nums.length - 1] - nums[0];
        while (l < r) {
            int mid = l + r >> 1;
            int t = 0;
            for (int i = 1, j = 0; i < nums.length; i ++ ) {
                while (j < i && nums[i] - nums[j] > mid) j ++ ;
                t += i - j;
            }
            if (t >= k) r = mid;
            else l = mid + 1; 
        }
        return l;
    }
}
~~~

#### [1283. 使结果不超过阈值的最小除数](https://leetcode.cn/problems/find-the-smallest-divisor-given-a-threshold/)

~~~
class Solution {
    public int smallestDivisor(int[] nums, int threshold) {
        int l = 1, r = 1000000;
        while (l < r) {
            int mid = l + r >> 1;
            long t = 0;
            for (int i : nums) t += (i + mid - 1) / mid;
            if (t <= threshold) r = mid;
            else l = mid + 1;
        }
        return l;
    }
}
~~~

#### [1870. 准时到达的列车最小时速](https://leetcode.cn/problems/minimum-speed-to-arrive-on-time/)

~~~
class Solution {
    public int minSpeedOnTime(int[] dist, double h) {
        int n = dist.length;
        int l = 1, r = 10000010;
        while (l < r) {
            int mid = l + r >> 1;
            double t = 0;
            for (int i = 0; i < n - 1; i ++ ) t += (dist[i] + mid - 1) / mid; 
            t += dist[n - 1] * 1.0 / mid;
            if (t <= h) r = mid;
            else l = mid + 1;
        }
        return l == 10000010 ? -1 : l;
    }
}
~~~

#### [1898. 可移除字符的最大数目](https://leetcode.cn/problems/maximum-number-of-removable-characters/)

~~~
class Solution {
    public int maximumRemovals(String s, String p, int[] removable) {
        int n = removable.length;
        int l = 0, r = n;
        while (l < r) {
            int mid = l + r >> 1;
            if (!check(s, p, mid, removable)) r = mid;
            else l = mid + 1;
        }
        return l;
    }
    public boolean check(String s1, String s2, int mid, int[] remove) {
        char[] ss = s1.toCharArray();
        for (int i = 0; i <= mid; i ++ ) ss[remove[i]] = ',';
        int n = s1.length(), m = s2.length();
        int j = 0;
        for (int i = 0; i < s1.length(); i ++ ) {
            if (j < m) {
                if (ss[i] == s2.charAt(j)) j ++ ;
            }
            else return true;
        }
        if (j == m) return true;
        return false;
    }
}
~~~

#### [1482. 制作 m 束花所需的最少天数](https://leetcode.cn/problems/minimum-number-of-days-to-make-m-bouquets/)

~~~
class Solution {
    public int minDays(int[] bloomDay, int m, int k) {
        if (bloomDay.length < m * k) return -1;
        int l = 1, r = 1000000010, n = bloomDay.length;
        while (l < r) {
            int mid = l + r >> 1;
            int t = 0, count = 0;
            for (int i = 0; i < n; i ++ ) {
                if (bloomDay[i] <= mid) count ++ ;
                else count = 0;
                if (count == k) {
                    count = 0;
                    t ++ ;
                }
            }
            if (t >= m) r = mid;
            else l = mid + 1;
        }
        return l;
    }
}
~~~

#### [275. H 指数 II](https://leetcode.cn/problems/h-index-ii/)

~~~
class Solution {
    public int hIndex(int[] ct) {
        int  n = ct.length;
        int l = 0, r = n;
        while (l < r) {
            int mid = l + r >> 1;
            // 满足右侧的 len - mid篇论文分别被引用了至少 len - mid 次.
            if (n - mid <= ct[mid]) r = mid;
            else l = mid + 1;
        }
        return n - l;
    }
}
~~~

#### [1818. 绝对差值和](https://leetcode.cn/problems/minimum-absolute-sum-difference/)

~~~
class Solution {
    public int minAbsoluteSumDiff(int[] nums1, int[] nums2) {
        long sum = 0;
        int max = 0, mod = (int)1e9 + 7;
        int[] nums = nums1.clone();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i ++ ) {
            int a = nums1[i], b = nums2[i];
            int x = Math.abs(a - b);
            sum += x;
            int l = 0, r = nums1.length - 1;
            while (l < r) {
                int mid = l + r >> 1;
                if (nums[mid] >= b) r = mid;
                else l = mid + 1;
            }
            int t = 0;
            t = Math.abs(nums[l] - b);
            if (l > 0) t = Math.min(t, Math.abs(b - nums[l - 1]));
            max = Math.max(max, Math.abs(x - t));
        }
        return (int) ((sum - max) % mod);
    }
}
~~~

#### [540. 有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

~~~
class Solution {
    public int singleNonDuplicate(int[] nums) {
        int l = 0, r = (nums.length - 1) / 2;
        while (l < r) {
            int mid = l + r >> 1;
            //两两分组后找第一组两个不同的，左边那个数就是要找的
            if (nums[mid * 2] != nums[mid * 2 + 1]) r = mid;
            else l = mid + 1;
        }
        return nums[l * 2];
    }
}
~~~

#### [1712. 将数组分成三个子数组的方案数](https://leetcode.cn/problems/ways-to-split-array-into-three-subarrays/)

~~~
class Solution {
    public int waysToSplit(int[] nums) {
        int n = nums.length, mod = (int) 1e9 + 7;
        long[] s = new long[n + 1];
        long ans = 0;
        for (int i = 0; i < n; i ++ ) s[i + 1] = s[i] + nums[i];
        //枚举mid的左边界
        for (int i = 0; i < n; i ++ ) {
            int l = i + 1, r = n - 1;
            //二分出mid右边界的范围
            int l1 = i + 1, r1 = n - 1;
            while (l1 < r1) {
                int mid = l1 + r1 >> 1;
                long a = s[i + 1];
                long b = s[mid + 1] - s[i + 1];
                if (a <= b) r1 = mid;
                else l1 = mid + 1;
            }
            while (l < r) {
                int mid = l + r >> 1;
                long b = s[mid + 1] - s[i + 1];
                long c = s[n] - s[mid + 1];
                if (b > c) r = mid;
                else l = mid + 1;
            }
            if (l >= l1) ans += l - l1;
            ans %= mod;
        }
        return (int) ans % mod;
    }
}
~~~

#### [1838. 最高频元素的频数](https://leetcode.cn/problems/frequency-of-the-most-frequent-element/)

~~~
class Solution {
    int[] nums, s;
    int n, k;
    public int maxFrequency(int[] _nums, int _k) {
        nums = _nums;
        k = _k;
        n = nums.length;
        Arrays.sort(nums);
        s = new int[n + 1];
        for (int i = 0; i < n; i ++ ) s[i + 1] = s[i] + nums[i];
        int l = 1, r = n;
        while (l < r) {
            int mid = l + r + 1 >> 1;
            if (check(mid)) l = mid;
            else r = mid - 1;
        }
        return l;
    }
    public boolean check(int len) {
        for (int l = 0; l + len - 1 < n; l ++ ) {
            int r = l + len - 1;
            int cur = s[r + 1] - s[l];
            int t = nums[r] * len;
            if (t - cur <= k) return true;
        }
        return false;
    }
}
~~~

#### [436. 寻找右区间](https://leetcode.cn/problems/find-right-interval/)

~~~
class Solution {
    public int[] findRightInterval(int[][] intervals) {
        int n = intervals.length;
        if (n == 1) return new int[]{-1};
        int[] res = new int[n];
        int[][] map = new int[n][2];
        for (int i = 0; i < n; i ++ ) {
            map[i][0] = intervals[i][0];
            map[i][1] = i;
        }
        Arrays.sort(map, (a, b) -> a[0] - b[0]);
        for (int i = 0; i < n; i ++ ) {
            int k = intervals[i][1];
            int l = 0, r = n - 1;
            while (l < r) {
                int mid = l + r >> 1;
                if (map[mid][0] >= k) r = mid;
                else l = mid + 1;
            }
            if (map[l][0] >= k) res[i] = map[l][1];
            else res[i] = -1;
        }
        return res;
    }
}
~~~

#### [826. 安排工作以达到最大收益](https://leetcode.cn/problems/most-profit-assigning-work/)

~~~
class Solution {
    public int maxProfitAssignment(int[] d, int[] p, int[] w) {
        int ans = 0, max = 0;
        Map<Integer, Integer> map = new HashMap<>();
        //将利润和难度进行关联
        for (int i = 0; i < d.length; i ++ ) {
            //d[i]会有重复，判断后存最大值
            if (!map.containsKey(d[i])) map.put(d[i], p[i]);
            else map.put(d[i], Math.max(map.get(d[i]), p[i]));
        }
        Arrays.sort(d);
        //预处理，防止利润难度不成正比
        for (int i = 0; i < d.length; i ++ ) {
            max = Math.max(max, map.get(d[i]));
            map.put(d[i], max);
        } 
        for (int i : w) {
            int l = 0, r = d.length - 1;
            while (l < r) {
                int mid = l + r >> 1;
                if (d[mid] >= i) r = mid;
                else l = mid + 1;
            }
            if (l == 0 && d[l] > i) continue;
            if (d[l] > i) ans += map.get(d[l - 1]);
            else ans += map.get(d[l]);
        }
        return ans;
    }
}
~~~

#### [*81. 搜索旋转排序数组 II](https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/)

~~~
class Solution {
    public boolean search(int[] nums, int target) {
        int n = nums.length;
        int l = 0, r = n - 1;
        //恢复二段性
        while (l < r && nums[l] == nums[r]) r -- ;
        if (l == r) return nums[r] == target;
        //二分找旋转点
        while (l < r) {
            int mid = l + r + 1 >> 1;
            if (nums[mid] >= nums[0]) l = mid;
            else r = mid - 1; 
        }
        if (target >= nums[0]) l = 0;
        else if (r + 1 < n) {
            l = r + 1;
            r = n - 1;
        }
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] >= target) r = mid;
            else l = mid + 1;
        }
        return nums[l] == target;
    }
}
~~~

#### [154. 寻找旋转排序数组中的最小值 II](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/)

~~~
class Solution {
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r && nums[r] == nums[0]) r -- ;
        if (nums[r] > nums[0]) return nums[0];
        while (l < r) { 
            int mid = l + r >> 1;
            if (nums[mid] < nums[0]) r = mid;
            else l = mid + 1;
        }
        return nums[l];
    }
}
~~~

#### [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

~~~
class Solution {
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        if (nums[r] > nums[0]) return nums[0];
        while (l < r) { 
            int mid = l + r >> 1;
            if (nums[mid] < nums[0]) r = mid;
            else l = mid + 1;
        }
        return nums[l];
    }
}
~~~

#### [710. 黑名单中的随机数](https://leetcode.cn/problems/random-pick-with-blacklist/)

~~~
class Solution {
    List<int[]> list = new ArrayList<>();
    int[] s = new int[100005];
    Random r = new Random();
    int sz;
    public Solution(int n, int[] bs) {
        Arrays.sort(bs);
        int m = bs.length;
        //我们不对「点」做离散化，而利用 bs 数据范围为 1e5，来对「线段」做离散化。
        if (m == 0) list.add(new int[]{0, n - 1});
        else {
            if (bs[0] != 0) list.add(new int[]{0, bs[0] - 1});
            for (int i = 1; i < m; i ++ ) {
                if (bs[i - 1] == bs[i] - 1) continue;
                list.add(new int[]{bs[i - 1] + 1, bs[i] - 1});
            }
            if (bs[m - 1] != n - 1) list.add(new int[]{bs[m - 1] + 1, n - 1});
        }
        sz = list.size();
        for (int i = 0; i < sz; i ++ ) {
            int[] info = list.get(i);
            s[i + 1] = s[i] + info[1] - info[0] + 1;
        }
    }
    
    public int pick() {
        int val = r.nextInt(s[sz]) + 1;
        int l = 1, r = sz;
        //二分找到val所在线段
        while (l < r) {
            int mid = l + r >> 1;
            if (s[mid] >= val) r = mid;
            else l = mid + 1;
        } 
        int[] info = list.get(l);
        //然后再利用该线段的左右端点的值，取出对应的点。
        int a = info[0], b = info[1], end = s[r];
        System.out.println(b);
        return b - (end - val);
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(n, blacklist);
 * int param_1 = obj.pick();
 */
~~~

#### [1498. 满足条件的子序列数目](https://leetcode.cn/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/)

~~~
class Solution {
    public int numSubseq(int[] nums, int target) {
        int mod = (int)1e9 + 7;
        int n = nums.length;
        int[] f = new int[n];
        for (int i = 0; i < n; i ++ ) {
            if (i == 0) f[i] = 1;
            else f[i] = (f[i - 1] << 1) % mod;
        }
        int ans = 0;
        Arrays.sort(nums);
        for (int i = 0; i < n; i ++ ) {
            int l = i, r = n - 1;
            while (l < r) {
                int mid = l + r + 1 >> 1;
                if (nums[mid] + nums[i] <= target) l = mid;
                else r = mid - 1;
            }
            if (nums[l] + nums[i] <= target) ans = (ans + f[r - i]) % mod;
        }
        return ans;
    }
}
~~~

#### [793. 阶乘函数后 K 个零](https://leetcode.cn/problems/preimage-size-of-factorial-zeroes-function/)

~~~
class Solution {
    //阶乘函数后k个0的个数取决于有几个5
    public int preimageSizeFZF(int k) {
        if (k <= 1) return 5;
        else return f(k) - f(k - 1);
    }
    public int f(int k) {
        long l = 0, r = (long)1e10;
        while (l < r) {
            long mid = l + r + 1 >> 1;
            if (getCur(mid) <= k) l = mid;
            else r = mid - 1;
        }
        return (int)l;
    }
    public int getCur(long x) {
        long res = 0;
        while (x != 0) {
            res += x / 5;
            x /= 5;
        }
        return (int)res;
    }
}
~~~

#### [878. 第 N 个神奇数字](https://leetcode.cn/problems/nth-magical-number/)

~~~
class Solution {
    int mod = (int) 1e9 + 7;

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }
    
    public int nthMagicalNumber(int n, int a, int b) {
        long l = 0l, r = (long) Math.max(a, b) * n;
        var lcm = a * b / gcd(a, b);
        while (l < r) {
            var mid = l + r >> 1;
            var k = mid /a + mid / b - mid / lcm;
            if (k >= n) r = mid;
            else l = mid + 1;
        }
        return (int) (l % mod);   
    }
}
~~~



## 前缀和

#### [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)

前缀和优化

~~~
class Solution {
    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        int[] s = new int[n + 1];
        int ans = 0;
        for (int i = 0; i < n; i ++ ) s[i + 1] = s[i] + nums[i];
        Map<Integer, Integer> map = new HashMap<>();
        //初始化时前缀和为0的数组出现次数设置为1，用来计算当前前缀和恰好为k的情况。
        map.put(0, 1);
        for (int i = 1; i <= n; i ++ ) {
            //s[j] - s[i] = k -- > s[i] = s[j] - k;
            /*
            *我们可以枚举区间的终点，用哈希表来记录终点前的前缀和出现次数，
            *以当前点为终点的和为k的子数组的出现次数即为当前前缀和减去k后
            *的前缀和的次数。
            */
            ans += map.getOrDefault(s[i] - k, 0);
            map.put(s[i], map.getOrDefault(s[i], 0) + 1);
        }
        return ans;
    }
}
~~~

#### [930. 和相同的二元子数组](https://leetcode.cn/problems/binary-subarrays-with-sum/)

~~~
class Solution {
    public int numSubarraysWithSum(int[] nums, int goal) {
        int n = nums.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i ++ ) s[i + 1] = s[i] + nums[i];
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int res = 0;
        for (int i = 1; i <= n; i ++ ) {
            res += map.getOrDefault(s[i] - goal, 0);
            map.put(s[i], map.getOrDefault(s[i], 0) + 1);
        }
        return res;
    }
}
~~~

#### [525. 连续数组](https://leetcode.cn/problems/contiguous-array/)

~~~
class Solution {
    public int findMaxLength(int[] nums) {
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 0);
        int res = 0;
        for (int i = 1, one = 0, zero = 0; i <= n; i ++ ) {
            int x = nums[i - 1];
            if (x == 0) zero ++ ;
            else one ++ ;
            int s = one - zero;
            if (map.containsKey(s)) res = Math.max(res, i - map.get(s));
            else map.put(s, i);
        }
        return res;
    }
}
~~~

#### [523. 连续的子数组和](https://leetcode.cn/problems/continuous-subarray-sum/)

~~~
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        int n = nums.length;
        if (k == 0) {
            for (int i = 1; i < n; i ++ ) {
                if (nums[i - 1] == 0 && nums[i] == 0) return true;
            }
            return false;
        }
        int[] s = new int[n + 1];
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < n; i ++ ) s[i + 1] = s[i] + nums[i]; 
        for (int i = 2; i <= n; i ++ ) {
        //两个数相减能被k整除等价于两个数除以k的余数相同
            set.add(s[i - 2] % k);
            if (set.contains(s[i] % k)) return true;
        }
        return false;
    }
}
~~~

#### [1248. 统计「优美子数组」](https://leetcode.cn/problems/count-number-of-nice-subarrays/)

~~~
class Solution {
    public int numberOfSubarrays(int[] nums, int k) {
        int n = nums.length;
        int ans = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int i = 0, cnt = 0; i < n; i ++ ) {
            int x = nums[i];
            if (x % 2 == 1) cnt ++ ;
            if (map.containsKey(cnt - k)) ans += map.get(cnt - k);
            map.put(cnt, map.getOrDefault(cnt, 0) + 1);
        }
        return ans;
    }
}
~~~

#### [974. 和可被 K 整除的子数组](https://leetcode.cn/problems/subarray-sums-divisible-by-k/)

~~~
class Solution {
    public int subarraysDivByK(int[] nums, int k) {
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int ans = 0;
        int sum = 0;
        for (int i = 1; i <= n; i ++ ) {
            sum += nums[i - 1];
            int r = (sum % k + k) % k;
            if (map.containsKey(r)) ans += map.get(r);
            map.put(r, map.getOrDefault(r, 0) + 1);
        }
        return ans;
    }
}
~~~

#### [221. 最大正方形](https://leetcode.cn/problems/maximal-square/)

**解法：二维前缀和 + 二分**

~~~
class Solution {
    public boolean check(int[][] s, int d) {
        int n = s.length - 1, m = s[0].length - 1;
        for (int i = 0; i <= n - d; i ++ ) {
            for (int j = 0; j <= m - d; j ++ ) {
                int x = i + d - 1, y = j + d - 1;
                if (s[x + 1][y + 1] - s[x + 1][j] - s[i][y + 1] + s[i][j] == d * d)
                    return true;
            }
        }
        return false;
    }

    public int maximalSquare(char[][] matrix) {
        int n = matrix.length, m = matrix[0].length;
        int[][] s = new int[n + 1][m + 1];
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ )
                s[i + 1][j + 1] = s[i + 1][j] + s[i][j + 1] - s[i][j] + matrix[i][j] - '0';
        int l = 0, r = Math.min(n, m);
        while (l < r) {
            int mid = l + r + 1 >> 1;
            if (check(s, mid))
                l = mid;
            else
                r = mid - 1;
        }
        return r * r;
    }
}
~~~

#### [6098. 统计得分小于 K 的子数组数目](https://leetcode.cn/problems/count-subarrays-with-score-less-than-k/)

~~~
class Solution {
    public long countSubarrays(int[] nums, long k) {
        int n = nums.length;
        long[] s = new long[n + 1];
        for (int i = 0; i < n; i ++ ) s[i + 1] = s[i] + nums[i] * 1l;
        long ans = 0;
        for (int i = 0, j = 0; i < n; i ++ ) {
            while (j <= i && (s[i + 1] - s[j]) * (i - j + 1) >= k) j ++ ;
            ans += i - j + 1;
        }
        return ans;
    }
}
~~~

#### [5229. 拼接数组的最大分数](https://leetcode.cn/problems/maximum-score-of-spliced-array/)

~~~
class Solution {
    public int work (int[] a, int[] b) {
        int sum = 0;
        for (int x : a) sum += x;
        int dt = 0, f = 0;
        //求最大连续子数组和
        for (int i = 0; i < a.length; i ++ ) {
            f = Math.max(f, 0) + b[i] - a[i];
            dt = Math.max(dt, f);
        }
        return sum + dt;
    }
    public int maximumsSplicedArray(int[] a, int[] b) {
        return Math.max(work(a, b), work(b, a));
    }
}
~~~

#### [AcWing4405. 统计子矩阵](https://www.acwing.com/problem/content/description/4408/)

**解法：二维前缀和**

~~~
import java.util.*;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), m = sc.nextInt();
        long k = sc.nextInt();
        int[][] g = new int[n + 1][m + 1];
        for (int i = 0; i < n; i ++ ) 
            for (int j = 0; j < m; j ++ ) 
                g[i + 1][j + 1] = g[i][j + 1] + g[i + 1][j] - g[i][j] + sc.nextInt();
        
        long res = 0;        
        for (int i = 0; i <= n; i ++ ) 
         for (int j = i + 1; j <= n; j ++ ) 
            for (int l = 0, r = 1; r <= m; r ++ ) {
                while (l < r && g[j][r] - g[j][l] - g[i][r] + g[i][l] > k) l ++ ;
                res += r - l;
            }
            
        System.out.print(res);
    }
}
~~~

#### [1314. 矩阵区域和](https://leetcode.cn/problems/matrix-block-sum/)

**解法：二维前缀和**

~~~~
class Solution {
    public int[][] matrixBlockSum(int[][] mat, int k) {
        int n = mat.length, m = mat[0].length;
        var s = new int[n + 1][m + 1];
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m ; j ++ )
                s[i + 1][j + 1] = s[i + 1][j] + s[i][j + 1] - s[i][j] + mat[i][j];
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ ) {
                int x1 = Math.max(0, i - k);
                int y1 = Math.max(0, j - k);
                int x2 = Math.min(n - 1, i + k);
                int y2 = Math.min(m - 1, j + k);
                mat[i][j] = s[x2 + 1][y2 + 1] - s[x2 + 1][y1] - s[x1][y2 + 1] + s[x1][y1];
            }
        return mat;
    }
}
~~~~



## Trie字典树

#### [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/)

~~~
//字典树 + dfs
class Trie {
    private static class TrieNode {
        TrieNode[] child;
        boolean    isEnd;
        public TrieNode() {
            this.child = new TrieNode[26];
            this.isEnd = false;
        }
    }

    private TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    /**
    * Inserts a word into the trie.
    */
    public void insert(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (p.child[c - 'a'] == null) {
                p.child[c - 'a'] = new TrieNode();
            }
            p = p.child[c - 'a'];
        }
        p.isEnd = true;
    }

    /**
    * Returns if the word is in the trie.
    */
    public boolean search(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (p.child[c - 'a'] == null) {
                return false;
            }
            p = p.child[c - 'a'];
        }
        return p.isEnd;
    }

    /**
    * Returns if there is any word in the trie that starts with the given prefix.
    */
    public boolean startsWith(String prefix) {
        TrieNode p = root;
        for (int i = 0; i < prefix.length(); i++) {
            char c = prefix.charAt(i);
            if (p.child[c - 'a'] == null) {
                return false;
            }
            p = p.child[c - 'a'];
        }
        return true;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
~~~

#### [676. 实现一个魔法字典](https://leetcode.cn/problems/implement-magic-dictionary/)

~~~
class MagicDictionary {
    int N = 100 * 100, M = 26, idx = 0;
    int[][] son = new int[N][M];
    boolean[] isEnd = new boolean[N * M];
    
    public void insert(String word) {
        char[] str = word.toCharArray();
        int p = 0;
        for (int i = 0; i < str.length; i ++ ) {
            int u = str[i] - 'a';
            if (son[p][u] == 0) son[p][u] = ++ idx;
            p = son[p][u];
        }
        isEnd[p] = true;
    }

    public boolean dfs (String s, int p, int u, int c) {
        if (isEnd[p] && u == s.length() && c == 1) return true;
        if (c > 1 || u == s.length()) return false;

        int x = s.charAt(u) - 'a';
        for (int i = 0; i < 26; i ++ ) {
            if (son[p][i] == 0) continue;
            if (dfs(s, son[p][i], u + 1, (x == i) ? c : c + 1)) 
                return true;
        }
        return false;
    }

    public MagicDictionary() {
        
    }
    
    public void buildDict(String[] dictionary) {
        for (String dic : dictionary) {
            insert(dic);
        }
    }
    
    public boolean search(String searchWord) {
        return dfs(searchWord, 0, 0, 0);
    }
}

/**
 * Your MagicDictionary object will be instantiated and called as such:
 * MagicDictionary obj = new MagicDictionary();
 * obj.buildDict(dictionary);
 * boolean param_2 = obj.search(searchWord);
 */
~~~

#### [745. 前缀和后缀搜索](https://leetcode.cn/problems/prefix-and-suffix-search/)

~~~
class WordFilter {
    public class TrieNode {
        int id;
        TrieNode[] son;
        TrieNode() {
            son = new TrieNode[27];
        }
    }
    TrieNode root = new TrieNode(); 
    public void insert(String s, int id) {
        TrieNode node = root;
        for (char c : s.toCharArray()) {
            int u = c == '#' ? 26 : c - 'a';
            if (node.son[u] == null) node.son[u] = new TrieNode();
            node = node.son[u];
            node.id = id;
        }
    }

    public int search(String s) {
        TrieNode node = root;
        for (char c : s.toCharArray()) {
            int u = c == '#' ? 26 : c - 'a';
            if (node.son[u] == null) return -1;
            node = node.son[u];
        }
        return node.id;
    }

    public WordFilter(String[] words) {
        TrieNode node = root;
        for (int i = 0; i < words.length; i ++ ) {
            String s = "#" + words[i];
            insert(s, i);
            for (int j = words[i].length() - 1; j >= 0; j -- ) {
                s = words[i].charAt(j) + s;
                insert(s, i);
            }
        }
    }
    
    public int f(String pref, String suff) {
        return search(suff + "#" + pref);
    }
}

/**
 * Your WordFilter object will be instantiated and called as such:
 * WordFilter obj = new WordFilter(words);
 * int param_1 = obj.f(pref,suff);
 */
~~~

#### [211. 添加与搜索单词 - 数据结构设计](https://leetcode.cn/problems/design-add-and-search-words-data-structure/)

~~~
class WordDictionary {

    TrieNode root = new TrieNode();

    public WordDictionary() {

    }
    
    public void addWord(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int u = c - 'a';
            if (node.son[u] == null) node.son[u] = new TrieNode();
            node = node.son[u];
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        return dfs(root, word, 0);
    }

    public boolean dfs(TrieNode p, String s, int i) {
        if (i == s.length()) return p.isEnd;
        if (s.charAt(i) == '.') {
            for (int k = 0; k < 26; k ++ ) {
                if (p.son[k] != null && dfs(p.son[k], s, i + 1)) 
                    return true; 
            }
            return false;
        } else {
            int u = s.charAt(i) - 'a';
            if (p.son[u] == null) return false;
            return dfs(p.son[u], s, i + 1);
        }
    }

    public class TrieNode {
        TrieNode[] son;
        boolean isEnd;
        public TrieNode() {
            son = new TrieNode[26];
        }
    }
}

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary obj = new WordDictionary();
 * obj.addWord(word);
 * boolean param_2 = obj.search(word);
 */
~~~

#### [AcWing143.最大异或对](https://www.acwing.com/problem/content/description/145/)

~~~
import java.io.*;
import java.util.*;
public class Main {
    static int N = 100010, M = 3000000;
    static int son[][] = new int[M][2];
    static int idx = 0;
    public static void insert(int x) {
        int p = 0;
        for (int i = 30; i >= 0; i -- ) {
            int s = (x >> i) & 1;
            if (son[p][s] == 0) son[p][s] = ++ idx;
            p = son[p][s];
        }
    }
    public static int search(int x) {
        int p = 0, res = 0;
        for (int i = 30; i >= 0; i -- ) {
            int s = (x >> i) & 1;
            if (son[p][1 - s] != 0) {
                res += 1 << i;
                p = son[p][1 - s];
            } else {
                p = son[p][s];
            }
        }
        return res;
    }
    public static void main(String[] args) throws Exception {
        in.nextToken();
        int n = (int) in.nval;
        int[] a = new int[n];
        for (int i = 0; i < n; i ++ ) {
            in.nextToken();
            a[i] = (int) in.nval;
        }
        for (int x : a) insert(x);
        int res = 0;
        for (int x : a) 
            res = Math.max(res, search(x));
        pw.print(res);
        pw.close();
    }
    static  StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
    static PrintWriter pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
}
~~~

#### [421. 数组中两个数的最大异或值](https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/)

~~~
class Solution {
    TrieNode root = new TrieNode();
    public void insert(int x) {
        var p = root;
        for (int i = 30; i >= 0; i -- ) {
            var k = (x >> i) & 1;
            if (p.son[k] == null) p.son[k] = new TrieNode();
            p = p.son[k];
        }
    }

    public int search(int x) {
        var p = root;
        var res = 0;
        for (int i = 30; i >= 0; i -- ) {
            var k = (x >> i) & 1;
            if (p.son[1 - k] != null) {
                res += 1 << i;
                p = p.son[1 - k];
            } else 
                p = p.son[k];
        }
        return res;
    }

    public int findMaximumXOR(int[] nums) {
        var res = 0;
        for (int x : nums) {
            insert(x);
            res = Math.max(res, search(x));
        }    
        return res;
    }

    public class TrieNode {
        TrieNode[] son;
        boolean isEnd;
        public TrieNode() {
            son = new TrieNode[2];
        }
    }
}
~~~

~~~
class Solution:
    def __init__(self):
        self.idx = 0
        self.son = list()
        self.son.append([0, 0])

    def findMaximumXOR(self, nums: List[int]) -> int:
        def insert(x):
            p = 0
            for i in range(30, -1, -1):
                k = (x >> i) & 1
                if not self.son[p][k]: 
                    self.idx += 1
                    self.son.append([0, 0])
                    self.son[p][k] = self.idx
                p = self.son[p][k]

        def search(x):
            p, res = 0, 0
            for i in range(30, -1, -1):
                k = (x >> i) & 1
                if self.son[p][1 - k]: 
                    res += 1 << i
                    p = self.son[p][1 - k]
                else:
                    p = self.son[p][k]
            return res
        res = 0
        for x in nums:
            insert(x)
            res = max(res, search(x))
        return res
~~~



## 哈希

#### [349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/)

11.1

#### [771. 宝石与石头](https://leetcode-cn.com/problems/jewels-and-stones/)

#### [1512. 好数对的数目](https://leetcode-cn.com/problems/number-of-good-pairs/)

#### [1684. 统计一致字符串的数目](https://leetcode-cn.com/problems/count-the-number-of-consistent-strings/)

#### [594. 最长和谐子序列](https://leetcode-cn.com/problems/longest-harmonious-subsequence/)

哈希计数（Java中的Map提供了getOrDefault()方法，对不存在的键值提供默认值的方法。）

#### [1995. 统计特殊四元组](https://leetcode-cn.com/problems/count-special-quadruplets/)

#### [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

~~~
//哈希排序
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) map.put(num, map.getOrDefault(num, 0) + 1);
        Set<Map.Entry<Integer, Integer>> entries = map.entrySet();
        // 根据map的value值正序排，相当于一个小顶堆
        PriorityQueue<Map.Entry<Integer, Integer>> queue = new PriorityQueue<>((o1, o2) -> o1.getValue() - o2.getValue());
        for (Map.Entry<Integer, Integer> entry : entries) {
            queue.offer(entry);
            if (queue.size() > k) {
                queue.poll();
            }
        }
        int[] res = new int[k];
        for (int i = 0; i < k; i ++ ) res[i] = queue.poll().getKey();
        return res;
    }
}
~~~

#### [205. 同构字符串](https://leetcode.cn/problems/isomorphic-strings/)

~~~
public boolean isIsomorphic(String s, String t) {
        int[] st = new int[128];
        int[] ts = new int[128];
        for (int i = 0; i < s.length(); i ++ ) {
            int a = s.charAt(i), b = t.charAt(i);
            if (st[a] != 0 && st[a] != b) return false;
            st[a] = b;
            if (ts[b] != 0 && ts[b] != a) return false;
            ts[b] = a;            
        }
        return true;
    }
~~~

#### [890. 查找和替换模式](https://leetcode.cn/problems/find-and-replace-pattern/)

~~~
class Solution {
    public List<String> findAndReplacePattern(String[] words, String pattern) {
        List<String> ans = new ArrayList<>();
        for (String s : words) {
            if (isIsomorphic(s, pattern)) ans.add(s);
        }
        return ans;
    }
    public boolean isIsomorphic(String s, String t) {
        int[] st = new int[128];
        int[] ts = new int[128];
        for (int i = 0; i < s.length(); i ++ ) {
            int a = s.charAt(i), b = t.charAt(i);
            if (st[a] != 0 && st[a] != b) return false;
            st[a] = b;
            if (ts[b] != 0 && ts[b] != a) return false;
            ts[b] = a;            
        }
        return true;
    }
}
~~~

#### [30. 串联所有单词的子串](https://leetcode.cn/problems/substring-with-concatenation-of-all-words/)

~~~
//朴素哈希O(n×m×w)
class Solution {
    public List<Integer> findSubstring(String s, String[] words) {
        int n = words.length, m = words[0].length();
        List<Integer> res = new ArrayList<>();
        HashMap<String, Integer> map = new HashMap<>();
        for (String word : words) map.put(word, map.getOrDefault(word, 0) + 1);
        out:for (int i = 0; i + m * n <= s.length(); i ++ ) {
            Map<String, Integer> cur = new HashMap<>();
            String sub = s.substring(i, i + n * m);
            for (int j = 0; j < sub.length(); j += m) {
                String item = sub.substring(j, j + m);
                if (!map.containsKey(item)) continue out;
                cur.put(item, cur.getOrDefault(item, 0) + 1);
            }
            if (cur.equals(map)) res.add(i);
        }
        return res;
    }
}
~~~

~~~
//哈希 + 分组 + 滑窗 O(nw)
class Solution {
    public List<Integer> findSubstring(String s, String[] words) {
        int n = s.length(), m = words.length, w = words[0].length();
        List<Integer> res = new ArrayList<>();
        HashMap<String, Integer> map = new HashMap<>();
        for (String word : words) map.put(word, map.getOrDefault(word, 0) + 1);
        //根据当前下标与单词长度的取余结果分为w组
        for (int i = 0; i < w; i ++ ) {
            Map<String, Integer> wd = new HashMap<>();
            //记录有效单词数
            int cnt = 0;
            //每次将下一个单词加入，上一个单词移除
            for (int j = i; j + w <= n; j += w) {
            	//窗口已达到最大，删除首个单词
                if (j >= i + m * w) {
                    String word = s.substring(j - m * w, j - m * w + w);
                    wd.put(word, wd.get(word) - 1);
                    if (map.containsKey(word) && wd.get(word) < map.get(word)) cnt -- ;
                }
                //添加新的单词
                String word = s.substring(j, j + w);
                wd.put(word, wd.getOrDefault(word, 0) + 1);
                if (map.containsKey(word) && wd.get(word) <= map.get(word)) cnt ++ ;
                if (cnt == m) res.add(j - (m - 1) * w);
            }
        }
        return res;
    }
}
~~~

#### [648. 单词替换](https://leetcode.cn/problems/replace-words/)

~~~
//哈希暴力
class Solution {
    public String replaceWords(List<String> dictionary, String sentence) {
        String[] ss = sentence.split(" ");
        int n = ss.length;
        StringBuilder ans = new StringBuilder();
        Set<String> set = new HashSet<>();
        for (String s : dictionary) set.add(s);
        for (int i = 0; i < n; i ++ ) {
            String s = ss[i];
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < s.length(); j ++ ) {
                sb.append(s.charAt(j));
                if (set.contains(sb.toString())) break;
            }
            ans.append(sb);
            if (i != n - 1)  ans.append(" ");
        }
        return ans.toString();
    }
}
~~~

~~~
//字符串哈希
class Solution {
    public String replaceWords(List<String> dictionary, String sentence) {
        int p = 131;
        Set<Long> set = new HashSet<>();
        for (String s : dictionary) {
            long h = 0;
            for (char c : s.toCharArray()) h = h * p + c;
            set.add(h);
        }
        StringBuilder ans = new StringBuilder();
        String[] ss = sentence.split(" ");
        for (int i = 0; i < ss.length; i ++ ) {
            StringBuilder sb = new StringBuilder();
            long h = 0;
            for (char c : ss[i].toCharArray()) {
                sb.append(c);
                h = h * p + c;
                if (set.contains(h)) break;
            }
            ans.append(sb).append(" ");
        }
        return ans.deleteCharAt(ans.length() - 1).toString();
    }
}
~~~

#### [676. 实现一个魔法字典](https://leetcode.cn/problems/implement-magic-dictionary/)

~~~
class MagicDictionary {
	//根据字符串长度进行哈希然后爆搜
    Map<Integer, List<String>> map;
    public MagicDictionary() {
        map = new HashMap<>();
    }
    
    public void buildDict(String[] dictionary) {
        for (String dic : dictionary) {
            int t = dic.length();
            List<String> list = map.getOrDefault(t, new ArrayList<>());
            list.add(dic);
            map.put(t, list);
        }
    }
    
    public boolean search(String searchWord) {
        int n = searchWord.length();
        if (!map.containsKey(n)) return false;
        List<String> list = map.get(n);
        out:for (int i = 0; i < list.size(); i ++ ) {
            int t = 0;
            String s = list.get(i);
            for (int j = 0; j < n; j ++ ) {
                if (searchWord.charAt(j) == s.charAt(j)) continue;
                else t ++ ;
                if (t > 1) continue out;
            }
            if (t == 1) return true;
        }
        return false;
    }
}

/**
 * Your MagicDictionary object will be instantiated and called as such:
 * MagicDictionary obj = new MagicDictionary();
 * obj.buildDict(dictionary);
 * boolean param_2 = obj.search(searchWord);
 */
~~~

#### [745. 前缀和后缀搜索](https://leetcode.cn/problems/prefix-and-suffix-search/)

~~~
//枚举每个前缀和后缀放入map
class WordFilter {
    Map<String, Integer> map = new HashMap<>();
    public WordFilter(String[] words) {
        for (int i = 0; i < words.length; i ++ ) {
            for (int j = 1; j <= words[i].length(); j ++ ) {
                for (int k = 1; k <= words[i].length(); k ++ ) {
                    String pref = words[i].substring(0, j);
                    String suff = words[i].substring(words[i].length()-k, words[i].length());
                    map.put(pref + " " + suff, i);
                }
            }
        }
    }
    
    public int f(String pref, String suff) {
        return map.getOrDefault(pref + " " + suff, -1);
    }
}

/**
 * Your WordFilter object will be instantiated and called as such:
 * WordFilter obj = new WordFilter(words);
 * int param_1 = obj.f(pref,suff);
 */
~~~

#### [692. 前K个高频单词](https://leetcode.cn/problems/top-k-frequent-words/)

~~~
//哈希排序
class Solution {
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> map = new HashMap<>();
        for (String s : words) map.put(s, map.getOrDefault(s, 0) + 1);
        PriorityQueue<Map.Entry<String, Integer>> q = new PriorityQueue<>((a, b) -> {
            if (a.getValue().equals(b.getValue())) 
                return b.getKey().compareTo(a.getKey());
            else
                return a.getValue() - b.getValue();
        });
        Set<Map.Entry<String, Integer>> entries = map.entrySet();
        for (Map.Entry<String, Integer> entry : entries) {
            q.offer(entry);
            if (q.size() > k) q.poll();
        }
        List<String> ans = new ArrayList<>();
        for (int i = 0; i < k; i ++ ) ans.add(q.poll().getKey());
        Collections.reverse(ans);
        return ans;
    }
}
~~~

#### [904. 水果成篮](https://leetcode.cn/problems/fruit-into-baskets/)

**解法：哈希**

~~~
class Solution {
	//求两个数的最长连续子序列
    public int totalFruit(int[] f) {
        int res = 0;
        int[] map = new int[f.length];
        for (int i = 0, j = 0, s = 0; i < f.length; i ++ ) {
            if ( ++ map[f[i]] == 1) s ++ ;
            while (s > 2) {
                if (-- map[f[j]] == 0) s -- ;
                j ++ ;
            }
            res = Math.max(res, i - j + 1);
        }
        return res;
    }
}
~~~

#### [1282. 用户分组](https://leetcode.cn/problems/group-the-people-given-the-group-size-they-belong-to/)

~~~
class Solution {
    public List<List<Integer>> groupThePeople(int[] g) {
        List<List<Integer>> res = new ArrayList<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < g.length; i ++ ) {
            int x = g[i];
            if (map.get(x) == null) map.put(x, new ArrayList<>());
            map.get(x).add(i);
            if (map.get(x).size() == x) {
                res.add(new ArrayList<>(map.get(x)));
                map.put(x, null);
            }
        }
        return res;
    }
}
~~~

#### [1224. 最大相等频率](https://leetcode.cn/problems/maximum-equal-frequency/)

~~~
class Solution {
    public int maxEqualFreq(int[] nums) {
        int n = nums.length;
        int res = 0, max = 0;
        //记录每个数字出现的频次
        int[] hash = new int[100010];
        //记录每个频次出现的次数
        int[] cnt = new int[100010];
        for (int i = 0; i < n; i ++ ) {
            int x = nums[i];
            if (hash[x] > 0) 
                cnt[hash[x]] -- ;
            hash[x] ++ ;
            cnt[hash[x]] ++ ;

            max = Math.max(max, hash[x]);
            if (max == 1 || max * cnt[max] == i || (max - 1) * (cnt[max - 1] + 1) == i)
                res = i + 1;
        }
        return res;
    }
}
~~~



#### [1371. 每个元音包含偶数次的最长子字符串](https://leetcode.cn/problems/find-the-longest-substring-containing-vowels-in-even-counts/)

**解法：状态压缩&前缀和&哈希**

~~~
class Solution {
    private static String v = "aeiou";
    public int findTheLongestSubstring(String s) {
        int res = 0;
        int n = s.length();
        //一共5位如果这一位为1就说明是奇数，为0就说明是偶数
        int state = 0; //00000
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < v.length(); j ++ ) 
                if (s.charAt(i) == v.charAt(j)) {
                    state ^= 1 << j;
                    break;
                }
            if (map.containsKey(state)) 
                res = Math.max(res, i - map.get(state));
            else map.put(state, i);
        }
        return res;
    }
}
~~~



#### 



## 原地哈希

***特点O(1)空间***

#### [41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/)

#### [442. 数组中重复的数据](https://leetcode.cn/problems/find-all-duplicates-in-an-array/)

#### [448. 找到所有数组中消失的数字](https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/)

## 模拟

#### [299. 猜数字游戏](https://leetcode-cn.com/problems/bulls-and-cows/)

#### [495. 提莫攻击](https://leetcode-cn.com/problems/teemo-attacking/)

#### [520. 检测大写字母](https://leetcode-cn.com/problems/detect-capital/)

#### [397. 整数替换](https://leetcode-cn.com/problems/integer-replacement/)

暴力模拟 + 备忘录防止重复 （根号n复杂度）

#### [384. 打乱数组](https://leetcode-cn.com/problems/shuffle-an-array/)

Knuth洗牌算法

对于下标为 0 位置，从 [0,n−1] 随机一个位置进行交换，共有 n 种选择；下标为 1 的位置，从 [1,n−1] 随机一个位置进行交换，共有n−1 种选择 ...

~~~
for (int i = 0; i < n; i++) {
	swap(ans, i, i + random.nextInt(n - i));
}
~~~

~~~
class Solution {
    int[] nums;
    int n;
    Random r;
    public Solution(int[] _nums) {
        nums = _nums;
        n = nums.length;
        r = new Random();
    }
    
    public int[] reset() {
        return nums;
    }
    
    public int[] shuffle() {
        int[] res = nums.clone();
        for (int i = 0; i < n; i ++ ) {
            swap(res, i, i + r.nextInt(n - i));
        }
        return res;
    }

    public void swap(int[] arr, int a, int b) {
        int t = arr[a];
        arr[a] = arr[b];
        arr[b] = t;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(nums);
 * int[] param_1 = obj.reset();
 * int[] param_2 = obj.shuffle();
 */
~~~



#### [859. 亲密字符串](https://leetcode-cn.com/problems/buddy-strings/)

#### [423. 从英文中重建数字](https://leetcode-cn.com/problems/reconstruct-original-digits-from-english/)

我们可以先对 `s` 进行词频统计，然后根据「英文单词中的字符唯一性」确定构建的顺序

#### [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

#### [506. 相对名次](https://leetcode-cn.com/problems/relative-ranks/)

#### [1816. 截断句子](https://leetcode-cn.com/problems/truncate-sentence/)

#### [794. 有效的井字游戏](https://leetcode-cn.com/problems/valid-tic-tac-toe-state/)

#### [748. 最短补全词](https://leetcode-cn.com/problems/shortest-completing-word/)

#### [2275. 按位与结果大于零的最长组合](https://leetcode.cn/problems/largest-combination-with-bitwise-and-greater-than-zero/)

~~~
class Solution {
    public int largestCombination(int[] nums) {
        /*对于一个序列，其中每一个元素转换为二进制后看成一个32位的数组，
         每一位不是0就是1。一个序列要想“与运算”后不为0，
         其实只要序列中某一位全都是1即可。于是我们可以对原数组每个元素按位求和，
         最大的那个和就是最长“与运算”后结果不为0的序列长度。
        */
        int n = nums.length, ans = 0;
        for (int i = 0; i < 32; i ++ ) {
            int cnt = 0;
            for (int j = 0; j < n; j ++ ) {
                if ((nums[j] >> i & 1) == 1) cnt ++ ;
            }
            ans = Math.max(ans, cnt);
        }
        return ans;
    }
}
~~~

#### [735. 行星碰撞](https://leetcode.cn/problems/asteroid-collision/)

~~~
class Solution {
    public int[] asteroidCollision(int[] asteroids) {
        ArrayDeque<Integer> s = new ArrayDeque<>();
        for (int x : asteroids) {
            if (x > 0) s.push(x);
            else {
                while (s.size() > 0 && s.peek() > 0 && s.peek() < -x) s.pop();
                if (s.size() > 0 && s.peek() == -x) s.pop();
                else if (s.size() == 0 || s.peek() < 0) s.push(x);
            }
        }
        if (s.size() == 0) return new int[]{};
        int[] res = new int[s.size()];
        for (int i = 0; i < res.length; i ++ ) res[i] = s.pollLast();
        return res;
    }
}
~~~

#### [118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/)

~~~
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ans = new ArrayList();
        for (int i = 0; i < numRows; i ++ ) {
            List<Integer> tem = new ArrayList<>();
            for (int j = 0; j <= i; j ++ ) {
                if (j == 0 || j == i) tem.add(1);
                else tem.add(ans.get(i - 1).get(j - 1) + ans.get(i - 1).get(j));
            }
            ans.add(tem);
        }
        return ans;
    }
}
~~~

#### [剑指 Offer II 041. 滑动窗口的平均值](https://leetcode.cn/problems/qIsx9U/)

~~~
class MovingAverage {
    int sum = 0, n;
    ArrayDeque<Integer> q = new ArrayDeque<>();
    /** Initialize your data structure here. */
    public MovingAverage(int size) {
        n = size;
    }
    
    public double next(int val) {
        if (q.size() >= n) sum -= q.poll();
        sum += val;
        q.add(val);
        return sum * 1.0 / q.size();
    }
}

/**
 * Your MovingAverage object will be instantiated and called as such:
 * MovingAverage obj = new MovingAverage(size);
 * double param_1 = obj.next(val);
 */
~~~

#### [1260. 二维网格迁移](https://leetcode.cn/problems/shift-2d-grid/)

~~~
class Solution {
    public List<List<Integer>> shiftGrid(int[][] g, int k) {
        List<List<Integer>> res = new ArrayList<>();
        int n = g.length, m = g[0].length;
        int[] arr = new int[m * n];
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                int idx = i * m + j + k;
                idx %= (m * n);
                arr[idx] = g[i][j];
            }
        }
        for (int i = 0; i < n; i ++ ) {
            List<Integer> tem = new ArrayList<>();
            for (int j = 0; j < m; j ++ ) {
                tem.add(arr[i * m + j]);
            }
            res.add(tem);
        }
        return res;
    }
}
~~~

#### [169. 多数元素](https://leetcode.cn/problems/majority-element/)

~~~
class Solution {
    public int majorityElement(int[] nums) {
        int r = 0, c = 0;
        for (int x : nums) {
            if (c == 0) {
                r = x;
                c = 1;
            } else if (r == x) {
                c ++ ;
            } else {
                c -- ;
            }
        }
        return r;
    }
}
~~~

#### [592. 分数加减运算](https://leetcode.cn/problems/fraction-addition-and-subtraction/)

**解法：模拟**

~~~
class Solution {
    public int gcd(int a, int b) {
        return b != 0 ? gcd(b, a % b) : a; 
    }
    public String fractionAddition(String ex) {
        if (ex.charAt(0) != '-') ex = "+" + ex;
        int a = 0, b = 1;
        for (int i = 0; i < ex.length(); ) {
            int t = i + 1;
            while (ex.charAt(t) != '/') t ++ ;
            int c = Integer.parseInt(ex.substring(i + 1, t));
            int j = t + 1;
            while (j < ex.length() && Character.isDigit(ex.charAt(j))) j ++ ;
            int d = Integer.parseInt(ex.substring(t + 1, j));
            if (ex.charAt(i) == '-') c = -c;
            a = a * d + b * c;
            b = b * d;
            i = j;
        } 
        int z = gcd(a, b);
        a /= z; b /= z;
        if (b < 0) {
            a = -a;
            b = -b;
        }
        return a + "/" + b;
    }
}
~~~

#### [622. 设计循环队列](https://leetcode.cn/problems/design-circular-queue/)

**解法：模拟**

~~~
class MyCircularQueue {
    int[] q;
    int hh = 0, tt = 0;
    public MyCircularQueue(int k) {
        //将长度开到k + 1，目的是区分队空和队满的情况
        q = new int[k + 1];
    }
    
    public boolean enQueue(int x) {
        if (isFull()) return false;
        q[tt ++ ] = x;
        if (tt == q.length) tt = 0;
        return true;
    }
    
    public boolean deQueue() {
        if (isEmpty()) return false;
        hh ++ ;
        if (hh == q.length) hh = 0;
        return true;
    }
    
    public int Front() {
        if (isEmpty()) return -1;
        return q[hh];
    }
    
    public int Rear() {
        if (isEmpty()) return -1;
        int t = tt - 1;
        return t < 0 ? q[q.length - 1] : q[t];
    }
    
    public boolean isEmpty() {
        return hh == tt;
    }
    
    public boolean isFull() {
        return (tt + 1) % q.length == hh;
    }
}
~~~

#### [224. 基本计算器](https://leetcode.cn/problems/basic-calculator/)

**解法：栈**

~~~
class Solution {
    public void calc(ArrayDeque<Integer> num, ArrayDeque<Character> op) {
        var b = num.pop();
        var a = num.pop();
        if (op.peek() == '+') num.push(a + b);
        else num.push(a - b);
        op.pop();
    }

    public int calculate(String rs) {
        StringBuilder sb = new StringBuilder();
        for (char x : rs.toCharArray())
            if (x != ' ')
                sb.append(x);
        String s = sb.toString();
        ArrayDeque<Integer> num = new ArrayDeque<>();
        ArrayDeque<Character> op = new ArrayDeque<>();
        for (int i = 0; i < s.length(); i ++ ) {
            char c = s.charAt(i);
            if (c == ' ') continue;
            if (c == '+' || c == '-') {
                //特殊符号处理
                if (i == 0 || s.charAt(i - 1) == '-' || s.charAt(i - 1) == '(') 
                    num.push(0);
                op.push(c);
            } else if (c == '(') {
                op.push(c);
            } else if (c == ')') {
                op.pop();
                while (!op.isEmpty() && op.peek() != '(') calc(num, op);
            } else {
                int k = i;
                while (k < s.length() && Character.isDigit(s.charAt(k))) k ++ ;
                num.push(Integer.parseInt(s.substring(i, k)));
                i = k - 1;
                while (!op.isEmpty() && op.peek() != '(') calc(num, op);
            }
        }
        return num.pop();
    }
}
~~~

#### [640. 求解方程](https://leetcode.cn/problems/solve-the-equation/)

**解法：模拟**

~~~
class Solution {
    public Pair<Integer, Integer> work(String s) {
        int a = 0, b = 0;
        if (s.charAt(0) != '-') s = '+' + s;
        for (int i = 0; i < s.length(); i ++ ) {
            int j = i + 1;
            while (j < s.length() && Character.isDigit(s.charAt(j))) j ++ ;
            int c = 1;
            if (i + 1 <= j - 1) c = Integer.parseInt(s.substring(i + 1, j));
            if (s.charAt(i) == '-') c *= -1;
            if (j < s.length() && s.charAt(j) == 'x') {
                a += c;
                i = j;
            }
            else {
                b += c;
                i = j - 1;
            }   
        }
        return new Pair(a, b);
    }
    public String solveEquation(String equation) {
        String[] ss = equation.split("=");
        //左右分别合并同类项，再分类讨论答案
        Pair<Integer, Integer> l = work(ss[0]), r = work(ss[1]);
        int A = l.getKey() - r.getKey(), B = r.getValue() - l.getValue();
        if (A == 0 && B == 0) return "Infinite solutions";
        else if (A == 0 && B != 0) return "No solution";
        else return "x=" + B / A;
    }
}
~~~

#### [2196. 根据描述创建二叉树](https://leetcode.cn/problems/create-binary-tree-from-descriptions/)

**解法：模拟**

~~~
class Solution {
    public TreeNode createBinaryTree(int[][] descriptions) {
        //用去重和构建二叉树
        Map<Integer, TreeNode> map = new HashMap<>();
        //用来确定根节点
        Set<Integer> set = new HashSet<>();
        for (int[] node : descriptions) {
            int p = node[0], c = node[1], left = node[2];
            set.add(c);
            if (!map.containsKey(p)) map.put(p, new TreeNode(p));
            if (!map.containsKey(c)) map.put(c, new TreeNode(c));
            if (left == 1) map.get(p).left = map.get(c);
            else map.get(p).right = map.get(c);
        }
        TreeNode res = null;
        for (int x : map.keySet())
            if (!set.contains(x))
                res = map.get(x);
        return res;
    }
}
~~~

#### [641. 设计循环双端队列](https://leetcode.cn/problems/design-circular-deque/)

**解法：模拟**

~~~
class MyCircularDeque {
    int[] q;
    int hh, tt, sz, n;
    public MyCircularDeque(int k) {
        n = k;
        sz = 0;
        hh = tt = 0;
        q = new int[k];
    }
    
    public boolean insertFront(int value) {
        if (isFull()) return false;
        hh = (hh + n - 1) % n;
        q[hh] = value;
        sz ++ ;
        return true;
    }
    
    public boolean insertLast(int value) {
        if (isFull()) return false;
        q[tt ++ ] = value;
        tt %= n;
        sz ++ ;
        return true;
    }
    
    public boolean deleteFront() {
        if (isEmpty()) return false;
        hh = (hh  + 1) % n;
        sz -- ;
        return true;
    }
    
    public boolean deleteLast() {
        if (isEmpty()) return false;
        tt = (tt - 1 + n) % n;
        sz -- ;
        return true;
    }
    
    public int getFront() {
        if (isEmpty()) return -1;
        return q[hh];
    }
    
    public int getRear() {
        if (isEmpty()) return -1;
        int t = tt - 1;
        return t < 0 ? q[q.length - 1] : q[t];
    }
    
    public boolean isEmpty() {
        return sz == 0;
    }
    
    public boolean isFull() {
        return sz == n;
    }
}

/**
 * Your MyCircularDeque object will be instantiated and called as such:
 * MyCircularDeque obj = new MyCircularDeque(k);
 * boolean param_1 = obj.insertFront(value);
 * boolean param_2 = obj.insertLast(value);
 * boolean param_3 = obj.deleteFront();
 * boolean param_4 = obj.deleteLast();
 * int param_5 = obj.getFront();
 * int param_6 = obj.getRear();
 * boolean param_7 = obj.isEmpty();
 * boolean param_8 = obj.isFull();
 */
~~~

#### [1329. 将矩阵按对角线排序](https://leetcode.cn/problems/sort-the-matrix-diagonally/)

~~~
class Solution {
    public int[][] diagonalSort(int[][] mat) {
        int n = mat.length, m = mat[0].length;
        int[] t = new int[m + n];
        for (int i = 0; i < n; i ++ ) {
            int idx = 0;
            for (int x = i, y = 0; x < n && y < m; x ++ , y ++ ) 
                t[idx ++ ] = mat[x][y];
            Arrays.sort(t, 0, idx);
            for (int j = 0, x = i, y = 0; j < idx; j ++ ) 
                mat[x ++ ][y ++ ] = t[j];
        }
        for (int i = 0; i < m; i ++ ) {
            int idx = 0;
            for (int x = 0, y = i; x < n && y < m; x ++ , y ++ ) 
                t[idx ++ ] = mat[x][y];
            Arrays.sort(t, 0, idx);
            for (int j = 0, x = 0, y = i; j < idx; j ++) 
                mat[x ++ ][y ++ ] = t[j];
        }
        return mat;
    }
}
~~~

#### [1360. 日期之间隔几天](https://leetcode.cn/problems/number-of-days-between-two-dates/)

~~~
class Solution {
    int[] month = new int[]{0,31,28,31,30,31,30,31,31,30,31,30,31};
    public int daysBetweenDates(String date1, String date2) {
        return Math.abs(get(date1) - get(date2));
    }
    public int isleap(int year) {
        //四年一闰，百年不闰，四百年再闰
        if (year % 4 == 0 && year % 100 != 0 || year % 400 == 0) return 1;
        return 0;
    }
    public int get(String date) {
        String[] s = date.split("-");
        int y = Integer.parseInt(s[0]);
        int m = Integer.parseInt(s[1]);
        int d = Integer.parseInt(s[2]);
        int res = 0;
        for (int i = 1971; i < y; i ++ ) 
            res += 365 + isleap(i);
        for (int i = 1; i < m; i ++ ) {
            if (i == 2) res += isleap(y);
            res += month[i];
        }
        res += d;
        return res;
    }
}
~~~

#### [38. 外观数列](https://leetcode.cn/problems/count-and-say/)

~~~
class Solution {
    public String countAndSay(int n) {
        if (n == 1) return "1";
        String s = countAndSay(n - 1);
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < s.length(); i ++ ) {
            int j = i;
            while (j < s.length() && s.charAt(i) == s.charAt(j)) j ++ ;
            res.append(j - i).append(s.charAt(i));
            i = j - 1;
        }
        return res.toString();
    }
}
~~~

#### [1073. 负二进制数相加](https://leetcode.cn/problems/adding-two-negabinary-numbers/)

~~~
class Solution {
    public void swap(int[] ss, int l, int r) {
        int t = ss[l];
        ss[l] = ss[r];
        ss[r] = t;
    }
    public void reverse(int[] ss, int l, int r) {
        while (l < r) {
            swap(ss, l, r);
            l ++ ;
            r -- ;
        }
    }
    public int[] addNegabinary(int[] A, int[] B) {
        reverse(A, 0, A.length - 1);
        reverse(B, 0, B.length - 1);
        var C = new ArrayList<Integer>();
        //1 + 1 = 110
        //11 + 1 = 0
        for (int i = 0, a = 0, b = 0, c = 0; i < Math.max(A.length, B.length) || a > 0 || b > 0; i ++ ) {
            if (a == 1 && b == 2) a = b = 0;
            c = b;
            b = a;
            a = 0;
            if (i < A.length) c += A[i];
            if (i < B.length) c += B[i];
            C.add(c & 1); 
            c >>= 1;
            a += c;
            b += c;
        }
        while (C.size() > 1 && C.get(C.size() - 1) == 0) C.remove(C.size() - 1);
        Collections.reverse(C);
        return C.stream().mapToInt(x -> x).toArray();
    }
}
~~~

#### [842. 将数组拆分成斐波那契序列](https://leetcode.cn/problems/split-array-into-fibonacci-sequence/)

~~~
class Solution {
    public List<Integer> splitIntoFibonacci(String s) {
        for (int i = 1; i <= 10 && i < s.length(); i ++ ) {
            for (int j = i + 1; j <= i + 10 && j < s.length(); j ++ ) {
                long a = Long.parseLong(s.substring(0, i));
                long b = Long.parseLong(s.substring(i, j));
                List<Integer> res = dfs(a, b, s);
                if (res.size() != 0) return res;
            }
        }
        return new ArrayList<>();
    }
    public List<Integer> dfs(long a, long b, String s) {
        var res = new ArrayList<Integer>();
        res.add((int)a);
        res.add((int)b);
        StringBuilder t = new StringBuilder().append(a).append(b);
        while (t.length() < s.length()) {
            long c = a + b;
            if (c >= Integer.MAX_VALUE) return new ArrayList<>();
            t.append(c);
            res.add((int)c);
            a = b;
            b = c;
        }
        if (!t.toString().equals(s)) return new ArrayList<>();
        return res;
    }
}
~~~

#### [289. 生命游戏](https://leetcode.cn/problems/game-of-life/)

~~~
class Solution {
    static int[][] ds = new int[][]{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, - 1}, {1, -1}};
    //最后一位为初始状态，第二位为新状态
    public void gameOfLife(int[][] board) {
        int n = board.length, m = board[0].length;
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                int live = 0;
                for (int[] d : ds) {
                    int x = i + d[0], y = j + d[1];
                    if (x >= 0 && x < n && y >= 0 && y < m)
                        live += board[x][y] & 1;
                }
                if ((board[i][j] & 1) > 0) {
                    if (live >= 2 && live <= 3)
                        board[i][j] = 3;
                } else if (live == 3) 
                    board[i][j] = 2;
            }
        }
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ )
                board[i][j] >>= 1;
    }
}
~~~

#### [950. 按递增顺序显示卡牌](https://leetcode.cn/problems/reveal-cards-in-increasing-order/)

~~~
class Solution {
    public int[] deckRevealedIncreasing(int[] deck) {
        Arrays.sort(deck);
        int n = deck.length;
        var q = new ArrayDeque<Integer>();
        for (int i = 0; i < n; i ++ ) q.add(i);
        int idx = 0;
        var res = new int[n];
        while (!q.isEmpty()) {
            int t = q.poll();
            res[t] = deck[idx ++ ];
            if (!q.isEmpty()) 
                q.add(q.poll());
        }
        return res;
    }
}
~~~

#### [647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/)

**解法：中心扩展**

~~~
class Solution {
    public int countSubstrings(String s) {
        var res = 0;
        var n = s.length();
        for (int i = 0; i < n * 2 - 1; i ++ ) {
            int l = i / 2, r = l + i % 2;
            while (l >= 0 && r < n && s.charAt(l) == s.charAt(r)) {
                res ++ ;
                l -- ;
                r ++ ;
            }
        }
        return res;
    }
}
~~~

#### [面试题 01.08. 零矩阵](https://leetcode.cn/problems/zero-matrix-lcci/)

~~~
class Solution {
    public void setZeroes(int[][] matrix) {
        int n = matrix.length, m = matrix[0].length;
        Set<Integer> row = new HashSet<>();
        Set<Integer> col = new HashSet<>();
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ )
                if (matrix[i][j] == 0) {
                    row.add(i);
                    col.add(j);
                }
        for (int i = 0; i < n; i ++ )
            if (row.contains(i))
                Arrays.fill(matrix[i], 0);
        for (int j = 0; j < m; j ++ )
            if (col.contains(j)) 
                for (int i = 0; i < n; i ++ )
                    matrix[i][j] = 0;   
    }
}
~~~

#### [912. 排序数组](https://leetcode.cn/problems/sort-an-array/)

**解法：手撕堆排**

~~~
class Solution {
    int[] h;
    public void swap(int a, int b) {
        int t = h[a];
        h[a] = h[b];
        h[b] = t;
    }
    public void down(int u, int k) {
        int t = u;
        if (u * 2 < k && h[t] > h[u * 2]) t = u * 2;
        if (u * 2 + 1 < k && h[t] > h[u * 2 + 1]) t = u * 2 + 1;
        if (u != t) {
            swap(u ,t);
            down(t, k);
        } 
    }
    public void buildHeap(int[] nums, int n) {
        h = new int[n + 1];
        //建堆
        for (int i = 1; i <= n; i ++ )
            h[i] = nums[i - 1];
        for (int i = n / 2; i > 0; i -- )
            down(i, h.length);
    }
    public void headSort(int[] nums) {
        int n = nums.length;
        buildHeap(nums, n);
        //每次取出堆顶并删除
        for (int i = 0; i < nums.length; i ++ ) {
            nums[i] = h[1];
            swap(1, n - i);
            down(1, n - i);
        }
    }
    public int[] sortArray(int[] nums) {
        headSort(nums);
        return nums;
    }
}
~~~



#### [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

11.11

#### [938. 二叉搜索树的范围和](https://leetcode-cn.com/problems/range-sum-of-bst/)

11.13

#### [617. 合并二叉树](https://leetcode-cn.com/problems/merge-two-binary-trees/)

#### [589. N 叉树的前序遍历](https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/)

#### [590. N 叉树的后序遍历](https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/)

11.15

#### [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

递归

#### [700. 二叉搜索树中的搜索](https://leetcode-cn.com/problems/search-in-a-binary-search-tree/)

11.16

#### [1302. 层数最深叶子节点的和](https://leetcode-cn.com/problems/deepest-leaves-sum/)

dfs/bfs

#### [222. 完全二叉树的节点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int countNodes(TreeNode root) {
        if (root == null) return 0;
        TreeNode l = root.left, r = root.right;
        int x = 1, y = 1;
        while (l != null) {
            l = l.left;
            x ++ ;
        }
        while (r != null) {
            r = r.right;
            y ++ ;
        }
        if (x == y) return (1 << x) - 1;
        return countNodes(root.left) + countNodes(root.right) + 1;     
    }
}
~~~



#### [面试题 04.02. 最小高度树](https://leetcode-cn.com/problems/minimum-height-tree-lcci/)

#### [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

#### [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

注意陷阱：左右子树中 一个为空，一个不为空的情况

11.17

#### [654. 最大二叉树](https://leetcode-cn.com/problems/maximum-binary-tree/)

#### [100. 相同的树](https://leetcode-cn.com/problems/same-tree/)

#### [783. 二叉搜索树节点最小距离](https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/)

#### [230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

二叉搜索树的中序遍历是有序的，因此我们可以直接对「二叉搜索树」进行中序遍历，保存遍历过程中的第k个数即可（783同理可找到最小距离）

#### [429. N 叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)

#### [563. 二叉树的坡度](https://leetcode-cn.com/problems/binary-tree-tilt/)

#### [559. N 叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-n-ary-tree/)

#### [897. 递增顺序搜索树](https://leetcode-cn.com/problems/increasing-order-search-tree/)

#### [99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/)

#### [109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/)

#### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

#### [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

#### [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

#### [404. 左叶子之和](https://leetcode-cn.com/problems/sum-of-left-leaves/)

#### [872. 叶子相似的树](https://leetcode-cn.com/problems/leaf-similar-trees/)

#### [701. 二叉搜索树中的插入操作](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

#### [671. 二叉树中第二小的节点](https://leetcode-cn.com/problems/second-minimum-node-in-a-binary-tree/)

#### [513. 找树左下角的值](https://leetcode-cn.com/problems/find-bottom-left-tree-value/)

~~~
//bfs
class Solution {
    public int findBottomLeftValue(TreeNode root) {
        ArrayDeque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        while (!q.isEmpty()) {
            root = q.poll();
            if (root.right != null) q.add(root.right);
            if (root.left != null) q.add(root.left);
        }
        return root.val;
    }
}
~~~

~~~
//dfs
class Solution {
    int res = 0, dep = -1;
    public int findBottomLeftValue(TreeNode root) {
        dfs(root, 0);
        return res;
    }
    public void dfs(TreeNode root, int depth) {
        if (root == null) return;
        if (depth > dep) {
            dep = depth;
            res = root.val;
        }
        dfs(root.left, depth + 1);
        dfs(root.right, depth + 1);
    }
}
~~~

#### [1325. 删除给定值的叶子节点](https://leetcode-cn.com/problems/delete-leaves-with-a-given-value/)

#### [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

#### [538. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

二叉搜索树中序遍历倒以下为递减序列

#### [173. 二叉搜索树迭代器](https://leetcode-cn.com/problems/binary-search-tree-iterator/)

#### [1609. 奇偶树](https://leetcode-cn.com/problems/even-odd-tree/)

#### [449. 序列化和反序列化二叉搜索树](https://leetcode.cn/problems/serialize-and-deserialize-bst/)

#### [331. 验证二叉树的前序序列化](https://leetcode.cn/problems/verify-preorder-serialization-of-a-binary-tree/)

~~~~
class Solution {
    public boolean isValidSerialization(String s) {
        int n = s.length();
        int num = 0;
        for (int i = n - 1; i >= 0; i -- ) {
            char c = s.charAt(i);
            if (c == ',') continue;
            else if (c =='#') num ++ ;
            else {
                while (i > 0 && s.charAt(i - 1) != ',') i -- ;
                if (num >= 2) num -- ;
                else {
                    if (num < 2) return false;
                } 
            }
        } 
        if (num != 1) return false;
        return true;
    }
}
~~~~

#### [剑指 Offer II 047. 二叉树剪枝](https://leetcode.cn/problems/pOCWxh/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode pruneTree(TreeNode root) {
        if (root == null) return null;
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);
        if (root.left == null && root.right == null && root.val == 0) root = null;
        return root;
    }
}
~~~

#### [113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)

~~~
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    ArrayDeque<Integer> path = new ArrayDeque<>();
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        dfs(root, targetSum);
        return res;
    }
    public void dfs(TreeNode root, int sum) {
        if (root == null) return;
        path.add(root.val);
        sum -= root.val;
        if (root.left == null && root.right == null && sum == 0)
            res.add(new ArrayList<>(path));
        dfs(root.left, sum);
        dfs(root.right, sum);
        path.pollLast();
    }
}
~~~

#### [437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/)

~~~
//dfs + 前缀和
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int ans = 0;
    Map<Integer, Integer> cnt = new HashMap<>();
    public int pathSum(TreeNode root, int targetSum) {
        cnt.put(0, 1);
        dfs(root, targetSum, 0);
        return ans;
    }
    public void dfs(TreeNode root, int target, int cur) {
        if (root == null) return;
        cur += root.val;
        //Si - Sj = T --> Sj = Si - T --> Sj = cnt - target
        if (cnt.containsKey(cur - target)) ans += cnt.get(cur - target);
        cnt.put(cur, cnt.getOrDefault(cur, 0) + 1);
        dfs(root.right, target, cur);
        dfs(root.left, target, cur);
        cnt.put(cur, cnt.get(cur) - 1);
    }
}
~~~

#### [515. 在每个树行中找最大值](https://leetcode.cn/problems/find-largest-value-in-each-tree-row/)

~~~
//bfs
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        ArrayDeque<TreeNode> q = new ArrayDeque<>();
        if (root == null) return res;
        q.add(root);
        while (!q.isEmpty()) {
            int count = q.size();
            int max = Integer.MIN_VALUE;
            while (count -- > 0) {
                TreeNode node = q.poll();
                max = Math.max(max, node.val);
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            res.add(max);
        }
        return res;
    }
}
~~~

~~~
//dfs
class Solution {
    List<Integer> res = new ArrayList<>();
    int max = Integer.MAX_VALUE;
    public List<Integer> largestValues(TreeNode root) {
        dfs(root, 0);
        return res;
    }
    public void dfs(TreeNode root, int depth) {
        if (root == null) return;
        if (depth == res.size()) res.add(root.val);
        else res.set(depth, Math.max(res.get(depth), root.val));
        dfs(root.left, depth + 1);
        dfs(root.right, depth + 1);
    }
}
~~~

#### [919. 完全二叉树插入器](https://leetcode.cn/problems/complete-binary-tree-inserter/)

**解法1：队列**

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class CBTInserter {
    ArrayDeque<TreeNode> q = new ArrayDeque<>();
    TreeNode root;
    public CBTInserter(TreeNode _root) {
        root = _root;
        TreeNode cur = root;
        if (cur != null) q.add(cur);
        while (!q.isEmpty()) {
            cur = q.peek();
            if (cur.left != null) q.add(cur.left);
            if (cur.right != null) q.add(cur.right);

            //如果该节点满了就poll
            if (cur.right != null) q.poll();
            else break;
        }
    }
    
    public int insert(int val) {
        TreeNode cur = q.peek();
        TreeNode node = new TreeNode(val);
        if (cur.left == null) cur.left = node;
        else {
            cur.right = node;
            q.poll();
        }
        q.add(node);
        return cur.val;
    }
    
    public TreeNode get_root() {
        return root;
    }
}

/**
 * Your CBTInserter object will be instantiated and called as such:
 * CBTInserter obj = new CBTInserter(root);
 * int param_1 = obj.insert(val);
 * TreeNode param_2 = obj.get_root();
 */
~~~

**解法2：list模拟**

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class CBTInserter {
    ArrayList<TreeNode> list = new ArrayList<>();
    int idx = 0;
    public CBTInserter(TreeNode root) {
        list.add(root);
        int cur = 0;
        while (cur < list.size()) {
            TreeNode node = list.get(cur);
            if (node.left != null) list.add(node.left);
            if (node.right != null) list.add(node.right);
            cur ++ ;
        }
    }
    
    public int insert(int val) {
        while (list.get(idx).left != null && list.get(idx).right != null) idx ++;
        TreeNode cur = list.get(idx);
        TreeNode node = new TreeNode(val);
        if (cur.left == null) cur.left = node;
        else {
            cur.right = node;
            idx ++ ;
        }
        list.add(node);
        return cur.val;
    }
    
    public TreeNode get_root() {
        return list.get(0);
    }
}

/**
 * Your CBTInserter object will be instantiated and called as such:
 * CBTInserter obj = new CBTInserter(root);
 * int param_1 = obj.insert(val);
 * TreeNode param_2 = obj.get_root();
 */
~~~

#### [1161. 最大层内元素和](https://leetcode.cn/problems/maximum-level-sum-of-a-binary-tree/)

**解法：层序遍历**

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int maxLevelSum(TreeNode root) {
        int res = 0, max = Integer.MIN_VALUE;
        //当前层数
        int t = 1;
        ArrayDeque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        while (!q.isEmpty()) {
            int cur = 0, num = q.size();
            while (num -- > 0) {
                TreeNode node = q.poll();
                cur += node.val;
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            if (cur > max) {
                max = cur;
                res = t;
            }
            t ++ ;
        }
        return res;
    }
}
~~~

**解法：深搜**

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int[] map = new int[10010];
    int depth = 0;
    public int maxLevelSum(TreeNode root) {
        dfs(root, 1);
        int res = 0, max = -10010;
        for (int i = 1; i <= depth; i ++ ) {
            if (max < map[i]) {
                max = map[i];
                res = i;
            }
        }
        return res;
    }
    public void dfs(TreeNode root, int l) {
        if (root == null) return;
        depth = Math.max(depth, l);
        map[l] += root.val;
        dfs(root.left, l + 1);
        dfs(root.right, l + 1);
    }
}
~~~

#### [889. 根据前序和后序遍历构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

**解法：深搜**

~~~
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    public TreeNode constructFromPrePost(int[] pre, int[] pos) {
        for (int i = 0; i < pre.length; i ++ )
            map.put(pos[i], i);
        return dfs(pre, pos, 0, pre.length - 1, 0, pos.length - 1);
    }
    public TreeNode dfs(int[] pre, int[] pos, int prl, int prr, int pol, int por) {
        if (prl > prr) return null;
        TreeNode root = new TreeNode(pre[prl]);
        if (prl == prr) return root;
        int k = map.get(pre[prl + 1]) - pol;
        root.left = dfs(pre, pos, prl + 1, prl + 1 + k, pol, pol + k);
        root.right = dfs(pre, pos, prl + 1 + k + 1, prr, pol + k + 1, por - 1);
        return root;
    }
}
~~~

#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

**解法：递归**

~~~
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    public TreeNode buildTree(int[] pre, int[] in) {
        for (int i = 0; i < in.length; i ++ ) 
            map.put(in[i], i);
        return dfs(pre, in, 0, pre.length - 1, 0, in.length - 1);
    }
    public TreeNode dfs(int[] pre, int[] in, int pl, int pr, int il, int ir) {
        if (il > ir) return null;
        int k = map.get(pre[pl]) - il;
        TreeNode root = new TreeNode(pre[pl]);
        root.left = dfs(pre, in, pl + 1, pl + k, il, il + k - 1);
        root.right = dfs(pre, in, pl + k + 1, pr, il + k + 1, ir);
        return root;
    }
}
~~~

#### [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

**解法：递归**

~~~
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        int n = inorder.length;
        for (int i = 0; i < n; i ++ ) map.put(inorder[i], i);
        return dfs(postorder, inorder, 0, n - 1, 0, n - 1);
    }
    public TreeNode dfs(int[] pos, int[] in, int pl, int pr, int il, int ir) {
        if (il > ir) return null;
        int k = map.get(pos[pr]) - il;
        TreeNode root = new TreeNode(pos[pr]);
        root.left = dfs(pos, in, pl, pl + k - 1, il, il + k - 1);
        root.right = dfs(pos, in, pl + k, pr - 1, il + k + 1, ir);
        return root;
    }
}
~~~

#### [1008. 前序遍历构造二叉搜索树](https://leetcode.cn/problems/construct-binary-search-tree-from-preorder-traversal/)

**解法：递归**

~~~
class Solution {
    public TreeNode bstFromPreorder(int[] pre) {
        return dfs(pre, 0, pre.length - 1);
    }
    public TreeNode dfs(int[] pre, int l, int r) {
        if (l > r) return null;
        TreeNode root = new TreeNode(pre[l]);
        int j = l;
        while (j + 1 <= r && pre[l] > pre[j + 1]) j ++ ;
        root.left = dfs(pre, l + 1, j);
        root.right = dfs(pre, j + 1, r);
        return root;
    }
}
~~~



#### [687. 最长同值路径](https://leetcode.cn/problems/longest-univalue-path/)

~~~
//与124类似
class Solution {
    private int res = 0;
    public int longestUnivaluePath(TreeNode root) {
        if (root == null) return 0;
        dfs(root, root.val);
        return res;
    }
    public int dfs(TreeNode root, int val) {
        if (root == null) return 0;
        int l = dfs(root.left, root.val);
        int r = dfs(root.right, root.val);
        res = Math.max(res, l + r);
        if (root.val == val) return Math.max(l, r) + 1;
        else return 0;
    }
}
~~~



#### [1339. 分裂二叉树的最大乘积](https://leetcode.cn/problems/maximum-product-of-splitted-binary-tree/)

**解法：记忆化dfs**

~~~
class Solution {
    int mod = (int)1e9 + 7;
    long res = 0;
    Map<TreeNode, Long> map = new HashMap<>();
    public int maxProduct(TreeNode root) {
        map.put(null, 0l);
        dfs(root);
        for (TreeNode node : map.keySet()) 
            res = Math.max(res, (map.get(root) - map.get(node)) * map.get(node));
        return (int) (res % mod);
    }

    public void dfs(TreeNode root) {
        if (root == null) return;
        dfs(root.left);
        dfs(root.right);
        map.put(root, root.val * 1l + map.get(root.left) + map.get(root.right));
    }
}
~~~

#### [1373. 二叉搜索子树的最大键值和](https://leetcode.cn/problems/maximum-sum-bst-in-binary-tree/)

~~~~
class Solution {
    int res = 0;
    public class Node {
        boolean isSearch;
        int mx, mn, sum;
        public Node(boolean isSearch, int mx, int mn, int sum) {
            this.isSearch = isSearch;
            this.mx = mx;
            this.mn = mn;
            this.sum = sum;
        }
    }
    public int maxSumBST(TreeNode root) {
        dfs(root);
        return res;
    }
    public Node dfs(TreeNode root) {
        if (root == null) return new Node(true, -100000, 100000, 0);
        var l = dfs(root.left);
        var r = dfs(root.right);
        if (l.isSearch && r.isSearch && l.mx < root.val && r.mn > root.val) {
            res = Math.max(res, root.val + l.sum + r.sum);
            return new Node(true, Math.max(r.mx, root.val), Math.min(l.mn, root.val), root.val + l.sum + r.sum);
        }
        else 
            return new Node(false, 0, 0, 0);
    }
}
~~~~

#### [921. 使括号有效的最少添加](https://leetcode.cn/problems/minimum-add-to-make-parentheses-valid/)

~~~
class Solution {
    public int minAddToMakeValid(String s) {
        int res = 0, cur = 0;
        for (char x : s.toCharArray()) {
            if (x == '(') cur ++ ;
            else {
                if (cur != 0) cur -- ;
                else res ++ ;
            }
        } 
        while (cur -- > 0) 
            res ++ ;
        return res;
    }
}
~~~

#### [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)

~~~
class MedianFinder {
    
    PriorityQueue<Integer> a, b;
    public MedianFinder() {
        b = new PriorityQueue<>();
        a = new PriorityQueue<>((o1, o2) -> o2 - o1);
    }
    
    public void addNum(int num) {
        a.add(num);
        b.add(a.poll());
        if (b.size() > a.size())
            a.add(b.poll());
    }
    
    public double findMedian() {
        if (a.size() == b.size())
            return (a.peek() + b.peek()) / 2.0;
        else
            return a.peek();
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
~~~

#### [856. 括号的分数](https://leetcode.cn/problems/score-of-parentheses/)

~~~
class Solution {
    public int scoreOfParentheses(String s) {
        var stk = new ArrayDeque<Integer>();
        stk.push(0);
        for (char c : s.toCharArray()) {
            if (c == '(') stk.push(0);
            else {
                int cur = stk.pop();
                stk.push(stk.pop() + (cur == 0 ? 1 : cur * 2));
            } 
        }
        return stk.peek();
    }
}
~~~

~~~
class Solution {
    public int scoreOfParentheses(String s) {
        int layer = 0, res = 0;
        for (int i = 0; i < s.length(); i ++ )
            if (s.charAt(i) == '(') layer ++ ;
            else {
                layer -- ;
                if (s.charAt(i - 1) == '(')
                    res += 1 << layer;
            }
        return res;
    }
}
~~~

#### [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)

~~~
//栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」
class Solution {
    public int longestValidParentheses(String s) {
        int res = 0;
        ArrayDeque<Integer> stk = new ArrayDeque<>();
        stk.push(-1);
        for (int i = 0, cur = 0; i < s.length(); i ++ ) {
            if (s.charAt(i) == '(') stk.push(i);
            else {
                stk.pop();
                if (stk.isEmpty())
                    stk.push(i);
                else 
                    res = Math.max(res, i - stk.peek());
            }
        }
        return res;
    }
}
~~~

~~~
class Solution {
    public int longestValidParentheses(String s) {
        int l = 0, r = 0, res = 0;
        for (int i = 0; i < s.length(); i ++ ) {
            if (s.charAt(i) == '(') l ++ ;
            else r ++ ;
            if (l == r) res = Math.max(res, r * 2);
            else if (r > l) r = l = 0;
        }

        l = r = 0;
        for (int i = s.length() - 1; i >= 0; i -- ) {
            if (s.charAt(i) == '(') l ++ ;
            else r ++ ;
            if (l == r) res = Math.max(res, r * 2);
            else if (l > r) r = l = 0;
        }
        return res;
    }
}
~~~

#### [2386. 找出数组的第 K 大和](https://leetcode.cn/problems/find-the-k-sum-of-an-array/)

**解法：堆**

~~~
class Solution {
    public long kSum(int[] nums, int k) {
        int n = nums.length;
        var sum = 0L;
        for (int i = 0; i < n; i ++ )
            if (nums[i] >= 0) sum += nums[i];
            else nums[i] = -nums[i];
        var q = new PriorityQueue<long[]>((a, b) -> Long.compare(b[0], a[0]));
        Arrays.sort(nums);
        q.add(new long[]{sum, 0L});
        while ( -- k > 0) {
            var p = q.poll();
            var s = p[0];
            var i = (int) p[1];
            if (i < nums.length) {
                q.add(new long[]{s - nums[i], i + 1});
                if (i > 0) q.add(new long[]{s + nums[i - 1] - nums[i], i + 1});
            }
        }
        return q.peek()[0];
    }
}
~~~

#### [769. 最多能完成排序的块](https://leetcode.cn/problems/max-chunks-to-make-sorted/)

~~~
class Solution {
    public int maxChunksToSorted(int[] arr) {
        int n = arr.length, res = 0;
        for (int i = 0, max = -1; i < n; i ++ ) {
            max = Math.max(max, arr[i]);
            if (max == i) res ++ ;
        }
        return res;
    }
}
~~~

#### [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

**解法1：排序**

~~~
class Solution {
    public int[][] merge(int[][] arr) {
        if (arr.length == 0 || arr == null) return arr;
        List<int[]> res = new ArrayList<>();
        Arrays.sort(arr, (a, b) -> a[0] - b[0]);
        int st = arr[0][0], end = arr[0][1];
        for (int i = 1; i < arr.length; i ++ ) {
            int a = arr[i][0], b = arr[i][1];
            //当前区间不能合并
            if (a > end) {
                res.add(new int[]{st, end});
                st = a;
                end = b;
            } else {
                //能合并需要更新有端点
                end = Math.max(end, b);
            }
        }
        res.add(new int[]{st, end});
        return res.toArray(new int[res.size()][2]);
    }
}
~~~



**解法2：BitSet位图**

~~~
//运用api
class Solution {
    public int[][] merge(int[][] arr) {
        BitSet bitSet = new BitSet();
        int max = 0;
        for (int[] ss : arr) {
            /*比如[1,4]和[5,6]两个区间在数轴上是不连续的，但在BitSet上却
            是连续的。乘2是为了让它们从BitSet上看也是不连续的*/
            // bitSet.set() 函数 [x,y)
            int temp = ss[1] * 2 + 1;
            bitSet.set(ss[0] * 2, temp, true);
            max = Math.max(max, temp);
        }

        int index = 0, count = 0;
        while (index < max) {
            int start = bitSet.nextSetBit(index);
            int end = bitSet.nextClearBit(start);

            int[] item = {start / 2, (end - 1) / 2};
            arr[count ++ ] = item;
            index = end;
        }

        int[][] ret = new int[count][2];
        for (int i = 0; i < count; i ++ ) {
            ret[i] = arr[i];
        }
        return ret;
    }
}
~~~

~~~
//手动实现
class Solution {
    boolean[] bitSet = new boolean[20010];
    //返回第一个设置为 true 的位的索引，这发生在指定的起始索引或之后的索引上。
    public int nextSetBit(int i) {
        while (bitSet[i] == false) i ++ ;
        return i;
    }
    //返回第一个设置为 false 的位的索引，这发生在指定的起始索引或之后的索引上。
    public int nextClearBit(int i) {
        while (bitSet[i] == true) i ++ ;
        return i;
    }
    public int[][] merge(int[][] arr) {
        int max = 0;
        for (int[] ss : arr) {
            int begin = ss[0] * 2;
            int end = ss[1] * 2 + 1;
            for (int i = begin; i < end; i ++ ) bitSet[i] = true;
            max = Math.max(max, end);
        }

        ArrayList<int[]> res = new ArrayList<>();
        int index = 0;
        //合并区间
        while (index < max) {
            int begin = nextSetBit(index);
            int end = nextClearBit(begin);
 
            res.add(new int[]{begin / 2, (end - 1) / 2};);
            index = end;
        }

        return res.toArray(new int[res.size()][2]);
    }
}
~~~



#### [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

~~~
class Solution { 
    public void nextPermutation(int[] nums) {
        int k = nums.length - 1;
        while (k > 0 && nums[k - 1] >= nums[k]) k -- ;
        if (k == 0) { 
            //不存在下一个排列
            Arrays.sort(nums);
        } else {
            int t = k;
            //找到大于nums[k - 1]的最小值的索引
            while (t < nums.length && nums[t] > nums[k - 1]) t ++ ;
            //交换以后对剩下的数升序排列
            swap(t - 1, k - 1, nums);
            Arrays.sort(nums, k, nums.length);
        }
    }
    public void swap(int a, int b, int[] nums) {
        int t = nums[a];
        nums[a] = nums[b];
        nums[b] = t;
    }
}
~~~

#### [剑指 Offer 66. 构建乘积数组](https://leetcode.cn/problems/gou-jian-cheng-ji-shu-zu-lcof/)

~~~
class Solution {
    public int[] constructArr(int[] a) {
        int n = a.length;
        int[] ans = new int[n];
        //先乘右边的
        for (int i = 0, cnt = 1; i < n; i ++ ) {
            ans[i] = cnt;
            cnt *= a[i];
        }
        //再乘左边的
        for (int i = n - 1, cnt = 1; i >= 0; i -- ) {
            ans[i] *= cnt;
            cnt *= a[i];
        }
        return ans;
    }
}
~~~

#### [581. 最短无序连续子数组](https://leetcode.cn/problems/shortest-unsorted-continuous-subarray/)

**解法：排序**

O($nlogn$),O($n$)

~~~
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        var n = nums.length;
        var temp = nums.clone();
        Arrays.sort(temp);
        int l = 0, r = n - 1;
        while (l < n && temp[l] == nums[l]) l ++ ;
        while (l < r && temp[r] == nums[r]) r -- ;
        return r - l + 1;
    }
}
~~~

**解法：扫描**

O($n$),O($1$)

~~~
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length, l = 0, r = n - 1;
        while (l + 1 < n && nums[l] <= nums[l + 1]) l ++ ;
        while (r > l && nums[r - 1] <= nums[r]) r -- ;
        if (l == r) return 0;
        for (int i = l + 1; i < n; i ++ )
            while (l >= 0 && nums[i] < nums[l])
                l -- ;
        for (int i = r - 1; i >= 0; i -- )
            while (r < n && nums[i] > nums[r])
                r ++ ;
        return r - l - 1;
    }
}
~~~

#### [238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)

~~~
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Arrays.fill(ans, 1);
        int l = 1, r = 1;
        for (int i = 0; i < n; i ++ ) {
            ans[i] *= l;
            l *= nums[i];

            ans[n - i - 1] *= r;
            r *= nums[n - i - 1];
        }
        return ans;
    }
}
~~~

#### [399. 除法求值](https://leetcode.cn/problems/evaluate-division/)

~~~
class Solution {
    //转为多源最短路问题解决a -> b = c, b -> a = 1 / c, a -> c = a -> b * b -> c;
    public double[] calcEquation(List<List<String>> e, double[] val, List<List<String>> qu) {
        Set<String> vers = new HashSet<>();
        Map<String, Map<String, Double>> map = new HashMap<>();
        for (int i = 0; i < e.size(); i ++ ) {
            List<String> list = e.get(i);
            String a = list.get(0), b = list.get(1);
            double c = val[i];
            Map<String, Double> m1 = map.getOrDefault(a, new HashMap<>());
            Map<String, Double> m2 = map.getOrDefault(b, new HashMap<>());
            m1.put(b, c);
            m2.put(a, 1/ c);
            map.put(a, m1);
            map.put(b, m2);
            vers.add(a);
            vers.add(b);
        }
        //弗洛伊德
        for (String k : vers) {
            for (String i : vers) {
                for (String j : vers) {
                    if (map.get(i).containsKey(k) && map.get(k).containsKey(j)) {
                        map.get(i).put(j, map.get(i).get(k) * map.get(k).get(j));
                    }
                }
            }
        }

        double[] ans = new double[qu.size()];
        for (int i = 0; i < qu.size(); i ++ ) {
            List<String> q = qu.get(i);
            String a = q.get(0), b = q.get(1);
            if (map.containsKey(a) && map.get(a).containsKey(b)) ans[i] = map.get(a).get(b);
            else ans[i] = -1.0;
        }
        return ans;
    }
}
~~~

#### [剑指 Offer 45. 把数组排成最小的数](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

~~~
class Solution {
    public String minNumber(int[] nums) {
        List<String> list = new ArrayList<>();
        for (int num : nums) {
            list.add(String.valueOf(num));
        }
        list.sort((o1, o2) -> (o1 + o2).compareTo(o2 + o1));
        return String.join("", list);
    }
}
~~~

#### [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode.cn/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

~~~
class Solution {
    /* 数字范围    数量  位数    占多少位
    1-9        9      1       9
    10-99      90     2       180
    100-999    900    3       2700
    1000-9999  9000   4       36000  ...

    例如 2901 = 9 + 180 + 2700 + 12 即一定是4位数,第12位   n = 12;
    数据为 = 1000 + (12 - 1)/ 4  = 1000 + 2 = 1002
    定位1002中的位置 = (n - 1) %  4 = 3    s.charAt(3) = 2;
    */
    public int findNthDigit(int n) {
        int d = 1;//n所在数字的位数
        long  st = 1;//数字范围开始的第一个数
        long  count = 9;//占多少位
        while (n > count) {
            n -= count;
            d ++ ;
            st *= 10;
            count = d * st * 9;
        }
        long num = st + (n - 1) / d;
        return Long.toString(num).charAt((n - 1) % d) - '0';
    }
}
~~~

#### [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

~~~
class Solution {
    public List<Integer> spiralOrder(int[][] g) {
        int[] dx = new int[]{0, 1, 0, -1}, dy = new int[]{1, 0, -1, 0};
        int m = g.length, n = g[0].length;
        boolean[][] st = new boolean[m][n];
        int t = m * n;
        List<Integer> ans = new ArrayList<>();
        for (int i = 0, x = 0, y = 0, d = 0; i < t; i ++ ) {
            ans.add(g[x][y]);
            st[x][y] = true;
            int a = x + dx[d], b = y + dy[d];
            //检验是否合法，不合法则更新方向，并重新计算下一个位置
            if (a < 0 || a >= m || b < 0 || b >= n || st[a][b]) {
                d ++ ;
                d %= 4;
                a = x + dx[d];
                b = y + dy[d];
            }
            x = a;
            y = b;
        }
        return ans;
    }
}
~~~

#### [59. 螺旋矩阵 II](https://leetcode.cn/problems/spiral-matrix-ii/)

~~~
class Solution {
    public int[][] generateMatrix(int n) {
        int[][] ans = new int[n][n];
        int[] dx = new int[]{0, 1, 0, -1}, dy = new int[]{1, 0, -1, 0};
        for (int i = 0, x = 0, y = 0, d = 0; i < n * n; i ++ ) {
            ans[x][y] = i + 1;
            int a = x + dx[d], b = y + dy[d];
            if (a < 0 || a >= n || b < 0 || b >= n || ans[a][b] != 0) {
                d ++ ;
                d %= 4;
                a = x + dx[d];
                b = y + dy[d];
            }
            x = a;
            y = b;
        }
        return ans;
    }
}
~~~

#### [57. 插入区间](https://leetcode.cn/problems/insert-interval/)

~~~
class Solution {
    public int[][] insert(int[][] arrs, int[] newArr) {
        List<int[]> res = new ArrayList<>();
        boolean has_in = false;
        for (int[] arr : arrs) {
            if (arr[0] > newArr[1]) {
                if (!has_in) {
                    res.add(newArr);
                    has_in = true;
                }
                res.add(arr);
            } else if (arr[1] < newArr[0]) {
                res.add(arr);
            } else {
                newArr[0] = Math.min(newArr[0], arr[0]);
                newArr[1] = Math.max(newArr[1], arr[1]);
            }
        }
        if (!has_in) res.add(newArr);
        return res.toArray(new int[res.size()][2]);
    }
}
~~~

#### [498. 对角线遍历](https://leetcode.cn/problems/diagonal-traverse/)

~~~
class Solution {
    public int[] findDiagonalOrder(int[][] mat) {
        int n = mat.length, m = mat[0].length;
        int[] ans = new int[n * m];
        int idx = 0;
        //以对角线坐标的和枚举
        for (int i = 0; i < n + m - 1; i ++ ) {
            if (i % 2 == 0) {
                for (int j = Math.min(i, n - 1); j >= Math.max(0, 1 - m + i); j -- ) 
                    ans[idx ++ ] = mat[j][i - j];
            } else {
                for (int j = Math.max(0, 1 - m + i); j <= Math.min(i, n - 1); j ++ )
                    ans[idx ++ ] = mat[j][i - j];
            }
        }
        return ans;
    }
}
~~~

#### [6111. 螺旋矩阵 IV](https://leetcode.cn/problems/spiral-matrix-iv/)

~~~
class Solution {
    public int[][] spiralMatrix(int m, int n, ListNode head) {
        int[][] ans = new int[m][n];
        for (int i = 0; i < m; i ++ ) Arrays.fill(ans[i], -2);
        int[] dx = new int[]{0, 1, 0, -1}, dy = new int[]{1, 0, -1, 0};
        for (int i = 0, x = 0, y = 0, d = 0; i < m * n; i ++ ) {
            ans[x][y] = head != null ? head.val : -1;
            if (head != null) head = head.next;
            int a = x + dx[d], b = y + dy[d];
            if (a < 0 || a >= m || b < 0 || b >= n || ans[a][b] != -2) {
                d ++ ;
                d %= 4;
                a = x + dx[d];
                b = y + dy[d];
            }
            x = a;
            y = b;
        }
        return ans;
    }
}
~~~

#### [565. 数组嵌套](https://leetcode.cn/problems/array-nesting/)

~~~左闭右开
class Solution {
    //每个点出度和入读都是1，采取原地标记
    public int arrayNesting(int[] nums) {
        int ans = 0;
        int n = nums.length;
        for (int i = 0; i < n; i ++ ) {
            if (nums[i] == -1) continue;
            int cur = 0, j = i;
            while (nums[j] != -1) {
                cur ++ ;
                int t = nums[j];
                nums[j] = -1;
                j = t;
            }
            ans = Math.max(ans, cur);
        }
        return ans;
    } 
}
~~~

#### [2420. 找到所有好下标](https://leetcode.cn/problems/find-all-good-indices/)

**解法：递推**

~~~
class Solution {
    public List<Integer> goodIndices(int[] nums, int k) {
        int n = nums.length;
        var f = new int[n];
        var g = new int[n];
        Arrays.fill(f, 1);Arrays.fill(g, 1);
        for (int i = 1; i < n; i ++ ) 
            if (nums[i] <= nums[i - 1])
                f[i] = f[i - 1] + 1;
        for (int i = n - 1; i > 0; i -- )
            if (nums[i] >= nums[i - 1]) 
                g[i - 1] = g[i] + 1;
        var res = new ArrayList<Integer>();
        for (int i = k; i < n - k; i ++ )
            if (f[i - 1] >= k && g[i + 1] >= k)
                res.add(i);
        return res;
    }
}
~~~

#### [915. 分割数组](https://leetcode.cn/problems/partition-array-into-disjoint-intervals/)

~~~java
class Solution {
    public int partitionDisjoint(int[] nums) {
        var n = nums.length;
        var l = new int[n];
        var r = new int[n];
        int p = 0, q = 1000010;
        for (int i = 0; i < n; i ++ ) {
            p = Math.max(p, nums[i]);
            l[i] = p;
        }
        for (int i = n - 1; i >= 0; i -- ) {
            q = Math.min(q, nums[i]);
            r[i] = q;
        }
        var res = 0;
        for (int i = 1; i < n; i ++ )
            if (l[i - 1] <= r[i]) {
                res = i;
                break;
            }
        return res;
    }
}
~~~



~~~python
class Solution:
    def partitionDisjoint(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = [0] * n, [0] * n
        p, q = 0, 1000010
        for i in range(0, n):
            p = max(p, nums[i])
            l[i] = p
        for i in range(n - 1, -1, -1):
            q = min(q, nums[i])
            r[i] = q
        res = 0
        for i in range(1, n):
            if l[i - 1] <= r[i]:p'y
                return i
        return -1
~~~



#### [1773. 统计匹配检索规则的物品数量](https://leetcode.cn/problems/count-items-matching-a-rule/)

~~~
class Solution:
    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        res, idx = 0, 0 if ruleKey[0] == 't' else (1 if ruleKey[0] == 'c' else 2)
        return sum(v[idx] == ruleValue for v in items)
~~~

#### [481. 神奇字符串](https://leetcode.cn/problems/magical-string/)

~~~
class Solution {
    public int magicalString(int n) {
        StringBuilder sb = new StringBuilder();
        sb.append("122");
        int num = 1;
        for (int i = 2; sb.length() <= n; i ++ ) {
            char c = sb.charAt(i);
            if (c == '1') sb.append(num);
            else sb.append(num).append(num);
            num ^= 3;
        }
        int res = 0;
        for (int i = 0; i < n; i ++ )
            if (sb.charAt(i) == '1')
                res ++ ;
        return res;
    }
}
~~~

#### [1106. 解析布尔表达式](https://leetcode.cn/problems/parsing-a-boolean-expression/)

~~~
class Solution {
    public char calc(char top, char cur, char op) {
        boolean x = top == 't', y = cur == 't';
        var res = op == '|' ? x | y : x & y;
        return res ? 't' : 'f';
    }
    public boolean parseBoolExpr(String s) {
        ArrayDeque<Character> nums = new ArrayDeque<>();
        ArrayDeque<Character> ops = new ArrayDeque<>();
        var n = s.length();
        for (char c : s.toCharArray()) {
            if (c == ',') continue;
            else if (c == 't' || c == 'f') nums.addLast(c);
            else if (c == '!' || c == '&' || c == '|') ops.addLast(c);
            else if (c == '(') nums.addLast(c);
            else {
                var op = ops.pollLast();
                var cur = nums.pollLast();
                while (!nums.isEmpty() && nums.peekLast() != '(') {
                    cur = calc(nums.pollLast(), cur, op);
                }
                if (op == '!') cur = cur == 't' ? 'f' : 't';
                nums.pollLast();
                nums.addLast(cur);
            }
        }
        return nums.pollLast() == 't';
    }
}
~~~

~~~
class Solution {
    String ex;
    int k;
    public boolean dfs() {
        if (ex.charAt(k) == 't' && k ++ > 0) return true;
        if (ex.charAt(k) == 'f' && k ++ > 0) return false;
        var op = ex.charAt(k);
        k += 2;
        //初始化答案，如果op为|的话需要初始化为false
        var res = op == '|' ? false : true;
        while (ex.charAt(k) != ')') {
            if (ex.charAt(k) == ',') {
                k ++ ;
            } else {
                var t = dfs();
                res = op == '|' ? res | t : res & t;
            }
        }
        k ++ ;
        return op == '!' ? !res : res;
    }
    public boolean parseBoolExpr(String expression) {
        ex = expression;
        k = 0;
        return dfs();
    }
}
~~~

#### [816. 模糊坐标](https://leetcode.cn/problems/ambiguous-coordinates/)

~~~
class Solution {
    String s;
    public List<String> ambiguousCoordinates(String _s) {
        s = _s.substring(1, _s.length() - 1);
        var n = s.length();
        var res = new ArrayList<String>();
        for (var i = 0; i < n - 1; i ++ ) {
            var a = search(0, i);
            var b = search(i + 1, n - 1);
            for (var x : a)
                for (var y : b)
                    res.add("(" + x + ", " + y + ")");
        }
        return res;
    }

    public List<String> search(int start, int end) {
        var res = new ArrayList<String>();
        if (start == end || s.charAt(start) != '0') res.add(s.substring(start, end + 1));
        //枚举小数点
        for (int i = start; i < end; i ++ ) {
            var a = s.substring(start, i + 1);
            var b = s.substring(i + 1, end + 1);
            if (a.length() > 1 && a.charAt(0) == '0') continue;
            if (b.charAt(b.length() - 1) == '0') continue;
            res.add(a + "." + b);
        }
        return res;
    }
}
~~~

#### [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

**解法：中心扩展**

~~~
class Solution {
    public String longestPalindrome(String s) {
        var n = s.length();
        var res = "";
        for (int i = 0; i < 2 * n - 1; i ++ ) {
            int l = i / 2, r = l + i % 2;
            while (l >= 0 && r < n && s.charAt(l) == s.charAt(r)) {
                l -- ;
                r ++ ;
            } 
            if (r - l - 1 >= res.length())
                res = s.substring(l + 1, r);
        }
        return res;
    }
}
~~~

#### [792. 匹配子序列的单词数](https://leetcode.cn/problems/number-of-matching-subsequences/)

~~~
class Solution {
    public int numMatchingSubseq(String s, String[] words) {
        var ss = s.toCharArray();
        var n = ss.length;
        //记录每个位置的下一个某个字母的下标
        var ne = new int[n + 1][26];
        var pos = new int[26];
        Arrays.fill(pos, -1);
        for (var i = 0; i <= n; i ++ ) Arrays.fill(ne[i], -1);
        for (int i = n - 1; i >= 0; i -- ) {
            pos[ss[i] - 'a'] = i;
            ne[i] = pos.clone();
        }
        var res = words.length;
        for (var word : words) {
            var pre = 0;
            for (var c : word.toCharArray()) {
                pre = ne[pre][c - 'a'];
                if (pre == -1) {
                    res -- ;
                    break;
                }
                pre ++ ;
            }
        }
        return res;
    }
}
~~~

#### [460. LFU 缓存](https://leetcode.cn/problems/lfu-cache/)

~~~
class LFUCache {
    int n;
    Block headBlock, tailBlock;
    Map<Integer, Block> blockCache = new HashMap<>();
    Map<Integer, Node> nodeCache = new HashMap<>();

    public LFUCache(int capacity) {
        n = capacity;
        headBlock = new Block(Integer.MAX_VALUE);
        tailBlock = new Block(0);
        headBlock.next = tailBlock;
        tailBlock.pre = headBlock;
    }
    
    private void insert(Block p) {
        Block block = new Block(p.cnt + 1);
        p.pre.next = block;
        block.pre = p.pre;
        block.next = p;
        p.pre = block;
    }

    private void remove(Block p) {
        p.pre.next = p.next;
        p.next.pre = p.pre;
    }

    public int get(int key) {
        Block block = blockCache.get(key);
        if (block == null) return -1;
        Node node = nodeCache.get(key);

        block.remove(node);
        // 需要插入到一个新的 block 中
        if (block.pre.cnt != block.cnt + 1) {
            insert(block);
        }

        // 将当前node 插入到新的 block中
        block.pre.insert(node);
        blockCache.put(key, block.pre);
        if (block.isEmpty()) remove(block);
        return node.val;
    }
    
    public void put(int key, int value) {
        if (n == 0) return;
        if (blockCache.containsKey(key)) {
            nodeCache.get(key).val = value;
            get(key);
        } else {
            if (nodeCache.size() == n) {
                Node node = tailBlock.pre.tail.pre;
                tailBlock.pre.remove(node);
                if (tailBlock.pre.isEmpty()) remove(tailBlock.pre);

                blockCache.remove(node.key);
                nodeCache.remove(node.key);
            }
            Node node = new Node(key, value);
            if (tailBlock.pre.cnt != 1) insert(tailBlock);

            tailBlock.pre.insert(node);
            blockCache.put(key, tailBlock.pre);
            nodeCache.put(key, node);
        }
    }

    public class Node {
        int key, val;
        Node pre, next;
        public Node(int key, int val) {
            this.key = key;
            this.val = val;
        }
    }

    public class Block {
        //使用次数
        int cnt;
        Node head, tail;
        Block pre, next;
        public Block(int cnt) {
            this.cnt = cnt;
            head = new Node(-1, -1);
            tail = new Node(-1, -1);
            head.next = tail;
            tail.pre = head;
        }

        public void remove(Node p) {
            p.pre.next = p.next;
            p.next.pre = p.pre;
        }

        public void insert(Node p) {
            head.next.pre = p;
            p.pre = head;
            p.next = head.next;
            head.next = p;
        }

        public boolean isEmpty() {
            return head.next == tail;
        }
    }
}
~~~



## 数论

#### [319. 灯泡开关](https://leetcode-cn.com/problems/bulb-switcher/)

数论问题

#### [343. 整数拆分](https://leetcode.cn/problems/integer-break/)

数论

#### [6168. 恰好移动 k 步到达某一位置的方法数目](https://leetcode.cn/problems/number-of-ways-to-reach-a-position-after-exactly-k-steps/)

**解法：逆元求组合数**

~~~
class Solution {
    private int mod = (int)1e9 + 7;
    
    public int qmi(int m, int k, int p) {
        long res = 1l % p, t = m * 1l;
        while (k != 0) {
            if ((k & 1) != 0) res = res * t % p;
            t = t * t % p;
            k >>= 1;
        }
        return (int)res;
    }

    public int numberOfWays(int startPos, int endPos, int k) {
        int m = Math.abs(endPos - startPos);
        if ((m - k) % 2 != 0 || k < m) return 0;
        int r = (k + m) / 2;
        long res = 1;
        for (int i = k; i > k - r; i -- ) 
            res = res * i % mod;
        for (int i = 1; i <= r; i ++ ) {
            res = res * qmi(i, mod - 2, mod) % mod;
        }
        return (int)res;
    }
}
~~~

#### [204. 计数质数](https://leetcode.cn/problems/count-primes/)

~~~
class Solution {
    public int countPrimes(int n) {
        var cnt = 0;
        var st = new boolean[n + 1];
        var primes = new int[n];
        for (int i = 2; i < n; i ++ ) {
            if (!st[i]) primes[cnt ++ ] = i;
            for (int j = 0; primes[j] <= n / i; j ++ ) {
                st[primes[j] * i] = true;
                if (i % primes[j] == 0) break;
            }     
        }
        return cnt;
    }
}
~~~

#### [754. 到达终点数字](https://leetcode.cn/problems/reach-a-number/)

根据超出t的步数分类讨论，超出偶数步可将（sum - t）/ 2处取反，奇数步的话看走完下一步超出的是奇数还是偶数，奇数的话位k+1，否则为k+2

~~~
class Solution {
    public int reachNumber(int target) {
        target = Math.abs(target);
        var k = 0;
        while (target > 0) {
            k ++ ;
            target -= k;
        }
        return target % 2 == 0 ? k : k + 1 + k % 2;
    }
}
~~~



## 滑动窗口

#### [594. 最长和谐子序列](https://leetcode-cn.com/problems/longest-harmonious-subsequence/)

Arrays.sort()排序 滑动窗口

#### [438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

#### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

#### [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/)

#### [713. 乘积小于 K 的子数组](https://leetcode-cn.com/problems/subarray-product-less-than-k/)

#### [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

#### [187. 重复的DNA序列](https://leetcode-cn.com/problems/repeated-dna-sequences/)

#### [220. 存在重复元素 III](https://leetcode-cn.com/problems/contains-duplicate-iii/)

>TreeSet的用法
>
>public boolean add(E e)：添加元素
>
>public  boolean addAll(Collection c)：通过集合添加元素
>
>public E ceiling(E e)：返回大于或等于给定键值的最小键值
>
>public void clear()：清空集合
>
>public Object clone()：克隆集合
>
>public Comparator comparator()：用于在此树映射中维护顺序的比较器，如果使用其键的自然顺序，则为null
>
>public boolean contains(Object o) ：是否包含这个元素
>
>public Iterator descendingIterator()：用于按降序迭代元素。
>
>public E first()：获取首点
>
>public E floor(E e)：返回小于或等于给定键值的最大键值
>
>public SortedSet headSet(E toElement)：返回key<=toElement集合
>
>public NavigableSet headSet(E toElement, boolean inclusive)：返回key<=toElement集合，inclusive=true返回的集合在原set中，会包含自己，否则不会包含
>
>public E higher(E e)：返回严格大于给定键值的最小键值
>
>public boolean isEmpty()：判断集合是否为空
>
>public Iterator iterator() ：迭代输出
>
>public E last()：获取最后的值
>
>public E lower(E e)：返回严格小于给定键值的最大键值
>
>public E pollFirst()：获取第一个值并移除第一个值
>
>public E pollLast()：获取最后值并移除这个值
>
>public boolean remove(Object o)：移除元素
>
>public int size() ：当前set容量
>
>public Spliterator spliterator() ： 方法用于拆分set元素，并逐个迭代它们。
>
>public NavigableSet subSet(E fromElement, boolean fromInclusive, E toElement,   boolean toInclusive)：返回from到to之间的值，fromInclusive和toInclusive代表是否包含当前值
>
>public SortedSet subSet(E fromElement, E toElement)：返回from到to之间的值，包含from，不包含to，即[左闭右开）
>
>public SortedSet tailSet(E fromElement)：返回>=fromElement值的集合元素

#### [395. 至少有 K 个重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)

~~~
class Solution {
    public int longestSubstring(String s, int k) {
        int[] map = new int[26];
        int n = s.length();
        int ans = 0;
        for (int p = 1; p <= 26; p ++ ) {
            Arrays.fill(map, 0);   
            // x 代表 [j, i] 区间所有的字符种类数量；
            // y 代表满足「出现次数不少于 k」的字符种类数量
            for (int i = 0, j = 0, x = 0, y = 0; i < n; i ++ ) {
                int u = s.charAt(i) - 'a';
                map[u] ++;
                // 如果添加到 cnt 之后为 1，说明字符总数 +1
                if (map[u] == 1) x ++ ;
                // 当区间所包含的字符种类数量 tot 超过了当前限定的数量 p，
                //那么我们要删除掉一些字母，即「左指针」右移
                if (map[u] == k) y ++ ;
                while (x > p) {
                    int t = s.charAt(j ++ ) - 'a';
                    map[t] -- ;
                    // 如果添加到 cnt 之后为 0，说明字符总数-1
                    if (map[t] == 0) x --;
                    // 如果添加到 cnt 之后等于 k - 1，说明该字符从达标变为不达标，达标数量 - 1
                    if (map[t] == k - 1) y -- ;
                }
                // 当所有字符都符合要求，更新答案
                if (x == y) ans = Math.max(ans, i - j + 1);
            }
        }
        return ans;
    }
}
~~~

#### [1695. 删除子数组的最大得分](https://leetcode-cn.com/problems/maximum-erasure-value/)

#### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

#### [643. 子数组最大平均数 I](https://leetcode-cn.com/problems/maximum-average-subarray-i/)

#### [1004. 最大连续1的个数 III](https://leetcode-cn.com/problems/max-consecutive-ones-iii/)

#### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

#### [1838. 最高频元素的频数](https://leetcode.cn/problems/frequency-of-the-most-frequent-element/)

~~~
class Solution {
    public int maxFrequency(int[] nums, int k) {
        Arrays.sort(nums);
        int res = 0, tem = 0;
        for (int i = 0, j = 0; i < nums.length; i ++ ) {
            while (nums[i] * (i - j) - tem > k) tem -= nums[j ++ ];
            tem += nums[i];
            res = Math.max(res, i - j + 1);
        }
        return res;
    }
}
~~~

#### [1156. 单字符重复子串的最大长度](https://leetcode.cn/problems/swap-for-longest-repeated-character-substring/)

~~~
class Solution {
    public int maxRepOpt1(String s) {
        List<Integer>[] p = new List[26];
        Arrays.setAll(p, e -> new ArrayList<>());
        for (int i = 0; i < s.length(); i ++ )
            p[s.charAt(i) - 'a'].add(i);
        int res = 0;
        for (List<Integer> q : p) {
            //中间有一个空位
            for (int i = 0, j = 0; i < q.size(); i ++ ) {
                //中间多于一个空间
                while (q.get(i) - q.get(j) > i - j + 1) j ++ ;
                int t = q.get(i) - q.get(j) + 1;
                if (i + 1 < q.size() || j > 0) {
                    res = Math.max(res, t);
                }
            }

            //中间没有空位
            for (int i = 0, j = 0; i < q.size(); i ++ ) {
                while (q.get(i) - q.get(j) > i - j) j ++ ;
                int t = q.get(i) - q.get(j) + 1;
                if (i + 1 < q.size() || j > 0) t ++ ;
                res = Math.max(res, t);
            }
        }
        return res;
    }
}
~~~



## 双指针

#### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

**解法：排序 + 双指针**

~~~
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n - 2; i ++ ) {
            //nums[i]如果大于0则三数之和必大于0
            if (nums[i] > 0) break;
            //去重
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            for (int j = i + 1, k = n - 1; j < k; j ++ ) {
                List<Integer> tem = new ArrayList<>();
                //去重
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                while (k > j && nums[i] + nums[j] + nums[k] > 0) k -- ;
                //没有合适的答案
                if (k == j) continue;
                if (nums[i] + nums[j] + nums[k] == 0) {
                    tem.add(nums[i]);tem.add(nums[j]);tem.add(nums[k]);
                    ans.add(tem);
                }
            }
        }
        return ans;
    }
}
~~~

#### 

#### [18. 四数之和](https://leetcode-cn.com/problems/4sum/)

#### [1574. 删除最短的子数组使剩余数组有序](https://leetcode.cn/problems/shortest-subarray-to-be-removed-to-make-array-sorted/)

#### [75. 颜色分类](https://leetcode.cn/problems/sort-colors/)

**解法1：常规排序**

**解法2：三指针排序**

~~~
class Solution {
    public void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }
    //j指向0的部分，k指向2的部分
    public void sortColors(int[] nums) {
        for (int i = 0, j = 0, k = nums.length - 1; i <= k; ) {
            if (nums[i] == 2) swap(nums, i ++ , k -- );
            else if (nums[i] == 0) swap(nums, i ++ , j ++ );
            else i ++ ;
        }
    }
}
~~~

#### 

#### [264. 丑数 II](https://leetcode.cn/problems/ugly-number-ii/)

~~~
//写法1，容量未固定内存消耗大
class Solution {
    public int nthUglyNumber(int n) {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        for (int i = 0, j = 0, k = 0; list.size() < n;) {
            int t = Math.min(list.get(i) * 2, Math.min(list.get(j) * 3, list.get(k) * 5));
            list.add(t);
            if (list.get(i) * 2 == t) i ++ ;
            if (list.get(j) * 3 == t) j ++ ;
            if (list.get(k) * 5 == t) k ++ ;
        }
        //for (int i : list) System.out.println(i);
        return list.get(n - 1);
    }
}
~~~

~~~
//三指针多路归并
//写法2固定容量，比写法1的动态扩容内存少一点
//s[i]表示第i + 1个丑数
//idx表示当前正在生成的丑数的下标
class Solution {
    public int nthUglyNumber(int n) {
        int[] s = new int[n];
        s[0] = 1;
        for (int i = 0, j = 0, k = 0, idx = 1; idx < n; idx ++ ) {
            int t = Math.min(s[i] * 2, Math.min(s[j] * 3, s[k] * 5));
            s[idx] = t;
            if (s[i] * 2 == t) i ++ ;
            if (s[j] * 3 == t) j ++ ;
            if (s[k] * 5 == t) k ++ ;
        }
        return s[n - 1];
    }
}
~~~

#### [16. 最接近的三数之和](https://leetcode.cn/problems/3sum-closest/)

~~~
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int ans = nums[0] + nums[1] + nums[n - 1];
        for (int i = 0; i < n; i ++ ) {
            int l = i + 1, r = n - 1;
            while (l < r) {
                int t = nums[i] + nums[l] + nums[r];
                if (Math.abs(t - target) < Math.abs(ans - target)) ans = t;
                if (t < target) l ++ ;
                else if (t > target) r --;
                else return t;     
            }
        }
        return ans;
    }
}
~~~

#### [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)

~~~
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int ans = Integer.MAX_VALUE;
        int n = nums.length;
        for (int i = 0, j = 0, count = 0; i < n; i ++ ) {
            count += nums[i];
            while (count >= target) {
                ans = Math.min(i - j + 1, ans);
                count -= nums[j ++ ];
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }
}
~~~

#### [1498. 满足条件的子序列数目](https://leetcode.cn/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/)

~~~
class Solution {
    public int numSubseq(int[] nums, int target) {
        int mod = (int)1e9 + 7;
        int n = nums.length;
        int[] f = new int[n];
        //预处理2的n次方
        for (int i = 0; i < n; i ++ ) {
            if (i == 0) f[i] = 1;
            else f[i] = (f[i - 1] << 1) % mod;
        }
        long ans = 0;
        Arrays.sort(nums);
        // 2. 双指针i和j表示最小元素和最大元素下标
        // 例如j为1，i为5，则j的右边有5-1=4个元素，以j为最小元素的子序列，就是这4个元素的子集，共有2的4次方个
        for (int i = n - 1, j = 0; i >= j;) {
            if (nums[i] + nums[j] > target) i -- ;
            else {
                ans = (ans + f[i - j]) % mod;
                j ++ ;
            }
        }
        return (int) ans;
    }
}
~~~

#### [6117. 坐上公交的最晚时间](https://leetcode.cn/problems/the-latest-time-to-catch-a-bus/)

~~~
class Solution {
    public int latestTimeCatchTheBus(int[] buses, int[] passengers, int capacity) {
        int n = buses.length, m = passengers.length;
        Arrays.sort(buses);
        Arrays.sort(passengers);
        Set<Integer> s = new HashSet<>();
        for (int x : passengers) s.add(x);
        int ans = 0;
        //双指针模拟上车 i 表示车， j 表示乘客
        for(int i = 0, j = 0; i < buses.length; i ++) {
            int cur = 0;
            while(cur < capacity && j < passengers.length && passengers[j] <= buses[i]) {  
                // 如果车 i 还有位置
                if (!s.contains(passengers[j] - 1) && passengers[j] - 1 <= buses[i]) {  
                    // 如果可以在某乘客前上车(passengers[j] - 1)
                    ans = Math.max(ans, passengers[j] - 1);
                }
                j ++ ;
                cur ++ ;
            }
            // 是否可以卡点上车(车来了再上车)
            if (cur < capacity && !s.contains(buses[i])){
                ans = Math.max(ans, buses[i]);
            }
        }
        return ans;
    }
}
~~~

#### [777. 在LR字符串中交换相邻字符](https://leetcode.cn/problems/swap-adjacent-in-lr-string/)

~~~
class Solution {
    public boolean canTransform(String start, String end) {
        char[] a = start.toCharArray(), b = end.toCharArray();
        int n = a.length;
        int j = 0;
        for (int i = 0; i < n; i ++ ) {
            if (a[i] == 'X') continue;
            while (j < n && b[j] == 'X') j ++ ;
            if (j == n) return false;
            if (a[i] != b[j]) return false;
            if (a[i] == 'L' && i < j) return false;
            if (a[i] == 'R' && i > j) return false;
            j ++ ;
        }
        for (int i = j; i < n; i ++ ) 
            if (b[i] != 'X') 
                return false;
        return true;
    }
}
~~~



## 图论

#### [面试题 08.10. 颜色填充](https://leetcode-cn.com/problems/color-fill-lcci/)

#### [1034. 边界着色](https://leetcode-cn.com/problems/coloring-a-border/)

#### [882. 细分图中的可到达节点](https://leetcode.cn/problems/reachable-nodes-in-subdivided-graph/)

**解法：dijkstra**

~~~
class Solution {
    public int reachableNodes(int[][] edges, int maxMoves, int n) {
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (var e : edges) {
            int a = e[0], b = e[1], v = e[2];
            g[a].add(new int[]{b, v + 1});
            g[b].add(new int[]{a, v + 1});
        }

        var dist = dijkstra(g, 0);
        var res = 0;
        for (var d : dist)
            if (d <= maxMoves)
                res ++ ;
        for (var e : edges) {
            int a = e[0], b = e[1], v = e[2];
            int x = Math.max(maxMoves - dist[a], 0);
            int y = Math.max(maxMoves - dist[b], 0);
            res += Math.min(x + y, v);
        }
        return res;
    }

    public int[] dijkstra(List<int[]>[] g, int start) {
        var dist = new int[g.length];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[start] = 0;
        var pq = new PriorityQueue<int[]>((a, b) -> a[1] - b[1]);
        pq.add(new int[]{start, 0});
        while (!pq.isEmpty()) {
            var p = pq.poll();
            int x = p[0], d = p[1];
            if (d > dist[x]) continue;
            for (var e : g[x]) {
                int y = e[0];
                int newDist = d + e[1];
                if (newDist < dist[y]) {
                    dist[y] = newDist;
                    pq.offer(new int[]{y, newDist});
                }
            }
        }
        return dist;
    }
}
~~~



## DP

#### [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

~~~
class Solution {
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i ++ ) {
            //要么从上个位置转移过来，要么从上上个位置转移过来
            dp[i] = dp[i - 2] + dp[i - 1];
        }
        return dp[n];
    }
}
~~~

#### [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/)

~~~
class Solution {
    public int minPathSum(int[][] g) {
        int n = g.length, m = g[0].length;
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                if (i == 0 && j == 0) continue;
                else if (i == 0) g[i][j] += g[i][j - 1];
                else if (j == 0) g[i][j] += g[i - 1][j];
                else g[i][j] += Math.min(g[i][j - 1], g[i - 1][j]);
            }
        }
        return g[n - 1][m - 1];
    }
}
~~~

#### [62. 不同路径](https://leetcode.cn/problems/unique-paths/)

~~~
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i ++ ) {
            for (int j = 0; j < n; j ++ ) {
                if (i == 0 && j == 0) dp[i][j] = 1;
                else {
                    if (i > 0) dp[i][j] = dp[i][j] + dp[i - 1][j];
                    if (j > 0) dp[i][j] = dp[i][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }
}
~~~

#### [63. 不同路径 II](https://leetcode.cn/problems/unique-paths-ii/)

~~~
class Solution {
    public int uniquePathsWithObstacles(int[][] g) {
        int m = g.length, n = g[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i ++ ) {
            for (int j = 0; j < n; j ++ ) {
                if (g[i][j] == 0) {
                    if (i == 0 && j == 0) dp[i][j] = 1;
                    else {
                        if (i > 0) dp[i][j] = dp[i][j] + dp[i - 1][j];
                        if (j > 0) dp[i][j] = dp[i][j] + dp[i][j - 1];
                    }
                }
            }
        }
        return dp[m - 1][n - 1];
    }
}
~~~

#### [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/)

#### [剑指 Offer 46. 把数字翻译成字符串](https://leetcode.cn/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

~~~
class Solution {
    public int translateNum(int num) {
        String str = String.valueOf(num);
        int n = str.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i ++ ) {
            String sub = str.substring(i - 2, i);
            if (sub.compareTo("10") >= 0 && sub.compareTo("25") <= 0) {
                dp[i] = dp[i - 1] + dp[i - 2];
            } else {
                //不能从下面两格转移上来
                dp[i] = dp[i - 1];
            }
        }
        return dp[n];
    }
}
~~~

#### [62. 不同路径](https://leetcode.cn/problems/unique-paths/)

~~~
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i ++ ) {
            for (int j = 0; j < n; j ++ ) {
                if (i == 0 && j == 0) dp[i][j] = 1;
                else {
                    if (i > 0) dp[i][j] = dp[i][j] + dp[i - 1][j];
                    if (j > 0) dp[i][j] = dp[i][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }
}
~~~

#### [63. 不同路径 II](https://leetcode.cn/problems/unique-paths-ii/)

~~~
class Solution {
    public int uniquePathsWithObstacles(int[][] g) {
        int m = g.length, n = g[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i ++ ) {
            for (int j = 0; j < n; j ++ ) {
                if (g[i][j] == 0) {
                    if (i == 0 && j == 0) dp[i][j] = 1;
                    else {
                        if (i > 0) dp[i][j] = dp[i][j] + dp[i - 1][j];
                        if (j > 0) dp[i][j] = dp[i][j] + dp[i][j - 1];
                    }
                }
            }
        }
        return dp[m - 1][n - 1];
    }
}
~~~

#### [198. 打家劫舍](https://leetcode.cn/problems/house-robber/)

~~~
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(dp[0], nums[1]);
        for (int i = 2; i < n; i ++ ) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[n - 1];
    }
}
~~~

#### [213. 打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/)

~~~
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        if (n == 1) return nums[0];
        int res = 0;
        //记录偷第一个
        int[] dp1 = new int[n];
        //记录不偷第一个
        int[] dp2 = new int[n];
        dp1[0] = nums[0];
        dp1[1] = Math.max(nums[0], nums[1]);
        dp2[0] = 0;
        dp2[1] = nums[1];
        for (int i = 2; i < n; i ++ ) {
            if (i == n - 1) {
                //偷了第一个就不能偷最后一个
                dp1[i] = dp1[i - 1];
                dp2[i] = Math.max(dp2[i - 1], dp2[i - 2] + nums[i]);
            } else {
                dp1[i] = Math.max(dp1[i - 1], dp1[i - 2] + nums[i]);
                dp2[i] = Math.max(dp2[i - 1], dp2[i - 2] + nums[i]);
            }
        }
        return Math.max(dp1[n - 1], dp2[n - 1]);
    }
}
~~~

#### [337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/)

~~~
//树形dp
/*我们使用一个大小为 2 的数组来表示 int[] res = new int[2] 0 代表不偷，1 代表偷
任何一个节点能偷到的最大钱的状态可以定义为
当前节点选择不偷：当前节点能偷到的最大钱数 = 左孩子能偷到的钱 + 右孩子能偷到的钱
当前节点选择偷：当前节点能偷到的最大钱数 = 左孩子选择自己不偷时能得到的钱 + 右孩子选择不偷时能得到的钱 + 当前节点的钱数
root[0] = Math.max(rob(root.left)[0], rob(root.left)[1]) + Math.max(rob(root.right)[0], rob(root.right)[1])
root[1] = rob(root.left)[0] + rob(root.right)[0] + root.val;
*/
class Solution {
    public int rob(TreeNode root) {
        int[] f = dfs(root);
        return Math.max(f[0], f[1]);
    }
    public int[] dfs(TreeNode root) {
        if (root == null) return new int[]{0, 0};
        int[] res = new int[2];
        int[] l = dfs(root.left);
        int[] r = dfs(root.right);

        res[0] = Math.max(l[0], l[1]) + Math.max(r[0], r[1]);
        res[1] = l[0] + r[0] + root.val;
        return res;
    }
}
~~~

#### [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

~~~
//O(n * n)
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        int ans = 1;
        //dp[i]表示以nums[i]结尾的最长递增子序列的长度
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        int res = 1;
        for (int i = 1; i < n; i ++ ) {
            for (int j = 0; j < i; j ++ ) {
                if (nums[j] < nums[i])
                    dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }
}
~~~

~~~
//O(n * logn)
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        //dp[i]表示最长子序列长度为i时的最小的结尾num值
        int[] dp = new int[n + 1];
        dp[1] = nums[0];
        int res = 1;
        for (int i = 1; i < n; i ++ ) {
            int l = 0, r = res;
            while (l < r) {
                int mid = l + r + 1 >> 1;
                if (dp[mid] < nums[i]) l = mid;
                else r = mid - 1;
            }
            dp[r + 1] = nums[i];
            res = Math.max(res, r + 1);
        }
        return res;
    }
}
~~~



#### [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)

~~~
class Solution {
    public int maxProduct(int[] nums) {
        int res = nums[0];
        int f = nums[0], g = nums[0];
        for (int i = 1; i < nums.length; i ++ ) {
            int a = nums[i], fa = a * f, ga = a * g;
            f = Math.max(a, Math.max(fa, ga));
            g = Math.min(a, Math.min(fa, ga));
            res = Math.max(res, f);
        }
        return res;
    }
}
~~~

#### [139. 单词拆分](https://leetcode.cn/problems/word-break/)

~~~
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        //dp[i]表示字符串s的前i个字符能否拆分成wordDict
        boolean[] dp = new boolean[n + 1];
        Set<String> set = new HashSet<>();
        dp[0] = true;
        for (String word : wordDict) set.add(word);
        for (int i = 1; i <= n; i ++ ) {
            for (int j = 0; j < i; j ++ ) {
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
}
~~~

#### [剑指 Offer 42. 连续子数组的最大和](https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

~~~
class Solution {
    public int maxSubArray(int[] nums) {
        int f = -101;
        int ans = -101;
        for (int i : nums) {
            f = Math.max(i, f + i);
            ans = Math.max(f, ans);
        }
        return ans;
    }
}
~~~

#### [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

~~~
class Solution {
    public int maxSubArray(int[] nums) {
        int ans = Integer.MIN_VALUE;
        for (int i = 0, last = 0; i < nums.length; i ++ ) {
            last = nums[i] + Math.max(last, 0);
            ans = Math.max(ans, last);
        }
        return ans;
    }
}
~~~



#### [1359. 有效的快递序列数目](https://leetcode.cn/problems/count-all-valid-pickup-and-delivery-options/)

~~~
class Solution {
    static final int MOD = (int)1e9 + 7;
    public int countOrders(int n) {
        long[] f = new long[n + 1];
        f[1] = 1;
        for (int i = 2; i <= n; i ++ ) {
            long x = 2l * i - 1;
            f[i] = (x * x + x) / 2 * f[i - 1];
            f[i] %= MOD;
        }
        return (int)f[n];
    }
}
~~~

**优化空间**

~~~
class Solution {
    static final int MOD = (int)1e9 + 7;
    public int countOrders(int n) {
        long res = 1;
        for (int i = 2; i <= n; i ++ ) {
            res *= i * (2 * i - 1l);
            res %= MOD;
        }
        return (int)res;
    }
}
~~~

#### [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

~~~
class Solution {
    public int longestCommonSubsequence(String a, String b) {
        int n = a.length(), m = b.length();
        //f[i][j]表示第一个字符串的前i个字符和第二个字符串的前j个字符的最长公共子序列
        int[][] f = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i ++ ) {
            for (int j = 1; j <= m; j ++ ) {
                f[i][j] = Math.max(f[i - 1][j], f[i][j - 1]);
                if (a.charAt(i - 1) == b.charAt(j - 1))
                    f[i][j] = Math.max(f[i][j], f[i - 1][j - 1] + 1);
            }
        }
        return f[n][m];
    }
}
~~~

#### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

**解法：记忆化搜索**

```
class Solution {
    int[] map;
    public int coinChange(int[] coins, int amount) {
        if (amount == 0) return 0;
        map = new int[amount + 1];
        return dfs(coins, amount);
    }

    public int dfs(int[] coins, int rem) {
        if (rem < 0) return -1;
        if (rem == 0) return 0;
        if (map[rem] != 0) return map[rem];
        int min = Integer.MAX_VALUE;
        for (int x : coins) {
            int res = dfs(coins, rem - x);
            if (res >= 0 && res < min)
                min = res + 1;
        }
        return map[rem] = min == Integer.MAX_VALUE ? -1 : min;
    } 
}
```
**解法：动态规划**

~~~
class Solution {
    public int coinChange(int[] coins, int m) {
        int[] f = new int[m + 1];
        Arrays.fill(f, 10010);
        f[0] = 0; 
        for (int v : coins)
            for (int j = v; j <= m; j ++ )
                f[j] = Math.min(f[j - v] + 1, f[j]);
        return f[m] == 10010 ? -1 : f[m];
    }
}
~~~

#### [940. 不同的子序列 II](https://leetcode.cn/problems/distinct-subsequences-ii/)

**解法：序列dp**

~~~
class Solution {
    public int distinctSubseqII(String s) {
        int mod = (int) 1e9 + 7;
        var n = s.length();
        //f[i][j]表示前i个字符j结尾的个数
        var f = new int[n + 1][26];
        for (int i = 1; i <= n; i ++ ) {
            int c = s.charAt(i - 1) - 'a';
            for (int j = 0; j < 26; j ++ ) {
                if (c != j) {
                    f[i][j] = f[i - 1][j];
                } else {
                    int cur = 1;
                    for (int k = 0; k < 26; k ++ )
                        cur = (cur + f[i - 1][k]) % mod;
                    f[i][j] = cur;
                }
            }
        }
        var res = 0;
        for (int i = 0; i < 26; i ++ ) res = (res + f[n][i]) % mod;
        return res;
    }
}
~~~

**优化**

~~~
class Solution {
    public int distinctSubseqII(String s) {
        int mod = (int) 1e9 + 7;
        var n = s.length();
        var f = new int[26];
        var res = 0;
        for (int i = 0; i < n; i ++ ) {
            var c = s.charAt(i) - 'a';
            //总数中不包含f[c]的个数，避免重复
            var other = res - f[c];
            f[c] = res + 1;
            res = ((f[c] + other) % mod + mod) % mod;
        }
        return res;
    }
}
~~~

#### [1235. 规划兼职工作](https://leetcode.cn/problems/maximum-profit-in-job-scheduling/)

**解法：序列DP&二分**

~~~
class Solution {
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        var job = new ArrayList<int[]>();
        var n = startTime.length;
        // f[i]为考虑前 i 个工作，所能取得的最大收益
        var f = new int[n + 10];
        for (var i = 0; i < n; i ++ )
            job.add(new int[]{startTime[i], endTime[i], profit[i]});
        Collections.sort(job, (a, b) -> a[1] - b[1]);
        for (int i = 1; i <= n; i ++ ) {
            var info = job.get(i - 1);
            int a = info[0], b = info[1], c = info[2];
            f[i] = Math.max(f[i - 1], c);
            int l = 0, r = i - 1;
            while (l < r) {
                var mid = l + r + 1 >> 1;
                if (job.get(mid)[1] <= a) l = mid;
                else r = mid - 1; 
            }
            if (job.get(l)[1] <= a) f[i] = Math.max(f[i], f[l + 1] + c);
        }
        return f[n];
    }
}
~~~

#### [1668. 最大重复子字符串](https://leetcode.cn/problems/maximum-repeating-substring/)

**解法：动态规划**

~~~~
class Solution {
    public int maxRepeating(String a, String b) {
        int n = a.length(), m = b.length();
        var f = new int[n];
        if (n < m) return 0;
        out:for (int i = m - 1; i < n; i ++ ) {
            boolean flag = true;
            for (int j = 0; j < m; j ++ ) {
                if (a.charAt(i - m + 1 + j) != b.charAt(j)) {
                    continue out;
                }
            }
            f[i] = (i == m - 1 ? 0 : f[i - m]) + 1;
        }
        return Arrays.stream(f).max().getAsInt();
    }
}
~~~~



#### [808. 分汤](https://leetcode.cn/problems/soup-servings/)

~~~
class Solution {
    public int g(int x) {
        return Math.max(x, 0);
    }
    public double soupServings(int n) {
        n = (n + 25 - 1) / 25;
        if (n >= 500) return 1.0;
        //  f[i][j] 为 汤A 剩余 i 毫升，汤B 剩余 j 毫升时的最终概率
        var f = new double[n + 1][n + 1];
        for (var i = 0; i <= n; i ++ ) {
            for (var j = 0; j <= n; j ++ ) {
                if (i == 0 && j == 0) f[i][j] = 0.5;
                else if (i == 0 && j != 0) f[i][j] = 1;
                else if (i != 0 && j == 0) f[i][j] = 0;
                else {
                    f[i][j] += f[g(i - 4)][j];
                    f[i][j] += f[g(i - 3)][g(j - 1)];
                    f[i][j] += f[g(i - 2)][g(j - 2)];
                    f[i][j] += f[g(i - 1)][g(j - 3)];
                    f[i][j] /= 4;
                }
            }
        }
        return f[n][n];
    }
}
~~~



## BFS

#### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

**解法：BFS**

~~~
class Solution {
    public int ladderLength(String start, String end, List<String> bank) {
        var set = new HashSet<String>();
        var map = new HashMap<String, Integer>();
        var q = new ArrayDeque<String>();
        for (String x : bank) set.add(x);
        map.put(start, 0);
        q.add(start);
        while (!q.isEmpty()) {
            var s = q.poll();
            var cs = s.toCharArray();
            var step = map.get(s);
            for (int i = 0; i < cs.length; i ++ ) {
                for (char c = 'a'; c <= 'z'; c ++ ) {
                    if (c == cs[i]) continue;
                    var t = cs[i];
                    cs[i] = c;
                    var sub = new String(cs);
                    if (set.contains(sub) && !map.containsKey(sub)) {
                        map.put(sub, step + 1);
                        if (end.equals(sub))
                            return map.get(sub) + 1;
                        q.add(sub);
                    }
                    cs[i] = t;
                }
            }
        }
        return 0;
    }
}
~~~

朴素的 BFS 可能会带来「搜索空间爆炸」的情况。

想象一下，如果我们的 wordList 足够丰富（包含了所有单词），对于一个长度为 10 的 beginWord 替换一次字符可以产生 10 * 25 个新单词（每个替换点可以替换另外 25 个小写字母），第一层就会产生 250 个单词；第二层会产生超过 6 * 10^4 个新单词 ...

随着层数的加深，这个数字的增速越快，这就是「搜索空间爆炸」问题。

**在朴素的 BFS 实现中，空间的瓶颈主要取决于搜索空间中的最大宽度。**

「双向 BFS」 可以很好的解决这个问题：

**同时从两个方向开始搜索，一旦搜索到相同的值，意味着找到了一条联通起点和终点的最短路径。**

「双向 BFS」的基本实现思路如下：

1、创建「两个队列」分别用于两个方向的搜索；
2、创建「两个哈希表」用于「解决相同节点重复搜索」和「记录转换次数」；
3、为了尽可能让两个搜索方向“平均”，每次从队列中取值进行扩展时，先判断哪个队列容量较少；
4、如果在搜索过程中「搜索到对方搜索过的节点」，说明找到了最短路径。

「双向 BFS」基本思路对应的伪代码大致如下：

~~~
d1、d2 为两个方向的队列
m1、m2 为两个方向的哈希表，记录每个节点距离起点的
    
// 只有两个队列都不空，才有必要继续往下搜索
// 如果其中一个队列空了，说明从某个方向搜到底都搜不到该方向的目标节点
while(!d1.isEmpty() && !d2.isEmpty()) {
    if (d1.size() <= d2.size()) {
        update(d1, m1, m2);
    } else {
        update(d2, m2, m1);
    }
}

// update 为将当前队列 d 中包含的元素取出，进行「一次完整扩展」的逻辑（按层拓展）
void update(Deque d, Map cur, Map other) {}
~~~

**解法：双向BFS**

~~~
class Solution {
    Set<String> set;
    public int bfs(ArrayDeque<String> q, Map<String, Integer> m1, Map<String, Integer> m2) {
        var size = q.size();
        while (size -- > 0) {
            var s = q.poll();
            var cs = s.toCharArray();
            var step = m1.get(s);
            for (int i = 0; i < cs.length; i ++ ) {
                for (char c = 'a'; c <= 'z'; c ++ ) {
                    if (c == cs[i]) continue;
                    var t = cs[i];
                    cs[i] = c;
                    var sub = new String(cs);
                    if (set.contains(sub) && !m1.containsKey(sub)) {
                        if (m2.containsKey(sub))
                            return step + 1 + m2.get(sub);
                        else {
                            m1.put(sub, step + 1);
                            q.add(sub);
                        }
                    }
                    cs[i] = t;
                }
            }
        }
        return -1;
    }

    public int ladderLength(String start, String end, List<String> bank) {
        set = new HashSet<String>();
        var q1 = new ArrayDeque<String>();
        var q2 = new ArrayDeque<String>();
        var m1 = new HashMap<String, Integer>();
        var m2 = new HashMap<String, Integer>();
        for (String x : bank) set.add(x);
        if (!set.contains(end)) return 0;
        m1.put(start, 0); m2.put(end, 0);
        q1.add(start); q2.add(end);
        while (!q1.isEmpty() && !q2.isEmpty()) {
            var t = -1;
            if (q1.size() <= q2.size()) t = bfs(q1, m1, m2);
            else t = bfs(q2, m2, m1);
            if (t != -1) return t + 1;
        }
        return 0;
    }
}
~~~



#### [433. 最小基因变化](https://leetcode-cn.com/problems/minimum-genetic-mutation/)

BFS

~~~
class Solution {
    public int minMutation(String start, String end, String[] bank) {
        Set<String> set = new HashSet<>();
        for (String s : bank) set.add(s);
        Map<String,Integer> map = new HashMap<>();
        Queue<String> q = new LinkedList<>();
        char[] item = new char[]{'A','C','G','T'};
        q.add(start);
        map.put(start, 0);
        while (!q.isEmpty()) {
            int size = q.size();
            while (size -- > 0) {
                String s = q.poll();
                char[] cs = s.toCharArray();
                int step = map.get(s);
                for (int i = 0; i < 8; i ++ ) {
                    for (char c : item) {
                        if (c == cs[i]) continue;
                        char[] st = cs.clone();
                        st[i] = c;
                        String sub = String.valueOf(st);
                        if (!set.contains(sub)) continue;
                        if (map.containsKey(sub)) continue;
                        if (sub.equals(end)) return step + 1;
                        q.add(sub);
                        map.put(sub, step + 1);
                    }
                }
            }
        }
        return -1;
    }
}
~~~

双向BFS

~~~
class Solution {
    Set<String> set = new HashSet<>();
    char[] item = new char[]{'A','C','G','T'};
    public int minMutation(String s, String e, String[] bank) {
        for (String t : bank) set.add(t);
        Queue<String> q1 = new LinkedList<>(), q2 = new LinkedList<>();
        Map<String, Integer> m1 = new HashMap<>(), m2 = new HashMap<>();
        q1.add(s);
        q2.add(e);
        m1.put(s, 0);
        m2.put(e,0); 
        int ans = -1;
        while (!q1.isEmpty() && !q2.isEmpty()) {
            int t = -1;
            if (q1.size() <= q2.size()) {
                t = update(q1, m1, m2);
            } else {
                t = update(q2, m2, m1);
            }
            if (t != -1) ans = t;
        }
        return ans;
    }
    public int update(Queue<String> q, Map<String, Integer> cur, Map<String, Integer> other) {
        int size = q.size();
        while (size -- > 0) {
            String s = q.poll();
            char[] cs = s.toCharArray();
            for (int i = 0; i < 8; i ++ ) {
                for (char c : item) {
                    char[] ss = cs.clone();
                    ss[i] = c;
                    String sub = String.valueOf(ss);
                    if (set.contains(sub) && !cur.containsKey(sub)) {
                        if (other.containsKey(sub)) {
                            return cur.get(s) + 1 + other.get(sub);
                        } else {
                            q.add(sub);
                            cur.put(sub, cur.get(s) + 1);
                        }
                    }
                }
            } 
        }
        return -1;
    }
}
~~~

#### [6081. 到达角落需要移除障碍物的最小数目](https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/)

~~~
class Solution {
    final static int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    public int minimumObstacles(int[][] g) {
        int m = g.length, n = g[0].length;
        int[][] dis = new int[m][n];
        for (int[] d : dis) Arrays.fill(d, Integer.MAX_VALUE);
        dis[0][0] = 0;
        Deque<int[]> q = new LinkedList<>();
        q.addFirst(new int[]{0, 0});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            int x = p[0], y = p[1];
            for (int[] d : dirs) {
                int nx = d[0] + x, ny = d[1] + y;
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int t = g[nx][ny];
                    if (dis[x][y] + t < dis[nx][ny]) {
                        dis[nx][ny] = dis[x][y] + t;
                        if (t == 0) q.addFirst(new int[]{nx, ny});
                        else q.addLast(new int[]{nx, ny});
                    }
                }
            }
        }
        return dis[m - 1][n - 1];
    }
}class Solution {
    final static int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    public int minimumObstacles(int[][] g) {
        int m = g.length, n = g[0].length;
        int[][] dis = new int[m][n];
        for (int[] d : dis) Arrays.fill(d, Integer.MAX_VALUE);
        dis[0][0] = 0;
        Deque<int[]> q = new LinkedList<>();
        q.addFirst(new int[]{0, 0});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            int x = p[0], y = p[1];
            for (int[] d : dirs) {
                int nx = d[0] + x, ny = d[1] + y;
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int t = g[nx][ny];
                    if (dis[x][y] + t < dis[nx][ny]) {
                        dis[nx][ny] = dis[x][y] + t;
                        //优先走短路，利用双端队列的性质极大地优化了时间复杂度
                        if (t == 0) q.addFirst(new int[]{nx, ny});
                        else q.addLast(new int[]{nx, ny});
                    }
                }
            }
        }
        return dis[m - 1][n - 1];
    }
}
~~~

#### [116. 填充每个节点的下一个右侧节点指针](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)

~~~
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}
    
    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};
*/

class Solution {
    public Node connect(Node root) {
        if (root == null) return root;
        Node last = root;
        while (last.left != null) {
            for (Node p = last; p != null; p = p.next) {
                p.left.next = p.right;
                if (p.next != null) p.right.next = p.next.left;
            }
            //从下一层的左边遍历
            last = last.left;
        }
        return root;
    }
}
~~~

#### [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/)

~~~
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}
    
    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};
*/

class Solution {
    public Node connect(Node root) {
        if (root == null) return root;
        Node cur = root;
        while (cur != null) {
        	//维护当前层的时候将下一层next的关系建出来
            Node dummy = new Node(-1);
            Node tail = dummy;
            for (Node p = cur; p != null; p = p.next) {
                if (p.left != null) tail = tail.next = p.left;
                if (p.right != null) tail = tail.next = p.right;
            }
            cur = dummy.next;
        }
        return root;
    }
}
~~~

#### [690. 员工的重要性](https://leetcode.cn/problems/employee-importance/)

~~~
/*
// Definition for Employee.
class Employee {
    public int id;
    public int importance;
    public List<Integer> subordinates;
};
*/

class Solution {
    public int getImportance(List<Employee> employees, int id) {
        int ans = 0;
        Map<Integer, Employee> map = new HashMap<>();
        for (Employee e : employees) map.put(e.id, e);
        ArrayDeque<Integer> q = new ArrayDeque<>();
        q.add(id);
        while (!q.isEmpty()) {
            int t = q.size();
            while (t -- > 0) {
                int i = q.poll();
                Employee e = map.get(i);
                ans += e.importance;
                List<Integer> list = e.subordinates;
                for (int p : list) q.add(p);
            }
        }
        return ans;
    }
}
~~~

#### [AcWing3675.逃离迷宫](https://www.acwing.com/problem/content/description/3678/)

~~~
import java.util.*;
import java.io.*;
public class Main {
    private int N = 110;
    private int k ,x1 , y1, x2, y2;
    private char[][] g;
    private boolean[][] st;
    private final int[][] dirs = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    
    
    public void solve(InputReader in, PrintWriter out) {
        g = new char[N][N];
        st = new boolean[N][N];
        int t = in.nextInt();
        out : for (int i = 0; i < t; i ++ ) {
            int m = in.nextInt(), n = in.nextInt();
            for (int j = 0; j < m; j ++ )
                    g[j] = in.next().toCharArray();
            k = in.nextInt();
            y1 = in.nextInt() - 1;
            x1 = in.nextInt() - 1;
            y2 = in.nextInt() - 1;
            x2 = in.nextInt() - 1;
            for (int j = 0; j < N; j ++ ) Arrays.fill(st[j], false);
            //只入队转弯的点
            ArrayDeque<Node> q = new ArrayDeque<>();
            q.add(new Node(x1, y1, -1));
            Node p, next = new Node(), temp = new Node();
            while (!q.isEmpty()) {
                p = q.poll();
                //System.out.println(p.x+":"+p.y);
                if (p.x == x2 && p.y == y2 && p.s <= k) {
                    //System.out.println("yes");
                    out.println("yes");
                    continue out;
                }
                for (int[] d : dirs) {
                    next.x = p.x + d[0];
                    next.y = p.y + d[1];
                    while (next.x >= 0 && next.x < m && next.y >=0 && next.y < n && g[next.x][next.y] == '.') {
                        if (!st[next.x][next.y]) {
                            next.s = p.s + 1;
                            st[next.x][next.y] = true;
                            q.add(new Node(next));
                        }
                        //往一个方向走
                        temp.x = next.x + d[0];
                        temp.y = next.y + d[1];
                        next = temp;
                    }
                }
            }
            //System.out.println("no");
            out.println("no");
        }
        out.close();
    }
    
    public static void main(String[] args) throws IOException {
        new Main().solve(new InputReader(System.in), new PrintWriter(System.out));
    }
    
    public static class Node {
        int x, y, s;
        public Node() {}
        public Node(int x, int y, int s) {
            this.x = x;
            this.y = y;
            this.s = s;
        }
        public Node(Node node) {
            this.x = node.x;
            this.y = node.y;
            this.s = node.s;
        }
    }
}
//手写快读类
class InputReader {
    private final BufferedReader reader;
    private StringTokenizer tokenizer;

    public InputReader(InputStream stream) {
        reader = new BufferedReader(new InputStreamReader(stream), 32768);
        tokenizer = null;
    }

    public String next() {
        while (tokenizer == null || !tokenizer.hasMoreTokens()) {
            try {
                tokenizer = new StringTokenizer(reader.readLine());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return tokenizer.nextToken();
    }

    public String nextLine() {
        String str;
        try {
            str = reader.readLine();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return str;
    }

    public boolean hasNext() {
        while (tokenizer == null || !tokenizer.hasMoreTokens()) {
            String nextLine;
            try {
                nextLine = reader.readLine();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            if (nextLine == null) {
                return false;
            }
            tokenizer = new StringTokenizer(nextLine);
        }
        return true;
    }

    public int nextInt() {
        return Integer.parseInt(next());
    }

    public long nextLong() {
        return Long.parseLong(next());
    }
}
~~~

#### [1345. 跳跃游戏 IV](https://leetcode.cn/problems/jump-game-iv/)

~~~
class Solution {
    public int minJumps(int[] arr) {
        int n = arr.length;
        if (n == 1) return 0;
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = n - 1; i >= 0; i -- ) {
            int x = arr[i];
            if (!map.containsKey(x)) map.put(x, new LinkedList<>());
            map.get(x).add(i); 
        }
        boolean[] st = new boolean[n];
        //存下标&步数
        ArrayDeque<int[]> q = new ArrayDeque<>();
        st[0] = true;
        q.add(new int[]{0, 0});
        while (!q.isEmpty()) {
            int a = q.peek()[0], b = q.peek()[1];
            q.poll();
            //System.out.println(a + ":" + b);
            if (a + 1 == n - 1) return b + 1;
            if (!st[a + 1]) {
                st[a + 1] = true;
                q.add(new int[]{a + 1, b + 1});
            } 
            if (a > 0 && !st[a - 1]) {
                st[a - 1] = true;
                q.add(new int[]{a - 1, b + 1});
            }
            if (map.containsKey(arr[a])) {
                for (int x : map.get(arr[a])) {
                    if (x != a && !st[x]) {
                        if (x == n - 1) return b + 1;
                        st[x] = true;
                        q.add(new int[]{x, b + 1});
                    }
                }
                map.remove(arr[a]);
            }
        }
        return -1;
    }
}
~~~

#### [785. 判断二分图](https://leetcode.cn/problems/is-graph-bipartite/)

~~~
class Solution {
    public boolean isBipartite(int[][] g) {
        int[] st = new int[g.length];
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < g.length; i ++ ) {
            if (st[i] != 0) continue;
            q.add(i);
            st[i] = 1;
            while (!q.isEmpty()) {
                int x = q.poll();
                for (int v : g[x]) {
                    if (st[v] == st[x]) return false;
                    if (st[v] == 0) {
                        st[v] = -st[x];
                        q.add(v);
                    }
                }
            }
        }
        return true;
    }
}
~~~

#### [854. 相似度为 K 的字符串](https://leetcode.cn/problems/k-similar-strings/)

~~~
class Solution {
    public int kSimilarity(String s1, String s2) {
        int n = s1.length();
        char[] c = s2.toCharArray();
        var map = new HashMap<String, Integer>();
        var q = new ArrayDeque<String>();
        q.add(s1);
        map.put(s1, 0);
        while (!q.isEmpty()) {
            var s = q.poll();
            char[] t = s.toCharArray();
            for (int i = 0; i < n; i ++ ) {
                if (t[i] != c[i]) {
                    for (int j = i + 1; j < n; j ++ ) {
                        if (t[j] == c[i]) {
                            swap(t, i, j);
                            var p = new String(t);
                            if (p.equals(s2))
                                return map.get(s) + 1;
                            if (!map.containsKey(p)) {
                                map.put(p, map.get(s) + 1);
                                q.add(p);
                            }
                            swap(t, i, j);
                        }
                    }
                    break;
                }
            }
        }
        return 0;
    }
    public void swap(char[] s, int a, int b) {
        char c = s[a];
        s[a] = s[b];
        s[b] = c;
    }
}
~~~

#### [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)

**解法：BFS**

~~~
class Solution {
    public int orangesRotting(int[][] g) {
        int n = g.length, m = g[0].length;
        var q = new ArrayDeque<Pair<Integer, Integer>>();
        int num = 0;
        int[] dx = new int[]{-1, 0, 1, 0}, dy = new int[]{0, 1, 0, -1};
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < m; j ++ )
                if (g[i][j] == 1) 
                    num ++ ;
                else if (g[i][j] == 2) 
                    q.add(new Pair<>(i, j));
        int t = 0;
        while (!q.isEmpty() && num > 0) {
            t ++ ;
            int size = q.size();
            while (size -- > 0) {
                int x = q.peek().getKey(), y = q.peek().getValue();
                q.poll();
                for (int i = 0; i < 4; i ++ ) {
                    int a = x + dx[i], b = y + dy[i];
                    if (a < 0 || a >= n || b < 0 || b >= m || g[a][b] != 1) continue;
                    g[a][b] = 2;
                    num -- ;
                    q.add(new Pair<>(a, b));
                }
            }
        }
        return num == 0 ? t : -1;
    }
}
~~~

~~~
class Solution:
    def orangesRotting(self, g: List[List[int]]) -> int:
        n, m, num = len(g), len(g[0]), 0
        q = collections.deque()
        for i, row in enumerate(g):
            for j, x in enumerate(row):
                if x == 1: num += 1
                elif  x == 2: q.append((i, j))
        t = 0
        while q and num:
            t += 1
            size = len(q)
            while size > 0:
                size -= 1
                x, y = q.popleft()
                for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                    if 0 <= nx < n and 0 <= ny < m and g[nx][ny] == 1:
                        q.append((nx, ny))
                        num -= 1
                        g[nx][ny] = 2
        return t if num == 0 else -1 


~~~



#### [934. 最短的桥](https://leetcode.cn/problems/shortest-bridge/)

**解法：BFS**

~~~
class Solution {
    static int[][] dirs = new int[][]{{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
    public int shortestBridge(int[][] g) {
        var n = g.length;
        var q = new ArrayDeque<int[]>();
        var p = new ArrayList<int[]>();
        for (var i = 0; i < n; i ++ ) {
            for (var j = 0; j < n; j ++ ) {
                if (g[i][j] == 1) {
                    q.add(new int[]{i, j});
                    g[i][j] = -1;
                    while (!q.isEmpty()) {
                        var cell = q.poll();
                        p.add(cell);
                        for (int[] d : dirs) {
                            int x = cell[0] + d[0], y = cell[1] + d[1];
                            if (x >= 0 && x < n && y >= 0 && y < n && g[x][y] == 1) {
                                q.add(new int[]{x, y});
                                g[x][y] = -1;
                            }
                        }
                    }
                }
                for (int[] x : p) q.add(x);
                int step = 0;
                while (!q.isEmpty()) {
                    var size = q.size();
                    while (size -- > 0) {
                        var cell = q.poll();
                        for (int[] d : dirs) {
                            int x = cell[0] + d[0], y = cell[1] + d[1];
                            if (x >= 0 && x < n && y >= 0 && y < n) {
                                if (g[x][y] == 0) {
                                    q.add(new int[]{x, y});
                                    g[x][y] = -1;
                                } else if (g[x][y] == 1) {
                                    return step;
                                }
                            }
                        }
                    } 
                    step ++ ;
                }
            }
        }
        return -1;
    }
}
~~~



**解法：并查集&双向BFS**

~~~
class Solution {
    static int N = 10010;
    static int[] p = new int[N];
    static int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int n;
    public int getIdx(int x, int y) {
        return x * n + y;
    }

    public int find(int x) {
        if (x != p[x]) p[x] = find(p[x]);
        return p[x];
    }

    public void union(int a, int b) {
        p[find(a)] = find(b);    
    }

    public int shortestBridge(int[][] g) {
        n = g[0].length;
        for (int i = 0; i <= n * n; i ++ ) p[i] = i;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ ) {
                if (g[i][j] == 0) continue;
                for (int[] d : dirs) {
                    int x = i + d[0], y = j + d[1];
                    if (x < 0 || x >= n || y < 0 || y >= n) continue;
                    if (g[x][y] == 0) continue;
                    union(getIdx(x, y), getIdx(i, j));
                }
            }
        int a = -1;
        //d1, d2存储a岛和b岛的点
        var d1 = new ArrayDeque<int[]>();
        var d2 = new ArrayDeque<int[]>();
        //m1, m2记录从该岛屿出发到该点的距离
        var m1 = new HashMap<Integer, Integer>();
        var m2 = new HashMap<Integer, Integer>();
        for (var i = 0; i < n; i ++ ) {
            for (var j = 0; j < n; j ++ ) {
                if (g[i][j] == 0) continue;
                int idx = getIdx(i, j), root = find(idx);
                if (a == -1) a = root;
                if (root == a) {
                    d1.add(new int[]{i, j});
                    m1.put(idx, 0);
                } else {
                    d2.add(new int[]{i, j});
                    m2.put(idx, 0);
                }
            }
        }
        //双向BFS求解最短通路
        while (!d1.isEmpty() && !d2.isEmpty()) {
            int t = -1;
            if (d1.size() < d2.size()) t = bfs(d1, m1, m2);
            else t = bfs(d2, m2, m1);
            if (t != -1) return t - 1;
        }
        return -1;
    }

    public int bfs(ArrayDeque<int[]> q, Map<Integer, Integer> m1, Map<Integer, Integer> m2) {
        int size = q.size();
        while (size -- > 0) {
            var info = q.poll();
            int x = info[0], y = info[1], idx = getIdx(x, y), step = m1.get(idx);
            for (int[] d : dirs) {
                int nx = x + d[0], ny = y + d[1], nidx = getIdx(nx, ny);
                if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                if (m1.containsKey(nidx)) continue;
                if (m2.containsKey(nidx)) return step + 1 + m2.get(nidx);
                q.add(new int[]{nx, ny});
                m1.put(nidx, step + 1);
            }
        }
        return -1;
    }
}
~~~

#### [864. 获取所有钥匙的最短路径](https://leetcode.cn/problems/shortest-path-to-get-all-keys/)

**解法：BFS&状态压缩**

~~~
class Solution {
    static int N = 35, K = 6, INF = 0x3f3f3f3f;
    static int[][][] dist = new int[N][N][1 << K];
    static int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    public int shortestPathAllKeys(String[] g) {
        int n = g.length, m = g[0].length(), cnt = 0;
        var q = new ArrayDeque<int[]>();
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                Arrays.fill(dist[i][j], INF);
                char c = g[i].charAt(j);
                if (c == '@') {
                    q.add(new int[]{i, j, 0});
                    dist[i][j][0] = 0;
                } else if (c >= 'a' && c <= 'z') cnt ++ ;
            }
        }

        while (!q.isEmpty()) {
            var info = q.poll();
            int x = info[0], y = info[1], cur = info[2], step = dist[x][y][cur];
            for (var d : dirs) {
                int nx = x + d[0], ny = y + d[1];
                if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
                var c = g[nx].charAt(ny);
                if (c == '#') continue;
                if (c >= 'A' && c <= 'Z' && (cur >> (c - 'A') & 1) == 0) continue;
                var ncur = cur;
                if (c >= 'a' && c <= 'z') ncur |= 1 << (c - 'a');
                if (ncur == (1 << cnt) - 1) return step + 1;
                if (step + 1 >= dist[nx][ny][ncur]) continue;
                dist[nx][ny][ncur] = step + 1;
                q.add(new int[]{nx, ny, ncur});
            }
        }
        return -1;
    }
}
~~~



## DFS

#### [17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

#### [39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

~~~
class Solution {
    List<List<Integer>> ans = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] cs, int target) {
        dfs(cs, 0, target);
        return ans;
    }

    public void dfs(int[] cs, int u, int target) {
        if(target == 0){
            //对象引用问题，不new 一个新的，后续 path 发生变化，结果也会变
            ans.add(new ArrayList<>(path));
            return;
        }
        if (u == cs.length) return;

        for (int i = 0; i * cs[u] <= target; i ++ ) {
            dfs(cs, u + 1, target - cs[u] * i);
            path.add(cs[u]);
        }

        for(int i = 0; i * cs[u] <= target; i ++){
            path.remove(path.size() - 1);
        }
    }
}
~~~

#### [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

~~~
class Solution {
    List<List<Integer>> ans;
    List<Integer> list;
    boolean[] st;
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        ans = new ArrayList<>();
        list = new ArrayList<>();
        st = new boolean[n];
        dfs(nums, 0);
        return ans;
    }

    public void dfs(int[] nums, int u) {
        if (u == nums.length) {
            ans.add(new ArrayList<>(list));
            return;
        }

        for (int i = 0; i < nums.length; i ++ ) {
            if (!st[i]) {
            	//保证相同数字要按顺序使用，即保证相同数字的相对顺序不变即可
                if (i > 0 && nums[i - 1] == nums[i] && !st[i - 1]) continue;
                st[i] = true;
                list.add(nums[i]);
                dfs(nums, u + 1);
                st[i] = false;
                list.remove(list.size() - 1);
            }
        }
    }
}
~~~

#### [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

floodfill

#### [698. 划分为k个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)

剪枝：

	1、从大到小枚举
	
	2、nums[i] == nums[i + 1] nums[i]失败，nums[i + 1]一定失败
	
	3、当前第一个数失败，直接跳过
	
	4、当前最后一个数失败，直接跳过

~~~
class Solution {
    int[] nums;
    boolean[] st;
    int n;
    int avg;
    public boolean canPartitionKSubsets(int[] _nums, int k) {
        nums = _nums;
        n = nums.length;
        st = new boolean[n];
        int sum = 0;
        for (int x : nums) sum += x;
        //可行性剪枝
        if (sum % k != 0) return false;
        avg = sum / k;
        //从大到小枚举
        nums = Arrays.stream(nums).boxed().sorted((a, b) -> b - a).mapToInt(i -> i).toArray();
        return dfs(0, 0, k);
    }
    /*
    *start：当前这组从第几个开始搜
    *cur：当前组总和
    *k：已经搜了几组
     */
    public boolean dfs(int start, int cur, int k) {
        //枚举完毕返回true
        if (k == 0) return true;
        //枚举完一组继续下一组
        if (cur == avg) return dfs(0, 0, k - 1);
        for (int i = start; i < n; i ++ ) {
            if (st[i] || cur + nums[i] > avg) continue;
            st[i] = true;
            if (dfs(i + 1, cur + nums[i], k)) return true;
            st[i] = false;
            //剪枝nums[i] == nums[i + 1] nums[i]失败，nums[i + 1]一定失败
            while (i + 1 < n && nums[i + 1] == nums[i]) i ++ ;
            //当前组的第一个数或者最后一个数失败那么一定失败
            if (cur == 0) return false;
        }
        return false;
    }
}
~~~

#### [剑指 Offer 12. 矩阵中的路径](https://leetcode.cn/problems/ju-zhen-zhong-de-lu-jing-lcof/)

~~~
class Solution {
    char[][] g;
    String word;
    int n, m;
    boolean[][] st;
    public boolean exist(char[][] board, String _word) {
        g = board;
        n = g.length;
        m = g[0].length;
        word = _word;
        st = new boolean[n][m];
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                if (g[i][j] == word.charAt(0)) {
                    st[i][j] = true;
                    if (dfs(i, j, 0)) return true;
                    st[i][j] = false;
                }
            }
        }
        return false;
    }
    int[] dx = new int[]{-1, 0, 1, 0}, dy = new int[]{0, 1, 0, -1};
    public boolean dfs(int i, int j, int u) {
        if (u == word.length() - 1) return true;
        for (int k = 0; k < 4; k ++ ) {
            int x = i + dx[k], y = j + dy[k];
            if (x < 0 || x >= n || y < 0 || y >= m) continue;
            if (g[x][y] != word.charAt(u + 1)) continue;
            if (!st[x][y]) {
                st[x][y] = true;
                if (dfs(x, y, u + 1)) return true;
                st[x][y] = false;
            }
        }
        return false;
    }
}
~~~

#### [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode.cn/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        dfs(root, target);
        return res;
    }
    public void dfs(TreeNode root, int t) {
        if (root == null) return;
        path.add(root.val);
        if (t == root.val && root.left == null && root.right == null) 
        	res.add(new ArrayList<>(path));
        dfs(root.left, t - root.val);
        dfs(root.right, t - root.val);
        path.remove(path.size() - 1);
    }
}
~~~

#### [剑指 Offer 26. 树的子结构](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) return false;
        return dfs(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }
    public boolean dfs(TreeNode A, TreeNode B) {
        if (B == null) return true;
        if (A == null) return false;
        return A.val == B.val && dfs(A.left, B.left) && dfs(A.right, B.right);
    }
}
~~~

#### [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode.cn/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

~~~
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,Node _left,Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
    Node pre, head;
    public Node treeToDoublyList(Node root) {
        if (root == null) return null;
        dfs(root);
        head.left = pre;
        pre.right = head;
        return head;
    }
    public void dfs(Node root) {
        if (root == null) return;
        dfs(root.left);
        //如果pre为空，则说明是第一节点，头节点，然后用head保存头节点
        if (pre == null) head = root;
        //如果不为空，那就说明是中间的节点，并且pre保存的是上一个节点
        //让上一个节点的右指针指向当前节点
        else if (pre != null) pre.right = root;
        //将当前节点的做指针指向父节点
        root.left = pre;
        //保存当前节点，用于下层递归创建
        pre = root;
        dfs(root.right);
    }
}
~~~

#### [剑指 Offer 38. 字符串的排列](https://leetcode.cn/problems/zi-fu-chuan-de-pai-lie-lcof/)

~~~
class Solution {
    List<String> list = new ArrayList<>();
    List<Character> path = new ArrayList<>();
    boolean[] st;
    char[] ss;
    public String[] permutation(String s) {
        int n = s.length();
        ss = s.toCharArray();
        st = new boolean[n];
        Arrays.sort(ss);
        dfs(ss, n);
        String[] ans = new String[list.size()];
        //for (List<Character> i : list) System.out.println(i);
        for (int i = 0; i < list.size(); i ++ ) ans[i] = list.get(i);
        return ans;
    }
    public void dfs(char[] ss, int k) {
        if (k == path.size()) {
            StringBuilder sb = new StringBuilder();
            for (char c : path) sb.append(c);
            list.add(sb.toString());
        }

        for (int i = 0; i < ss.length; i ++ ) {
            if (!st[i]) {
                if (i > 0 && ss[i] == ss[i - 1] && !st[i - 1]) continue;
                st[i] = true;
                path.add(ss[i]);
                dfs(ss, k);
                st[i] = false;
                path.remove(path.size() - 1);
            }
        }
    }
}
~~~

#### [494. 目标和](https://leetcode.cn/problems/target-sum/)

~~~
//dfs
class Solution {
    public int findTargetSumWays(int[] nums, int t) {
        return dfs(nums, t, 0);
    }
    public int dfs(int[] nums, int t, int u) {
        if (u == nums.length) return t == 0 ? 1 : 0;
        int l = dfs(nums, t - nums[u], u + 1);
        int r = dfs(nums, t + nums[u], u + 1);
        return l + r;
    }
}
~~~

~~~
//dfs + 记忆化搜索
class Solution {
    public int findTargetSumWays(int[] nums, int t) {
        return dfs(nums, t, 0);
    }
    public int dfs(int[] nums, int t, int u) {
        if (u == nums.length) return t == 0 ? 1 : 0;
        int l = dfs(nums, t - nums[u], u + 1);
        int r = dfs(nums, t + nums[u], u + 1);
        return l + r;
    }
}
~~~

#### [403. 青蛙过河](https://leetcode.cn/problems/frog-jump/)

~~~
//记忆化搜索
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    Map<String, Boolean> cache = new HashMap<>();
    public boolean canCross(int[] stones) {
        int n = stones.length;
        for (int i = 0; i < n; i ++ ) map.put(stones[i], i);
        if (stones[1] != 1) return false;
        return dfs(stones, n, 1, 1);
    }
    /**
     * 判定是否能够跳到最后一块石子
     * @param ss 石子列表【不变】
     * @param n  石子列表长度【不变】
     * @param u  当前所在的石子的下标
     * @param k  上一次是经过多少步跳到当前位置的
     * @return 是否能跳到最后一块石子
     */
    public boolean dfs(int[] ss, int n, int u, int k) {
        if (u == n - 1) return true;
        String key = u + "_" + k;
        if (cache.containsKey(key)) return cache.get(key);
        else 
        for (int i = -1; i <= 1; i ++ ) {
            if (i + k == 0) continue;
            //理论上下一个石头的编号
            int next = ss[u] + i + k;
            if (map.containsKey(next)) {
                boolean cur = dfs(ss, n, map.get(next), k + i);
                cache.put(key, cur);
                if (cur) return true;
            }
        }
        cache.put(key, false);
        return false;
    }
}
~~~

#### [337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/)

~~~
//记忆化搜索
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    Map<TreeNode, Integer> map = new HashMap<>();
    public int rob(TreeNode root) {
        if (root == null) return 0;
        if (map.containsKey(root)) return map.get(root);
        int a = root.val;
        if (root.left != null) a += rob(root.left.left) + rob(root.left.right);
        if (root.right != null) a += rob(root.right.left) + rob(root.right.right);
        int b = rob(root.left) + rob(root.right);
        map.put(root, Math.max(a, b));
        return Math.max(a, b);
    }
}
~~~

#### [394. 字符串解码](https://leetcode.cn/problems/decode-string/)

~~~
class Solution {
    int u = 0;
    public String decodeString(String s) {
        return dfs(s);
    }
    public String dfs(String s) {
        StringBuilder res = new StringBuilder();
        while (u < s.length() && s.charAt(u) != ']') {
            if (s.charAt(u) >= 'a' && s.charAt(u) <= 'z' || s.charAt(u) >= 'A' && s.charAt(u) <= 'Z') {
                res.append(s.charAt(u ++ ));
            }
            else if (s.charAt(u) >= '0' && s.charAt(u) <= '9') {
                int k = u;
                while (s.charAt(k) >= '0' && s.charAt(k) <= '9') k ++ ;
                int x = Integer.parseInt(s.substring(u, k));
                u = k + 1;
                String y = dfs(s);
                //过滤掉右括号
                u ++ ;
                //System.out.println(s.charAt(u));
                while (x > 0) {
                    res.append(y);
                    x -- ;
                }
            }
        }
        return res.toString();
    }
}
~~~

#### [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)

~~~
class Solution {
    List<String> ans = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        dfs(n, 0, 0, "");
        return ans;
    }
    //任意前缀中"("数量>=")"数量
    //左右括号数量相等
    public void dfs(int n, int l, int r, String s) {
        if (l == n && r == n) ans.add(s);
        else {
            if (l < n) dfs(n, l + 1, r, s + "(");
            if (r < n && l > r) dfs(n, l, r + 1, s + ")");
        }
    }
}
~~~

#### [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

~~~
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return verify(postorder, 0, postorder.length - 1);
    }
    public boolean verify(int[] pos, int l, int r) {
        if (l >= r) return true;
        int m = l;
        //划分左右子树
        while (pos[m] < pos[r]) m ++ ;
        for (int i = m; i < r; i ++ ) {
            if (pos[i] < pos[r]) return false;
        }
        return verify(pos, l, m - 1) && verify(pos, m, r - 1);
    }
}
~~~

#### [297. 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {

    private StringBuilder sb = new StringBuilder();
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        dfs1(root);
        return sb.toString();
    }
    public void dfs1(TreeNode root) {
        if (root == null) {
            sb.append("null.");
            return;
        }
        sb.append(root.val).append(".");
        dfs1(root.left);
        dfs1(root.right);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] t = data.split("\\.");
        List<String> list = new LinkedList<>(Arrays.asList(t));
        return dfs2(list);
    }
    public TreeNode dfs2(List<String> list) {
        if (list.get(0).equals("null")) {
            list.remove(0);
            return null;
        }
        TreeNode res = new TreeNode(Integer.valueOf(list.get(0)));
        list.remove(0);
        res.left = dfs2(list);
        res.right = dfs2(list);
        return res;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec ser = new Codec();
// Codec deser = new Codec();
// TreeNode ans = deser.deserialize(ser.serialize(root));
~~~

#### [5289. 公平分发饼干](https://leetcode.cn/problems/fair-distribution-of-cookies/)

~~~
class Solution {
    int[] sum = new int[10], q;
    int ans = (int)1e9, n;
    public void dfs(int u) {
        if (u == q.length) {
            int mx = 0;
            for (int i = 0; i < n; i ++ ) mx = Math.max(mx, sum[i]);
            ans = Math.min(ans, mx);
            return; 
        }
        for (int i = 0; i < n; i ++ ) {
            sum[i] += q[u];
            //剪枝，当前组如果大于1e9则不可能优化最优解，就不进行递归
            if (sum[i] <= ans) {
                dfs(u + 1);
            }
            sum[i] -= q[u];
        }
    }
    public int distributeCookies(int[] cookies, int k) {
        n = k;
        q = cookies;
        dfs(0);
        return ans;
    }
}
~~~

#### [90. 子集 II](https://leetcode.cn/problems/subsets-ii/)

~~~~
class Solution {
    List<List<Integer>> ans = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    int[] nums;
    public List<List<Integer>> subsetsWithDup(int[] _nums) {
        nums = _nums;
        Arrays.sort(nums);
        dfs(0);
        return ans;
    }
    public void dfs(int u) {
        if (u == nums.length) {
            ans.add(new ArrayList<>(path));
            return;
        }
        int k = u + 1;
        while (k < nums.length && nums[k] == nums[u]) k ++ ;
        
        for (int i = 0; i <= k - u; i ++ ) {
            dfs(k);
            path.add(nums[u]);
        }
        for (int i = 0; i <= k - u; i ++ ) {
            path.remove(path.size() - 1);
        }
    }
}
~~~~

#### [508. 出现次数最多的子树元素和](https://leetcode.cn/problems/most-frequent-subtree-sum/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    int max = 0;
    public int[] findFrequentTreeSum(TreeNode root) {
        dfs(root);
        List<Integer> list = new ArrayList<>();
        for (int t : map.keySet()) if (map.get(t) == max) list.add(t);
        int[] ans = new int[list.size()];
        for (int i = 0; i < list.size(); i ++ ) ans[i] = list.get(i);
        return ans;
    }
    public int dfs(TreeNode root) {
        if (root == null) return 0;
        int l = dfs(root.left), r = dfs(root.right);
        int ans = l + r + root.val;
        map.put(ans, map.getOrDefault(ans, 0) + 1);
        max = Math.max(max, map.get(ans));
        return ans;
    }
}
~~~

~~~
class Solution:
    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        map = Counter()
        def dfs(root):
            if not root: return 0
            l, r = dfs(root.left), dfs(root.right)
            res = l + r + root.val
            map[res] += 1
            return res
        dfs(root)
        mx = max(map.values())
        return [x for x, c in map.items() if c == mx]
~~~



#### [241. 为运算表达式设计优先级](https://leetcode.cn/problems/different-ways-to-add-parentheses/)

~~~
class Solution {
    char cs[];
    public List<Integer> diffWaysToCompute(String expression) {
        cs = expression.toCharArray();
        return dfs(0, cs.length - 1);
    }
    public List<Integer> dfs(int l, int r) {
        List<Integer> ans = new ArrayList<>();
        for (int i = l; i <= r; i ++ ) {
            if (cs[i] >= '0' && cs[i] <='9') continue;
            List<Integer> l1 = dfs(l,i - 1), r1 = dfs(i + 1, r);
            for (int a : l1) {
                for (int b : r1) {
                    int cur = 0;
                    if (cs[i] == '+') cur = a + b;
                    else if (cs[i] == '-') cur = a - b;
                    else cur = a * b;
                    ans.add(cur);
                }
            }
        }
        if (ans.isEmpty()) {
            int cur = 0;
            for (int i = l; i <= r; i ++ ) cur = cur * 10 + (cs[i] - '0');
            ans.add(cur);
        }
        return ans;
    }
}
~~~

#### [95. 不同的二叉搜索树 II](https://leetcode.cn/problems/unique-binary-search-trees-ii/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<TreeNode> generateTrees(int n) {
        return dfs(1, n);
    }
    public List<TreeNode> dfs(int l, int r) { 
        List<TreeNode> ans = new ArrayList<>();
        if (l > r) {
            ans.add(null);
            return ans;
        }
        for (int i = l; i <= r; i ++ ) {
            List<TreeNode> l1 = dfs(l, i - 1), l2 = dfs(i + 1, r);
            for (TreeNode ll : l1) {
                for (TreeNode rr: l2) {
                    TreeNode root = new TreeNode(i);
                    root.left = ll;
                    root.right = rr;
                    ans.add(root);
                }
            }
        }
        return ans;
    }
}
~~~

#### [6110. 网格图中递增路径的数目](https://leetcode.cn/problems/number-of-increasing-paths-in-a-grid/)

~~~
//记忆化搜索
class Solution {
    int mod = (int) 1e9 + 7;
    int m, n;
    //定义f[i][j]表示以i行j列的格子位起点的路径数
    int[][] g, f;
    int[] dx = new int[]{-1, 0, 1, 0}, dy = new int[]{0, 1, 0, -1};
    public int countPaths(int[][] grid) {
        g = grid;
        m = g.length;
        n = g[0].length;
        f = new int[m][n];
        for (int i = 0; i < m; i ++ ) Arrays.fill(f[i], -1);
        long ans = 0;
        //枚举所有起点
        for (int i = 0; i < m; i ++ ) {
            for (int j = 0; j < n; j ++ ) {
                ans = (ans + dfs(i, j)) % mod;
            }
        }
        return (int) ans;
    }
    public int dfs(int i, int j) {
        if (f[i][j] != -1) return f[i][j];
        int res = 1;
        for (int k = 0; k < 4; k ++ ) {
            int x = i + dx[k], y = j + dy[k];
            if (x < 0 || x >= m || y < 0 || y >= n) continue;
            if (g[x][y] <= g[i][j]) continue;
            res = (res + dfs(x, y)) % mod; 
        }
        return f[i][j] = res;
    }
}
~~~

#### [129. 求根节点到叶节点数字之和](https://leetcode.cn/problems/sum-root-to-leaf-numbers/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int ans = 0;
    public int sumNumbers(TreeNode root) {
        dfs(root, 0);
        return ans;
    }
    public void dfs (TreeNode root, int cur) {
        cur = cur * 10 + root.val;
        if (root.left == null && root.right == null) ans += cur;
        if (root.left != null) dfs(root.left, cur);
        if (root.right != null) dfs(root.right, cur);
    }
}
~~~

#### [116. 填充每个节点的下一个右侧节点指针](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)

~~~
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}
    
    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};
*/

class Solution {
    public Node connect(Node root) {
        Node ans = dfs(root);
        return ans;
    }
    public Node dfs(Node root) {
        if (root == null || root.right == null) return root;
        //树的左子树指向右子树
        root.left.next = root.right;
        //右子树的右节点指向下一个节点
        if (root.next != null) root.right.next = root.next.left;
        dfs(root.left);dfs(root.right);
        return root;
    }
}
~~~

#### [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/)

~~~
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}
    
    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};
*/

class Solution {
    public Node connect(Node root) {
        dfs(root);
        return root;
    }
    public Node dfs(Node root) {
        if (root == null || (root.left == null && root.right == null)) return root;
        //左右都不空
        if (root.left != null && root.right != null) {
            root.left.next = root.right;
            root.right.next = getNextNode(root);
        }
        //左空右不空
        if (root.left == null) root.right.next = getNextNode(root);
        //左不空右不空
        if (root.right == null) root.left.next = getNextNode(root);
        //注意优先建立右子树关系
        //[2,1,3,0,7,9,1,2,null,1,0,null,null,8,8,null,null,null,null,7]
        dfs(root.right);dfs(root.left);
        return root;
    }
    //找到下一个具有左或右子节点的Node
    public Node getNextNode(Node root) {
        while (root.next != null) {
            if (root.next.left != null) return root.next.left;
            if (root.next.right != null) return root.next.right;
            root = root.next;
        }
        return null;
    }
}
~~~

#### [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<Integer> ans = new ArrayList<>();
    public List<Integer> rightSideView(TreeNode root) {
        dfs(root, 0);
        return ans;
    }
    public void dfs(TreeNode root, int u) {
        if (root == null) return;
        if (ans.size() == u) ans.add(root.val);
        dfs(root.right, u + 1);dfs(root.left, u + 1);
    }
}
~~~

#### [130. 被围绕的区域](https://leetcode.cn/problems/surrounded-regions/)

~~~
class Solution {
    int n, m;
    char[][] g;
    boolean[][] st;
    //先标记不会被覆盖的
    public void solve(char[][] board) {
        g = board;
        n = board.length;
        m = board[0].length;
        st = new boolean[n][m];
        for (int i = 0; i < n; i ++ ) {
            if (g[i][0] == 'O') dfs(i, 0);
            if (g[i][m - 1] == 'O') dfs(i, m - 1);
        }
        
        for (int j = 0; j < m; j ++ ) {
            if (g[0][j] == 'O') dfs(0, j);
            if (g[n - 1][j] == 'O') dfs(n - 1, j);
        }

        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                if (!st[i][j]) board[i][j] = 'X';
            }
        }
    }
    int[] dx = new int[]{-1, 0, 1, 0}, dy = new int[]{0, 1, 0, -1};
    public void dfs(int i, int j) {
        st[i][j] = true;
        for (int k = 0; k < 4; k ++ ) {
            int a = i + dx[k], b = j + dy[k];
            if (a < 0 || a >= n) continue;
            if (b < 0 || b >= m) continue;
            if (st[a][b]) continue;
            if (g[a][b] != 'O') continue;
            dfs(a, b);
        }
    }
}
~~~

#### [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int res = -Integer.MAX_VALUE;
    public int maxPathSum(TreeNode root) {
        dfs(root);
        return res;
    }
    public int dfs(TreeNode root) {
        if (root == null) return 0;
        int l = Math.max(0, dfs(root.left)),r = Math.max(0, dfs(root.right));
        res = Math.max(res, root.val + l + r);
        return root.val + Math.max(l, r);
    }
}
~~~

#### [329. 矩阵中的最长递增路径](https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/)

~~~
class Solution {
    int n, m;
    int[][] g, f;
    public int longestIncreasingPath(int[][] matrix) {
        g = matrix;
        n = g.length;
        m = g[0].length;
        f = new int[n][m];
        int res = 0;
        for (int i = 0; i < n; i ++ ) Arrays.fill(f[i], -1);
        for (int i = 0; i < n; i ++ ) { 
            for (int j = 0; j < m; j ++) {
                res = Math.max(res, dfs(i, j));
            }
        }
        return res;
    }
    int[][] dirs = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    public int dfs(int i, int j) {
        if (f[i][j] != -1) return f[i][j];
        f[i][j] = 1;
        for (int[] d : dirs) {
            int x = i + d[0], y = j + d[1];
            if (x < 0 || x >= n || y < 0 || y >= m) continue;
            if (g[x][y] <= g[i][j]) continue;
            f[i][j] = Math.max(f[i][j], dfs(x, y) + 1);
        }
        return f[i][j]; 
    }
}
~~~

#### [419. 甲板上的战舰](https://leetcode.cn/problems/battleships-in-a-board/)

~~~
class Solution {
    char[][] g;
    public int countBattleships(char[][] board) {
        g = board;
        int ans = 0;
        for (int i = 0; i < g.length; i ++ ) {
            for (int j = 0; j < g[0].length; j ++ ) {
                if (g[i][j] == 'X') {
                    ans ++ ;
                    dfs(i, j);
                }
            }
        }
        return ans;
    }
    int[][] dirs = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    public void dfs(int i, int j) {
        g[i][j] = '.';
        for (int[] d : dirs) {
            int x = i + d[0], y = j + d[1];
            if (x < 0 || x >= g.length || y < 0 || y >= g[0].length) continue;
            if (g[x][y] != 'X') continue;
            dfs(x, y);
        }
    }
}
~~~

#### [1038. 从二叉搜索树到更大和树](https://leetcode.cn/problems/binary-search-tree-to-greater-sum-tree/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int t;
    public TreeNode bstToGst(TreeNode root) {
        dfs(root);
        return root;
    }
    public void dfs(TreeNode root) {
        if (root == null) return;
        dfs(root.right);
        t += root.val;
        root.val = t;
        dfs(root.left);
    }
}
~~~

#### [1026. 节点与其祖先之间的最大差值](https://leetcode.cn/problems/maximum-difference-between-node-and-ancestor/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int ans = 0;
    public int maxAncestorDiff(TreeNode root) {
        dfs(root, root.val, root.val);
        return ans;
    }
    //记录当前节点所有父节点的最大值和最小值
    public void dfs(TreeNode root, int max, int min) {
        if (root == null) return;
        max = Math.max(root.val, max);
        min = Math.min(root.val, min);
        dfs(root.left, max, min);dfs(root.right, max, min);
        ans = Math.max(ans, max - min);
    }
}
~~~

#### [1080. 根到叶路径上的不足节点](https://leetcode.cn/problems/insufficient-nodes-in-root-to-leaf-paths/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode sufficientSubset(TreeNode root, int limit) {
        TreeNode res = dfs(root, 0, limit);
        return res;
    }
    //如果该点删除那么他的左右子树也一定删除
    public TreeNode dfs(TreeNode root, int sum, int limit) {
        sum += root.val;
        if (root.left == null && root.right == null) {
            if (sum < limit) root = null;
        } else {
            if (root.left != null) 
                root.left = dfs(root.left, sum, limit);
            if (root.right != null) 
                root.right = dfs(root.right, sum, limit);
            if (root.left == null && root.right == null) 
                root = null;
        }
        return root;
    }
}
~~~

#### [652. 寻找重复的子树](https://leetcode.cn/problems/find-duplicate-subtrees/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<TreeNode> ans = new ArrayList<>();
    Map<String, Integer> map = new HashMap<>();
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        dfs(root);
        return ans;
    }
    public String dfs(TreeNode root) {
        if (root == null) return "";
        StringBuilder sb = new StringBuilder();
        sb.append(root.val).append(",");
        sb.append(dfs(root.left)).append(",");
        sb.append(dfs(root.right));
        String cur = sb.toString();
        map.put(cur, map.getOrDefault(cur, 0) + 1);
        if (map.get(cur) == 2) ans.add(root);
        return cur;
    } 
}
~~~

#### [669. 修剪二叉搜索树](https://leetcode.cn/problems/trim-a-binary-search-tree/)

~~~
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode trimBST(TreeNode root, int low, int high) {
        TreeNode res = dfs(root, low, high);
        return res;
    }
    public TreeNode dfs(TreeNode root, int l, int r) {
        if (root == null) return root;
        //修建二叉树
        if (root.val < l) return dfs(root.right, l, r);
        if (root.val > r) return dfs(root.left, l, r);
        //处理正常的节点
        root.left = dfs(root.left, l, r);
        root.right = dfs(root.right, l, r);
        return root;
    }
}
~~~

#### [690. 员工的重要性](https://leetcode.cn/problems/employee-importance/)

~~~
/*
// Definition for Employee.
class Employee {
    public int id;
    public int importance;
    public List<Integer> subordinates;
};
*/

class Solution {
    Map<Integer, Employee> map = new HashMap<>();
    public int getImportance(List<Employee> employees, int id) {
        for (Employee e : employees) map.put(e.id, e);
        int res = dfs(id);
        return res;
    }
    public int dfs(int id) {
        Employee e = map.get(id);
        int res = e.importance;
        for (int i : e.subordinates) res += dfs(i);
        return res;
    }
}
~~~

#### [814. 二叉树剪枝](https://leetcode.cn/problems/binary-tree-pruning/)

~~~
class Solution {
    public TreeNode pruneTree(TreeNode root) {
        if (!dfs(root)) return null;
        return root;
    }
    public boolean dfs(TreeNode root) {
        if (root == null) return false;
        if (!dfs(root.left)) root.left = null;
        if (!dfs(root.right))  root.right = null;
        return root.val == 1 || root.left != null || root.right != null;
    }
}
~~~

#### [133. 克隆图](https://leetcode.cn/problems/clone-graph/)

~~~
class Solution {
    Map<Node, Node> hash = new HashMap<>();
    public void dfs(Node node) {
        hash.put(node, new Node(node.val));
        for (var x : node.neighbors) {
            if (!hash.containsKey(x)) {
                dfs(x);
            }
        }
    }
    public Node cloneGraph(Node node) {
        if (node == null) return null;
        //复制所有点
        dfs(node);
        for (var x : hash.keySet()) 
            for (var y : x.neighbors) 
                hash.get(x).neighbors.add(hash.get(y));
        
        return hash.get(node);
    }
}
~~~

#### [655. 输出二叉树](https://leetcode.cn/problems/print-binary-tree/)

~~~
class Solution {
    String[][] res;
    public int[] dfs(TreeNode root) {
        if (root == null) return new int[]{0, 0};
        int[] l = dfs(root.left), r = dfs(root.right);
        return new int[]{Math.max(l[0], r[0]) + 1, Math.max(l[1], r[1]) * 2 + 1};
    }
    public void print(TreeNode root, int h, int l, int r) {
        if (root == null) return;
        int mid = l + r >> 1;
        res[h][mid] = String.valueOf(root.val);
        print(root.left, h + 1, l, mid - 1);
        print(root.right, h + 1, mid + 1, r);
    }
    public List<List<String>> printTree(TreeNode root) {
        int[] t = dfs(root);
        int h = t[0], w = t[1];
        res = new String[h][w];
        for (int i = 0; i < h; i ++ ) 
            Arrays.fill(res[i], "");
        print(root, 0, 0, w - 1);
        List<List<String>> list = new ArrayList<>();
        for (int i = 0; i < h; i ++ )
            list.add(Arrays.asList(res[i]));
        return list;
    }
}
~~~

#### [785. 判断二分图](https://leetcode.cn/problems/is-graph-bipartite/)

~~~
class Solution {
    int[] st;
    int[][] g;
    public boolean isBipartite(int[][] _g) {
        g = _g;
        st = new int[g.length];
        for (int i = 0; i < g.length; i ++ ) {
            if (st[i] == 0 && !dfs(i, 1))
                return false;
        }
        return true;
    }
    public boolean dfs(int u, int colour) {
        st[u] = colour;
        for (int x : g[u]) {
            if (st[x] != 0) {
                if (st[x] == colour)
                    return false;
            } else if (!dfs(x, -colour)) return false;
        }
        return true;
    }
}
~~~

#### [491. 递增子序列](https://leetcode.cn/problems/increasing-subsequences/)

~~~
class Solution {
    List<List<Integer>> res;
    List<Integer> path;
    public List<List<Integer>> findSubsequences(int[] nums) {
        res = new ArrayList<>();
        path = new ArrayList<>();
        dfs(nums, 0);
        return res;
    }
    
    public void dfs(int[] nums, int u) {
        if (path.size() >= 2) res.add(new ArrayList<>(path));
        if (u == nums.length) return;
        Set<Integer> set = new HashSet<>();
        for (int i = u; i < nums.length; i ++ ) {
            if (path.isEmpty() || nums[i] >= path.get(path.size() - 1)) {
                //如果这个数已经被选过，那么所有选这个数的情况已经被枚举过了
                if (set.contains(nums[i])) continue;
                set.add(nums[i]);
                path.add(nums[i]);
                dfs(nums, i + 1);
                path.remove(path.size() - 1);
            }
        }
    }
}
~~~

#### [865. 具有所有最深节点的最小子树](https://leetcode.cn/problems/smallest-subtree-with-all-the-deepest-nodes/)

~~~
class Solution {
    public TreeNode subtreeWithAllDeepest(TreeNode root) {
        return dfs(root).root;
    }
    public Pair dfs(TreeNode root) {
        if (root == null) return new Pair(null, 0);
        Pair l = dfs(root.left), r = dfs(root.right);
        if (l.depth == r.depth) return new Pair(root, l.depth + 1);
        if (l.depth > r.depth) return new Pair(l.root, l.depth + 1);
        else return new Pair(r.root, r.depth + 1);
    }
    public class Pair {
        TreeNode root;
        int depth;
        public Pair(TreeNode root, int depth) {
            this.root = root;
            this.depth = depth;
        }
    }
}
~~~

#### [216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)

~~~
class Solution {
    List<List<Integer>> res;
    List<Integer> path;
    public List<List<Integer>> combinationSum3(int k, int n) {
        res = new ArrayList<>();
        path = new ArrayList<>();
        dfs(1, n, k);
        return res;
    }
    public void dfs(int start, int n, int k) {
        if (n == 0) {
            if (k == 0)
                res.add(new ArrayList<>(path));
        } else {
            for (int i = start; i <= 9; i ++ ) {
                if (n >= i) {
                    path.add(i);
                    dfs(i + 1, n - i, k - 1);
                    path.remove(path.size() - 1);
                }
            }
        }
    }
}
~~~

#### [572. 另一棵树的子树](https://leetcode.cn/problems/subtree-of-another-tree/)

~~~
class Solution {
    public boolean isSubtree(TreeNode p, TreeNode q) {
        if (p == null) return false;
        return dfs(p, q) || isSubtree(p.left, q) || isSubtree(p.right, q);
    }

    public boolean dfs(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        if (p.val != q.val) return false;
        return dfs(p.left, q.left) && dfs(p.right, q.right);
    }
}
~~~

#### [37. 解数独](https://leetcode.cn/problems/sudoku-solver/)

~~~
class Solution {
    boolean[][] row, col, block;
    public boolean dfs(char[][] board, int i, int j) {
        while (board[i][j] != '.') {
            if ( ++ j == 9) {
                i ++ ;
                j = 0;
            }
            if (i == 9) return true;
        }
        for (int k = 0; k < 9; k ++ ) {
            int x = i / 3 * 3 + j / 3;
            if (!row[i][k] && !col[j][k] && !block[x][k]) {
                board[i][j] = (char)(k + '1');
                row[i][k] = true;
                col[j][k] = true;
                block[x][k] = true;
                if (dfs(board, i, j)) return true;
                else {
                    board[i][j] = '.';
                    row[i][k] = false;
                    col[j][k] = false;
                    block[x][k] = false;
                }
            }
        }
        return false;
    }
    public void solveSudoku(char[][] board) {
        col = new boolean[9][9];
        row = new boolean[9][9];
        block = new boolean[9][9];
        for (int i = 0; i < 9; i ++ )
            for (int j = 0; j < 9; j ++ ) 
                if (board[i][j] != '.') {
                    int k = board[i][j] - '1';
                    row[i][k] = true;
                    col[j][k] = true;
                    block[i / 3 * 3 + j / 3][k] = true;
                }
        
        dfs(board, 0, 0);
    }
}
~~~

#### [4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

~~~
class Solution {
    public double findMedianSortedArrays(int[] a, int[] b) {
        int tot = a.length + b.length;
        if (tot % 2 == 0) {
            int left = find(a, 0, b, 0, tot / 2);
            int right = find(a, 0, b, 0, tot / 2 + 1);
            return (left + right) / 2.0;
        } else {
            return find(a, 0, b, 0, tot / 2 + 1);
        }
    }

    public int find(int[] a, int i, int[] b, int j, int k) {
        if (a.length - i > b.length - j) return find(b, j, a, i, k);
        if (i == a.length) return b[j + k - 1];
        if (k == 1) return Math.min(a[i], b[j]);
        int si = Math.min(i + k / 2, a.length), sj = j + k / 2;
        if (a[si - 1] > b[sj - 1])
            return find(a, i, b, sj, k - (sj - j));
        else 
            return find(a, si, b, j, k - (si - i));
    } 
}
~~~

#### [886. 可能的二分法](https://leetcode.cn/problems/possible-bipartition/)

**解法：染色法判断二分图**

~~~
class Solution {
    int N = 2010, M =  2 * 10010;
    int[] h, e, ne, colour;
    int idx;

    public void add(int a, int b) {
        e[idx] = b;
        ne[idx] = h[a];
        h[a] = idx ++ ;
    }

    public boolean dfs(int u, int cur) {
        colour[u] = cur;
        for (int i = h[u]; i != -1; i = ne[i]) {
            int j = e[i];
            if (colour[j] == cur) return false;
            if (colour[j] == 0 && !dfs(j, 3 - cur)) return false;
        }
        return true;
    }

    public boolean possibleBipartition(int n, int[][] dislikes) {
        h = new int[N];
        e = new int[M];
        ne = new int[M];
        colour = new int[N];
        Arrays.fill(h, -1);
        for (int[] e : dislikes) {
            int a = e[0], b = e[1];
            add(a, b);
            add(b, a);
        }

        for (int i = 1; i <= n; i ++ ) {
            if (colour[i] != 0) continue;
            if (!dfs(i, 1)) return false;
        }
        return true;
    }
}
~~~

#### [112. 路径总和](https://leetcode.cn/problems/path-sum/)

~~~
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        return dfs(root, targetSum);
    }
    public boolean dfs(TreeNode root, int target) {
        if (root == null) return false;
        target -= root.val;
        if (root.left == null && root.right == null && target == 0)
            return true;
        return dfs(root.left, target) || dfs(root.right, target);
    }
}
~~~

#### [784. 字母大小写全排列](https://leetcode.cn/problems/letter-case-permutation/)

~~~
class Solution {
    List<String> res = new ArrayList<>();
    public List<String> letterCasePermutation(String S) {
        dfs(new StringBuilder(S), 0);
        return res;
    }

    public void dfs(StringBuilder s, int u) {
        if (u == s.length()) {
            res.add(s.toString());
        } else {
            dfs(s, u + 1);
            if (Character.isLetter(s.charAt(u))) {
                s.setCharAt(u, (char)(s.charAt(u) ^ 32));
                dfs(s, u + 1);
                s.setCharAt(u, (char)(s.charAt(u) ^ 32));
            }
        }
    }
}
~~~

#### [6240. 树上最大得分和路径](https://leetcode.cn/problems/most-profitable-path-in-a-tree/)

~~~
class Solution {
    int n;
    int[] bt, p, w;
    List<Integer>[] g;
    public void dfs1(int u, int fa) {
        for (var v : g[u]) {
            if (v == fa) continue;
            p[v] = u;
            dfs1(v, u);
        }
    }

    public int dfs2(int u, int fa, int t) {
        var val = 0;
        if (bt[u] == -1 || t < bt[u]) val = w[u];
        else if (bt[u] == t) val = w[u] / 2;
        var mx = (int) -2e9;
        for (var v : g[u]) {
            if (v == fa) continue;
            mx = Math.max(mx, dfs2(v, u, t + 1));
        }
        if (mx == (int) -2e9) mx = 0;
        return mx + val;
    }

    public int mostProfitablePath(int[][] edges, int bob, int[] amount) {
        w = amount;
        n = amount.length;
        bt = new int[n];
        p = new int[n];
        g = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (var e : edges) {
            int a = e[0], b = e[1];
            g[a].add(b);
            g[b].add(a);
        }
        //找到每个点的父节点
        dfs1(0, -1);
        Arrays.fill(bt, -1);
        var t = 0;
        while (true) {
            bt[bob] = t;
            t ++ ;
            if (bob == 0) break;
            bob = p[bob];
        }
        return dfs2(0, -1, 0);
    }
}
~~~



## 并查集

#### [547. 省份数量](https://leetcode-cn.com/problems/number-of-provinces/)

~~~
class Solution {
    int[] p;
    public int find(int x) {
        if (x != p[x]) p[x] = find(p[x]);
        return p[x];
    }
    public void union(int a, int b) {
        p[find(a)] = find(b);
    }
    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        p = new int[n];
        for (int i = 0; i < n; i ++ ) p[i] = i;
        int res = n;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ )
                if (isConnected[i][j] == 1 && find(i) != find(j)) {
                    union(i, j);
                    res -- ;
                }
        return res;
    }
}
~~~



#### [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

~~~
class Solution {
    int[] p, size;
    public int find(int x) {
        if (x != p[x]) p[x] = find(p[x]);
        return p[x];
    }
    public void union(int a, int b) {
        size[find(b)] += size[find(a)];
        p[find(a)] = find(b);
    }
    public int longestConsecutive(int[] nums) {
        int n = nums.length;
        p = new int[n];
        size = new int[n];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i ++ ) {
            p[i] = i;
            map.put(nums[i], i);
        }
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i ++ ) 
            if (map.containsKey(nums[i] + 1)) 
                if (find(map.get(nums[i])) != find(map.get(nums[i] + 1)))
                    union(map.get(nums[i]), map.get(nums[i] + 1));
            
        int res = 0;
        for (int x : size) 
            res = Math.max(res, x);
        return res;
    }
}
~~~



#### [684. 冗余连接](https://leetcode-cn.com/problems/redundant-connection/)

~~~
class Solution {
    int[] p;
    public int find(int x) {
        if (x != p[x]) p[x] = find(p[x]);
        return p[x];
    } 
    public void union(int a, int b) {
        p[find(a)] = p[find(b)];
    }
    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        p = new int[n + 1];
        for (int i = 0; i <= n; i ++ ) p[i] = i;
        for (int[] e : edges) {
            int a = e[0], b = e[1];
            if (find(a) == find(b)) return e;
            else union(a, b);
        }
        return new int[]{};
    }
}
~~~

~~~
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        p = [0] * (n + 1)
        def find(x):
            if x != p[x]: p[x] = find(p[x])
            return p[x]
        def union(a, b):
            p[find(a)] = find(b)
         
        for i in range(0, n + 1):
            p[i] = i
        for e in edges:
            a, b = e[0], e[1]
            if find(a) == find(b): return e
            else: union(a, b)
        return []
~~~



#### [1020. 飞地的数量](https://leetcode-cn.com/problems/number-of-enclaves/)

**解法：dfs&并查集**

~~~
class Solution {
    int[] p;
    int[][] g;
    int[] dx = new int[]{-1, 0, 1, 0}, dy = new int[]{0, 1, 0, -1};
    int n, m;
    public int find(int x) {
        if (x != p[x]) p[x] = find(p[x]); 
        return p[x];
    }

    public void union(int a, int b) {
        p[find(a)] = find(b);
    }

    public int getIdx(int x, int y) {
        return x * m + y;
    }
    public int numEnclaves(int[][] _g) {
        g = _g;
        n = g.length;
        m = g[0].length;
        p = new int[500 * 500];
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                p[getIdx(i, j)] = getIdx(i, j);
            }
        }
        int ans = 0;
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                if (i == 0 || j == 0 || i == n - 1 || j == m - 1) {
                    if (g[i][j] == 1 && find(getIdx(i, j)) != 0) {
                        dfs(i, j);
                    }
                }
            }
        }

        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                if (g[i][j] == 1 && find(getIdx(i, j)) != 0) ans ++ ;
            }
        }
        return ans;
    }

    public void dfs(int x, int y) {
        union(getIdx(x, y), 0);
        for (int i = 0; i < 4; i ++ ) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && g[a][b] == 1 && find(getIdx(a, b)) != 0) {
                dfs(a, b);
            }
        }
    }
}
~~~

~~~
//dfs
class Solution {
    int[][] g;
    int n, m;
    public int numEnclaves(int[][] grid) {
        g = grid;
        n = g.length;
        m = g[0].length;
        for (int i = 0; i < n; i ++ ) {
            if (g[i][0] == 1) dfs(i, 0); 
            if (g[i][m - 1] == 1) dfs(i, m - 1);
        }
        for (int i = 0; i < m; i ++ ) {
            if (g[0][i] == 1) dfs(0, i);
            if (g[n - 1][i] == 1) dfs(n - 1, i);
        }
        int ans = 0;
        for (int i = 0; i < n; i ++ ) 
            for (int j = 0; j < m; j ++ ) 
                if (g[i][j] == 1) ans ++ ;
        return ans;
    }
    int[][] dirs = new int[][]{{-1, 0}, {0, 1},{1, 0}, {0, -1}};
    public void dfs(int i, int j) {
        g[i][j] = 0;
        for (int[] d : dirs) {
            int x = i + d[0], y = j + d[1];
            if (x < 0 || x >= n) continue;
            if (y < 0 || y >= m) continue;
            if (g[x][y] != 1) continue;
            dfs(x, y);
        }
    }
}
~~~



#### [6106. 统计无向图中无法互相到达点对数](https://leetcode.cn/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/)

~~~
class Solution {
    int[] p, size;
    public int find(int x) {
        if (x != p[x]) p[x] = find(p[x]); 
        return p[x];
    }

    public void union(int a, int b) {
        size[find(b)] += size[find(a)];
        p[find(a)] = find(b);
    }
    public long countPairs(int n, int[][] edges) {
        p = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i ++ ) {
            p[i] = i;
            size[i] = 1;
        }
        for (int[] e : edges) {
            if (find(e[0]) != find(e[1])) {
                union(e[0], e[1]);
            } 
        }
        long ans =  n * (n - 1l);
        //总数减去能互相到达的就是互相不能到达的
        for(int i = 0; i < n; i++ ) if (find(i) == i) ans -= size[i] * (size[i] - 1l);
        return ans >> 1;
    }
}
~~~

#### [952. 按公因数计算最大组件大小](https://leetcode.cn/problems/largest-component-size-by-common-factor/)

~~~
class Solution {
    int[] p, s;
    public int find(int x) {
        if (x != p[x]) p[x] = find(p[x]); 
        return p[x];
    }

    public void union(int a, int b) {
        s[find(b)] += s[find(a)];
        p[find(a)] = find(b);
    }
    public int largestComponentSize(int[] nums) {
        int n = nums.length;
        p = new int[n];
        s = new int[n];
        for (int i = 0; i < n; i ++ ) {
            p[i] = i;
            s[i] = 1;
        }
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < n; i ++ ) {
            int x = nums[i];
            for (int j = 1; j * j <= x; j ++ ) {
                if (x % j == 0) {
                    List<Integer> l1 = map.getOrDefault(j, new ArrayList<>());
                    List<Integer> l2 = map.getOrDefault(x / j, new ArrayList<>());
                    if (j > 1) l1.add(i);
                    map.put(j, l1);
                    l2.add(i);
                    map.put(x / j, l2);
                }
            }
        }
        int res = 1;
        for (int k : map.keySet()) {
            List<Integer> list = map.get(k);
            for (int i = 1; i < list.size(); i ++ ) {
                int a = list.get(0), b = list.get(i);
                if (find(a) != find(b))
                    union(a, b);
                res = Math.max(res, s[find(a)]);
            }
        }
        return res;
    }
}
~~~



#### [785. 判断二分图](https://leetcode.cn/problems/is-graph-bipartite/)

~~~
class Solution {
    int[] p;
    public int find(int x) {
        if (x != p[x]) x = find(p[x]);
        return p[x];
    }

    public void union(int a, int b) {
        p[find(a)] = find(b);
    }

    public boolean isBipartite(int[][] g) {
        p = new int[g.length];
        for (int i = 0; i < g.length; i ++ ) p[i] = i;

        for (int i = 0; i < g.length; i ++ ) {
            for (int x : g[i]) {
                if (find(x) == find(i)) return false;
                else union(g[i][0], x);
            }
        }
        return true;
    }
}
~~~

#### [827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/)

~~~
class Solution {
    int[] p, sz;
    int[][] dirs = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    int n;
    public int find(int x) {
        if (p[x] != x) p[x] = find(p[x]);
        return p[x];
    }
    public void union(int a, int b) {
        sz[find(b)] += sz[find(a)];
        p[find(a)] = find(b);
    }
    public int get(int x, int y) {
        return x * n + y;
    }
    public int largestIsland(int[][] g) {
        n = g.length;
        int res = 0;
        p = new int[n * n];
        sz = new int[n * n];
        for (int i = 0; i < n * n; i ++ ) {
            p[i] = i;
            sz[i] = 1;
        }
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ )
                if (g[i][j] == 1) {
                    int a = get(i, j);
                    for (int[] d : dirs) {
                        int x = i + d[0], y = j + d[1];
                        int b = get(x, y);
                        if (x >= 0 && x < n && y >= 0 && y < n && g[x][y] == 1)
                            if (find(a) != find(b))
                                union(a, b);
                    }
                    res = Math.max(res, sz[find(a)]);
                }
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ ) 
                if (g[i][j] == 0) {
                    var set = new HashSet<Integer>();
                    int tot = 1;
                    for (int[] d : dirs) {
                        int x = i + d[0], y = j + d[1];
                        int t = get(x, y);
                        if (x >= 0 && x < n && y >= 0 && y < n) 
                            if (g[x][y] == 1)
                                if (!set.contains(find(t))) {
                            tot += sz[find(t)];
                            set.add(find(t));
                        }
                    }
                    res = Math.max(res, tot);
                }
        return res;
    }
}
~~~



#### [695. 岛屿的最大面积](https://leetcode.cn/problems/max-area-of-island/)

~~~
class Solution {
    int[] p, sz;
    int[][] dirs = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    int n, m;
    public int get(int x, int y) {
        return x * m + y;
    }
    public int find(int x) {
        if (p[x] != x) p[x] = find(p[x]);
        return p[x];
    }
    public void union(int a, int b) {
        sz[find(b)] += sz[find(a)];
        p[find(a)] = find(b);
    }
    public int maxAreaOfIsland(int[][] g) {
        n = g.length;
        m = g.length;
        p = new int[n * m];
        sz = new int[n * m];
        for (int i = 0; i < n * m; i ++ ) {
            p[i] = i;
            sz[i] = 1;
        }
        for (int i = 0; i < n; i ++ ) 
            for (int j = 0; j < m; j ++ )
                if (g[i][j] == 1) {
                    int a = get(i, j);
                    for (int[] d : dirs) {
                        int x = i + d[0], y = j + d[1];
                        int b = get(x, y);
                        if (x >= 0 && x < n && y > 0 && y < m)
                            if (g[x][y] == 1)
                                if (find(a) != find(b))
                                    union(a, b);
                    }
                }
        int res = 0;
        for (int x : sz) res = Math.max(res, x);
        return res;
    }
}
~~~

#### [886. 可能的二分法](https://leetcode.cn/problems/possible-bipartition/)

~~~
class Solution {
    int[] p;
    List<Integer>[] g;

    public int find(int x) {
        if (x != p[x]) p[x] = find(p[x]);
        return p[x];
    }

    public void union(int a, int b) {
        p[find(a)] = find(b);
    }

    public boolean possibleBipartition(int n, int[][] dislikes) {
        p = new int[n + 1];
        g = new List[n + 1];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (int i = 0; i <= n; i ++ ) p[i] = i;
        for (int[] e : dislikes) {
            g[e[0]].add(e[1]);
            g[e[1]].add(e[0]);
        }

        for (int i = 1; i <= n; i ++ ) {
            for (int w : g[i]) {
                if (find(w) == find(i)) return false;
                else union(g[i].get(0), w);
            }
        }
        return true;
    }
}
~~~



## 拓扑排序

#### [207. 课程表](https://leetcode.cn/problems/course-schedule/)

~~~
class Solution {
    public boolean canFinish(int n, int[][] prerequisites) {
        //统计每个点入度
        int[] d = new int[n];
        List<Integer>[] g = new List[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        //每个点插入邻接表
        for (int[] e : prerequisites) {
            int b = e[0], a = e[1];
            g[a].add(b);
            d[b] ++ ;
        }
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i ++ ) if (d[i] == 0) q.add(i);
        int cnt = 0;
        while (!q.isEmpty()) {
            int t = q.poll();
            cnt ++ ;
            //枚举当前点的所有后继节点
            for (int i : g[t]) {
                if (-- d[i] == 0) q.add(i);
            }
        }
        //如果不等于n说明不存在拓扑序
        return cnt == n;
    }
}
~~~

~~~
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g = collections.defaultdict(list)
        d = [0] * numCourses
        for e in prerequisites:
            b, a = e[1], e[0]
            g[a].append(b)
            d[b] += 1
        q = collections.deque([x for x in range(numCourses) if d[x] == 0])
        visit = 0
        while q:
            visit += 1
            cur = q.popleft()
            for v in g[cur]:
                d[v] -= 1
                if d[v] == 0:
                    q.append(v)
        return visit == numCourses
~~~



#### [210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii/)

~~~
class Solution {
    public int[] findOrder(int n, int[][] pre) {
        int[] d = new int[n];//每个点的入度
        List<Integer>[] g = new List[n];
        for (int i = 0; i < n; i ++ ) g[i] = new ArrayList<>();
        for (int[] p : pre) {
            int b = p[0], a = p[1];
            d[b] ++ ;
            g[a].add(b);
        }
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i ++ ) if (d[i] == 0) q.add(i);
        int[] res = new int[n];
        int idx = 0;
        while (!q.isEmpty()) {
            int cur = q.poll();
            res[idx ++ ] = cur;
            for (int i : g[cur]) 
                if (-- d[i] == 0)
                    q.add(i);
        }
        return idx == n ? res : new int[0];
    }
}
~~~

#### [802. 找到最终的安全状态](https://leetcode.cn/problems/find-eventual-safe-states/)

~~~
class Solution {
    public List<Integer> eventualSafeNodes(int[][] graph) {
        int n = graph.length;
        List<Integer> ans = new ArrayList<>();
        int[] d = new int[n];
        List<List<Integer>> g = new ArrayList<>();
        for (int i = 0; i < n; i ++ ) g.add(new ArrayList<>());
        //拓扑排序找的是入度为0的点，该题找的是出度为0
        //建立反向图
        for (int i = 0; i < n; i ++ ) {
            for (int b : graph[i]) {
                int a = i;
                g.get(b).add(a);
                d[a] ++ ;
            }
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; i ++ ) if (d[i] == 0) q.add(i);
        while (!q.isEmpty()) {
            int t = q.poll();
            for (int i : g.get(t)) {
                if (-- d[i] == 0) q.add(i);
            }
        }
        for (int i = 0; i < n; i ++ ) if (d[i] == 0) ans.add(i);
        return ans;
    }
}
~~~

#### [310. 最小高度树](https://leetcode.cn/problems/minimum-height-trees/)

~~~
class Solution {
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) return List.of(0);
        List<List<Integer>> g = new ArrayList<>();
        int[] d = new int[n];
        boolean[] st = new boolean[n];
        for (int i = 0; i < n; i ++ ) g.add(new ArrayList<>());
        for (int[] e : edges) {
            int b = e[0], a = e[1];
            g.get(a).add(b);
            g.get(b).add(a);
            d[a] ++ ;
            d[b] ++ ;
        }
        Queue<Integer> q = new LinkedList<>();
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < n; i ++ ) if (d[i] == 1) q.add(i);
        int cnt = 0;
        while (cnt < n - 2) {
            int m = q.size();
            cnt += m; 
            for (int i = 0; i < m; i ++ ) {
                int t = q.poll();
                st[t] = true;
                for (int u : g.get(t)) {
                    if (st[u]) continue;
                    if (-- d[u] == 1) q.add(u);
                }
            }
        }
        while (!q.isEmpty()) ans.add(q.poll());
        return ans;
    }
}
~~~

#### [1462. 课程表 IV](https://leetcode.cn/problems/course-schedule-iv/)

~~~
class Solution {
    public List<Boolean> checkIfPrerequisite(int n, int[][] prerequisites, int[][] queries) {
        List<Boolean> ans = new ArrayList<>();
        List<List<Integer>> g = new ArrayList<>();
        List<Set<Integer>> pre = new ArrayList<>();
        int[] d = new int[n];
        for (int i = 0; i < n; i ++ ) {
            g.add(new ArrayList<>());
            pre.add(new HashSet<>());
        }
        for (int[] e : prerequisites) {
            int a = e[0], b = e[1];
            g.get(a).add(b);
            d[b] ++ ;
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; i ++ ) if (d[i] == 0) q.add(i);
        while (!q.isEmpty()) {
            int t = q.poll();
            for (int u : g.get(t)) {
                pre.get(u).addAll(pre.get(t));
                pre.get(u).add(t);
                if (-- d[u] == 0) q.add(u);
            }
        }
        for (int[] querie : queries) {
            int b = querie[0], a = querie[1];
            ans.add(pre.get(a).contains(b));
        }
        return ans;
    }
}
~~~

#### [剑指 Offer II 115. 重建序列](https://leetcode.cn/problems/ur2n8P/)

~~~
class Solution {
    public boolean sequenceReconstruction(int[] nums, int[][] sequences) {
        int n = nums.length;
        ArrayDeque<Integer> q = new ArrayDeque<>();
        int[] d = new int[n + 1];//每个点的入度
        List<Integer>[] g = new List[n + 1];
        for (int i = 1; i <= n; i ++ ) 
            g[i] = new ArrayList<>();
        for (int[] s : sequences) {
            for (int i = 0; i < s.length - 1; i ++ ) {
                d[s[i + 1]] ++ ;
                g[s[i]].add(s[i + 1]);
            }
        }
        for (int i = 1; i <= n; i ++ ) 
            if (d[i] == 0) q.add(i);
        int idx = 0;
        while (!q.isEmpty()) {
            if (q.size() > 1) return false;
            int cur = q.poll();
            if (nums[idx ++ ] != cur) return false;
            for (int x : g[cur]) {
                d[x] -- ;
                if (d[x] == 0) q.add(x);
            }
        }
        return idx == n;
    }
}
~~~

#### [2360. 图中的最长环](https://leetcode.cn/problems/longest-cycle-in-a-graph/)

**解法：拓扑排序找环**

~~~
class Solution {
    public int longestCycle(int[] p) {
        int n = p.length, res = -1;
        var st = new boolean[n];
        var q = new ArrayDeque<Integer>();//辅助队列
        var d = new int[n];//入度数组
        for (int x : p)
            if (x != -1)
                d[x] ++ ;
        for (int i = 0; i < n; i ++ )
            if (d[i] == 0) {
                q.add(i);
                st[i] = true;
            }
        while (!q.isEmpty()) {
            int t = q.poll();
            int ne = p[t];
            if (ne != -1 && -- d[ne] == 0) {
                q.add(ne);
                st[ne] = true;
            }
        }

        //剩下的就是环上的点，统计答案
        for (int i = 0; i < n; i ++ ) {
            if (st[i]) continue;//访问过的就跳过
            int j = i, cur = 0;
            while (!st[j]) {
                cur ++ ;
                st[j] = true;
                j = p[j];
            }
            res = Math.max(res, cur);
        }
        return res;
    }
}
~~~



## 贪心

#### [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)

~~~
class Solution {
    public boolean canJump(int[] nums) {
        int n = nums.length;
        for (int i = 0, j = 0; i < n; i ++ ) {
            if (j < i) return false;
            j = Math.max(j, i + nums[i]);
        }
        return true;
    }
}
~~~

#### [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/)

~~~
class Solution {
    public int jump(int[] nums) {
        int n = nums.length;
        //f[i]表示跳到点i的最小步数。
        int[] f = new int[n];
        for (int i = 1, j = 0; i < n; i ++ ) {
        	//j + nums[j] 跳的距离不到i的时候就可以往后走
            while (j + nums[j] < i) j ++ ;
            f[i] = f[j] + 1;
        }
        return f[n - 1];
    }
}
~~~

#### [406. 根据身高重建队列](https://leetcode.cn/problems/queue-reconstruction-by-height/)

~~~
class Solution {
    /**
     * 解题思路：先排序再插入
     * 1.排序规则：按照先H高度降序，K个数升序排序
     * 2.遍历排序后的数组，根据K插入到K的位置上
     *
     * 核心思想：高个子先站好位，矮个子插入到K位置上，前面肯定有K个高个子，矮个子再插到前面也满足K的要求
     * LinkedList:
     *      add(int index, E element) :在此列表中的指定位置插入指定的元素。
     * @param people
     * @return
     */
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (o1, o2) -> o1[0] == o2[0] ? o1[1] - o2[1] : o2[0] - o1[0]);
        LinkedList<int[]> list = new LinkedList<>();
        for (int[] i : people) {
            list.add(i[1], i);
        }
        return list.toArray(new int[list.size()][2]);
    }
}
~~~

#### [134. 加油站](https://leetcode.cn/problems/gas-station/)

~~~
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
    	//rest表示走完后剩余的油量，g为当前油量， st为开始的起点
        int g = 0, st = 0, rest = 0;
        for (int i = 0; i < gas.length; i ++ ) {
            g += gas[i] - cost[i];
            rest += gas[i] - cost[i];;
            if (g < 0) {
                g = 0;
                st = i + 1;
            }
        }
        return rest < 0 ? -1 : st;
    }
}
~~~

#### [2311. 小于等于 K 的最长二进制子序列](https://leetcode.cn/problems/longest-binary-subsequence-less-than-or-equal-to-k/)

~~~
class Solution {
    public int longestSubsequence(String s, int k) {
        int res = 0, n = s.length();
        long t = 0;
        for (int i = n - 1; i >= 0; i -- ) {
            if (s.charAt(i) == '0') res ++ ;
            else {     
            	//超过30后此时1 << 30 大于1e9,之后不用统计1的个数
                if (res <= 30 && t + (1L << res) <= k) {
                    t += 1L << res;
                    res ++ ;
                }
            }
        }
        return res;
    }
}
~~~

#### [6105. 操作后的最大异或和](https://leetcode.cn/problems/maximum-xor-after-operations/)

~~~
class Solution {
    public int maximumXOR(int[] nums) {
        int ans = 0;
        for (int x : nums) ans |= x;
        return ans;
    }
}
~~~

#### [871. 最低加油次数](https://leetcode.cn/problems/minimum-number-of-refueling-stops/)

~~~
class Solution {
    /*想象成不是只在加油站才能加油，而是只要现在需要油，并且之前有加油站
    还没有加油，那么此时就可以加油。这样一来，如果要使得加油次数最少，那么
    只要加油就加最多的油，为了保证时间效率，这里用堆来维护前面的未用过的加油站
    里的油量。需要加油而没有油时(也就是堆为空)，那么就不能够到达，此时返回-1。
    */
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        if(stations.length == 0) return startFuel >= target ? 0 :-1;
        PriorityQueue<Integer> q = new PriorityQueue<Integer>((o1, o2) -> o2 - o1);
        int sum = startFuel;
        int ans = 0;
        for (int i = 0; i < stations.length; i ++ ) {
            while (sum < stations[i][0]) {
                if (q.isEmpty()) return -1;
                else {
                    sum += q.poll();
                    ans ++ ;
                }
            }
            q.add(stations[i][1]);
        }
        while (sum < target) {
            if (q.isEmpty()) return -1;
            else {
                sum += q.poll();
                ans ++ ;
            }
        }
        return ans;
    }
}
~~~

#### [556. 下一个更大元素 III](https://leetcode.cn/problems/next-greater-element-iii/)

~~~
/*1、从数组末尾往前找，找到 第一个 位置 k，使得 ss[k] <= ss[k - 1]
  2、如果不存在这样的 k，则说明数组是不递增的，直接将数组逆转即可。
  3、如果存在这样的 j，则从末尾找到第一个位置 t > k，使得 nums[t] > nums[k - 1]。
  4、交换 nums[t] 与 nums[k - 1]，然后将数组从 j + 1 到末尾部分逆转(也可以直接sort排序k后面的部分)。
*/
class Solution {
    public int nextGreaterElement(int n) {
        String s = String.valueOf(n);
        char[] ss = s.toCharArray();
        int k = s.length() - 1;
        while (k > 0 && ss[k] <= ss[k - 1]) k -- ;
        if (k == 0) return -1;
        int t = k;
        while (t + 1 < s.length() && ss[t + 1] > ss[k - 1]) t ++ ;
        swap(ss, k - 1, t);
        reverse(ss, k, ss.length - 1);
        long ans = 0;
        for (char c : ss) ans = ans * 10 + (c - '0');
        if (ans > Integer.MAX_VALUE) return -1;
        return (int) ans;
    }
    public void swap(char[] ss, int l, int r) {
        char t = ss[l];
        ss[l] = ss[r];
        ss[r] = t;
    }
    public void reverse(char[] ss, int l, int r) {
        while (l < r) {
            swap(ss, l, r);
            l ++ ;
            r -- ;
        }
    }
}
~~~

#### [31. 下一个排列](https://leetcode.cn/problems/next-permutation/)

~~~
class Solution {
    public void nextPermutation(int[] nums) {
        int k = nums.length - 1;
        while (k > 0 && nums[k - 1] >= nums[k]) k -- ;
        if (k == 0) Arrays.sort(nums);
        else {
            int t = k;
            while (t + 1 < nums.length && nums[t + 1] > nums[k - 1]) t ++ ;
            swap(nums, k - 1, t);
            reverse(nums, k, nums.length - 1);
        }
    }
    public void swap(int[] nums, int l, int r) {
        int t = nums[l];
        nums[l] = nums[r];
        nums[r] = t;
    }
    public void reverse(int[] nums, int l, int r) {
        while (l < r) {
            swap(nums, l, r);
            l ++ ;
            r -- ;
        }
    }
}
~~~

#### [1217. 玩筹码](https://leetcode.cn/problems/minimum-cost-to-move-chips-to-the-same-position/)

~~~
class Solution {
    //相当于先把奇数放一起，再把偶数放一起，把少的个数移到多的个数上去
    public int minCostToMoveChips(int[] position) {
        int odd = 0, even = 0;
        for (int p : position) {
            if (p % 2 == 0) odd ++ ;
            else even ++ ;
        }
        return Math.min(odd, even);
    }
}
~~~

#### [6118. 最小差值平方和](https://leetcode.cn/problems/minimum-sum-of-squared-difference/)

~~~
//多路归并贪心
class Solution {
    public long minSumSquareDiff(int[] a, int[] b, int k1, int k2) {
        int n = a.length, m = k1 + k2;
        for (int i = 0; i < n; i ++ ) a[i] = Math.abs(a[i] - b[i]);

        //二分求出切割线
        int l = 0, r = (int)1e5;
        while (l < r) {
            int mid = l + r >> 1;
            long sum = 0;
            for (int x : a) if (x > mid) sum += x - mid;
            if (sum <= m) r = mid;
            else l = mid + 1;
        }

        //求出剩余的操作数
        int sum = 0;
        for (int x : a) if (x > r) sum += x - r;
        m -= sum;
        
        //计算答案
        long ans = 0;
        for (int i = 0; i < n; i ++ ) {
            //如果比分割线阈值大
            if (a[i] >= r) {
                //在操作数剩余情况下可以进一步切割
                if (r > 0 && m > 0) {
                    ans += (long)(r - 1) * (r - 1);
                    m -- ;
                }
                else ans += (long)r * r; 
            } else {
                ans += (long)a[i] * a[i];
            }
        }
        return ans;
    }
}
~~~

#### [757. 设置交集大小至少为2](https://leetcode.cn/problems/set-intersection-size-at-least-two/)

**解法：贪心**

~~~
class Solution {
    public int intersectionSizeTwo(int[][] intervals) {
        //将数组按interval[1]升序列，interval[0]降序排列
    	//这是因为后续的线段的右端点大于当前线段右端点，如果
    	//选择一个当前线段前面位置上的点能被后面的线段利用，
    	//当前线段的右端点也一定能被后面的线段利用，即不会得到更坏的结果
        Arrays.sort(intervals, (a, b) -> {
            return a[1] != b[1] ? a[1] - b[1] : b[0] - a[0];
        });
        int cnt = 0;
        int v1 = -1, v2 = -1;
        //基于上述思想，我们也知道我们集合S中的点也是按照升序加入进来的，
        //接下来我们考虑已经加入的点集合S共cnt个点中最后两个点S[cnt - 1],S[cnt]和新线段interval的关系
        for (int[] interval : intervals) {
            //如果A[0] > S[cnt],说明当前已经选择的点必然不能不被A利用，所以我们要将A中最后两个点加入集合S中
            if (interval[0] > v2) {
                v1 = interval[1] - 1;
                v2 = interval[1];
                cnt += 2;
            } else if (interval[0] > v1) {
            //如果A[0] <= S[cnt - 1]，那么说明S[cnt - 1],S[cnt]都在线段A中，我们无需加入额外的点。
			//否则的话S[cnt - 1] < A[0] <= S[cnt]，这时候
			//S[cnt]在线段A中，我们只需要将A中最后一个端点加入进来就可以了
                v1 = v2;
                v2 = interval[1];
                cnt ++ ;
            }
        }
        return cnt;
    }
}
~~~

#### [1675. 数组的最小偏移量](https://leetcode.cn/problems/minimize-deviation-in-array/)

**解法：贪心**

~~~
class Solution {
    public int minimumDeviation(int[] nums) {
        TreeSet<Integer> set = new TreeSet<>();
        for (int x : nums) {
            if (x % 2 == 1) x *= 2;
            set.add(x);
        }
        int res = set.last() - set.first();
        while (set.last() % 2 == 0) {
            int x = set.last();
            set.remove(x);
            set.add(x / 2);
            res = Math.min(res, set.last() - set.first());
        }
        return res;
    }
}
~~~

~~~
class Solution {
    public int minimumDeviation(int[] nums) {
        PriorityQueue<Integer> q = new PriorityQueue<>((o1, o2) -> o2 - o1);
        int min = Integer.MAX_VALUE;
        Set<Integer> set = new HashSet();
        for (int x : nums) {
            if (x % 2 != 0) x *= 2;
            set.add(x);
            min = Math.min(min, x);
        }
        for (int x : set) q.add(x);
        int max = q.peek();
        int res = max - min;
        while (!q.isEmpty() && q.peek() % 2 == 0) {
            int x = q.poll();
            x /= 2;
            min = Math.min(min, x);
            q.add(x);
            if (!q.isEmpty()) max = q.peek();
            res = Math.min(res, max - min);
        }
        return res;
    }
}
~~~

#### [6174. 任务调度器 II](https://leetcode.cn/problems/task-scheduler-ii/)

**解法：贪心**

~~~
class Solution {
    public long taskSchedulerII(int[] q, int space) {
        int n = q.length;
        long res = 0;
        Map<Integer, Long> map = new HashMap<>();
        for (int x : q) {
            res ++ ;
            if (!map.containsKey(x))
                map.put(x, res);
            else {
                if (res - map.get(x) <= space) 
                    res = map.get(x) + space + 1;
                map.put(x, res);
            }
        }
        return res;
    }
}
~~~

#### [6144. 将数组排序的最少替换次数](https://leetcode.cn/problems/minimum-replacements-to-sort-the-array/)

**解法：贪心**

~~~
class Solution {
    public long minimumReplacement(int[] nums) {
        long res = 0;
        int n = nums.length, last = nums[n - 1];
        for (int i = n - 2; i >= 0; i -- ) {
            if (nums[i] > last) {
            	//拆的次数
                int x = (nums[i] + last - 1) / last;
                res += x - 1;
                //拆出来的最小数，nums[i]/(拆的次数)向下取整
                last = nums[i] / x;
            } else {
                last = nums[i];
            }
        }
        return res;
    }
}

~~~

#### [179. 最大数](https://leetcode.cn/problems/largest-number/)

**解法：排序贪心**

~~~
class Solution {
    public String largestNumber(int[] nums) {
        int n = nums.length;
        String[] arr = new String[n];
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < n; i ++ )
            arr[i] = String.valueOf(nums[i]);
        Arrays.sort(arr, (o1, o2) -> (o2 + o1).compareTo(o1 + o2));
        for (String x : arr) 
            res.append(x);
        if (res.charAt(0) == '0') return "0";
        return res.toString();
    }
}
~~~

#### [316. 去除重复字母](https://leetcode.cn/problems/remove-duplicate-letters/)

**解法：贪心&单调栈**

~~~
class Solution {
    public String removeDuplicateLetters(String s) {
        StringBuilder res = new StringBuilder();
        boolean[] st = new boolean[128];
        int[] last = new int[128];
        for (int i = 0; i < s.length(); i ++ ) last[s.charAt(i)] = i;
        for (int i = 0; i < s.length(); i ++ ) {
            char x = s.charAt(i);
            if (st[x]) continue;
            while (res.length() > 0 && res.charAt(res.length() - 1) > x && last[res.charAt(res.length() - 1)] > i) {
                st[res.charAt(res.length() - 1)] = false;
                res.deleteCharAt(res.length() - 1);
            }
            res.append(x);
            st[x] = true;
        }
        return res.toString();
    }
}
~~~

#### [435. 无重叠区间](https://leetcode.cn/problems/non-overlapping-intervals/)

**解法：排序&贪心**

~~~
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> a[1] - b[1]);
        int res = 0;
        for (int i = 1, pre = intervals[0][1]; i < intervals.length; i ++ ) {
            int l = intervals[i][0], r = intervals[i][1];
            if (l < pre) {
                res ++ ;
            } else pre = r;
        }
        return res;
    }
}
~~~

#### [330. 按要求补齐数组](https://leetcode.cn/problems/patching-array/)

~~~
class Solution {
    public int minPatches(int[] nums, int n) {
        int res = 0;
        //记录可以遍历到的最大数
        long s = 0;
        int pos = 0;
        //遍历目标数
        for (long i = 1; i <= n; ) {
            if (pos >= nums.length || i < nums[pos]) {
                res ++ ;
                s += i;
            } else {
                s += nums[pos];
                pos ++ ;
            }
            i = s + 1;
        }
        return res;
    }
}
~~~



#### [321. 拼接最大数](https://leetcode.cn/problems/create-maximum-number/)

~~~
class Solution {
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int n = nums1.length, m = nums2.length;
        int[] res = new int[k];
        for (int s = Math.max(0, k - m); s <= Math.min(n, k);  s ++ ) {
            int[] a = select(nums1, s);
            int[] b = select(nums2, k - s);
            int[] c = merge(a, b);
            if (greater(c, 0, res, 0)) res = c;
        }
        return res;
    }

    public int[] merge(int[] a, int[] b) {
        int i = 0, j = 0, k = 0;
        int[] res = new int[a.length + b.length];
        while (i < a.length && j < b.length) {
            if (greater(a, i, b, j)) res[k ++ ] = a[i ++ ];
            else res[k ++ ] = b[j ++ ];
            //System.out.println(res[k - 1]);
        }
        while (i < a.length) res[k ++ ] = a[i ++ ];
        while (j < b.length) res[k ++ ] = b[j ++ ];
        return res;
    }

    public int[] select(int[] arr, int k) {
        int[] stack = new int[k];
        int n = arr.length;
        int tt = 0;
        for (int i = 0; i < n; i ++ ) {
            int x = arr[i];
            //当前选择的数+后面能选的数大于k才去删除
            while (tt > 0 && stack[tt - 1] < x && tt + n - i > k) tt -- ;
            if (tt < k) stack[tt ++ ] = x; 
        }
        return stack;
    }
	//数组的比较方法
    public boolean greater(int[] a, int i, int[] b, int j) {
        int n = a.length, m = b.length;
        while (i < n && j < m && a[i] == b[j]) {
            i ++ ;
            j ++ ;
        }
        if (j == m || (i < n && a[i] > b[j])) return true;
        return false;
    } 
}
~~~



#### [452. 用最少数量的箭引爆气球](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/)

**解法：排序贪心**

~~~
class Solution {
    public int findMinArrowShots(int[][] p) {
        Arrays.sort(p, (a, b) -> {
            return a[1] < b[1] ? -1 : 1;
        });
        int res = 1, last = p[0][1];
        for (int i = 1; i < p.length; i ++ ) {
            if (last < p[i][0]) {
                res ++ ;
                last = p[i][1];
            }
        }
        return res;
    }
}
~~~



#### [632. 最小区间](https://leetcode.cn/problems/smallest-range-covering-elements-from-k-lists/)

**解法：贪心&优先队列**

~~~
class Solution {
    public int[] smallestRange(List<List<Integer>> nums) {
        int[] res = new int[]{(int)-1e5, (int)1e5};
        PriorityQueue<int[]> q = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));
        int max_r = 0;
        for (int i = 0; i < nums.size(); i ++ ) {
            q.add(new int[]{nums.get(i).get(0), i, 0}); 
            max_r = Math.max(max_r, nums.get(i).get(0));
        }
        
        while (!q.isEmpty()) {
            int[] t = q.poll();
            int l = t[0], r = max_r;
            if (res[1] - res[0] > r - l) res = new int[]{l, r};
            int i = t[1], j = t[2] + 1;
            if (j < nums.get(i).size()) {
                int x = nums.get(i).get(j);
                q.add(new int[]{x, i, j});
                max_r = Math.max(max_r, x);
            } else break;
        }
        
        return res;
    }
}
~~~



#### [738. 单调递增的数字](https://leetcode.cn/problems/monotone-increasing-digits/)

**解法：贪心**

~~~
class Solution {
    /*
    * 从右往左扫，若当前位比前一位小，将该位及其后边设为9，前一位减一
     */
    public int monotoneIncreasingDigits(int n) {
        char[] s = String.valueOf(n).toCharArray();
        int j = s.length;
        for (int i = s.length - 1; i > 0; i -- ) {
            if (s[i] < s[i - 1]) {
                j = i;
                s[i - 1] -- ;
            }
        } 
        for (int i = j; i < s.length; i ++ ) s[i] = '9';
        return Integer.parseInt(new String(s));
    }
}
~~~



#### [646. 最长数对链](https://leetcode.cn/problems/maximum-length-of-pair-chain/)

~~~
class Solution {
    public int findLongestChain(int[][] p) {
        Arrays.sort(p, (a, b) -> a[1] - b[1]);
        int res = 1, b = p[0][1];
        for (int i = 0; i < p.length; i ++ ) 
            if (p[i][0] > b) {
                res ++ ;
                b = p[i][1];
            }
        return res;
    }
}
~~~



#### [1386. 安排电影院座位](https://leetcode.cn/problems/cinema-seat-allocation/)

**解法：贪心&位运算&哈希**

~~~
class Solution {
    public int maxNumberOfFamilies(int n, int[][] rs) {
        n <<= 1;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < rs.length; i ++ ) 
            map.put(rs[i][0], map.getOrDefault(rs[i][0], 0) | (1 << rs[i][1] - 1));
        for (int state : map.values()) {
            if ((state & 510) == 0) continue;
            else if ((state & 30) == 0||(state & 120) == 0||(state & 480) == 0) n -- ;
            else n -= 2;
        }
        return n;
    }
}
~~~

#### [857. 雇佣 K 名工人的最低成本](https://leetcode.cn/problems/minimum-cost-to-hire-k-workers/)

~~~
class Solution {
    public double mincostToHireWorkers(int[] qs, int[] ws, int k) {
        int n = qs.length;
        List<Pair<Double, Integer>> ds = new ArrayList<>();
        for (int i = 0; i < n; i ++ )
            ds.add(new Pair<>(ws[i] * 1.0 / qs[i], i));
        Collections.sort(ds, (a, b) -> {
            if (a.getKey() == b.getKey()) return 0;
            else return a.getKey() > b.getKey() ? 1 : -1;
        });
        PriorityQueue<Integer> q = new PriorityQueue<>((a, b) -> b - a);
        double res = 1e18;
        for (int i = 0, tot = 0; i < n; i ++ ) {
            int cur = qs[ds.get(i).getValue()];
            tot += cur;
            q.add(cur);
            if (q.size() > k) tot -= q.poll();
            if (q.size() == k) res = Math.min(res, ds.get(i).getKey() * tot);
        } 
        return res;
    }
}
~~~

#### [2182. 构造限制重复的字符串](https://leetcode.cn/problems/construct-string-with-repeat-limit/)

~~~
class Solution {
    public String repeatLimitedString(String s, int repeatLimit) {
        var map = new int[26];
        for (char c : s.toCharArray()) map[c - 'a'] ++ ;
        var res = new StringBuilder();
        for (int i = 25; i >= 0; i -- ) {
            while (map[i] > 0) {
                int min = Math.min(map[i], repeatLimit);
                map[i] -= min;
                while (min -- > 0) res.append((char)(i + 'a'));
                if (map[i] == 0) break;
                int k = i - 1;
                while (k >= 0 && map[k] == 0) k -- ;
                if (k < 0) break;
                else {
                    map[k] -- ;
                    res.append((char)(k + 'a'));
                }
            }
        }
        return res.toString();
    }
}
~~~



## 位运算

#### [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

~~~
class Solution {
    public int[] singleNumbers(int[] nums) {
        int t = 0;
        int n = nums.length;
        for (int i : nums) t ^= i;
        //得到最低为的1，从而进行分组
        int flag = t & (-t);
        int res = 0;
        for (int i : nums) if ((flag & i) != 0) res ^= i;
        //利用自反得到另一个答案
        return new int[]{res, t ^ res};
    }
}
~~~

#### [剑指 Offer 65. 不用加减乘除做加法](https://leetcode.cn/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

~~~
class Solution {
    public int add(int a, int b) {
        /*
        1、我们发现按位加法（不进位）分四种情况 1 + 1 = 0, 0 + 0 = 0,1 + 0 = 1,
            0 + 1 = 1 刚好与位运算的异或运算结果相同。
        2、进位值1 + 1 = 1,0 + 0 = 0,1 + 0 = 1,0 + 1 = 1 刚好与我们位运算的与运算结果相同。
        进位值需要向前进一位，与位运算左移运算符结果相同。
        3、再接着用代替后的运算方式继续运算 1000 + 01110
        第一位0 ^ 0 = 0, 第二位 0 ^ 1 = 1,第三位 0 ^ 1 = 1,第四位 1 ^ 1 = 0，第五位 0 ^ 0 = 0，结果为 00110
        进位（1）0（个位不需进位），进位（2）0 与 0 = 0,进位（3）0 与 1 = 0,进位（4）0 与 1 = 0,进位（5）
        1 与 1 = 1,结果为 10000
        继续计算 00110 + 10000（省略计算过程）
        得到 10110 与 00000 结果位 10110 （22）
         */
        while (b != 0) {
            var c = a ^ b;
            b = (a & b) << 1;
            a = c;
        }
        return a;
    }
}
~~~

#### [231. 2 的幂](https://leetcode.cn/problems/power-of-two/)

~~~
//lowbit运算
class Solution {
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & -n) == n;
    }
}
~~~

#### [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)

**解法：位运算**

~~~
class Solution {
    public int singleNumber(int[] nums) {
        for (int i = 1; i < nums.length; i ++ ) 
            nums[0] ^= nums[i];
        return nums[0];
    }
}
~~~

#### [491. 递增子序列](https://leetcode.cn/problems/increasing-subsequences/)

**解法：二进制枚举&哈希**

~~~
class Solution {
    public List<List<Integer>> findSubsequences(int[] nums) {
        int n = nums.length;
        Set<List<Integer>> set = new HashSet<>();
        out:for (int k = 0; k <= 1 << n; k ++ ) {
            List<Integer> temp = new ArrayList<>();
            for (int i = 0, pre = -101; i < n; i ++ ) {
                if ((k >> i & 1) != 0) {
                    if (nums[i] < pre) continue out;
                    else {
                        pre = nums[i];
                        temp.add(nums[i]);
                    }
                }
            }
            if (temp.size() >= 2) set.add(new ArrayList<>(temp));
        }
        return new ArrayList<>(set);
    }
}
~~~

#### [201. 数字范围按位与](https://leetcode.cn/problems/bitwise-and-of-numbers-range/)

~~~
class Solution {
    public int rangeBitwiseAnd(int n, int m) {
        var res = 0;
        for (int i = 30; i >= 0; i -- ) {
            if ((n >> i & 1) != (m >> i & 1)) break;
            if ((n >> i & 1) == 1) res += 1 << i;
        }
        return res;
    }
}
~~~

#### [137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii/)

~~~
class Solution {
    public int singleNumber(int[] nums) {
        var res = 0;
        for (var i = 0; i < 32; i ++ ) {
            var counter = 0;
            for (var x : nums)
                counter += (x >> i) & 1;
            res += (counter % 3) << i;
        }
        return res;   
    }
}
~~~

~~~
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for i in range(32):
            counter = sum((x >> i & 1) for x in nums)
            if counter % 3:
                # 这里需要对最高位特殊判断
                if i == 31:
                    res -= 1 << i
                else:
                    res += 1 << i
        return res
~~~



#### [260. 只出现一次的数字 III](https://leetcode.cn/problems/single-number-iii/)

~~~
class Solution {
    public int[] singleNumber(int[] nums) {
        var xor = 0;
        for (int x : nums)
            xor ^= x;
        var k = xor & -xor;
        int num1 = 0, num2 = 0;
        for (int x : nums) {
            if ((x & k) == 0)
                num1 ^= x;
            else 
                num2 ^= x;
        }
        return new int[]{num1, num2};
    }
}
~~~

~~~
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        xor = 0
        for x in nums:
            xor ^= x
        k = xor & -xor
        a, b = 0, 0
        for x in nums:
            if x & k: a ^= x
            else: b ^= x
        return [a, b]
~~~





## 每日一题系列

#### [500. 键盘行](https://leetcode-cn.com/problems/keyboard-row/)

11.1

#### [575. 分糖果](https://leetcode-cn.com/problems/distribute-candies/)

11.8

#### [299. 猜数字游戏](https://leetcode-cn.com/problems/bulls-and-cows/)

11.13

#### [520. 检测大写字母](https://leetcode-cn.com/problems/detect-capital/)

11.15

#### [319. 灯泡开关](https://leetcode-cn.com/problems/bulb-switcher/)

11.29

#### [786. 第 K 个最小的素数分数](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/)

#### [400. 第 N 位数字](https://leetcode-cn.com/problems/nth-digit/)

#### [1005. K 次取反后最大化的数组和](https://leetcode-cn.com/problems/maximize-sum-of-array-after-k-negations/)

#### [383. 赎金信](https://leetcode-cn.com/problems/ransom-note/)

#### [372. 超级次方](https://leetcode-cn.com/problems/super-pow/)

#### [1816. 截断句子](https://leetcode-cn.com/problems/truncate-sentence/)

#### [1034. 边界着色](https://leetcode-cn.com/problems/coloring-a-border/)

dfs + flood fill

#### [748. 最短补全词](https://leetcode-cn.com/problems/shortest-completing-word/)

#### [911. 在线选举](https://leetcode-cn.com/problems/online-election/)

#### [807. 保持城市天际线](https://leetcode-cn.com/problems/max-increase-to-keep-city-skyline/)

#### [630. 课程表 III](https://leetcode-cn.com/problems/course-schedule-iii/)

#### [1518. 换酒问题](https://leetcode-cn.com/problems/water-bottles/)

#### [997. 找到小镇的法官](https://leetcode-cn.com/problems/find-the-town-judge/)

#### [1705. 吃苹果的最大数目](https://leetcode-cn.com/problems/maximum-number-of-eaten-apples/)

#### [825. 适龄的朋友](https://leetcode-cn.com/problems/friends-of-appropriate-ages/)

排序 + 双指针

#### [1995. 统计特殊四元组](https://leetcode-cn.com/problems/count-special-quadruplets/)

哈希

#### [846. 一手顺子](https://leetcode-cn.com/problems/hand-of-straights/)

堆 + 哈希计数

#### [507. 完美数](https://leetcode-cn.com/problems/perfect-number/)

缩小枚举范围

#### [2022. 将一维数组转变成二维数组](https://leetcode-cn.com/problems/convert-1d-array-into-2d-array/)

#### [1185. 一周中的第几天](https://leetcode-cn.com/problems/day-of-the-week/)

#### [89. 格雷编码](https://leetcode-cn.com/problems/gray-code/)

#### [1614. 括号的最大嵌套深度](https://leetcode-cn.com/problems/maximum-nesting-depth-of-the-parentheses/)

#### [1576. 替换所有的问号](https://leetcode-cn.com/problems/replace-all-s-to-avoid-consecutive-repeating-characters/)

#### [1629. 按键持续时间最长的键](https://leetcode-cn.com/problems/slowest-key/)

#### [334. 递增的三元子序列](https://leetcode-cn.com/problems/increasing-triplet-subsequence/)

#### [747. 至少是其他数字两倍的最大数](https://leetcode-cn.com/problems/largest-number-at-least-twice-of-others/)

#### [373. 查找和最小的K对数字](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/)

#### [382. 链表随机节点](https://leetcode-cn.com/problems/linked-list-random-node/)

#### [219. 存在重复元素 II](https://leetcode-cn.com/problems/contains-duplicate-ii/)

滑动窗口 + 哈希表

#### [2029. 石子游戏 IX](https://leetcode-cn.com/problems/stone-game-ix/)

#### [1332. 删除回文子序列](https://leetcode-cn.com/problems/remove-palindromic-subsequences/)

#### [2034. 股票价格波动](https://leetcode-cn.com/problems/stock-price-fluctuation/)

#### [1763. 最长的美好子字符串](https://leetcode-cn.com/problems/longest-nice-substring/)

~~~
 public String longestNiceSubstring(String s) {
        int n = s.length();
        String ans = "";
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (j - i + 1 > ans.length() && check(s.substring(i, j + 1))) ans = s.substring(i, j + 1);
            }
        }
        return ans;
    }
    boolean check(String s) {
        Set<Character> set = new HashSet<>();
        for (char c : s.toCharArray()) set.add(c);
        for (char c : s.toCharArray()) {
            char a = Character.toLowerCase(c), b = Character.toUpperCase(c);
            if (!set.contains(a) || !set.contains(b)) return false;
        }
        return true;
    }

~~~

#### [2000. 反转单词前缀](https://leetcode-cn.com/problems/reverse-prefix-of-word/)

#### [479. 最大回文数乘积](https://leetcode-cn.com/problems/largest-palindrome-product/)

~~~
class Solution {
    public int largestPalindrome(int n) {
        if (n == 1) return 9;
        int max = (int)Math.pow(10, n) - 1;
        for (int i = max; i >= 0; i -- ) {
            String s = String.valueOf(i);
            int m = s.length();
            //枚举回文串前半部分
            for (int j = m - 1; j >= 0; j -- ) s += String.valueOf(s.charAt(j));
            long num = Long.parseLong(s);
            //检查回文串是否能够分解
            for (long j = max; j * j >= num; j -- ) 
                if (num % j == 0) return (int)(num % 1337);
        }
        return -1;
    }
}
~~~

~~~
class Solution {
    public int largestPalindrome(int n) {
        if (n == 1) return 9;
        int max = (int)Math.pow(10,n) - 1;
        for (int i = max; i >= 0; i -- ) {
            long num = i, t = i;
            while (t != 0) {
                num = num * 10 + t % 10;
                t /= 10;
            }
            for (long j = max; j * j >= num; j --) {
                if (num % j == 0) return (int)(num % 1337);
            }
        }
        return -1;
    }
}
~~~



