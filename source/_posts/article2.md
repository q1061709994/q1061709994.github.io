---
icon: article
title: 周赛总结
author: huan
date: 2022-01-02
category: 算法笔记
tag: 
    - 数据结构与算法
star: true
---
## 82场双周赛

#### Problem A - [计算布尔二叉树的值](https://leetcode-cn.com/problems/evaluate-boolean-binary-tree/)

**解法：递归&模拟**

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
    public boolean evaluateTree(TreeNode root) {
        boolean ans = dfs(root);
        return ans;
    }
    public boolean dfs(TreeNode root) {
        if (root.left == null && root.right == null) return root.val == 1 ? true : false;
        else {
            boolean l = dfs(root.left), r = dfs(root.right);
            if (root.val == 2) return l || r;
            else return l && r;
        }
    } 
}
~~~

#### Problem B - [坐上公交的最晚时间](https://leetcode-cn.com/problems/the-latest-time-to-catch-a-bus/)

**解法：贪心&双指针**

**由于求的是最晚到达公交站的时间，根据贪心容易得到，这个时间要么是某个公交的发车时间，要么比某个乘客早到 11 单位时间。因此我们通过 two pointers 的方式模拟上车过程，并枚举所有可能的答案：**

**1、当一个乘客在 t 时刻到达时，我们尝试抢先在它之前上车。只要不存在 (t - 1)(t−1) 时刻到达的乘客即可；**
**2、当公交发车时，若当前公交没有坐满，且不存在发车时到达的乘客，我们可以在这个时刻上车。**

**在所有可以上车的时刻中取最大值即可。复杂度 O(*n*log*n*)。**

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

#### Problem C - [最小差值平方和](https://leetcode-cn.com/problems/minimum-sum-of-squared-difference/)

**解法：二分&贪心**

~~~
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

#### Problem D - [元素值大于变化阈值的子数组](https://leetcode-cn.com/problems/subarray-with-elements-greater-than-varying-threshold/)

**解法：枚举 & 并查集 & 双指针**

## 301场周赛

#### Problem A - [装满杯子需要的最短总时长](https://leetcode-cn.com/problems/minimum-amount-of-time-to-fill-cups/)

**解法：贪心&优先队列 || 排序**

~~~
class Solution {
    public int fillCups(int[] amount) {
        PriorityQueue<Integer> q = new PriorityQueue<>((a, b) -> b - a);
        for (int x : amount) if (x != 0) q.add(x);
        int ans = 0;
        while (!q.isEmpty()) {
            if (q.size() == 1) {
                ans += q.poll();
                break;
            }
            int a = q.poll(), b = q.poll();
            a --; b -- ;
            if (a != 0) q.add(a);
            if (b != 0) q.add(b);
            ans ++ ;
        }
        return ans;
    }
}
~~~

#### Problem B - [无限集中的最小数字](https://leetcode-cn.com/problems/smallest-number-in-infinite-set/)

**解法：模拟&优先队列**

~~~
class SmallestInfiniteSet {
    PriorityQueue<Integer> q = new PriorityQueue<>();
    public SmallestInfiniteSet() {
        for (int i = 1; i <= 1000; i ++ ) q.add(i);
    }
    
    public int popSmallest() {
        return q.poll();
    }
    
    public void addBack(int num) {
        if (!q.contains(num)) {
            q.add(num);
        }
    }
}

/**
 * Your SmallestInfiniteSet object will be instantiated and called as such:
 * SmallestInfiniteSet obj = new SmallestInfiniteSet();
 * int param_1 = obj.popSmallest();
 * obj.addBack(num);
 */

~~~

#### Problem C - [移动片段得到字符串](https://leetcode-cn.com/problems/move-pieces-to-obtain-a-string/)

**解法：双指针**

~~~
class Solution {
    public boolean canChange(String start, String target) {
        int n = start.length();
        char[] s = start.toCharArray(), t = target.toCharArray();
        int j = 0;
        for (int i = 0; i < n; i ++ ) {
            if (s[i] == '_') continue;
            while (j < n && t[j] == '_') j ++ ;
            if (j == n) return false;
            if (s[i] != t[j]) return false;
            if (t[j] == 'L' && j > i) return false;  
            if (t[j] == 'R' && j < i) return false;
            j ++ ;
        }
        for (int i = j; i < n; i ++ ) 
            if (t[i] != '_') 
                return false;
        return true;
    }
}
~~~

#### Problem D - [统计理想数组的数目](https://leetcode-cn.com/problems/count-the-number-of-ideal-arrays/)

**解法：数学 & 递推**

## 302场周赛

#### Problem A - [数组能形成多少数对](https://leetcode-cn.com/problems/maximum-number-of-pairs-in-array/)

**解法：模拟**

~~~
class Solution {
    public int[] numberOfPairs(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : nums) map.put(x, map.getOrDefault(x, 0) + 1);
        int[] res = new int[2];
        for (int k : map.keySet()) {
            res[0] += map.get(k) / 2;
            res[1] += map.get(k) % 2;
        }
        return res;
    }
}
~~~

#### Problem B - [数位和相等数对的最大和](https://leetcode-cn.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/)

**解法：枚举**

~~~
class Solution {
    public int maximumSum(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = -1;
        for (int x : nums) {
            int s = 0, y = x;
            while (y != 0) {
                s += y % 10;
                y /= 10;
            }
            if (map.containsKey(s)) {
                res = Math.max(res, x + map.get(s));
                map.put(s, Math.max(map.get(s), x));
            } else {
                map.put(s, x);
            }
        }
        return res;
    }
}
~~~

#### Problem C - [裁剪数字后查询第 K 小的数字](https://leetcode-cn.com/problems/query-kth-smallest-trimmed-number/)

**解法：字符串排序&第二关键字排序**

~~~
class Solution {
    public int[] smallestTrimmedNumbers(String[] a, int[][] b) {
        int n = a.length, m = a[0].length();
        int[] ans = new int[b.length];
        for (int i = 0; i < b.length; i ++ ) {
            int k = b[i][0], trim = b[i][1];
            String[][] ss = new String[n][2];
            for (int j = 0; j < n; j ++ ) {
                ss[j][0] = a[j].substring(m - trim);
                ss[j][1] = String.valueOf(j);
            }
            Arrays.sort(ss, (o1, o2) -> {
                int x = o1[0].compareTo(o2[0]);
                return x == 0 ? Integer.parseInt(o1[1]) -  Integer.parseInt(o2[1]): x;
            });
            ans[i] = Integer.parseInt(ss[k - 1][1]);
        }
        return ans;
    }
}
~~~



#### Problem D - [使数组可以被整除的最少删除次数](https://leetcode-cn.com/problems/minimum-deletions-to-make-array-divisible/)

**解法：模拟**

~~~
class Solution {
    public int gcd(int a, int b) {
        return b != 0 ? gcd(b, a % b) : a;
    }
    public int minOperations(int[] a, int[] b) {
        int res = 0;
        Arrays.sort(a);
        int d = 0;
        for (var x: b) d = gcd(d, x);
        for (int i = 0; i < a.length; i ++ ) {
            if (d % a[i] == 0) break;
            res ++ ;
        }
        if (res == a.length) res = -1;
        return res;
    }
}
~~~

## 83场双周赛

#### Problem A - [最好的扑克手牌](https://leetcode-cn.com/problems/best-poker-hand/)

**解法：模拟**

~~~
class Solution {
    public String bestHand(int[] ranks, char[] suits) {
        // 记录花色种类的 set
        Set<Character> st = new HashSet();
        // 记录每个数字出现几次
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : ranks) map.put(x, map.getOrDefault(x, 0) + 1);
        for (char c : suits) st.add(c);
        // 只有一种花色
        if (st.size() == 1) return "Flush";
        int mx = 0;
        for (int x : map.keySet()) mx = Math.max(mx, map.get(x));
        // 判断出现最多的数字出现了几次
        if (mx >= 3) return "Three of a Kind";
        else if (mx == 2) return "Pair";
        else return "High Card";
    }
}
~~~

#### Problem B - [全 0 子数组的数目](https://leetcode-cn.com/problems/number-of-zero-filled-subarrays/)

**解法：枚举**

~~~
class Solution {
    public long zeroFilledSubarray(int[] nums) {
        long res = 0;
        int n = nums.length;
        for (int i = 0; i < n; i ++ ) {
            if (nums[i] != 0) continue;
            if (nums[i] == 0) res ++ ;
            int j = i;
            while (j + 1 < n && nums[j + 1] == 0) {
                j ++ ;
                res += j - i + 1;
            }
            i = j;
        }
        return res;
    }
}
~~~

#### Problem C - [设计数字容器系统](https://leetcode-cn.com/problems/design-a-number-container-system/)

**解法：模拟**

~~~
class NumberContainers {
    Map<Integer, Integer> m1 = new HashMap<>();
    Map<Integer, TreeSet<Integer>> m2 = new HashMap<>();
    public NumberContainers() {
        
    }
    
    public void change(int index, int number) {
        if (m1.containsKey(index)) {
            int t = m1.get(index);
            TreeSet<Integer> S = m2.get(t);
            S.remove(index);
            if (S.size() > 0) m2.put(t, S);
            else m2.remove(t);
        }
        m1.put(index, number);
        TreeSet<Integer> q = m2.getOrDefault(number, new TreeSet<Integer>());
        q.add(index);
        m2.put(number, q);
    }
    
    public int find(int number) {
        if (m2.containsKey(number)) return m2.get(number).first();
        else return -1;
    }
}

/**
 * Your NumberContainers object will be instantiated and called as such:
 * NumberContainers obj = new NumberContainers();
 * obj.change(index,number);
 * int param_2 = obj.find(number);
 */
~~~

#### Problem D - [不可能得到的最短骰子序列](https://leetcode-cn.com/problems/shortest-impossible-sequence-of-rolls/)

**解法：脑筋急转弯**

## 303场周赛

#### Problem A - [第一个出现两次的字母](https://leetcode-cn.com/problems/first-letter-to-appear-twice/)

**解法：模拟**

~~~
class Solution {
    public char repeatedCharacter(String s) {
        int[] map = new int[128];
        for (char x : s.toCharArray()) {
            if (map[x] == 1) return x;
            map[x] ++ ;
        }
        return 'a';
    }
}
~~~



#### Problem B - [相等行列对](https://leetcode-cn.com/problems/equal-row-and-column-pairs/)

**解法：模拟**

~~~
class Solution {
    public int equalPairs(int[][] g) {
        int n = g.length, m = g[0].length;
        int res = 0;
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                int t = 0;
                while (t < n && g[i][t] == g[t][j]) t ++ ;
                if (t == n) res ++ ;
            }
        }
        return res;
    }
}
~~~



#### Problem C - [设计食物评分系统](https://leetcode-cn.com/problems/design-a-food-rating-system/)

**解法：模拟**

~~~
class FoodRatings {
    Map<String, TreeSet<Pair<Integer, String>>> hash = new HashMap<>();
    Map<String, String> c = new HashMap<>();
    Map<String, Integer> r = new HashMap<>();
    public FoodRatings(String[] foods, String[] cuisines, int[] ratings) {
        for (int i = 0; i < foods.length; i ++ ) {
            c.put(foods[i], cuisines[i]);
            r.put(foods[i], ratings[i]);
            TreeSet<Pair<Integer, String>> S = hash.getOrDefault(cuisines[i], new TreeSet<>((o1, o2) -> {
                return o1.getKey().equals(o2.getKey()) ? o1.getValue().compareTo(o2.getValue()) : o1.getKey() - o2.getKey();
            }));
            S.add(new Pair<>(-ratings[i], foods[i]));
            hash.put(cuisines[i], S);
        }
    }
    
    public void changeRating(String food, int newRating) {
        String cuisine = c.get(food);
        hash.get(cuisine).remove(new Pair(-r.get(food), food));
        r.put(food, newRating);
        TreeSet<Pair<Integer, String>> S = hash.get(cuisine);
        S.add(new Pair(-newRating, food));
        hash.put(cuisine, S);
    }
    
    public String highestRated(String cuisine) {
        return hash.get(cuisine).first().getValue();
    }
}

/**
 * Your FoodRatings object will be instantiated and called as such:
 * FoodRatings obj = new FoodRatings(foods, cuisines, ratings);
 * obj.changeRating(food,newRating);
 * String param_2 = obj.highestRated(cuisine);
 */
~~~



#### Problem D - [优质数对的数目](https://leetcode-cn.com/problems/number-of-excellent-pairs/)

**解法：模拟**

第i位在两个数里出现几次那么它对答案的贡献就是几

~~~
class Solution {
    public long countExcellentPairs(int[] nums, int k) {
        TreeSet<Integer> set = new TreeSet<>();
        for (int x : nums) set.add(x);
        int[] cnt = new int[30];
        for (int x : set) {
            int t = 0;
            while (x != 0) {
                t += x & 1;
                x >>= 1;
            }
            cnt[t] ++ ;
        }
        long res = 0;
        for (int i = 0; i < 30; i ++ ) {
            for (int j = 0; j < 30; j ++ ) {
                if (i + j >= k) 
                    res += cnt[i] * cnt[j];
            }
        }
        return res;
    }
}
~~~



## 304场周赛

#### Problem A - [使数组中所有元素都等于零](https://leetcode-cn.com/problems/make-array-zero-by-subtracting-equal-amounts/)

**解法：脑筋急转弯**

~~~
class Solution {
    public int minimumOperations(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int x : nums) 
            if (x > 0) 
                set.add(x);
        return set.size();
    }
}
~~~



#### Problem B - [分组的最大数量](https://leetcode-cn.com/problems/maximum-number-of-groups-entering-a-competition/)

**解法：贪心**

~~~
class Solution {
    public int maximumGroups(int[] g) {
        int res = 0, cnt = 1, n = g.length;
        while (cnt <= n) {
            res ++ ;
            n -= cnt;
            cnt ++ ;
        }
        return res;
    }
}
~~~



#### Problem C - [找到离给定两个节点最近的节点](https://leetcode-cn.com/problems/find-closest-node-to-given-two-nodes/)

**解法：dfs**

~~~
class Solution {
    public int closestMeetingNode(int[] p, int x, int y) {
        int n = p.length;
        int[] d1 = new int[n], d2 = new int[n];
        Arrays.fill(d1, -1); Arrays.fill(d2, -1);
        d1[x] = 0; d2[y] = 0;
        //两次dfs求距离
        while (p[x] != -1) {
            if (d1[p[x]] != -1) break;
            d1[p[x]] = d1[x] + 1;
            x = p[x];
        }
        while (p[y] != -1) {
            if (d2[p[y]] != -1) break;
            d2[p[y]] = d2[y] + 1;
            y = p[y];
        }
        int max = -1, res = -1;
        for (int i = 0; i < n; i ++ ) {
            int a = d1[i], b = d2[i];
            if (a != -1 && b != -1) {
                if (max == -1 || max > Math.max(a, b)) {
                    max = Math.max(a, b);
                    res = i;
                }
            }
        }  
        return res;
    }
}
~~~



#### Problem D - [图中的最长环](https://leetcode-cn.com/problems/longest-cycle-in-a-graph/)

**解法：dfs**

~~~
class Solution {
    int[] p, in_stk;
    boolean[] st;
    int n, res = -1;
    public int longestCycle(int[] _p) {
        n = _p.length;
        p = _p;
        st = new boolean[n];
        in_stk = new int[n];
        for (int i = 0; i < n; i ++ ) 
            if (!st[i])
                dfs(i, 1);
        return res;
    }
    //u：当前搜到的点的下标
    //depth：当前点的深度
    public void dfs(int u, int depth) {
        st[u] = true;
        in_stk[u] = depth;
        int j = p[u];
        if (j != -1) {
            if (!st[j]) 
                dfs(j, depth + 1);
            else if (in_stk[j] > 0)
                // 当in_stk[j]大于0说明搜到了环，并且in_stk[j]为环的起点的深度
                res = Math.max(res, depth + 1 - in_stk[j]);
        }
        in_stk[u] = 0;
    }
}
~~~



## 84场双周赛

#### Problem A - [合并相似的物品](https://leetcode-cn.com/problems/merge-similar-items/)

**解法：模拟**

~~~
class Solution {
    public List<List<Integer>> mergeSimilarItems(int[][] a, int[][] b) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int[] x : a) {
            map.put(x[0], map.getOrDefault(x[0], 0) + x[1]);
        }
        for (int[] x : b) {
            map.put(x[0], map.getOrDefault(x[0], 0) + x[1]);
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int k : map.keySet()) {
            List<Integer> t = new ArrayList<>();
            t.add(k);
            t.add(map.get(k));
            res.add(t);
        }
        return res;
    }
}
~~~



#### Problem B - [统计坏数对的数目](https://leetcode-cn.com/problems/count-number-of-bad-pairs/)

**解法：枚举**

~~~
class Solution {
    public long countBadPairs(int[] nums) {
        long n = nums.length;
        for (int i = 0; i < n; i ++ ) nums[i] -= i;
        long res = n * (n - 1l) / 2;
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : nums) map.put(x, map.getOrDefault(x, 0) + 1);
        for (int k : map.keySet()) {
            int v = map.get(k);
            if (v == 1) continue;
            else {
                res -= v * (v - 1l) / 2;
            }
        }
        return res;
    }
}
~~~



#### Problem C - [任务调度器 II](https://leetcode-cn.com/problems/task-scheduler-ii/)

**解法：模拟&贪心**

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



#### Problem D - [将数组排序的最少替换次数](https://leetcode-cn.com/problems/minimum-replacements-to-sort-the-array/)

**解法：贪心&数学**

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

## 305场周赛

#### Problem A - [算术三元组的数目](https://leetcode.cn/problems/number-of-arithmetic-triplets/)

**解法：模拟**

~~~
class Solution {
    public int arithmeticTriplets(int[] nums, int diff) {
        int n = nums.length;
        int res = 0;
        for (int i = 0; i < n - 2; i ++ ) 
            for (int j = i + 1; j < n - 1; j ++ )
                for (int k = j + 1; k < n; k ++ ) {
                    if ((nums[j] - nums[i] == diff) && (nums[k] - nums[j] == diff)) 
                        res ++ ;
                }
        return res;
    }
}
~~~



#### Problem B - [ 受限条件下可到达节点的数目](https://leetcode.cn/problems/reachable-nodes-with-restrictions/)

**解法：dfs**

~~~
class Solution {
    List<Integer>[] g;
    boolean[] r;
    int res = 0;
    public void dfs(int i, int j) {
        res ++ ;
        for (int x : g[j]) 
            if (x != i && !r[x]) 
                dfs(j, x);
    }
    public int reachableNodes(int n, int[][] edges, int[] restricted) {
        g = new List[n];
        for (int i = 0; i < n; i ++ ) g[i] = new ArrayList<>();
        r = new boolean[n];
        //标记受限节点
        for (int x : restricted) r[x] = true;
        //建图
        for (int[] x : edges) {
            int a = x[0], b = x[1];
            g[a].add(b);
            g[b].add(a);
        }
        dfs(0, 0);
        return res;
    }
}
~~~

**解法：bfs**

~~~
class Solution {
    public int reachableNodes(int n, int[][] edges, int[] restricted) {
        List<Integer>[] g = new List[n];
        boolean[] r;
        int res = 1;
        for (int i = 0; i < n; i ++ ) g[i] = new ArrayList<>();
        r = new boolean[n];
        for (int x : restricted) r[x] = true;
        //建图
        for (int[] x : edges) {
            int a = x[0], b = x[1];
            g[a].add(b);
            g[b].add(a);
        }
        ArrayDeque<Integer> q = new ArrayDeque<>();
        q.add(0);
        r[0] = true;
        while (!q.isEmpty()) {
            int cur = q.poll();
            for (int x : g[cur]) {
                if (!r[x]) {
                    res ++ ;
                    r[x] = true;
                    q.add(x);
                }
            }
        }
        return res;
    }
}
~~~



#### Problem C - [检查数组是否存在有效划分](https://leetcode.cn/problems/check-if-there-is-a-valid-partition-for-the-array/)

**解法：线性dp**

~~~
class Solution {
    public boolean validPartition(int[] nums) {
        var n = nums.length;
        //设 f(i) 表示以 i 结尾的数组是否存在有效划分。i 的有效下标从 1 开始
        var f = new boolean[n + 1];
        f[0] = true;
        for (int i = 2; i <= n; i ++ ) {
            if (nums[i - 1] == nums[i - 2]) 
                f[i] = f[i - 2];
            
            if (i >= 3 && nums[i - 1] == nums[i - 2] && nums[i - 1] == nums[i - 3])
                f[i] = f[i] || f[i - 3];
            
            if (i >= 3 && nums[i - 1] == nums[i - 2] + 1 && nums[i - 2] == nums[i - 3] + 1)
                f[i] = f[i] || f[i - 3];
        }
        return f[n];
    }
}
~~~

#### Problem D - [最长理想子序列](https://leetcode.cn/problems/longest-ideal-subsequence/)

**解法：线性dp**

~~~
class Solution {
    public int longestIdealString(String s, int k) {
        //记 f[d] 表示以字符 d 为结尾的最长理想子序列。
        var f = new int[26];
        var n = s.length();
        for (var c : s.toCharArray()) {
            var x = c - 'a';
            var t = 0;
            //满足Math.abs(x - y) <= k就能将字符x接在以y结尾的子序列后边
            for (int y = 0; y < 26; y ++ ) 
                if (Math.abs(x - y) <= k)
                    t = Math.max(t, f[y] + 1);
            f[x] = Math.max(f[x], t);
        }
        return Arrays.stream(f).max().getAsInt();
    }
}
~~~

## 306场周赛

#### Problem A - [矩阵中的局部最大值](https://leetcode.cn/problems/largest-local-values-in-a-matrix/)

**解法：模拟**

~~~
class Solution {
    public int[][] largestLocal(int[][] g) {
        int n = g.length, m = g[0].length;
        int[][] res = new int[n - 2][m - 2];
        for (int i = 0; i < n - 2; i ++ ) 
            for (int j = 0; j < m - 2; j ++ )
                for (int x = 0; x < 3; x ++ )
                    for (int y = 0; y < 3; y ++ ) 
                        res[i][j] = Math.max(res[i][j], g[i + x][j + y]);
        return res;
    }
}
~~~



#### Problem B - [边积分最高的节点](https://leetcode.cn/problems/node-with-highest-edge-score/)

**解法：模拟**

~~~
class Solution {
    public int edgeScore(int[] edges) {
        int n = edges.length;
        long[] f = new long[n];
        for (int i = 0; i < n; i ++ ) {
            f[edges[i]] += i;
        }
        int res = 0;
        for (int i = 0; i < n; i ++ ) 
            if (f[i] > f[res]) 
                res = i;
        return res;
    }
}
~~~



#### Problem C - [根据模式串构造最小数字](https://leetcode.cn/problems/construct-smallest-number-from-di-string/)

**解法：贪心**

~~~
class Solution {
    public String smallestNumber(String pattern) {
        int n = pattern.length();
        int[] num = new int[n + 1];
        for (int i = 0; i <= n; i ++ ) 
            num[i] = i + 1;
        for (int i = 0; i < n; i ++ ) {
            if (pattern.charAt(i) == 'D') {
                int j = i;
                while (j < n && pattern.charAt(j) == 'D') j ++ ;
                reverse(num, i, j);
                i = j - 1;
            }
        }
        StringBuilder res = new StringBuilder();
        for (int x : num) res.append(x);
        return res.toString();
    }
    public void reverse(int[] nums, int i, int j) {
        while (i < j) {
            var x = nums[i];
            nums[i] = nums[j];
            nums[j] = x;
            i ++ ;
            j -- ;
        }
    }
}
~~~

**解法：暴力**

~~~
class Solution {
    boolean[] st;
    String p;
    public String smallestNumber(String _p) {
        p = _p;
        int n = p.length();
        int[] nums = new int[n + 1];
        for (int i = 0; i < n + 1; i ++ ) 
            nums[i] = i + 1;

        while (!check(nums)) {
            nextPermutation(nums);
        }
        var res = "";
        for (var x : nums) res += x;
        return res;
    }
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
    public boolean check(int[] nums) {
        for (int i = 0; i < p.length(); i ++ ) {
            char c = p.charAt(i);
            if (c == 'I' && nums[i] >= nums[i + 1])
                return false; 
            if (c == 'D' && nums[i] <= nums[i + 1])
                return false; 
        }
        return true;
    }
}
~~~



#### Problem D - [统计特殊整数](https://leetcode.cn/problems/count-special-integers/)

**解法：数位dp**



## 84场双周赛

#### Problem A - [得到 K 个黑块的最少涂色次数](https://leetcode.cn/problems/minimum-recolors-to-get-k-consecutive-black-blocks/)

**解法：滑动窗口**

~~~
class Solution {
    public int minimumRecolors(String s, int k) {
        int res = 100;
        for (int i = 0, j = 0, cnt = 0; i < s.length(); i ++ ) {
            if (s.charAt(i) == 'W') cnt ++ ;
            if (i - j + 1 > k) {
                if (s.charAt(j) == 'W') cnt -- ;
                j ++ ;
            }
            if (i >= k - 1) res = Math.min(cnt, res);
        }
        return res;
    }
}
~~~

#### Problem B - [二进制字符串重新安排顺序需要的时间](https://leetcode.cn/problems/time-needed-to-rearrange-a-binary-string/)

**解法：模拟**

~~~
class Solution {
    public int secondsToRemoveOccurrences(String s) {
        int res = 0;
        while (s.contains("01")) {
           res ++ ;
           s = s.replaceAll("01", "10");
        }  
        return res;
    }
}
~~~

#### Problem C - [字母移位 II](https://leetcode.cn/problems/shifting-letters-ii/)

**解法：差分**

~~~
class Solution {
    public String shiftingLetters(String s, int[][] shifts) {
        char[] c = s.toCharArray();
        int n = s.length();
        int[] t = new int[n + 1];
        for (int[] p : shifts) {
            int a = p[0], b = p[1], w = p[2];
            if (w == 0) w = -1;
            t[a] += w;
            t[b + 1] -= w;
        }
        for (int i = 0, cur = 0; i < n; i ++ ) {
            cur += t[i];
            cur %= 26;
            int p = c[i] - 'a';
            p += cur;
            p = (p + 26) % 26;
            c[i] = (char) (p + 'a');
        }
        return new String(c);
    }
}
~~~

#### Problem D - [删除操作后的最大子段和](https://leetcode.cn/problems/maximum-segment-sum-after-removals/)

**解法：模拟**



## 307场周赛

#### Problem A - [赢得比赛需要的最少训练时长](https://leetcode.cn/problems/minimum-hours-of-training-to-win-a-competition/)

**解法：贪心**

~~~
class Solution {
    public int minNumberOfHours(int a, int b, int[] c, int[] d) {
        int n = c.length;
        int res = 0;
        for (int i = 0; i < n; i ++ ) {
            if (a > c[i]) a -= c[i];
            else {
                int x = c[i] + 1 - a;
                a += x - c[i];
                res += x;
            }

            if (b > d[i]) b += d[i];
            else {
                int x = d[i] + 1 - b;
                b += x + d[i];
                res += x;
            }
        }
        return res;
    }
}
~~~

#### Problem B - [最大回文数字](https://leetcode.cn/problems/largest-palindromic-number/)

**解法：模拟**

~~~
class Solution {
    public String largestPalindromic(String s) {
        int[] map = new int[10];
        for (char x : s.toCharArray()) {
            int t = x - '0';
            map[t] ++ ;
        }
        int a = 0;
        //找出可能要追加的数
        for (int i = 1; i < 10; i ++ )
            if (map[i] % 2 == 1)
                a = i;
        StringBuilder sb = new StringBuilder();
        //拼凑答案的前半部分
        for (int i = 9; i >= 0; i -- ) {
            int t = map[i] / 2;
            if (i == 0 && sb.length() == 0) break;
            for (int j = 0; j < t; j ++ )
                sb.append(i);
        }

        StringBuilder x = new StringBuilder(sb).reverse();
        //如果可以的话，中间追加一位
        if (map[a] != 0) sb.append(a);
        return sb.toString() + x.toString();
    }
}
~~~

#### Problem C - [感染二叉树需要的总时间](https://leetcode.cn/problems/amount-of-time-for-binary-tree-to-be-infected/)

**解法：dfs + bfs**

链式前向星建图

~~~
class Solution {
    int N = 100010;
    int[] h, e, ne;
    //记录该点是否进入过队列
    boolean[] st;
    int idx;
    public void add(int a, int b) {
        e[idx] = b;
        ne[idx] = h[a];
        h[a] = idx ++ ;
    }
    //dfs建图
    public void dfs(TreeNode root) {
        if (root.left != null) {
            add(root.val, root.left.val);
            add(root.left.val, root.val);
            dfs(root.left);
        }
        if (root.right != null) {
            add(root.val, root.right.val);
            add(root.right.val, root.val);
            dfs(root.right);
        }
    }
    public int amountOfTime(TreeNode root, int start) {
        h = new int[N];
        e = new int[N * 2];
        ne = new int[N * 2];
        st = new boolean[N];
        idx = 0;
        Arrays.fill(h, -1);
        dfs(root);
        ArrayDeque<Integer> q = new ArrayDeque<>();
        q.add(start);
        int res = 0;
        while (!q.isEmpty()) {
            int count = q.size();
            while (count -- > 0) {
                var t = q.poll();
                st[t] = true;           
                for (int i = h[t]; i != -1; i = ne[i]) {
                    int j = e[i];
                    if (st[j]) continue;
                    else q.add(j);
                }
            }
            res ++ ;
        }
        return res - 1;
    }
}
~~~

**解法：dfs + bfs**

Map建图

~~~
class Solution {
    int N = 100010;
    Map<Integer, LinkedList<Integer>> g = new HashMap<>();
    boolean[] st;
    public void dfs(TreeNode root) {
        if (root == null) return;
        LinkedList<Integer> list = g.getOrDefault(root.val, new LinkedList<>());
        if (root.left != null) {
            list.add(root.left.val);
            LinkedList<Integer> temp = new LinkedList<>();
            temp.add(root.val);
            g.put(root.left.val, temp);
        }
        if (root.right != null) {
            list.add(root.right.val);
            LinkedList<Integer> temp = new LinkedList<>();
            temp.add(root.val);
            g.put(root.right.val, temp);
        }
        g.put(root.val, list);
        dfs(root.left);
        dfs(root.right);
    }
    public int amountOfTime(TreeNode root, int start) {
        dfs(root);
        st = new boolean[N];
        ArrayDeque<Integer> q = new ArrayDeque<>();
        q.add(start);
        int res = -1;
        while (!q.isEmpty()) {
            int count = q.size();
            while (count -- > 0) {
                int x = q.poll();
                st[x] = true;
                List<Integer> list = g.get(x);
                if (list != null) {
                    for (int a : list) 
                    if (!st[a])
                        q.add(a);
                }
            }
            res ++ ;
        }
        return res;
    }
}
~~~

#### Problem D - [找出数组的第 K 大和](https://leetcode.cn/problems/find-the-k-sum-of-an-array/)

**解法：多路归并**



## 308场周赛

#### Problem A - [和有限的最长子序列](https://leetcode.cn/problems/longest-subsequence-with-limited-sum/)

**解法：前缀和 &二分**

~~~
class Solution {
    public int[] answerQueries(int[] nums, int[] q) {
        Arrays.sort(nums);
        int n = nums.length, m = q.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i ++ )
            s[i + 1] = s[i] + nums[i];
        for (int i = 0; i < m; i ++ ) {
            int k = q[i];
            int l = 0, r = n;
            while (l < r) {
                int mid = l + r + 1 >> 1;
                if (s[mid] <= k) l = mid;
                else r = mid - 1;
            }
            q[i] = l;
        }
        return q;
    }
}
~~~

#### Problem B - [从字符串中移除星号](https://leetcode.cn/problems/removing-stars-from-a-string/)

**解法：模拟**

~~~
class Solution {
    public String removeStars(String s) {
        StringBuilder sb = new StringBuilder();
        for (char x : s.toCharArray()) {
            if (x == '*') sb.deleteCharAt(sb.length() - 1);
            else sb.append(x);
        }
        return sb.toString();
    }
}
~~~

#### Problem C - [收集垃圾的最少总时间](https://leetcode.cn/problems/minimum-amount-of-time-to-collect-garbage/)

**解法：贪心**

~~~
class Solution {
    public int garbageCollection(String[] g, int[] t) {
        Map<Character, Integer> map = new HashMap<>();
        int a = 0, b = 0, c = 0;
        for (int i = 0; i < g.length; i ++) {
            for (char x : g[i].toCharArray()) {
                if (x == 'M') a ++ ;
                else if (x == 'P') b ++ ;
                else c ++ ;
                map.put(x, i);
            }
        }
        int res = a + b + c;
        int[] s = new int[t.length + 1];
        for (int i = 0; i < t.length; i ++ ) 
            s[i + 1] = s[i] + t[i];
        for (char x : map.keySet()) res += s[map.get(x)];
        return res;
    }
}
~~~

#### Problem D - [给定条件下构造矩阵](https://leetcode.cn/problems/build-a-matrix-with-conditions/)

**解法：拓扑排序**

~~~
class Solution {
    public int[] topoSort(int k, int[][] edges) {
        List<Integer>[] g = new List[k];
        Arrays.setAll(g, e -> new ArrayList<>());
        int[] d = new int[k];
        for (var e : edges) {
            int a = e[0] - 1, b = e[1] - 1;
            g[a].add(b);
            d[b] ++ ;
        }
        ArrayList<Integer> res = new ArrayList<>();
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < k; i ++ )
            if (d[i] == 0)
                q.add(i);
        while (!q.isEmpty()) {
            int t = q.poll();
            res.add(t);
            for (int x : g[t])
                if (-- d[x] == 0)
                    q.add(x);
        }
        return res.stream().mapToInt(x -> x).toArray();
    }
    public int[][] buildMatrix(int k, int[][] row, int[][] col) {
        int[] x = topoSort(k, row), y = topoSort(k, col);
        if (x.length < k || y.length < k) return new int[][]{};
        int[][] res = new int[k][k];
        for (int i = 0; i < k; i ++ )
            for (int j = 0; j < k; j ++ )
                if (x[i] == y[j]) res[i][j] = x[i] + 1;
        return res;
    }
}
~~~



## 86场双周赛

#### Problem A - [和相等的子数组](https://leetcode.cn/problems/find-subarrays-with-equal-sum/)

**解法：枚举**

~~~
class Solution {
    public boolean findSubarrays(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int n = nums.length;
        for (int i = 0; i < n - 1; i ++ ) {
            int x = nums[i] + nums[i + 1];
            if (set.contains(x)) return true;
            set.add(x);
        }
        return false;
    }
}
~~~



#### Problem B - [严格回文的数字](https://leetcode.cn/problems/strictly-palindromic-number/)

**解法：模拟**

~~~
class Solution {
    public boolean f(String s) {
        for (int i = 0, j = s.length() - 1; i < j; i ++ , j -- ) 
            if (i < j && s.charAt(i) != s.charAt(j)) 
                return false;
        
        return true;
    }
    
    public boolean isStrictlyPalindromic(int n) {
        for (int i = 2; i <= n - 2; i ++ ) {
            StringBuilder sb = new StringBuilder();
            int t = n;
            while (t != 0) {
                sb.append(t % i);
                t /= i;
            }
            if (!f(sb.reverse().toString())) return false;
        }
        return true;
    }
}
~~~

**解法：脑筋急转弯**

~~~
class Solution {
    public boolean isStrictlyPalindromic(int n) {
    	return false;
    }
}
~~~



#### Problem C - [被列覆盖的最多行数](https://leetcode.cn/problems/maximum-rows-covered-by-columns/)

**解法：二进制枚举**

~~~
class Solution {
    public int maximumRows(int[][] g, int cols) {
        int m = g.length, n = g[0].length;
        int res = 0;
        for (int k = 0; k < (1 << n); k ++ ) {
            if (Integer.bitCount(k) != cols) continue;
            int t = 0;
            out:for (int i = 0; i < m; i ++ ) {
                    for (int j = 0; j < n; j ++ )
                        if ((k >> j & 1) == 0 && g[i][j] == 1)
                            continue out;
                t ++ ;
            }
            res = Math.max(res, t);
        }
        return res;
    }
}
~~~

**解法：dfs**、

~~~
class Solution {
    int[][] g;
    int res = 0;
    int n, m;
    public int maximumRows(int[][] mat, int cols) {
        n = mat.length;
        m = mat[0].length;
        g = mat;
        dfs(0, cols,new HashSet<Integer>());
        return res;
    }
    public void dfs(int u, int k, Set<Integer> set) {
        if (set.size() == k) {
            int t = 0;
            out:for (int i = 0; i < n; i ++ ) {
                for (int j = 0; j < m; j ++ )
                    if (g[i][j] == 1 && !set.contains(j))
                        continue out;
                t ++ ;
            }
            res = Math.max(res, t);
        } else {
            for (int i = u; i < m; i ++ ) {
                set.add(i);
                dfs(i + 1, k, set);
                set.remove(i);
            }
        }
    }
}
~~~



#### Problem D - [预算内的最多机器人数目](https://leetcode.cn/problems/maximum-number-of-robots-within-budget/)

**解法：双指针&单调队列**



## 309场周赛

#### Problem A - [检查相同字母间的距离](https://leetcode.cn/problems/check-distances-between-same-letters/)

**解法：模拟**

~~~
class Solution {
    public boolean checkDistances(String s, int[] d) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < s.length(); i ++ ) {
            char x = s.charAt(i);
            if (!map.containsKey(x - 'a')) map.put(x - 'a', new ArrayList<>());
            map.get(x - 'a').add(i);
        }
        for (int k : map.keySet()) {
            if (map.get(k).get(1) - map.get(k).get(0) == d[k] + 1) continue;
            else return false;
        }
        return true;
    }
}

~~~

~~~
class Solution {
    public boolean checkDistances(String s, int[] distance) {
        int[] a = new int[26];
        Arrays.fill(a, -1);
        for (int i = 0; i < s.length(); i ++ ) {
            int x = s.charAt(i) - 'a';
            if (a[x] == -1) a[x] = i;
            else if (i - a[x] != distance[x] + 1) return false;
        }
        return true;
    }
}
~~~



#### Problem B - [恰好移动 k 步到达某一位置的方法数目](https://leetcode.cn/problems/number-of-ways-to-reach-a-position-after-exactly-k-steps/)

**解法：数论**

动态规划求组合数

~~~
class Solution {
    private int mod = (int)1e9 + 7;
    public int numberOfWays(int startPos, int endPos, int k) {
        int d = Math.abs(endPos - startPos);
        if (d > k || (d + k) % 2 == 1) return 0;
        int[][] f = new int[k + 1][k + 1];
        for (int i = 0; i <= k; i ++ ) {
            f[i][0] = 1;
            for (int j = 1; j <= i; j ++ )
                f[i][j] = (f[i - 1][j] + f[i - 1][j - 1]) % mod;
        }
        return f[k][(d + k) / 2];
    }
}
~~~

逆元求组合数

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



#### Problem C - [最长优雅子数组](https://leetcode.cn/problems/longest-nice-subarray/)

**解法：位运算&滑动窗口&状态压缩**

~~~
class Solution {
    public int longestNiceSubarray(int[] nums) {
        int n = nums.length;
        int res = 0;
        int s = 0;
        for (int i = 0, j = 0; i < n; i ++ ) {
            while ((s & nums[i]) > 0) 
                s ^= nums[j ++ ];
            s ^= nums[i];
            res = Math.max(res, i - j + 1);    
        }
        return res;
    }
}
~~~



#### Problem D - [会议室 III](https://leetcode.cn/problems/meeting-rooms-iii/)

**解法：模拟**



## 310场周赛

#### Problem A - [出现最频繁的偶数元素](https://leetcode.cn/problems/most-frequent-even-element/)

**解法：模拟**

~~~
class Solution {
    public int mostFrequentEven(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : nums) {
            if (x % 2 == 0) 
                map.put(x, map.getOrDefault(x, 0) + 1);
        }
        int res = -1;
        for (int k : map.keySet()) {
            int v = map.get(k);
            if (res == -1 || map.get(res) < v || (map.get(res) == v && res > k))
                    res = k;
        }
        return res;
    }
}
~~~



#### Problem B - [子字符串的最优划分](https://leetcode.cn/problems/optimal-partition-of-string/)

**解法：贪心**

~~~
class Solution {
    public int partitionString(String s) {
        int[] map = new int[26];
        int res = 0;
        for (int i = 0; i < s.length(); i ++ ) {
            int x = s.charAt(i) - 'a';
            map[x] ++ ;
            if (map[x] > 1) {
                res ++ ;
                i -- ;
                map = new int[26];
            }   
        }
        return res + 1;
    }
}
~~~



#### Problem C - [将区间分为最少组数](https://leetcode.cn/problems/divide-intervals-into-minimum-number-of-groups/)

**解法：贪心**

~~~
class Solution {
    public int minGroups(int[][] g) {
        int n = g.length;
        Arrays.sort(g, (a, b) -> a[0] - b[0]);
        PriorityQueue<Integer> q = new PriorityQueue<>();

        for (int i = 0; i < n; i ++ ) {
            int[] r = g[i];
            //需要创建新组
            if (q.isEmpty() || q.peek() >= r[0]) q.add(r[1]);
            else {
                //将该段放入更新当前组最大值
                int t = q.poll();
                q.add(r[1]);
            }
        }
        return q.size();
    }
}
~~~



#### Problem D - [最长递增子序列 II](https://leetcode.cn/problems/longest-increasing-subsequence-ii/)

**解法：线段树**



## 87场双周赛

#### Problem A - [统计共同度过的日子数](https://leetcode.cn/problems/count-days-spent-together/)

**解法：模拟**

~~~
class Solution {
    int[] t = new int[]{0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    public int get(String s) {
        String[] ss = s.split("-");
        int a = Integer.parseInt(ss[0]), b = Integer.parseInt(ss[1]);
        int res = 0;
        for (int i = 0; i < a; i ++ ) res += t[i];
        return res + b;
    }
    public int countDaysTogether(String s1, String s2, String s3, String s4) {
        int a = get(s1), b = get(s2), c = get(s3), d = get(s4);
        int res = Math.min(b, d) - Math.max(a, c) + 1;
        return Math.max(0, res);
    }
}
~~~

#### Problem B - [运动员和训练师的最大匹配数](https://leetcode.cn/problems/maximum-matching-of-players-with-trainers/)

**解法：贪心**

~~~
class Solution {
    public int matchPlayersAndTrainers(int[] a, int[] b) {
        Arrays.sort(a);
        Arrays.sort(b);
        int res = 0;
        for (int i = 0, j = 0; i < a.length; i ++ ) {
            while (j < b.length && b[j] < a[i]) j ++ ;
            if (j == b.length) break;
            if (a[i] <= b[j]) {
                j ++ ;
                res ++ ;
            }
        }
        return res;
    }
}
~~~

#### Problem C - [按位或最大的最小子数组长度](https://leetcode.cn/problems/smallest-subarrays-with-maximum-bitwise-or/)

**解法：贪心**

~~~
class Solution {
    public int[] smallestSubarrays(int[] nums) {
        int n = nums.length;
        int[] p = new int[30];
        int[] res = new int[n];
        Arrays.fill(p, n);
        for (int i = n - 1; i >= 0; i -- ) {
            int t = i;
            for (int j = 0; j < 30; j ++ )
                if ((nums[i] >> j & 1) == 1) p[j] = i;
                else if (p[j] != n) t = Math.max(p[j], t);
            res[i] = t - i + 1;
        }
        return res;
    }
}
~~~

#### Problem D - [完成所有交易的初始最少钱数](https://leetcode.cn/problems/minimum-money-required-before-transactions/)

**解法：贪心**

~~~
class Solution {
    public long minimumMoney(int[][] transactions) {
        long sum = 0;
        for (int[] p : transactions) {
            int a = p[0], b = p[1];
            if (a > b) sum += a - b;
        }
        long res = 0;
        for (int[] p : transactions) {
            int a = p[0], b = p[1];
            long s = sum;
            if (a > b) s -= a - b;
            res = Math.max(res, s + a);
        }
        return res;
    }
}
~~~



## 311场周赛

#### Problem A - [最小偶倍数](https://leetcode.cn/problems/smallest-even-multiple/)

**解法：数学**

~~~
class Solution {
    public int smallestEvenMultiple(int n) {
        if (n % 2 == 0) return n;
        else return n * 2;
    }
}
~~~

#### Problem B - [最长的字母序连续子字符串的长度](https://leetcode.cn/problems/length-of-the-longest-alphabetical-continuous-substring/)

**解法：枚举**

~~~
class Solution {
    public int longestContinuousSubstring(String s) {
        char[] ss = s.toCharArray();
        int res = 0;
        for (int i = 0; i < ss.length; i ++ ) {
            int j = i + 1;
            while (j < ss.length && ss[j] == ss[j - 1] + 1) j ++ ;
            res = Math.max(res, j - i);
            i = j - 1;
        }
        return res;
    }
}
~~~



#### Problem C - [反转二叉树的奇数层](https://leetcode.cn/problems/reverse-odd-levels-of-binary-tree/)

**解法：bfs**

~~~
class Solution {
    public void reverse(List<TreeNode> list) {
        int l = 0, r = list.size() - 1;
        while (l < r) {
            TreeNode a = list.get(l), b = list.get(r);
            int t = a.val;
            a.val = b.val;
            b.val = t;
            l ++ ;
            r -- ;
        }
    } 
    public TreeNode reverseOddLevels(TreeNode root) {
        var q = new ArrayDeque<TreeNode>();
        q.add(root);
        int depth = -1;
        while (!q.isEmpty()) {
            int cur = q.size();
            var list = new ArrayList<TreeNode>();
            while (cur -- > 0) {
                TreeNode node = q.poll();
                list.add(node);
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            depth ++ ;
            if (depth % 2 == 1) 
                reverse(list);
        }
        return root;
    }
}
~~~

**解法：dfs**

~~~
class Solution {
    public void swap(TreeNode a, TreeNode b) {
        int t = a.val;
        a.val = b.val;
        b.val = t;
    }

    public void dfs(TreeNode a, TreeNode b, int d) {
        if (a == null) return;
        if (d % 2 == 1) swap(a, b);
        dfs(a.left, b.right, d + 1);
        dfs(a.right, b.left, d + 1); 
    }

    public TreeNode reverseOddLevels(TreeNode root) {
        dfs(root.left, root.right, 1);
        return root;
    }
}
~~~



#### Problem D - [字符串的前缀分数和](https://leetcode.cn/problems/sum-of-prefix-scores-of-strings/)

**解法：字典树**

~~~
class Solution {
    
    public class TrieNode {
        TrieNode[] child;
        int cnt;
        TrieNode() {
            child = new TrieNode[27];
            cnt = 1;
        }
    }
    
    TrieNode root = new TrieNode();
    
    public void insert(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (p.child[c - 'a'] == null) {
                p.child[c - 'a'] = new TrieNode();
            } else {
                p.child[c - 'a'].cnt ++ ;
            }
            p = p.child[c - 'a'];
        }
    }

    public int search(String word) {
        TrieNode p = root;
        int res = 0;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (p.child[c - 'a'] != null) {
                p = p.child[c - 'a'];
                res += p.cnt;
            }
        }
        return res;
    }

    public int[] sumPrefixScores(String[] words) {
        int n = words.length;
        for (String x : words) {
            insert(x);
        }
        int[] res = new int[n];
        for (int i = 0; i < n; i ++ ) {
            String word = words[i];
            res[i] = search(word);
        }
        return res;
    }
}
~~~



## 312场周赛

#### Problem A - [按身高排序](https://leetcode.cn/problems/sort-the-people/)

**解法：模拟**

~~~
class Solution {
    public String[] sortPeople(String[] names, int[] h) {
        var map = new HashMap<Integer, String>();
        for (int i = 0; i < names.length; i ++ )
            map.put(h[i], names[i]);
        Arrays.sort(h);
        var list = new ArrayList<String>();
        for (int i = h.length - 1; i >= 0; i -- )
            list.add(map.get(h[i]));
        return list.toArray(new String[h.length]);
    }
}
~~~

#### Problem B - [按位与最大的最长子数组](https://leetcode.cn/problems/longest-subarray-with-maximum-bitwise-and/)

**解法：脑筋急转弯**

~~~
class Solution {
    public int longestSubarray(int[] nums) {
        int mx = 0, n = nums.length;
        for (int x : nums)
            mx = Math.max(mx, x);
        int res = 0, cur = 0;
        for (int i = 0; i < n; i ++ )
            if (nums[i] == mx) {
                int j = i + 1;
                while (j < n && nums[j] == mx) j ++ ;
                res = Math.max(res, j - i);
                i = j;
            } 
        return res;
    }
}
~~~



#### Problem C - [找到所有好下标](https://leetcode.cn/problems/find-all-good-indices/)

**解法：枚举**

~~~
class Solution {
    public List<Integer> goodIndices(int[] nums, int k) {
        var n = nums.length;
        int[] f = new int[n], g = new int[n];
        Arrays.fill(f, 1); Arrays.fill(g, 1);
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



#### Problem D - [好路径的数目](https://leetcode.cn/problems/number-of-good-paths/)

**解法：并查集**

## 88场双周赛

#### Problem A - [删除字符使频率相同](https://leetcode.cn/problems/remove-letter-to-equalize-frequency/)

**解法：模拟**

~~~
class Solution {
    public boolean equalFrequency(String s) {
        int[] map = new int[26];
        for (char x : s.toCharArray()) map[x - 'a'] ++ ;
        for (int i = 0; i < 26; i ++ )
            if (map[i] > 0) {
                map[i] -- ;
                int t = 0;
                boolean res = true;
                for (int j = 0; j < 26; j ++ ) 
                    if (map[j] > 0) {
                        if (t == 0) t = map[j];
                        else if (map[j] != t) res = false;
                    }
                map[i] ++ ;
                if (res) return res;
            }
        return false;
    }
}
~~~



#### Problem B - [最长上传前缀](https://leetcode.cn/problems/longest-uploaded-prefix/)

**解法：模拟**

~~~
class LUPrefix {
    boolean[] st;
    int n, now = 0;
    public LUPrefix(int _n) {
        n = _n;
        st = new boolean[n + 1];
    }
    
    public void upload(int x) {
        st[x] = true;
        while (now < n && st[now + 1]) now ++ ;
    }
    
    public int longest() {
        return now;
    }
}

/**
 * Your LUPrefix object will be instantiated and called as such:
 * LUPrefix obj = new LUPrefix(n);
 * obj.upload(video);
 * int param_2 = obj.longest();
 */
~~~



#### Problem C - [所有数对的异或和](https://leetcode.cn/problems/bitwise-xor-of-all-pairings/)

**解法：脑筋急转弯**

~~~
class Solution {
    public int xorAllNums(int[] a, int[] b) {
        int n = a.length, m = b.length;
        int res = 0;
        if (n % 2 == 1) 
            for (int x : b) 
                res ^= x;        
        if (m % 2 == 1) 
            for (int x : a) 
                res ^= x;    
        return res;
    }
}
~~~



#### Problem D - [满足不等式的数对数目](https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/)

**解法：树状数组**

## 313场周赛

#### Problem A - [公因子的数目](https://leetcode.cn/problems/number-of-common-factors/)

**解法：枚举**

~~~
class Solution {
    public int commonFactors(int a, int b) {
        int res = 0;
        for (int i = 1; i <= Math.min(a, b); i ++ ) {
            if (a % i == 0 && b % i == 0)
                res ++ ;
        }
        return res;
    }
}
~~~



#### Problem B - [沙漏的最大总和](https://leetcode.cn/problems/maximum-sum-of-an-hourglass/)

**解法：前缀和**

~~~
class Solution {
    public int maxSum(int[][] g) {
        int n = g.length, m = g[0].length;
        int[][] s = new int[n + 1][m + 1];
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m; j ++ ) {
                s[i + 1][j + 1] = s[i + 1][j] + s[i][j + 1] - s[i][j] + g[i][j];
            }
        }
        int res = 0;
        for (int i = 2; i < n; i ++ )
            for (int j = 2; j < m; j ++ )
                res = Math.max(res, s[i + 1][j + 1] - s[i + 1][j - 2] - s[i - 2][j + 1] + s[i - 2][j - 2] - g[i - 1][j] - g[i - 1][j - 2]);
        return res;
    }
}
~~~



#### Problem C - [最小 XOR](https://leetcode.cn/problems/minimize-xor/)

**解法：贪心**

~~~
class Solution {
    public int minimizeXor(int a, int b) {
        int n = Integer.bitCount(b);
        int res = 0;
        for (int i = 32 - 1; i >= 0 && n != 0; i -- ) 
            if ((a >> i & 1) == 1) {
                res += 1 << i;
                n -- ;
            }
        for (int i = 0; i < 32 && n != 0; i ++ )
            if ((res >> i & 1) == 0) {
                res += 1 << i;
                n -- ;
            }
        
        return res;
    }
}
~~~



#### Problem D - [对字母串可执行的最大删除数](https://leetcode.cn/problems/maximum-deletions-on-a-string/)

**解法：线性dp&字符串哈希**

~~~
class Solution {
    int P = 131, n;
    long[] h, p;
    public long get(int l, int r) {
        return h[r] - h[l - 1] * p[r - l + 1];
    }
    public int deleteString(String s) {
        n = s.length();
        h = new long[n + 1];
        p = new long[n + 1];
        p[0] = 1;
        for (int i = 1; i <= n; i ++ ) {
            p[i] = p[i - 1] * P;
            h[i] = h[i - 1] * P + s.charAt(i - 1);
        }
        //表示将[i, n]这个后缀所需最大操作数
        int[] f = new int[n + 1];
        Arrays.fill(f, 1);
        for (int i = n; i > 0; i -- ) 
            for (int j = 1; j <= (n - i + 1) / 2; j ++ )
                if (get(i, i + j - 1) == get(i + j, i + j * 2 - 1))
                    f[i] = Math.max(f[i], f[i + j] + 1);
        
        return f[1];
    }
}
~~~

## 314场周赛

#### Problem A - [处理用时最长的那个任务的员工](https://leetcode.cn/problems/the-employee-that-worked-on-the-longest-task/)

**解法：模拟**

~~~
class Solution {
    public int hardestWorker(int n, int[][] logs) {
        var list = new ArrayList<int[]>();
        int pre = 0;
        for (int[] p : logs) {
            int a = p[0], b = p[1];
            list.add(new int[]{a, b - pre});
            pre = b;
        }
        list.sort((a, b) -> {
            return a[1] == b[1] ? a[0] - b[0] : b[1] - a[1];
        });
        return list.get(0)[0];
    }
}
~~~

#### Problem B - [找出前缀异或的原始数组](https://leetcode.cn/problems/find-the-original-array-of-prefix-xor/)

**解法：位运算**

~~~
class Solution {
    public int[] findArray(int[] p) {
        int n = p.length;
        int[] res = new int[n];
        res[0] = p[0];
        for (int i = 1; i < n; i ++ ) {
            res[i] = p[i] ^ p[i - 1];
        }
        return res;
    }   
}
~~~

#### Problem C - [使用机器人打印字典序最小的字符串](https://leetcode.cn/problems/using-a-robot-to-print-the-lexicographically-smallest-string/)

**解法：贪心**

~~~
class Solution {
    public String robotWithString(String s) {
        var map = new int[26];
        var res = new StringBuilder();
        var stk = new ArrayDeque<Character>();
        var min = 0;
        for (char c : s.toCharArray()) map[c - 'a'] ++ ;
        for (char c : s.toCharArray()) {
            -- map[c - 'a'];
            while (min < 26 && map[min] == 0) min ++ ;
            stk.push(c);
            while (!stk.isEmpty() && stk.peek() - 'a' <= min)
                res.append(stk.pop());
        }
        return res.toString();
    }
}
~~~

#### Problem D - [矩阵中和能被 K 整除的路径](https://leetcode.cn/problems/paths-in-matrix-whose-sum-is-divisible-by-k/)

**解法：动态规划**

## 89场双周赛

#### Problem A - [有效时间的数目](https://leetcode.cn/problems/number-of-valid-clock-times/)

**解法：枚举**

~~~
class Solution {
        public int countTime(String time) {
            String[] temp = time.split(":");
            int ans1 = 1, ans2 = 1, ans3 = 1, ans4 = 1;
            if (temp[0].equals("??")) {
                ans1 = 3;
                ans2 = 8;
            } else if (temp[0].charAt(0) == '?' && temp[0].charAt(1) != '?') {
                if (temp[0].charAt(1) >= '4' && temp[0].charAt(1) <= '9') ans1 = 2;
                else ans1 = 3;
            } else if (temp[0].charAt(1) == '?' && temp[0].charAt(0) != '?') {
                if (temp[0].charAt(0) == '2') ans2 = 4;
                else ans2 = 10;
            }
            if (temp[1].charAt(0) == '?') {
                ans3 = 6;
            }
            if (temp[1].charAt(1) == '?') {
                ans4 = 10;
            }
            return ans1 * ans2 * ans3 * ans4;
        }
    }
~~~

#### Problem B - [二的幂数组中查询范围内的乘积](https://leetcode.cn/problems/range-product-queries-of-powers/)

**解法：模拟**

~~~
class Solution {
    int mod = (int) 1e9 + 7;
    public int[] productQueries(int n, int[][] q) {
        var m = q.length;
        var p = new ArrayList<Integer>();
        for (var i = 0; i < 30; i ++ ) 
            if ((n >> i & 1) != 0)
                p.add(1 << i);
        var res = new int[m];
        for (var i = 0; i < m; i ++ ) {
            int l = q[i][0], r = q[i][1];
            var cur = 1l;
            for (var j = l; j <= r; j ++ )
                cur = (cur * p.get(j)) % mod;
            res[i] = (int) cur;
        }
        return res;
    }
}
~~~



#### Problem C - [最小化数组中的最大值](https://leetcode.cn/problems/minimize-maximum-of-array/)

**解法：二分**

~~~
class Solution {
    public int minimizeArrayValue(int[] nums) {
        int n = nums.length;
        int k = 0;
        for (int x : nums) k = Math.max(k, x);
        int l = 0, r = k;
        while (l < r) {
            int mid = l + r >> 1;
            var A = new long[n];
            for (int i = 0; i < n; i ++ )
                A[i] = nums[i] * 1l;
            for (int i = n - 1; i > 0; i -- )
                if (A[i] > mid) {
                    var p = A[i] - mid * 1l;
                    A[i] = mid;
                    A[i - 1] += p;
                }
            if (mid >= A[0]) r = mid;
            else l = mid + 1;
        }
        return r;
    }
}
~~~



#### Problem D - [创建价值相同的连通块](https://leetcode.cn/problems/create-components-with-same-value/)

**解法：枚举**



## 315场周赛

#### Problem A - [与对应负数同时存在的最大正整数](https://leetcode.cn/problems/largest-positive-integer-that-exists-with-its-negative/)

**解法：模拟**

~~~
class Solution {
    public int findMaxK(int[] nums) {
        var set = new HashSet<>();
        for (int x : nums) 
            if (x < 0)
                set.add(x);
        var res = 0;
        for (int x: nums)
            if (x > 0 && set.contains(-x))
                res = Math.max(res, x);
        return res;
    }
}
~~~



#### Problem B - [反转之后不同整数的数目](https://leetcode.cn/problems/count-number-of-distinct-integers-after-reverse-operations/)

**解法：模拟**

~~~
class Solution {
    public int countDistinctIntegers(int[] nums) {
        var set = new HashSet<Integer>();
        for (int x : nums) {
            set.add(x);
            var y = 0;
            for (int k = x; k != 0; k /= 10)
                y = y * 10 + k % 10;
            set.add(y);
        }
        return set.size();
    }
}
~~~



#### Problem C - [反转之后的数字和](https://leetcode.cn/problems/sum-of-number-and-its-reverse/)

**枚举**

~~~
class Solution {
    public boolean sumOfNumberAndReverse(int num) {
        for (int x = 0; x <= num; x ++ ) {
            int y = 0;
            for (int j = x; j != 0; j /= 10)
                y = y * 10 + j % 10;
            if (x + y == num) return true;
        }
        return false;
    }
}
~~~



#### Problem D - [统计定界子数组的数目](https://leetcode.cn/problems/count-subarrays-with-fixed-bounds/)

**解法：滑动窗口**

~~~
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        var res = 0l;
        int n = nums.length, x = -1, y = -1, k = -1;
        for (int i = 0; i < n; i ++ ) {
            int v = nums[i];
            if (v == minK) x = i;
            if (v == maxK) y = i;
            if (v < minK || v > maxK) k = i;
            res += Math.max(Math.min(x, y) - k, 0);
        }
        return res;
    }
}
~~~

## 316场周赛

#### Problem A - [判断两个事件是否存在冲突](https://leetcode.cn/problems/determine-if-two-events-have-conflict/)

**解法：模拟**

~~~
class Solution {
    public int get(String s) {
        String[] ss = s.split(":");
        return Integer.parseInt(ss[0]) * 60 + Integer.parseInt(ss[1]);
    }
    public boolean haveConflict(String[] a, String[] b) {
        int t1 = get(a[0]), t2 = get(a[1]);
        int t3 = get(b[0]), t4 = get(b[1]);
        if (t2 < t3 || t4 < t1) return false;
        return true;
    }
}
~~~

#### Problem B - [最大公因数等于 K 的子数组数目](https://leetcode.cn/problems/number-of-subarrays-with-gcd-equal-to-k/)

**解法：枚举**

~~~
class Solution {
    int gcd(int a, int b) {
        return b != 0 ? gcd(b, a % b) : a;
    }
    public int subarrayGCD(int[] nums, int k) {
        int n = nums.length, res = 0;
        for (int i = 0; i < n; i ++ ) {
            int p = nums[i];
            for (int j = i; j < n; j ++ ) {
                p = gcd(p, nums[j]);
                if (p == k) res ++ ;
            }
        }
        return res;
    }
}
~~~



#### Problem C - [使数组相等的最小开销](https://leetcode.cn/problems/minimum-cost-to-make-array-equal/)

**解法：数学**

~~~
class Solution {
    public long minCost(int[] nums, int[] cost) {
        var n = nums.length;
        var list = new ArrayList<int[]>();
        for (int i = 0; i < n; i ++ )
            list.add(new int[]{nums[i], cost[i]});
        Collections.sort(list, (a, b) -> {
            return a[0] == b[0] ? a[1] - b[1] : a[0] - b[0];
        });
        var tot = 0l;
        for (int x : cost)
            tot += x;
        var note = 0l;
        var choose = 0;
        for (int[] p : list) {
            int num = p[0], c = p[1];
            note += c;
            if (note * 2 >= tot) {
                choose = num;
                break;
            }
        }
        var res = 0l;
        for (int[] p : list) {
            int num = p[0], c = p[1];
            res += c * 1l * Math.abs(num - choose);
        }
        return res;
    }
}
~~~



~~~
class Solution:
    def minCost(self, nums: List[int], cost: List[int]) -> int:
        tem = sorted(zip(nums, cost))
        tot, note = sum(cost), 0
        for num, c in tem:
            note += c
            if note * 2 >= tot:
                choose = num
                break
        return sum(c * abs(num - choose) for num, c in tem)
~~~



#### Problem D - [使数组相似的最少操作次数](https://leetcode.cn/problems/minimum-number-of-operations-to-make-arrays-similar/)

**解法：数学**

~~~
class Solution {
    public long makeSimilar(int[] nums, int[] target) {
        var n = nums.length;
        var a = new Integer[n];
        var b = new Integer[n];
        for(var i = 0; i < n; i ++ ){
            a[i] = nums[i];
            b[i] = target[i];
        }
        Arrays.sort(a, (o1, o2)-> o1 % 2 == o2 % 2 ? o1 - o2 : o1 % 2 - o2 % 2);
        Arrays.sort(b, (o1, o2)-> o1 % 2 == o2 % 2 ? o1 - o2 : o1 % 2 - o2 % 2);
        var res = 0l;
        for (var i = 0; i < n; i ++ )
            res += Math.abs(a[i] - b[i]);
        return res >> 2;
    }
}
~~~



~~~
class Solution:
    def makeSimilar(self, nums: List[int], target: List[int]) -> int:
        nums.sort(key=lambda x: (x % 2, x))
        target.sort(key=lambda x: (x % 2, x))
        return sum(abs(x - y) for x, y in zip(nums, target)) // 4
~~~

## 90场双周赛

#### Problem A - [差值数组不同的字符串](https://leetcode.cn/problems/odd-string-difference/)

**解法：模拟**

~~~
class Solution {
    public String oddString(String[] words) {
        int n = words.length, m = words[0].length();
        int[][] d = new int[n][m - 1];
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < m - 1; j ++ ) {
                d[i][j] = words[i].charAt(j + 1) - words[i].charAt(j);
            }
        }
        Map<String, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < n; i ++ ) {
            var sb =  new StringBuilder();
            for (int c : d[i])
                sb.append(c).append(",");
            var s = sb.toString();
            if (!map.containsKey(s)) map.put(s, new ArrayList<>());
            map.get(s).add(i);
        }
        for (var k : map.keySet()) {
            if (map.get(k).size() == 1)
                return words[map.get(k).get(0)];
        }
        return "";
    }
}
~~~



#### Problem B - [距离字典两次编辑以内的单词](https://leetcode.cn/problems/words-within-two-edits-of-dictionary/)

**解法：枚举**

~~~
class Solution {
    public List<String> twoEditWords(String[] q, String[] d) {
        var res = new ArrayList<String>();
        for (var s1 : q) {
            for (var s2 : d) {
                int k = 0;
                for (int i = 0; i < s1.length(); i ++ ) {
                    if (s1.charAt(i) != s2.charAt(i)) k ++ ;
                    if (k > 2) break;
                }
                if (k <= 2) {
                    res.add(s1);
                    break;
                }
            }
        }
        return res;
    }
}
~~~



#### Problem C - [摧毁一系列目标](https://leetcode.cn/problems/destroy-sequential-targets/)

**解法：枚举**

~~~
class Solution {
    public int destroyTargets(int[] nums, int space) {
        var map = new TreeMap<Integer, Integer>();
        Arrays.sort(nums);
        for (int x : nums) {
            map.put(x % space, map.getOrDefault(x % space, 0) + 1);
        }
        var n = nums.length;
        int res = 0, p = 0;
        for (var i = 0; i < n; i ++ ) {
            int x = nums[i] % space;
            if (map.get(x) > p) {
                res = nums[i];
                p = map.get(x);
            }
        }
        return res;
    }
}
~~~



#### Problem D - [下一个更大元素 IV](https://leetcode.cn/problems/next-greater-element-iv/)

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

## 317场周赛

#### Problem A - [可被三整除的偶数的平均值](https://leetcode.cn/problems/average-value-of-even-numbers-that-are-divisible-by-three/)

**解法：模拟**

~~~
class Solution {
    public int averageValue(int[] nums) {
        int sum = 0, cnt = 0;
        for (var x : nums)
            if (x % 2 == 0 && x % 3 == 0) {
                sum += x;
                cnt ++ ;
            }
        return cnt == 0 ? 0 : sum / cnt;
    }
}
~~~

#### Problem B - [最流行的视频创作者](https://leetcode.cn/problems/most-popular-video-creator/)

**解法：模拟**

~~~
class Solution {
    public List<List<String>> mostPopularCreator(String[] creators, String[] ids, int[] views) {
        int n = creators.length;
        var max = 0l;
        List<List<String>> res = new ArrayList<>();
        Map<String, Long> m1 = new HashMap<>();
        Map<String, Integer> m2 = new HashMap<>();
        for (int i = 0; i < n; i ++ ) {
            var creator = creators[i];
            var id = ids[i];
            var view = views[i];
            m1.put(creator, m1.getOrDefault(creator, 0l) + view);
            max = Math.max(max, m1.get(creator));
            if (m2.containsKey(creator)) {
                if (view > views[m2.get(creator)]) {
                    m2.put(creator, i);
                } else if (view == views[m2.get(creator)]) {
                    if (ids[m2.get(creator)].compareTo(id) > 0) {
                        m2.put(creator, i);
                    }
                }
            } else {
                m2.put(creator, i);
            }
        }
        for (var creator : m1.keySet()) {
            if (m1.get(creator) == max) {
                res.add(List.of(creator, ids[m2.get(creator)]));
            }
        }
        return res;
    }
}
~~~



#### Problem C - [美丽整数的最小增量](https://leetcode.cn/problems/minimum-addition-to-make-integer-beautiful/)

**解法：数学&模拟**

~~~
class Solution {
    public int f(List<Integer> A) {
        var sm = 0;
        for (var x : A) sm += x;
        return sm;
    }
    public long makeIntegerBeautiful(long n, int target) {
        var A = new ArrayList<Integer>();
        var t = n;
        while (t != 0) {
            A.add((int)(t % 10));
            t /= 10;
        }
        A.add(0);
        if (f(A) <= target) return 0;
        var res = 0l;
        var p = 1l;
        for (var i = 0; i + 1 < A.size(); i ++, p *= 10) {
            res += (10l - A.get(i)) * p;
            A.set(i, 0);
            A.set(i + 1, A.get(i + 1) + 1);
            if (A.get(i + 1) == 10) continue;
            if (f(A) <= target) break;
        }
        return res;
    }
}
~~~



#### Problem D - [移除子树后的二叉树高度](https://leetcode.cn/problems/height-of-binary-tree-after-subtree-removal-queries/)

**解法：dfs**

## 318场周赛

#### Problem A - [对数组执行操作](https://leetcode.cn/problems/apply-operations-to-an-array/)

**解法：模拟**

~~~
class Solution {
    public int[] applyOperations(int[] nums) {
        var n = nums.length;
        for (var i = 0; i < n - 1; i ++ ) {
            if (nums[i] == nums[i + 1]) {
                nums[i] *= 2;
                nums[i + 1] = 0;
            }
        }
        var idx = 0;
        for (int i = 0; i < n; i ++ ) 
            if (nums[i] != 0)
                nums[idx ++ ] = nums[i];
        while (idx < n) nums[idx ++ ] = 0;
        return nums;
    }
}
~~~



#### Problem B - [长度为 K 子数组中的最大和](https://leetcode.cn/problems/maximum-sum-of-distinct-subarrays-with-length-k/)

**解法：滑动窗口**

~~~
class Solution {
    public long maximumSubarraySum(int[] nums, int k) {
        var res = 0l;
        var n = nums.length;
        var map = new int[100010];
        var s = new long[n + 1];
        for (int i = 0; i < n; i ++ )
            s[i + 1] = s[i] + nums[i] * 1l;
        for (int i = 0, j = 0; i < n; i ++ ) {
            map[nums[i]] ++ ;
            while (map[nums[i]] > 1 || i - j + 1 > k)
                map[nums[j ++ ]] -- ;
            if (i - j + 1 == k)
                res = Math.max(res, s[i + 1] - s[j]);
        }
        return res;
    }
}
~~~



#### Problem C - [雇佣 K 位工人的总代价](https://leetcode.cn/problems/total-cost-to-hire-k-workers/)

**解法：模拟**

~~~
class Solution {
    public long totalCost(int[] costs, int k, int candidates) {
        var qf = new PriorityQueue<Integer>();
        var qe = new PriorityQueue<Integer>();
        var n = costs.length;
        for (int i = 0; i < candidates; i ++ ) 
            qf.add(costs[i]);
        for (int i = Math.max(n - candidates, candidates); i < n; i ++ )
            qe.add(costs[i]);
        int i = candidates, j = n - candidates - 1;
        var res = 0l;
        while (k -- > 0) {
            if (qe.isEmpty() || (!qf.isEmpty() && qf.peek() <= qe.peek())) {
                res += qf.poll();
                if (i <= j) 
                    qf.add(costs[i ++ ]);
            } else {
                res += qe.poll();
                if (i <= j) 
                    qe.add(costs[j -- ]);
            }
        }
        return res;
    }
}
~~~



#### Problem D - [最小移动总距离](https://leetcode.cn/problems/minimum-total-distance-traveled/)

**解法：动态规划**



## 91场双周赛

#### Problem A - [不同的平均值数目](https://leetcode.cn/problems/number-of-distinct-averages/)

**解法：模拟**

~~~
class Solution {
    public int distinctAverages(int[] nums) {
        Arrays.sort(nums);
        var n = nums.length;
        var set = new HashSet<Integer>();
        for (int i = 0, j = n - 1; i < j; i ++ , j -- ) {
            set.add(nums[i] + nums[j]);
        }
        return set.size();
    }
}
~~~



#### Problem B - [统计构造好字符串的方案数](https://leetcode.cn/problems/count-ways-to-build-good-strings/)

**解法：递推**

~~~
class Solution {
    int MOD = (int) 1e9 + 7;
    public int countGoodStrings(int l, int h, int zero, int one) {
        int res = 0;
        var f = new long[h + 1];
        for (int i = 1; i <= h; i ++ ) {
            if (i - zero >= 0)
                f[i] = (f[i] + f[i - zero] + 1) % MOD;
            if (i - one >= 0)
                f[i] = (f[i] + f[i - one] + 1) % MOD; 
        }
        return (int)(f[h] + MOD - f[l - 1]) % MOD;
    }
}
~~~



#### Problem C - [树上最大得分和路径](https://leetcode.cn/problems/most-profitable-path-in-a-tree/)

**解法：枚举&dfs**

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

**链式前向星建图**

~~~
class Solution {
    int n;
    int idx;
    int[] e, h, ne;
    int[] bt, p, w;
    public void add(int a, int b) {
        e[idx] = b;
        ne[idx] = h[a];
        h[a] = idx ++ ;
    }

    public void dfs1(int u, int fa) {
        for (var i = h[u]; i != -1; i = ne[i]) {
            var v = e[i];
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
        for (var i = h[u]; i != -1; i = ne[i]) {
            var v = e[i];
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
        e = new int[n * 2];
        ne = new int[n * 2];
        h = new int[n];
        Arrays.fill(h, -1);
        for (var e : edges) {
            int a = e[0], b = e[1];
            add(a, b);
            add(b, a);
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



#### Problem D - [根据限制分割消息](https://leetcode.cn/problems/split-message-based-on-limit/)

**解法：枚举 & 模拟**



## 319场周赛

#### Problem A - [温度转换](https://leetcode.cn/problems/convert-the-temperature/)

**解法：模拟**

~~~
class Solution {
    public double[] convertTemperature(double celsius) {
        return new double[]{celsius + 273.15, celsius * 1.80 +};
    }
}
~~~



#### Problem B - [最小公倍数为 K 的子数组数目](https://leetcode.cn/problems/number-of-subarrays-with-lcm-equal-to-k/)

**解法：模拟**

~~~
class Solution {
    int gcd(int a, int b) {
        return b != 0 ? gcd(b, a % b) : a;
    }

    int lcm(int a, int b) {
        return a * b / gcd(a, b);
    }

    public int subarrayLCM(int[] nums, int k) {
        int n = nums.length, res = 0;
        for (int i = 0; i < n; i ++ ) {
            for (int j = i, p = 1; j < n; j ++ ) {
                p = lcm(p, nums[j]);
                if (k % p != 0) break; //剪枝，lcm必须是k的因子
                if (p == k) res ++ ;
            }
        }
        return res;
    }
}
~~~



#### Problem C - [逐层排序二叉树所需的最少操作数目](https://leetcode.cn/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/)

**解法：模拟**

~~~
class Solution {
    public void swap(int[] arr, int a, int b) {
        int t = arr[a];
        arr[a] = arr[b];
        arr[b] = t;
    }

    public int caclSortedNum(int[] a) {
        var map = new HashMap<Integer, Integer>();
        var b = a.clone();
        var n = a.length;
        var res = 0;
        Arrays.sort(b);
        for (int i = 0; i < n; i ++ ) map.put(b[i], i);
        for (int i = 0; i < n; i ++ ) {
            while (a[i] != b[i]) {
                swap(a, i, map.get(a[i]));
                res ++ ;
            }
        }
        return res;
    }

    public int minimumOperations(TreeNode root) {
        if (root == null) return 0;
        var q = new ArrayDeque<TreeNode>();
        var res = 0;
        q.add(root);
        while (!q.isEmpty()) {
            var size = q.size();
            while (size -- > 0) {
                var cur = q.poll();
                if (cur.left != null) q.add(cur.left);
                if (cur.right != null) q.add(cur.right);
            }
            var temp = q.stream().mapToInt(x -> x.val).toArray();
            res += caclSortedNum(temp);
        }
        return res;
    }
}
~~~



####  Problem D - [不重叠回文子字符串的最大数目](https://leetcode.cn/problems/maximum-number-of-non-overlapping-palindrome-substrings/)

**解法：DP&中心扩展**

~~~
class Solution {
    public int maxPalindromes(String S, int k) {
        var s = S.toCharArray();
        var n = s.length;
        var f = new int[n + 1];
        for (var i = 0; i < 2 * n - 1; i ++ ) {
            int l = i / 2, r = l + i % 2; // 中心扩展法
            f[l + 1] = Math.max(f[l + 1], f[l]);
            for (; l >= 0 && r < n && s[l] == s[r]; l -- , r ++ )
                if (r - l + 1 >= k)
                    f[r + 1] = Math.max(f[r + 1], f[l] + 1);
        }
        return f[n];
    }
}

~~~

## 320场周赛

#### Problem A - [数组中不等三元组的数目](https://leetcode.cn/problems/number-of-unequal-triplets-in-array/)

**解法：模拟**

~~~~
class Solution {
    public int unequalTriplets(int[] nums) {
        var res = 0;
        var n = nums.length;
        for (var i = 0; i < n; i ++ )
            for (var j = i + 1; j < n; j ++ )
                for (var k = j + 1; k < n; k ++ )
                    if (nums[i] != nums[j] && nums[i] != nums[k] && nums[j] != nums[k])
                        res ++ ;
        return res;
    }
}
~~~~



#### Problem B - [二叉搜索树最近节点查询](https://leetcode.cn/problems/closest-nodes-queries-in-a-binary-search-tree/)

**解法：二分**

~~~
class Solution {
    List<Integer> arr;
    public void dfs(TreeNode root) {
        if (root == null) return;
        dfs(root.left);
        arr.add(root.val);
        dfs(root.right);
    }
    public List<List<Integer>> closestNodes(TreeNode root, List<Integer> queries) {
        arr = new ArrayList<>();
        dfs(root);
        var res = new ArrayList<List<Integer>>();
        for (var q : queries) {
            int l = 0, r = arr.size() - 1;
            var tem = new ArrayList<Integer>();
            while (l < r) {
                var mid = l + r + 1 >> 1;
                if (arr.get(mid) <= q) l = mid;
                else r = mid - 1;
            }
            if (arr.get(l) <= q) tem.add(arr.get(l));
            else tem.add(-1);
            l = 0;
            r = arr.size() - 1;
            while (l < r) {
                var mid = l + r >> 1;
                if (arr.get(mid) >= q) r = mid;
                else l = mid + 1;
            }
            if (arr.get(r) >= q) tem.add(arr.get(r));
            else tem.add(-1);
            res.add(tem);
        }
        return res;
    }
}
~~~

**解法：二叉搜索树**

~~~
class Solution {
    TreeSet<Integer> set;
    public void dfs(TreeNode root) {
        if (root == null) return;
        dfs(root.left);
        set.add(root.val);
        dfs(root.right);
    }
    public List<List<Integer>> closestNodes(TreeNode root, List<Integer> queries) {
        set = new TreeSet<>();
        dfs(root);
        var res = new ArrayList<List<Integer>>();
        for (var q : queries) {
            var t = new ArrayList<Integer>();
            t.add(set.floor(q) == null ? -1 : set.floor(q));
            t.add(set.ceiling(q) == null ? -1 : set.ceiling(q));
            res.add(t);
        }
        return res;
    }
}
~~~



#### Problem C - [到达首都的最少油耗](https://leetcode.cn/problems/minimum-fuel-cost-to-report-to-the-capital/)

**解法：贪心**

~~~
class Solution {
    int seats;
    long res = 0l;
    List<Integer>[] g;

    public int dfs(int x, int fa) {
        var size = 1;
        for (var p : g[x])
            if (p != fa)
                size += dfs(p, x);
        if (x != 0)
            res += (size + seats - 1) / seats;
        return size;
    }

    public long minimumFuelCost(int[][] roads, int _seats) {
        var n = roads.length + 1;
        g = new List[n];
        seats = _seats;
        Arrays.setAll(g, e -> new ArrayList<>());
        for (var road : roads) {
            int a = road[0], b = road[1];
            g[a].add(b);
            g[b].add(a);
        }
        dfs(0, -1);
        return res;
    }
}
~~~



#### Problem D - [完美分割的方案数](https://leetcode.cn/problems/number-of-beautiful-partitions/)

**解法：dp**



## 92场双周赛

#### Problem A - [分割圆的最少切割次数](https://leetcode.cn/problems/minimum-cuts-to-divide-a-circle/)

**解法：分类讨论**

~~~
class Solution {
    public int numberOfCuts(int n) {
        if (n == 1) return 0;
        else if ((n & 1) == 1) return n;
        else return n / 2;
    }
}
~~~



#### Problem B - [行和列中一和零的差值](https://leetcode.cn/problems/difference-between-ones-and-zeros-in-row-and-column/)

**解法：模拟**

~~~
class Solution {
    public int[][] onesMinusZeros(int[][] g) {
        int n = g.length, m = g[0].length;
        var cols = new int[n];
        var rows = new int[m];
        for (var i = 0; i < n; i ++ ) 
            for (var j = 0; j < m; j ++ ) {
                cols[i] += g[i][j];
                rows[j] += g[i][j];
            }
        for (var i = 0; i < n; i ++ ) 
            for (var j = 0; j < m; j ++ ) 
                g[i][j] = cols[i] + rows[j] - (n - cols[i]) - (m - rows[j]);
        return g;
    }
}
~~~



#### Problem C - [商店的最少代价](https://leetcode.cn/problems/minimum-penalty-for-a-shop/)

**解法：枚举&前缀和**

~~~
class Solution {
    public int bestClosingTime(String c) {
        var n = c.length();
        var f = new int[n + 1];
        var g = new int[n + 1];
        for (var i = 0; i < n; i ++ ) f[i + 1] = f[i] + (c.charAt(i) == 'N' ? 1 : 0);
        for (var i = n; i > 0; i -- ) g[i - 1] = g[i] + (c.charAt(i - 1) == 'Y' ? 1 : 0);
        int res = 0, cnt = (int) 1e5;
        for (int i = 0; i <= n; i ++ ) {
            if (cnt > f[i] + g[i]) {
                res = i;
                cnt = f[i] + g[i];
            }
        }
        return res;
    }
}
~~~



#### Problem D - [统计回文子序列数目](https://leetcode.cn/problems/count-palindromic-subsequences/)

**解法：枚举 & 递推**



## 321场周赛

#### Problem A - [找出中枢整数](https://leetcode.cn/problems/find-the-pivot-integer/)

**解法：枚举**

~~~
class Solution {
    public int pivotInteger(int n) {
        var k = (1 + n) * n / 2;
        for (int i = 1, p = 0; i <= n; i ++ ) {
            p += i;
            if (p == k)
                return i;
            k -= i;
        }
        return -1;
    }
}
~~~



#### Problem B - [追加字符以获得子序列](https://leetcode.cn/problems/append-characters-to-string-to-make-subsequence/)

**解法：贪心**

~~~
class Solution {
    public int appendCharacters(String s, String t) {
        int k = 0;
        for (char c : s.toCharArray()) {
            if (c == t.charAt(k)) k ++ ;
            if (k == t.length()) break;
        }
        return t.length() - k;
    }
}
~~~



#### Problem C - [从链表中移除节点](https://leetcode.cn/problems/remove-nodes-from-linked-list/)

**解法：模拟**

~~~
class Solution {
    public ListNode removeNodes(ListNode head) {
        var list = new ArrayList<Integer>();
        for (var p = head; p != null; p = p.next) {
            list.add(p.val);
        }
        var list2 = new ArrayList<Integer>();
        for (int i = list.size() - 1, mx = 0; i >= 0; i -- ) {
            mx = Math.max(mx, list.get(i));
            if (list.get(i) >= mx) list2.add(list.get(i));
        }
        var dummy = new ListNode(-1);
        var cur = dummy;
        for (var i = list2.size() - 1; i >= 0; i -- ) {
            cur = cur.next = new ListNode(list2.get(i));
        }
        return dummy.next;
    }
}
~~~

**解法：递归**

~~~
class Solution {
    int mx = 0;
    public ListNode removeNodes(ListNode head) {
        return dfs(head);
    }

    public ListNode dfs(ListNode head) {
        if (head == null) return head;
        var res = dfs(head.next);
        mx = Math.max(mx, head.val);
        if (head.val >= mx) {
            head.next = res;
            return head;
        } else {
            return res;
        } 
    }
}
~~~

#### Problem D - [统计中位数为 K 的子数组](https://leetcode.cn/problems/count-subarrays-with-median-k/)

**解法：枚举&中心扩展**

~~~
class Solution {
    public int countSubarrays(int[] nums, int k) {
        var res = 0;
        var n = nums.length;
        var p = 0;
        while (nums[p] != k) p ++ ;
        var map = new HashMap<Integer, Integer>();
        map.put(0, 1);
        for (int i = p - 1, cur = 0; i >= 0; i -- ) {
            cur += nums[i] > k ? 1 : -1;
            map.put(cur, map.getOrDefault(cur, 0) + 1);
        }
        res += map.get(0);
        res += map.getOrDefault(1, 0);
        for (int i = p + 1, cur = 0; i < n; i ++ ) {
            cur += nums[i] > k ? 1 : -1;
            res += map.getOrDefault(-cur, 0);
            res += map.getOrDefault(-(cur - 1), 0);
        }
        return res;
    }
}
~~~

