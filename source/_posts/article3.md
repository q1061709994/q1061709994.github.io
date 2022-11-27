---
icon: article
title: 算法模板
author: huan
date: 2022-01-02
category: 算法笔记
tag: 
    - 数据结构与算法
star: true
---

## 一、基础算法

### 1.快速排序算法模板

[O(nlogn)]	空间[O(logn)]

~~~
void quick_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j)
    {
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }
    quick_sort(q, l, j), quick_sort(q, j + 1, r);
}
~~~

快速选择算法

[O(n)]	空间[O(logn)]

~~~
public void swap(int[] nums, int i, int j) {
	int t = nums[i];
    nums[i] = nums[j];
    nums[j] = t;
}
public int quick_select(int[] nums, int l, int r, int k) {
	if (l == r) return nums[k];
    int x = nums[new Random().nextInt(l, r + 1)], i = l - 1, j = r + 1;
    while (i < j) {
        do i ++ ; while (nums[i] > x);
        do j -- ; while (nums[j] < x);
        if (i < j) swap(nums, i, j);
    }
    if (k <= j) return quick_select(nums, l, j, k);
    else return quick_select(nums, j + 1, r, k);
}
~~~



### 2.归并排序算法模板

递归[O(nlogn)]	[O(logn)] 

~~~
void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int mid = l + r >> 1;
    merge_sort(q, l, mid);
    merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else tmp[k ++ ] = q[j ++ ];

    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];

    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
}
~~~
迭代	[O(nlogn)]	 [O(1)]
~~~
 public static void merge_sort(int[] k, int len) {
 		//用于存放排序的临时变量
        int[] temp = new int[len];
        //next是用来标志temp数组下标的
        int next;
        /*
        *每次归并都是对两段数据进行对比排序
        *left\right分别代表左面和右面(前面和后面)的两段数据
        *min和max分别代表各段数据的最前和最后下标
        */
        int left_min, left_max, right_min, right_max;
        for (int i = 1; i < len; i = i << 1) {
        	//每次步长递增，都从头开始归并处理
            for (left_min = 0; left_min < len - 1; left_min = right_max) {
            	//两段数据和步长之间关系
                right_min = left_max = left_min + i;
                right_max = left_max + i;
                //最后的下标不能超过n,否则无意义
                if (right_max > len) {
                    right_max = len;
                }
                //每次的内层循环都会将排列好的数据返回到k数组,因此next指针需每次清零
                next = 0;
                //两端数据均未排完
                while (left_min < left_max && right_min < right_max) {
                    if (k[left_min] < k[right_min]) temp[next++] = k[left_min++];
                    else temp[next++] = k[right_min++];
                }
                /*
                *上面的归并排序循环结束后,可能有一段数据尚未完全被排列带temp数组中 剩下未排列到temp中的数据一定是按照升序排					*列的最大的一部分数据
            	*此时有两种情况:left未排列完成,right未排列完成
        		*若是left未排列完成(left_min<left_max),则对于这一段数据省去temp数组的中转,直接赋值到k数组,即从right_max				*开始倒着赋值 
			 	*若是right未排列完成,则可以想到,那一段数据本就在应该放置的位置,则无需处理 
			 	*上面分析应该从right_max开始倒着赋值,但是实际因为右边的数据段已经全部排列,故此时right_min=right_max
                *且这里将right_min移动到需要的位置,方便下面赋值时使用
			 	*/
                while (left_min < left_max) k[--right_min] = k[--left_max];
                /*把排列好的数据段赋值给k数组
                *这里可以直接用上面经过--right_min倒数过来的right_min值
                *经过上面倒数的处理,right_min恰好在需要赋值和不需要赋值的数据段的分界处
                */
                while (next != 0) k[--right_min] = temp[--next];
            }
        }
    }
~~~



### 3.整数二分算法模板

~~~
bool check(int x) {/* ... */} // 检查x是否满足某种性质

// 区间[l, r]被划分成[l, mid]和[mid + 1, r]时使用：
int bsearch_1(int l, int r)
{
    while (l < r)
    {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;    // check()判断mid是否满足性质
        else l = mid + 1;
    }
    return l;
}
// 区间[l, r]被划分成[l, mid - 1]和[mid, r]时使用：
int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = l + r + 1 >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}
~~~

~~~
 		int[] arr = new int[]{0, 1, 3, 7, 8, 9, 11};
        //int[] arr = new int[]{1, 2, 2, 4};
        int n = arr.length;
        int a = 2, b = 7;
        int l = 0, r = n - 1;
        //找大于等于a的左边界
//        while (l < r) {
//            int mid = l + r >> 1;
//            if (arr[mid] >= a) r = mid;
//            else l = mid + 1;
//        }
        //找大于等于a的右边界
//        while (l < r) {
//            int mid = l + r + 1 >> 1;
//            if (arr[mid] >= a) l = mid;
//            else r = mid - 1;
//        }
        //找arr中小于等于b的第一个位置
//        while (l < r) {
//            int mid = l + r + 1 >> 1;
//            if (arr[mid] <= b) l = mid;
//            else r = mid - 1;
//        }

        //找arr中小于b的最后一个数
        while (l < r) {
            int mid = l + r >> 1;
            if (arr[mid] <= b) r = mid;
            else l = mid + 1;
        }
~~~

找右边界

[*81. 搜索旋转排序数组 II](https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/)



### 4.浮点数二分算法模板 

~~~
bool check(double x) {/* ... */} // 检查x是否满足某种性质

double bsearch_3(double l, double r)
{
    const double eps = 1e-6;   // eps 表示精度，取决于题目对精度的要求
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
~~~

### 5.高精度

~~~
高精度加法 —— 模板题 AcWing 791. 高精度加法
// C = A + B, A >= 0, B >= 0
vector<int> add(vector<int> &A, vector<int> &B)
{
    if (A.size() < B.size()) return add(B, A);

    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i ++ )
    {
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    
    if (t) C.push_back(t);
    return C;

}
高精度减法 —— 模板题 AcWing 792. 高精度减法
// C = A - B, 满足A >= B, A >= 0, B >= 0
vector<int> sub(vector<int> &A, vector<int> &B)
{
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); i ++ )
    {
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;

}
高精度乘低精度 —— 模板题 AcWing 793. 高精度乘法
// C = A * b, A >= 0, b >= 0
vector<int> mul(vector<int> &A, int b)
{
    vector<int> C;

    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ )
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }
    
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    
    return C;

}
高精度除以低精度 —— 模板题 AcWing 794. 高精度除法
// A / b = C ... r, A >= 0, b > 0
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
~~~

### 6.前缀和

#### 1.一维前缀和

S[i] = a[1] + a[2] + ... a[i]
a[l] + ... + a[r] = S[r] - S[l - 1]

#### 2.二维前缀和 

S[i, j] = 第i行j列格子左上部分所有元素的和
以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵的和为：
S[x2, y2] - S[x1 - 1, y2] - S[x2, y1 - 1] + S[x1 - 1, y1 - 1]

~~~
int[][] s = new int[n][m];
for (int i = 0; i < n; i ++ )
	for (int j = 0; i < m; j ++ )
		s[i + 1][j + 1] = s[i + 1][j] + s[i][j + 1] - s[i][j] + g[i][j];
public int get(int x1, int y1, int x2, int y2) {
	return s[x2 + 1][y2 + 1] - s[x1][y2 + 1] - s[x2 + 1][y1] + s[x1][y1];
}
~~~



### 7.差分

### 1.一维差分

给区间[l, r]中的每个数加上c：B[l] += c, B[r + 1] -= c

### 2.二维差分

给以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵中的所有元素加上c：
S[x1, y1] += c, S[x2 + 1, y1] -= c, S[x1, y2 + 1] -= c, S[x2 + 1, y2 + 1] += c
位运算 —— 模板题 AcWing 801. 二进制中1的个数
求n的第k位数字: n >> k & 1
返回n的最后一位1：lowbit(n) = n & -n

### 8.双指针算法 

~~~
for (int i = 0, j = 0; i < n; i ++ )
{
    while (j < i && check(i, j)) j ++ ;

    // 具体问题的逻辑
    
}
~~~

常见问题分类：
    (1) 对于一个序列，用两个指针维护一段区间
    (2) 对于两个序列，维护某种次序，比如归并排序中合并两个有序序列的操作

### 9.离散化

~~~
vector<int> alls; // 存储所有待离散化的值
sort(alls.begin(), alls.end()); // 将所有值排序
alls.erase(unique(alls.begin(), alls.end()), alls.end());   // 去掉重复元素

// 二分求出x对应的离散化的值
int find(int x) // 找到第一个大于等于x的位置
{
    int l = 0, r = alls.size() - 1;
    while (l < r)
    {
        int mid = l + r >> 1;
        if (alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return r + 1; // 映射到1, 2, ...n
}
~~~



### 10.区间合并 

**解法1：双指针**

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

### 11.位运算

### 1.n的二进制表示第k位是几

### 2.lowbit运算

### 3.2的x次  			

​	1 << x

### 4.^运算

异位或运算

0 ^ 0 = 0; 0 ^ 1 = 1;

1 ^ 0 = 1; 1 ^ 1 = 0;

## 二、数据结构

### 单链表 

~~~
// head存储链表头，e[]存储节点的值，ne[]存储节点的next指针，idx表示当前用到了哪个节点
int head, e[N], ne[N], idx;

// 初始化
void init()
{
    head = -1;
    idx = 0;
}

// 在链表头插入一个数a
void insert(int a)
{
    e[idx] = a, ne[idx] = head, head = idx ++ ;
}

// 将头结点删除，需要保证头结点存在
void remove()
{
    head = ne[head];
}
~~~

### 双链表 

~~~
// e[]表示节点的值，l[]表示节点的左指针，r[]表示节点的右指针，idx表示当前用到了哪个节点
int e[N], l[N], r[N], idx;

// 初始化
void init()
{
    //0是左端点，1是右端点
    r[0] = 1, l[1] = 0;
    idx = 2;
}

// 在节点a的右边插入一个数x
void insert(int a, int x)
{
    e[idx] = x;
    l[idx] = a, r[idx] = r[a];
    l[r[a]] = idx, r[a] = idx ++ ;
}

// 删除节点a
void remove(int a)
{
    l[r[a]] = l[a];
    r[l[a]] = r[a];
}
~~~



### 栈 

~~~
// tt表示栈顶
int stk[N], tt = 0;

// 向栈顶插入一个数
stk[ ++ tt] = x;

// 从栈顶弹出一个数
tt -- ;

// 栈顶的值
stk[tt];

// 判断栈是否为空
if (tt > 0)
{

}
~~~

### 队列 

~~~

1. 普通队列：
// hh 表示队头，tt表示队尾
int q[N], hh = 0, tt = -1;

// 向队尾插入一个数
q[ ++ tt] = x;

// 从队头弹出一个数
hh ++ ;

// 队头的值
q[hh];

// 判断队列是否为空
if (hh <= tt)
{

}
2. 循环队列
// hh 表示队头，tt表示队尾的后一个位置
int q[N], hh = 0, tt = 0;

// 向队尾插入一个数
q[tt ++ ] = x;
if (tt == N) tt = 0;

// 从队头弹出一个数
hh ++ ;
if (hh == N) hh = 0;

// 队头的值
q[hh];

// 判断队列是否为空
if (hh != tt)
{

}
~~~
### 单调栈

~~~
常见模型：找出每个数左边离它最近的比它大/小的数
int tt = 0;
for (int i = 1; i <= n; i ++ )
{
    while (tt && check(stk[tt], i)) tt -- ;
    stk[ ++ tt] = i;
}
~~~

### 单调队列 

~~~
常见模型：找出滑动窗口中的最大值/最小值
int hh = 0, tt = -1;
for (int i = 0; i < n; i ++ )
{
    while (hh <= tt && check_out(q[hh])) hh ++ ;  // 判断队头是否滑出窗口
    while (hh <= tt && check(q[tt], i)) tt -- ;
    q[ ++ tt] = i;
}
~~~

### KMP 

~~~
// s[]是长文本，p[]是模式串，n是s的长度，m是p的长度
求模式串的Next数组：
for (int i = 2, j = 0; i <= m; i ++ )
{
    while (j && p[i] != p[j + 1]) j = ne[j];
    if (p[i] == p[j + 1]) j ++ ;
    ne[i] = j;
}

// 匹配
for (int i = 1, j = 0; i <= n; i ++ )
{
    while (j && s[i] != p[j + 1]) j = ne[j];
    if (s[i] == p[j + 1]) j ++ ;
    if (j == m)
    {
        j = ne[j];
        // 匹配成功后的逻辑
    }
}


~~~

### Trie树

~~~
int son[N][26], cnt[N], idx;
// 0号点既是根节点，又是空节点
// son[][]存储树中每个节点的子节点
// cnt[]存储以每个节点结尾的单词数量

// 插入一个字符串
void insert(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++ ;
}

// 查询字符串出现的次数
int query(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}
~~~

### 并查集 

将两个集合合并

询问两个元素是否在一个集合当中

基本原理：每个集合用一颗树来表示，树根的标号就是整个集合的编号，每个节点存储它的父节点，p[x]表示x的父节点

问题1：如何判断树根：if(p[x] == x)

问题2：如何求x的集合编号：while (p[x]  != x)  x = p[x];

问题3：如何合并两个集合：px是x的集合编号，py是y的集合编号，px = y

### 1.朴素并查集

~~~
    int p[N]; //存储每个点的祖宗节点
    
    // 返回x的祖宗节点
    public int find(int x) {
    	if (x != p[x]) p[x] = find(p[x]); 
    	return p[x];
    }
    
    // 合并a和b所在的两个集合：
    public void union(int a, int b) {
    	p[find(a)] = find(b);
    }
    
    // 初始化，假定节点编号是1 ~ n
    for (int i = 1; i <= n; i ++ ) p[i] = i;

~~~

### 2.维护size的并查集
~~~
    int p[N], size[N];
    //p[]存储每个点的祖宗节点, size[]只有祖宗节点的有意义，表示祖宗节点所在集合中的点的数量
    
    // 返回x的祖宗节点
    public int find(int x) {
    	if (x != p[x]) p[x] = find(p[x]); 
    	return p[x];
    }
    
    // 合并a和b所在的两个集合：
    public void union(int a, int b) {
    	size[find(b)] += size[find(a)];
    	p[find(a)] = find(b);
    }
    
    // 初始化，假定节点编号是1~n
    for (int i = 1; i <= n; i ++ )
    {
        p[i] = i;
        size[i] = 1;
    }
~~~

### 3.维护到祖宗节点距离的并查集

~~~
    int p[N], d[N];
    //p[]存储每个点的祖宗节点, d[x]存储x到p[x]的距离
    
    // 返回x的祖宗节点
    int find(int x)
    {
        if (p[x] != x)
        {
            int u = find(p[x]);
            d[x] += d[p[x]];
            p[x] = u;
        }
        return p[x];
    }
    
    // 初始化，假定节点编号是1~n
    for (int i = 1; i <= n; i ++ )
    {
        p[i] = i;
        d[i] = 0;
    }
    
    // 合并a和b所在的两个集合：
    p[find(a)] = find(b);
    d[find(a)] = distance; // 根据具体问题，初始化find(a)的偏移量
~~~

### 堆

~~~
// h[N]存储堆中的值, h[1]是堆顶，x的左儿子是2x, 右儿子是2x + 1
// ph[k]存储第k个插入的点在堆中的位置
// hp[k]存储堆中下标是k的点是第几个插入的
int h[N], ph[N], hp[N], size;

// 交换两个点，及其映射关系
void heap_swap(int a, int b)
{
    swap(ph[hp[a]],ph[hp[b]]);
    swap(hp[a], hp[b]);
    swap(h[a], h[b]);
}

void down(int u)
{
    int t = u;
    if (u * 2 <= size && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= size && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (u != t)
    {
        heap_swap(u, t);
        down(t);
    }
}

void up(int u)
{
    while (u / 2 && h[u] < h[u / 2])
    {
        heap_swap(u, u / 2);
        u >>= 1;
    }
}

// O(n)建堆
for (int i = n / 2; i; i -- ) down(i);
~~~



### 一般哈希 

### 1.拉链法
~~~
    int h[N], e[N], ne[N], idx;
    // 向哈希表中插入一个数
    void insert(int x)
    {
        int k = (x % N + N) % N;
        e[idx] = x;
        ne[idx] = h[k];
        h[k] = idx ++ ;
    }
    
    // 在哈希表中查询某个数是否存在
    bool find(int x)
    {
        int k = (x % N + N) % N;
        for (int i = h[k]; i != -1; i = ne[i])
            if (e[i] == x)
                return true;
    
        return false;
    }
~~~
### 2.开放寻址法
~~~
     int h[N];
    // 如果x在哈希表中，返回x的下标；如果x不在哈希表中，返回x应该插入的位置
    int find(int x)
    {
        int t = (x % N + N) % N;
        while (h[t] != null && h[t] != x)
        {
            t ++ ;
            if (t == N) t = 0;
        }
        return t;
    }
~~~
### 字符串哈希

核心思想：将字符串看成P进制数，P的经验值是131或13331，取这两个值的冲突概率低
小技巧：取模的数用2^64，这样直接用unsigned long long存储，溢出的结果就是取模的结果

~~~
typedef unsigned long long ULL;
ULL h[N], p[N]; // h[k]存储字符串前k个字母的哈希值, p[k]存储 P^k mod 2^64

// 初始化
p[0] = 1;
for (int i = 1; i <= n; i ++ )
{
    h[i] = h[i - 1] * P + str[i];
    p[i] = p[i - 1] * P;
}

// 计算子串 str[l ~ r] 的哈希值
ULL get(int l, int r)
{
    return h[r] - h[l - 1] * p[r - l + 1];
}
~~~



### C++ STL简介
~~~
    vector, 变长数组，倍增的思想
        size()  返回元素个数
        empty()  返回是否为空
        clear()  清空
        front()/back()
        push_back()/pop_back()
        begin()/end()
        []
        支持比较运算，按字典序
    
    pair<int, int>
        first, 第一个元素
        second, 第二个元素
        支持比较运算，以first为第一关键字，以second为第二关键字（字典序）
    
    string，字符串
        size()/length()  返回字符串长度
        empty()
        clear()
        substr(起始下标，(子串长度))  返回子串
        c_str()  返回字符串所在字符数组的起始地址
    
    queue, 队列
        size()
        empty()
        push()  向队尾插入一个元素
        front()  返回队头元素
        back()  返回队尾元素
        pop()  弹出队头元素
    
    priority_queue, 优先队列，默认是大根堆
        size()
        empty()
        push()  插入一个元素
        top()  返回堆顶元素
        pop()  弹出堆顶元素
        定义成小根堆的方式：priority_queue<int, vector<int>, greater<int>> q;
    
    stack, 栈
        size()
        empty()
        push()  向栈顶插入一个元素
        top()  返回栈顶元素
        pop()  弹出栈顶元素
    
    deque, 双端队列
        size()
        empty()
        clear()
        front()/back()
        push_back()/pop_back()
        push_front()/pop_front()
        begin()/end()
        []
    
    set, map, multiset, multimap, 基于平衡二叉树（红黑树），动态维护有序序列
        size()
        empty()
        clear()
        begin()/end()
        ++, -- 返回前驱和后继，时间复杂度 O(logn)
    set/multiset
        insert()  插入一个数
        find()  查找一个数
        count()  返回某一个数的个数
        erase()
            (1) 输入是一个数x，删除所有x   O(k + logn)
            (2) 输入一个迭代器，删除这个迭代器
        lower_bound()/upper_bound()
            lower_bound(x)  返回大于等于x的最小的数的迭代器
            upper_bound(x)  返回大于x的最小的数的迭代器
    map/multimap
        insert()  插入的数是一个pair
        erase()  输入的参数是pair或者迭代器
        find()
        []  注意multimap不支持此操作。 时间复杂度是 O(logn)
        lower_bound()/upper_bound()
        unordered_set, unordered_map, unordered_multiset, unordered_multimap, 哈希表
        和上面类似，增删改查的时间复杂度是 O(1)
        不支持 lower_bound()/upper_bound()， 迭代器的++，--
    
    bitset, 圧位
        bitset<10000> s;
        ~, &, |, ^
        >>, <<
        ==, !=
        []
    count()  返回有多少个1
    
    any()  判断是否至少有一个1
    none()  判断是否全为0
    
    set()  把所有位置成1
    set(k, v)  将第k位变成v
    reset()  把所有位变成0
    flip()  等价于~
    flip(k) 把第k位取反
~~~
树与图的存储
树是一种特殊的图，与图的存储方式相同。
对于无向图中的边ab，存储两条有向边a->b, b->a。
因此我们可以只考虑有向图的存储。

(1) 邻接矩阵：g[a] [b] 存储边a->b

(2) 邻接表：

~~~
// 对于每个点k，开一个单链表，存储k所有可以走到的点。h[k]存储这个单链表的头结点
int h[N], e[N], ne[N], idx;

// 添加一条边a->b
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

// 初始化
idx = 0;
memset(h, -1, sizeof h);
~~~





## 三、搜索与图论

~~~~
1. 有向带权图最小环:Dijkstra
2. 有向带权图最大环:边变为负数+spfa 求最短路
3. 无向带权图最小环:Dijkstra
4. 无向带权图最大环:边变为负数+spfa 求最短路
5. 有向无权图最小环:拓扑排序找到所有环分组/Tarjan 缩点成拓扑图
6. 有向无权图最大环:拓扑排序找到所有环分组/Tarjan 缩点成拓扑图
7. 无向无权图最小环:拓扑排序找到所有环分组/dfs 找到所有环分组/Tarjan 缩点成树
8. 无向无权图最大环:拓扑排序找到所有环分组/dfs 找到所有环分组/Tarjan 缩点成树

- Dijkstra、spfa:枚举所有边，删除这条边之后以这条边的端点为起点终点跑一次最短路
~~~~



### 树与图的遍历

时间复杂度 O(n+m)O(n+m), nn 表示点数，mm 表示边数

### dfs深度优先遍历

~~~
int dfs(int u)
{
    st[u] = true; // st[u] 表示点u已经被遍历过

    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j]) dfs(j);
    }

}
~~~



### bfs宽度优先遍历 

~~~c++
queue<int> q;
st[1] = true; // 表示1号点已经被遍历过
q.push(1);

while (q.size())
{
    int t = q.front();
    q.pop();

    for (int i = h[t]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j])
        {
            st[j] = true; // 表示点j已经被遍历过
            q.push(j);
        }
    }

}
~~~

### 树的前序遍历

递归

~~~
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> preorderTraversal(TreeNode root) {
        dfs(root);
        return res;
    }
    public void dfs(TreeNode root){
        if (root == null) return;
        res.add(root.val);
        dfs(root.left);
        dfs(root.right);
    }
}
~~~

迭代

~~~
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                res.add(root.val);
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            root = root.right;
        }
        return res;
    }
}
~~~



### 树的中序遍历

递归

~~~
class Solution {
	List<Integer> res = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root){
        dfs(root);
        return res;
    }
    public void dfs(TreeNode root){
        if(root == null) return;
        dfs(root.left);
        res.add(root.val);
        dfs(root.right);
    }
}
~~~

迭代

~~~
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            res.add(root.val);
            root = root.right;
        }
        return res;
    }
}
~~~

### 树的后序遍历

递归

~~~
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> postorderTraversal(TreeNode root) {
        dfs(root);
        return res;
    }
    public void dfs(TreeNode root){
        if (root == null) return;
        dfs(root.left);
        dfs(root.right);
        res.add(root.val);
    }
}
~~~

迭代

~~~
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        Stack<TreeNode> resStack = new Stack<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                resStack.push(root);
                stack.push(root);
                root = root.right;
            }
            root = stack.pop();
            root = root.left;
        }
        while (!resStack.isEmpty()) {
            root = resStack.pop();
            res.add(root.val);
        }
        return res;
    }
}
~~~



### 树的层序遍历

~~~
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null)  queue.add(root);
        while (!queue.isEmpty()) {
            int count = queue.size();
            List<Integer> level = new ArrayList<>();
            while (count != 0) {
                TreeNode node = queue.poll();
                level.add(node.val);
                if (node.left != null) 	queue.add(node.left);
                if (node.right != null) queue.add(node.right);
                count--;
            }
            res.add(level);
        }
        return res;
    }
}
~~~



### 拓扑排序 

~~~
时间复杂度 O(n+m)O(n+m), nn 表示点数，mm 表示边数
bool topsort()
{
    int hh = 0, tt = -1;

    // d[i] 存储点i的入度
    for (int i = 1; i <= n; i ++ )
        if (!d[i])
            q[ ++ tt] = i;
    
    while (hh <= tt)
    {
        int t = q[hh ++ ];
    
        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (-- d[j] == 0)
                q[ ++ tt] = j;
        }
    }
    
    // 如果所有点都入队了，说明存在拓扑序列；否则不存在拓扑序列。
    return tt == n - 1;

}
~~~

### 朴素dijkstra算法 

~~~
时间复杂是 O(n2+m)O(n2+m), nn 表示点数，mm 表示边数
int g[N][N];  // 存储每条边
int dist[N];  // 存储1号点到每个点的最短距离
bool st[N];   // 存储每个点的最短路是否已经确定

// 求1号点到n号点的最短路，如果不存在则返回-1
int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    for (int i = 0; i < n - 1; i ++ )
    {
        int t = -1;     // 在还未确定最短路的点中，寻找距离最小的点
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;
    
        // 用t更新其他点的距离
        for (int j = 1; j <= n; j ++ )
            dist[j] = min(dist[j], dist[t] + g[t][j]);
    
        st[t] = true;
    }
    
    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];

}
~~~

~~~
static final int N = 510;
static int n, m;
static int[][] g = new int[N][N];
// 距离
static int[] dist = new int[N];
// 每个点最短路是否确定
static boolean[] st = new boolean[N];

public static int dijkstra() {
    Arrays.fill(dist,0x3f3f3f3f);
    dist[1] = 0;
    for (int i = 0; i < n; i ++ ) {
        int t = -1;
        for (int j = 1; j <= n; j ++ ) {
            if (!st[j] && (t == -1 || dist[t] > dist[j])) {
                t = j;
            }
        }
        if (t == n) break;
        st[t] = true;
        for (int j = 1; j <= n; j ++ ) {
            dist[j] = Math.min(dist[j], dist[t] + g[t][j]);
        }
    }
    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}
~~~



### 堆优化版dijkstra 

~~~
    时间复杂度 O(mlogn)O(mlogn), nn 表示点数，mm 表示边数
    typedef pair<int, int> PII;
    
    int n;      // 点的数量
    int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
    int dist[N];        // 存储所有点到1号点的距离
    bool st[N];     // 存储每个点的最短距离是否已确定
    
    // 求1号点到n号点的最短距离，如果不存在，则返回-1
    int dijkstra()
    {
        memset(dist, 0x3f, sizeof dist);
        dist[1] = 0;
        priority_queue<PII, vector<PII>, greater<PII>> heap;
        heap.push({0, 1});      // first存储距离，second存储节点编号
        while (heap.size())
        {
            auto t = heap.top();
            heap.pop();
    
            int ver = t.second, distance = t.first;
    
            if (st[ver]) continue;
            st[ver] = true;
    
            for (int i = h[ver]; i != -1; i = ne[i])
            {
                int j = e[i];
                if (dist[j] > distance + w[i])
                {
                    dist[j] = distance + w[i];
                    heap.push({dist[j], j});
                }
            }
        }
    
        if (dist[n] == 0x3f3f3f3f) return -1;
        return dist[n];
    }
~~~

~~~
static final int N = 100010;
static int n, m;
static int[] w = new int[N], h = new int[N], e = new int[N], ne = new int[N];
static int idx;
// 距离
static int[] dist = new int[N];
// 每个点最短路是否确定
static boolean[] st = new boolean[N];
static PriorityQueue<int[]> heap = new PriorityQueue<>((a, b)->a[0]-b[0]);

    private static void add(int a, int b, int c) {
        e[idx] = b;
        w[idx] = c;
        ne[idx] = h[a];
        h[a] = idx ++ ;
    }

    public static int dijkstra() {
        Arrays.fill(dist,0x3f3f3f3f);
        dist[1] = 0;
        heap.add(new int[]{0, 1});
        while (!heap.isEmpty()) {
            int[] t = heap.poll();
            int ver = t[1], distance = t[0];
            if (st[ver]) continue;

            for (int i = h[ver]; i != -1; i = ne[i]) {
                int j = e[i];
                if (dist[j] > distance + w[i]) {
                    dist[j] = distance + w[i];
                    heap.add(new int[]{dist[j], j});
                }
            }
        }
        if (dist[n] == 0x3f3f3f3f) return -1;
        return dist[n];
    }
~~~

~~~
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
~~~



### Bellman-Ford算法 



```c++
时间复杂度 O(nm)O(nm), nn 表示点数，mm 表示边数
注意在模板题中需要对下面的模板稍作修改，加上备份数组，详情见模板题。

int n, m;       // n表示点数，m表示边数
int dist[N];        // dist[x]存储1到x的最短路距离

struct Edge     // 边，a表示出点，b表示入点，w表示边的权重
{
    int a, b, w;
}edges[M];

// 求1到n的最短路距离，如果无法从1走到n，则返回-1。
int bellman_ford()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
// 如果第n次迭代仍然会松弛三角不等式，就说明存在一条长度是n+1的最短路径，由抽屉原理，路径中至少存在两个相同的点，说明图中存在负权回路。
for (int i = 0; i < n; i ++ )
{
    for (int j = 0; j < m; j ++ )
    {
        int a = edges[j].a, b = edges[j].b, w = edges[j].w;
        if (dist[b] > dist[a] + w)
            dist[b] = dist[a] + w;
    }
}

	if (dist[n] > 0x3f3f3f3f / 2) return -1;
	return dist[n];
}
```

~~~
private static int bellman_ford() {
        Arrays.fill(dist, 0x3f3f3f3f);
        dist[1] = 0;
        for (int i = 0; i < k; i ++ ) {
            //备份：存放上一次迭代的结果，防止串联
            backup = dist.clone();
            for (int j = 0; j < m; j ++ ) {
                int a = edges[j].a, b = edges[j].b, w = edges[j].w;
                dist[b] = Math.min(dist[b], backup[a] + w);
            }
        }
        if (dist[n] > 0x3f3f3f3f >> 1) return -1;
        return dist[n];
    }
~~~





### spfa 算法（队列优化的Bellman-Ford算法） 

~~~
    时间复杂度 平均情况下 O(m)O(m)，最坏情况下 O(nm)O(nm), nn 表示点数，mm 表示边数
    int n;      // 总点数
    int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
    int dist[N];        // 存储每个点到1号点的最短距离
    bool st[N];     // 存储每个点是否在队列中
    
    // 求1号点到n号点的最短路距离，如果从1号点无法走到n号点则返回-1
    int spfa()
    {
        memset(dist, 0x3f, sizeof dist);
        dist[1] = 0;
    queue<int> q;
    q.push(1);
    st[1] = true;
    
    while (q.size())
    {
        auto t = q.front();
        q.pop();
    
        st[t] = false;
    
        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > dist[t] + w[i])
            {
                dist[j] = dist[t] + w[i];
                if (!st[j])     // 如果队列中已存在j，则不需要将j重复插入
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }
    	if (dist[n] == 0x3f3f3f3f) return -1;
    	return dist[n];
    }
~~~
~~~
static final int N = 100010;
    static int n, m;
    static int[] w = new int[N], h = new int[N], e = new int[N], ne = new int[N];
    static int idx;
    // 距离
    static int[] dist = new int[N];
    // 每个点最短路是否确定
    static boolean[] st = new boolean[N];
    
    public static int spfa() {
        Arrays.fill(dist, 0x3f3f3f3f);
        dist[1] = 0;
        Queue<Integer> q = new LinkedList<>();
        q.add(1);
        st[1] = true;

        while (!q.isEmpty()) {
            int t = q.poll();

            st[t] = false;
            for (int i = h[t]; i != -1; i = ne[i]) {
                int j = e[i];
                if (dist[j] > dist[t] + w[i]) {
                    dist[j] = dist[t] + w[i];
                    if (!st[j]) {
                        q.add(j);
                        st[j] = true;
                    }
                }
            }
        }
        if (dist[n] == 0x3f3f3f3f) return -1;
        return dist[n];
    }
~~~



### spfa判断图中是否存在负环 
~~~
    时间复杂度是 O(nm)O(nm), nn 表示点数，mm 表示边数
    int n;      // 总点数
    int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
    int dist[N], cnt[N];        // dist[x]存储1号点到x的最短距离，cnt[x]存储1到x的最短路中经过的点数
    bool st[N];     // 存储每个点是否在队列中
    
    // 如果存在负环，则返回true，否则返回false。
    bool spfa()
    {
        // 不需要初始化dist数组
        // 原理：如果某条最短路径上有n个点（除了自己），那么加上自己之后一共有n+1个点，由抽屉原理一定有两个点相同，所以存在环。
    queue<int> q;
    for (int i = 1; i <= n; i ++ )
    {
        q.push(i);
        st[i] = true;
    }
    
    while (q.size())
    {
        auto t = q.front();
        q.pop();
    
        st[t] = false;
    
        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > dist[t] + w[i])
            {
                dist[j] = dist[t] + w[i];
                cnt[j] = cnt[t] + 1;
                if (cnt[j] >= n) return true;       // 如果从1号点到x的最短路中包含至少n个点（不包括自己），则说明存在环
                if (!st[j])
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }
    
    return false;
    }

~~~
### floyd算法 

~~~
时间复杂度是 O(n3)O(n3), nn 表示点数
初始化：
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            if (i == j) d[i][j] = 0;
            else d[i][j] = INF;

// 算法结束后，d[a][b]表示a到b的最短距离
void floyd()
{
    for (int k = 1; k <= n; k ++ )
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ )
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
}
~~~



### 朴素版prim算法 

~~~
时间复杂度是 O(n2+m)O(n2+m), nn 表示点数，mm 表示边数
int n;      // n表示点数
int g[N][N];        // 邻接矩阵，存储所有边
int dist[N];        // 存储其他点到当前最小生成树的距离
bool st[N];     // 存储每个点是否已经在生成树中


// 如果图不连通，则返回INF(值是0x3f3f3f3f), 否则返回最小生成树的树边权重之和
int prim()
{
    memset(dist, 0x3f, sizeof dist);

    int res = 0;
    for (int i = 0; i < n; i ++ )
    {
        int t = -1;
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;
    
        if (i && dist[t] == INF) return INF;
    
        if (i) res += dist[t];
        st[t] = true;
    
        for (int j = 1; j <= n; j ++ ) dist[j] = min(dist[j], g[t][j]);
    }
    
    return res;

}
~~~



### Kruskal算法 

~~~
时间复杂度是 O(mlogm)O(mlogm), nn 表示点数，mm 表示边数
int n, m;       // n是点数，m是边数
int p[N];       // 并查集的父节点数组

struct Edge     // 存储边
{
    int a, b, w;

    bool operator< (const Edge &W)const
    {
        return w < W.w;
    }

}edges[M];

int find(int x)     // 并查集核心操作
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int kruskal()
{
    sort(edges, edges + m);

    for (int i = 1; i <= n; i ++ ) p[i] = i;    // 初始化并查集
    
    int res = 0, cnt = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;
    
        a = find(a), b = find(b);
        if (a != b)     // 如果两个连通块不连通，则将这两个连通块合并
        {
            p[a] = b;
            res += w;
            cnt ++ ;
        }
    }
    
    if (cnt < n - 1) return INF;
    return res;

}
~~~



### 染色法判别二分图 

~~~
时间复杂度是 O(n+m)O(n+m), nn 表示点数，mm 表示边数
int n;      // n表示点数
int h[N], e[M], ne[M], idx;     // 邻接表存储图
int color[N];       // 表示每个点的颜色，-1表示未染色，0表示白色，1表示黑色

// 参数：u表示当前节点，c表示当前点的颜色
bool dfs(int u, int c)
{
    color[u] = c;
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (color[j] == -1)
        {
            if (!dfs(j, !c)) return false;
        }
        else if (color[j] == c) return false;
    }

    return true;

}

bool check()
{
    memset(color, -1, sizeof color);
    bool flag = true;
    for (int i = 1; i <= n; i ++ )
        if (color[i] == -1)
            if (!dfs(i, 0))
            {
                flag = false;
                break;
            }
    return flag;
}
~~~



### 匈牙利算法

~~~
时间复杂度是 O(nm)O(nm), nn 表示点数，mm 表示边数
int n1, n2;     // n1表示第一个集合中的点数，n2表示第二个集合中的点数
int h[N], e[M], ne[M], idx;     // 邻接表存储所有边，匈牙利算法中只会用到从第一个集合指向第二个集合的边，所以这里只用存一个方向的边
int match[N];       // 存储第二个集合中的每个点当前匹配的第一个集合中的点是哪个
bool st[N];     // 表示第二个集合中的每个点是否已经被遍历过

bool find(int x)
{
    for (int i = h[x]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j])
        {
            st[j] = true;
            if (match[j] == 0 || find(match[j]))
            {
                match[j] = x;
                return true;
            }
        }
    }

    return false;

}

// 求最大匹配数，依次枚举第一个集合中的每个点能否匹配第二个集合中的点
int res = 0;
for (int i = 1; i <= n1; i ++ )
{
    memset(st, false, sizeof st);
    if (find(i)) res ++ ;
}
~~~

## 四、数学知识

### 1.试除法判定质数

~~~
bool is_prime(int x)
{
    if (x < 2) return false;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
            return false;
    return true;
}
~~~

### 2.试除法分解质因数

~~~
void divide(int x)
{
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            int s = 0;
            while (x % i == 0) x /= i, s ++ ;
            cout << i << ' ' << s << endl;
        }
    if (x > 1) cout << x << ' ' << 1 << endl;
    cout << endl;
}
~~~

### 3.朴素筛法求素数 

~~~
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (st[i]) continue;
        primes[cnt ++ ] = i;
        for (int j = i + i; j <= n; j += i)
            st[j] = true;
    }
}
~~~

### 4.线性筛法求素数

~~~
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
    	//当前数如果没有被筛过，说明它就是素数
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            //primes[j]一定是i的最小质因子
            if (i % primes[j] == 0) break;
        }
    }
}
~~~

### 5.试除法求所有约数

~~~
vector<int> get_divisors(int x)
{
    vector<int> res;
    for (int i = 1; i <= x / i; i ++ )
        if (x % i == 0)
        {
            res.push_back(i);
            if (i != x / i) res.push_back(x / i);
        }
    sort(res.begin(), res.end());
    return res;
}
~~~

### 6.约数个数和约数之和

~~~~
如果 N = p1^c1 * p2^c2 * ... *pk^ck
约数个数： (c1 + 1) * (c2 + 1) * ... * (ck + 1)
约数之和： (p1^0 + p1^1 + ... + p1^c1) * ... * (pk^0 + pk^1 + ... + pk^ck)
~~~~

### 7.欧几里得算法

~~~
int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}
~~~

### 8.求欧拉函数 

~~~
int phi(int x)
{
    int res = x;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            res = res / i * (i - 1);
            while (x % i == 0) x /= i;
        }
    if (x > 1) res = res / x * (x - 1);

    return res;
}
~~~

### 9.筛法求欧拉函数 

~~~
int primes[N], cnt;     // primes[]存储所有素数
int euler[N];           // 存储每个数的欧拉函数
bool st[N];         // st[x]存储x是否被筛掉

void get_eulers(int n)
{
    euler[1] = 1;
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i])
        {
            primes[cnt ++ ] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            int t = primes[j] * i;
            st[t] = true;
            if (i % primes[j] == 0)
            {
                euler[t] = euler[i] * primes[j];
                break;
            }
            euler[t] = euler[i] * (primes[j] - 1);
        }
    }
}
~~~



### 10.快速幂

求 m^k mod p，时间复杂度 O(logk)。

~~~
public int qmi(int m, int k, int p) {
	long res = 1l % p, t = m * 1l;
	while (k != 0) {
    if ((k & 1) != 0) res = res * t % p;
        t = t * t % p;
        k >>= 1;
    }
	return (int)res;
}
~~~

### 11.扩展欧几里得算法 

~~~
// 求x, y，使得ax + by = gcd(a, b)
int exgcd(int a, int b, int &x, int &y)
{
    if (!b)
    {
        x = 1; y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= (a/b) * x;
    return d;
}
~~~

### 12.高斯消元

~~~
// a[N][N]是增广矩阵
int gauss()
{
    int c, r;
    for (c = 0, r = 0; c < n; c ++ )
    {
        int t = r;
        for (int i = r; i < n; i ++ )   // 找到绝对值最大的行
            if (fabs(a[i][c]) > fabs(a[t][c]))
                t = i;

        if (fabs(a[t][c]) < eps) continue;

        for (int i = c; i <= n; i ++ ) swap(a[t][i], a[r][i]);      // 将绝对值最大的行换到最顶端
        for (int i = n; i >= c; i -- ) a[r][i] /= a[r][c];      // 将当前行的首位变成1
        for (int i = r + 1; i < n; i ++ )       // 用当前行将下面所有的列消成0
            if (fabs(a[i][c]) > eps)
                for (int j = n; j >= c; j -- )
                    a[i][j] -= a[r][j] * a[i][c];

        r ++ ;
    }

    if (r < n)
    {
        for (int i = r; i < n; i ++ )
            if (fabs(a[i][n]) > eps)
                return 2; // 无解
        return 1; // 有无穷多组解
    }

    for (int i = n - 1; i >= 0; i -- )
        for (int j = i + 1; j < n; j ++ )
            a[i][n] -= a[i][j] * a[j][n];

    return 0; // 有唯一解
}
~~~

### 13.递归法求组合数 

~~~
// c[a][b] 表示从a个苹果中选b个的方案数
for (int i = 0; i < N; i ++ )
    for (int j = 0; j <= i; j ++ )
        if (!j) c[i][j] = 1;
        else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
~~~

### 14.通过预处理逆元的方式求组合数 

~~~
首先预处理出所有阶乘取模的余数fact[N]，以及所有阶乘取模的逆元infact[N]
如果取模的数是质数，可以用费马小定理求逆元
int qmi(int a, int k, int p)    // 快速幂模板
{
    int res = 1;
    while (k)
    {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

// 预处理阶乘的余数和阶乘逆元的余数
fact[0] = infact[0] = 1;
for (int i = 1; i < N; i ++ )
{
    fact[i] = (LL)fact[i - 1] * i % mod;
    infact[i] = (LL)infact[i - 1] * qmi(i, mod - 2, mod) % mod;
}
~~~

### 15.Lucas定理 

~~~
若p是质数，则对于任意整数 1 <= m <= n，有：
    C(n, m) = C(n % p, m % p) * C(n / p, m / p) (mod p)

int qmi(int a, int k, int p)  // 快速幂模板
{
    int res = 1 % p;
    while (k)
    {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

int C(int a, int b, int p)  // 通过定理求组合数C(a, b)
{
    if (a < b) return 0;

    LL x = 1, y = 1;  // x是分子，y是分母
    for (int i = a, j = 1; j <= b; i --, j ++ )
    {
        x = (LL)x * i % p;
        y = (LL) y * j % p;
    }

    return x * (LL)qmi(y, p - 2, p) % p;
}

int lucas(LL a, LL b, int p)
{
    if (a < p && b < p) return C(a, b, p);
    return (LL)C(a % p, b % p, p) * lucas(a / p, b / p, p) % p;
}
~~~

### 16.分解质因数法求组合数 

~~~
当我们需要求出组合数的真实值，而非对某个数的余数时，分解质因数的方式比较好用：
    1. 筛法求出范围内的所有质数
    2. 通过 C(a, b) = a! / b! / (a - b)! 这个公式求出每个质因子的次数。 n! 中p的次数是 n / p + n / p^2 + n / p^3 + ...
    3. 用高精度乘法将所有质因子相乘

int primes[N], cnt;     // 存储所有质数
int sum[N];     // 存储每个质数的次数
bool st[N];     // 存储每个数是否已被筛掉


void get_primes(int n)      // 线性筛法求素数
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}


int get(int n, int p)       // 求n！中的次数
{
    int res = 0;
    while (n)
    {
        res += n / p;
        n /= p;
    }
    return res;
}


vector<int> mul(vector<int> a, int b)       // 高精度乘低精度模板
{
    vector<int> c;
    int t = 0;
    for (int i = 0; i < a.size(); i ++ )
    {
        t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }

    while (t)
    {
        c.push_back(t % 10);
        t /= 10;
    }
    
    return c;

}

get_primes(a);  // 预处理范围内的所有质数

for (int i = 0; i < cnt; i ++ )     // 求每个质因数的次数
{
    int p = primes[i];
    sum[i] = get(a, p) - get(b, p) - get(a - b, p);
}

vector<int> res;
res.push_back(1);

for (int i = 0; i < cnt; i ++ )     // 用高精度乘法将所有质因子相乘
    for (int j = 0; j < sum[i]; j ++ )
        res = mul(res, primes[i]);
~~~



### 17.卡特兰数 

~~~
给定n个0和n个1，它们按照某种顺序排成长度为2n的序列，满足任意前缀中0的个数都不少于1的个数的序列的数量为： Cat(n) = C(2n, n) / (n + 1)
~~~

### 18.NIM游戏 

给定N堆物品，第i堆物品有Ai个。两名玩家轮流行动，每次可以任选一堆，取走任意多个物品，可把一堆取光，但不能不取。取走最后一件物品者获胜。两人都采取最优策略，问先手是否必胜。

我们把这种游戏称为NIM博弈。把游戏过程中面临的状态称为局面。整局游戏第一个行动的称为先手，第二个行动的称为后手。若在某一局面下无论采取何种行动，都会输掉游戏，则称该局面必败。
所谓采取最优策略是指，若在某一局面下存在某种行动，使得行动后对面面临必败局面，则优先采取该行动。同时，这样的局面被称为必胜。我们讨论的博弈问题一般都只考虑理想情况，即两人均无失误，都采取最优策略行动时游戏的结果。
NIM博弈不存在平局，只有先手必胜和先手必败两种情况。

定理： NIM博弈先手必胜，当且仅当 A1 ^ A2 ^ … ^ An != 0

公平组合游戏ICG
若一个游戏满足：

由两名玩家交替行动；
在游戏进程的任意时刻，可以执行的合法行动与轮到哪名玩家无关；
不能行动的玩家判负；
则称该游戏为一个公平组合游戏。
NIM博弈属于公平组合游戏，但城建的棋类游戏，比如围棋，就不是公平组合游戏。因为围棋交战双方分别只能落黑子和白子，胜负判定也比较复杂，不满足条件2和条件3。

### 19.有向图游戏

给定一个有向无环图，图中有一个唯一的起点，在起点上放有一枚棋子。两名玩家交替地把这枚棋子沿有向边进行移动，每次可以移动一步，无法移动者判负。该游戏被称为有向图游戏。
任何一个公平组合游戏都可以转化为有向图游戏。具体方法是，把每个局面看成图中的一个节点，并且从每个局面向沿着合法行动能够到达的下一个局面连有向边。

### 20.Mex运算

设S表示一个非负整数集合。定义mex(S)为求出不属于集合S的最小非负整数的运算，即：
mex(S) = min{x}, x属于自然数，且x不属于S

### 21.SG函数

在有向图游戏中，对于每个节点x，设从x出发共有k条有向边，分别到达节点y1, y2, …, yk，定义SG(x)为x的后继节点y1, y2, …, yk 的SG函数值构成的集合再执行mex(S)运算的结果，即：
SG(x) = mex({SG(y1), SG(y2), …, SG(yk)})
特别地，整个有向图游戏G的SG函数值被定义为有向图游戏起点s的SG函数值，即SG(G) = SG(s)。

有向图游戏的和 —— 模板题 AcWing 893. 集合-Nim游戏
设G1, G2, …, Gm 是m个有向图游戏。定义有向图游戏G，它的行动规则是任选某个有向图游戏Gi，并在Gi上行动一步。G被称为有向图游戏G1, G2, …, Gm的和。
有向图游戏的和的SG函数值等于它包含的各个子游戏SG函数值的异或和，即：
SG(G) = SG(G1) ^ SG(G2) ^ … ^ SG(Gm)

### 22.定理

有向图游戏的某个局面必胜，当且仅当该局面对应节点的SG函数值大于0。
有向图游戏的某个局面必败，当且仅当该局面对应节点的SG函数值等于0。

## 五、十大排序


### 1插入排序
~~~
void insert_sort()
{
    for (int i = 1; i < n; i ++ )
    {
        int x = a[i];
        int j = i-1;

        while (j >= 0 && x < a[j])
        {
            a[j+1] = a[j];
            j -- ;
        }
        a[j+1] = x;
    }
}
~~~
### 2选择排序
~~~
void select_sort()
{
    for (int i = 0; i < n; i ++ )
    {
        int k = i;
        for (int j = i+1; j < n; j ++ )
        {
            if (a[j] < a[k])
                k = j;
        }
        swap(a[i], a[k]);
    }

}
~~~
### 3冒泡排序
~~~
void bubble_sort()
{
    for (int i = n-1; i >= 1; i -- )
    {
        bool flag = true;
        for (int j = 1; j <= i; j ++ )
            if (a[j-1] > a[j])
            {
                swap(a[j-1], a[j]);
                flag = false;
            }
        if (flag) return;
    }
}
~~~
### 4希尔排序
~~~
void shell_sort()
{
    for (int gap = n >> 1; gap; gap >>= 1)
    {
        for (int i = gap; i < n; i ++ )
        {
            int x = a[i];
            int j;
            for (j = i; j >= gap && a[j-gap] > x; j -= gap)
                a[j] = a[j-gap];
            a[j] = x;
        }
    }
}
~~~
### 5快速排序（最快）
~~~
void quick_sort(int l, int r)
{
    if (l >= r) return ;

    int x = a[l+r>>1], i = l-1, j = r+1;
    while (i < j)
    {
        while (a[++ i] < x);
        while (a[-- j] > x);
        if (i < j) swap(a[i], a[j]);
    }
    sort(l, j), sort(j+1, r);
}
~~~
### 6归并排序
~~~
void merge_sort(int l, int r)
{
    if (l >= r) return;
    int temp[N];
    int mid = l+r>>1;
    merge_sort(l, mid), merge_sort(mid+1, r);
    int k = 0, i = l, j = mid+1;
    while (i <= mid && j <= r)
    {
        if (a[i] < a[j]) temp[k ++ ] = a[i ++ ];
        else temp[k ++ ] = a[j ++ ];
    }
    while (i <= mid) temp[k ++ ] = a[i ++ ];
    while (j <= r) temp[k ++ ] = a[j ++ ];
    for (int i = l, j = 0; i <= r; i ++ , j ++ ) a[i] = temp[j];
}
~~~
### 7堆排序

(须知此排序为使用了模拟堆，为了使最后一个非叶子节点的编号为n/2，数组编号从1开始)

https://www.cnblogs.com/wanglei5205/p/8733524.html
~~~
void down(int u)
{
    int t = u;
    if (u<<1 <= n && h[u<<1] < h[t]) t = u<<1;
    if ((u<<1|1) <= n && h[u<<1|1] < h[t]) t = u<<1|1;
    if (u != t)
    {
        swap(h[u], h[t]);
        down(t);
    }
}

int main()
{
    for (int i = 1; i <= n; i ++ ) cin >> h[i];
    for (int i = n/2; i; i -- ) down(i);
    while (true)
    {
        if (!n) break;
        cout << h[1] << ' ';
        h[1] = h[n];
        n -- ;
        down(1);
    }
    return 0;
}
~~~
### 8基数排序
~~~
int maxbit()
{
    int maxv = a[0];
    for (int i = 1; i < n; i ++ )
        if (maxv < a[i])
            maxv = a[i];
    int cnt = 1;
    while (maxv >= 10) maxv /= 10, cnt ++ ;

    return cnt;
}
void radixsort()
{
    int t = maxbit();
    int radix = 1;

    for (int i = 1; i <= t; i ++ )
    {
        for (int j = 0; j < 10; j ++ ) count[j] = 0;
        for (int j = 0; j < n; j ++ )
        {
            int k = (a[j] / radix) % 10;
            count[k] ++ ;
        }
        for (int j = 1; j < 10; j ++ ) count[j] += count[j-1];
        for (int j = n-1; j >= 0; j -- )
        {
            int k = (a[j] / radix) % 10;
            temp[count[k]-1] = a[j];
            count[k] -- ;
        }
        for (int j = 0; j < n; j ++ ) a[j] = temp[j];
        radix *= 10;
    }

}
~~~
### 9计数排序
~~~
void counting_sort()
{
    int sorted[N];
    int maxv = a[0];
    for (int i = 1; i < n; i ++ )
        if (maxv < a[i])
            maxv = a[i];
    int count[maxv+1];
    for (int i = 0; i < n; i ++ ) count[a[i]] ++ ;
    for (int i = 1; i <= maxv; i ++ ) count[i] += count[i-1];
    for (int i = n-1; i >= 0; i -- )
    {
        sorted[count[a[i]]-1] = a[i];
        count[a[i]] -- ;
    }
    for (int i = 0; i < n; i ++ ) a[i] = sorted[i];
}
~~~
### 10桶排序

（基数排序是桶排序的特例，优势是可以处理浮点数和负数，劣势是还要配合别的排序函数）

~~~
vector<int> bucketSort(vector<int>& nums) {
    int n = nums.size();
    int maxv = *max_element(nums.begin(), nums.end());
    int minv = *min_element(nums.begin(), nums.end());
    int bs = 1000;
    int m = (maxv-minv)/bs+1;
    vector<vector<int> > bucket(m);
    for (int i = 0; i < n; ++i) {
        bucket[(nums[i]-minv)/bs].push_back(nums[i]);
    }
    int idx = 0;
    for (int i = 0; i < m; ++i) {
        int sz = bucket[i].size();
        bucket[i] = quickSort(bucket[i]);
        for (int j = 0; j < sz; ++j) {
            nums[idx++] = bucket[i][j];
        }
    }
    return nums;
}
~~~


## 六、其他常用模板

### 链表反转

### 1.迭代

~~~
public ListNode reverseList(ListNode head) {
        ListNode cur = head, pre = null;
        while (cur != null) {
            ListNode nex = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nex;
        }
        return pre;
    }
~~~

### 2.递归

~~~
public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) 					return head;
        //定义返回的新的头节点
        ListNode tail = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return tail;
    }
~~~

### 3.头插法

~~~
public ListNode reverseList(ListNode head) {
        ListNode dummy = new ListNode(-1);
        while (head != null) {
            ListNode node = new ListNode(head.val);
            node.next = dummy.next;
            dummy.next = node;
            head = head.next;
        }
        return dummy.next;
    }
~~~

### 取整问题

​	(n / k) 上取整等于(n + k - 1) / k下取整或者(n - 1) / k + 1

### Flood Fill

~~~
//https://leetcode-cn.com/problems/color-fill-lcci/
class Solution {
    int[][] g;
    int[][] st; //0：未搜过，1：搜过
    int[] dx = new int[]{-1, 0, 1, 0}, dy = new int[]{0, 1, 0, -1};
    int n, m;
    public void dfs(int x, int y) {
        for (int i = 0; i < 4; i ++ ) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && g[a][b] == g[x][y]){
                if (st[a][b] == 0) {
                    st[a][b] = 1;
                    dfs(a, b);
                }
            }
        }
    }
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        n = image.length;
        m = image[0].length;
        g = image;
        st = new int[n][m];
        st[sr][sc] = 1;
        dfs(sr, sc);
        for (int i = 0; i < n; i ++ ) 
            for (int j = 0; j < m; j ++ )
                if (st[i][j] == 1)
                    image[i][j] = newColor;
        return image;
    }
}
~~~

蓄水池抽样算法

### [382. 链表随机节点](https://leetcode-cn.com/problems/linked-list-random-node/)

~~~
class Solution {
    private ListNode h;
    public Solution(ListNode head) {
        h = head;
    }
    
    public int getRandom() {
        int res = -1, n = 0;
        Random random = new Random();
        for (ListNode node = h; node != null; node = node.next) {
            n ++ ;
            if (random.nextInt() % n == 0) res = node.val;
        }
        return res;
    }
}
~~~

### 求最大公约数

~~~
public static int gcd(int a, int b) {
        //递归方式性能差些，但是写法简单
        //return b == 0 ? a : gcd(b, a % b);
        int t;
        while(b != 0) {
            t = a % b;
            a = b;
            b = t;
        }
        return a;
}
~~~

~~~
public static int gcd(int a, int b) {
	return b == 0 ? a : gcd(b, a % b);
}
~~~



### 求最小公倍数

~~~
 public static int lcm(int a, int b) {
        return a * b / gcd(a, b);
}
~~~

### 求a/b上取整

~~~
(a + b  - 1)  /  b
~~~

### 表达式求值

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



### Map排序

~~~
	//通过堆来排序
	Map<Integer, Integer> map = new HashMap<>();
	Set<Map.Entry<Integer, Integer>> entries = map.entrySet();
	// 根据map的value值正序排，相当于一个小顶堆
	PriorityQueue<Map.Entry<Integer, Integer>> queue = new PriorityQueue<>((o1, o2) -> o1.getValue() - o2.getValue());
	for (Map.Entry<Integer, Integer> entry : entries) queue.offer(entry);
~~~

### 保留两位小数

~~~
DecimalFormat df = new DecimalFormat("$%.2f"); 
num = df.format(num);
~~~

### 判断字符串是否是数字

~~~
public static boolean isNum(String str) {
	for (int i = 0; i < str.length(); i++) {
		if (!Character.isDigit(str.charAt(i))) {
			return false;
		}
	}
	return true;
} 
~~~

### 求数组最大值

~~~
Arrays.stream(f).max().getAsInt();
~~~

### java快读快写模板

~~~
static BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));
static StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
static PrintWriter pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
~~~

~~~
while (in.nextToken() != -1) {
	//code
}
~~~

~~~
public InputReader = new InputReader(System.in);
//手写快读
public class InputReader {
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

### 求n的阶乘质因数p的个数

~~~
public int getCur(int n, int p) {
    int res = 0;
    while (n != 0) {
        res += n / p;
        n /= p;
    }
    return res;
}
~~~



### 递推法预处理所有组合数

$C^j_i=C^{j - 1}_{i - 1} + C^j_{i−1}$
~~~
// 预处理所有的组合数，要注意取 mod，否则会溢出
    private static long[][] dp = new long[1001][1001];
    static {
        for (int i = 0; i <= 1000; i++)
            dp[i][0] = 1;
        for (int i = 1; i <= 1000; i++)
            for (int j = 1; j <= i; j++)
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % mod;
    }

~~~



## 由数据范围反推算法复杂度和算法内容

一般ACM或者笔试题的时间限制是1秒或2秒。
在这种情况下，C++代码中的操作次数控制在 $10^7$~$10^8$ 为最佳。

下面给出在不同数据范围下，代码的时间复杂度和算法该如何选择：

![image-20220819233725308](./_images/image-20220819233725308.png)

