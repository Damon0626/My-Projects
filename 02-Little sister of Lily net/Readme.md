### 前言
快到七夕节了，找了点百合网征婚小姐姐的信息，尝试从中发现妹子眼中期望的的男生是什么样子，当然重要的还是两情相悦。希望广大的朋友早日找到属于自己的另一半。加油吧！
### 数据来源
[百合网](http://search.baihe.com/),根据首页中的信息，爬取了近万名小姐姐的征婚信息，包括地区、年龄、收入、爱情宣言等等。目前信息只有北京范围的，感兴趣的可以尝试全国范围的信息。
### 网页
可以发现在首页中，返回的信息中包含了很多的json格式信息，利用json库很方便的进行处理
![image](https://github.com/Damon0626/My-Projects/blob/master/02-Little%20sister%20of%20Lily%20net/Lily%20net.jpg)
### 数据处理
#### 地图密度分布
根据地图密度分布，发现征婚的小姐姐们大致集中分布在东城区、西城区、朝阳区、海淀区、丰台区、通州区、昌平区\
\
![image](https://github.com/Damon0626/My-Projects/blob/master/02-Little%20sister%20of%20Lily%20net/map-dist.png)
#### 年龄分布
目前征婚的女性主力军集中在27、28、29、30岁之间，俗话说“女大三,抱金砖;女小三,男当官”，24-33岁之间的男同胞们，机会大大的，加加油啦\
\
![image](https://github.com/Damon0626/My-Projects/blob/master/02-Little%20sister%20of%20Lily%20net/age-distribution.png)
#### 城区分布
朝阳区可是独领风骚，人数远远超过平均水平2307人，海淀区紧随其后，人数也达到了7921人，超过平均水平的3倍\
![image](https://github.com/Damon0626/My-Projects/blob/master/02-Little%20sister%20of%20Lily%20net/loc-distribution.png)
#### 收入与城区关系
收入中，5k-10k, 1w-2w是绝大多数小姐姐的收入水平。朝阳区小姐姐普遍收入5k-2w,也代表了整体数据中的收入水平。密云区发现部分高收入小姐姐，薪资在2w-5w。崇文区的小姐姐工资好像不是很乐观，也说不定瞒报，考验我们真心程度呢。
![image](https://github.com/Damon0626/My-Projects/blob/master/02-Little%20sister%20of%20Lily%20net/income%26loc-distribution.png)
#### 年龄与学习分布
征婚的博士小姐姐好像都分布在22-25岁之间，想想自己这个年纪还在大学里肆意地挥霍，想想好气啊，同样是九年义务教育，为什么她们这么优秀。不得不感慨下啊。各个年龄，专本硕博比例相对均衡，也看出目前我们整体的学历水平还是杠杠的，哈哈，怎么着也是个大学生。
![image](https://github.com/Damon0626/My-Projects/blob/master/02-Little%20sister%20of%20Lily%20net/age%26edu-dist.png)
#### 爱情宣言词云
小姐姐理想中的另一半要求积极、阳光、有责任心、善良、旅游、相扶相持、自信等等；\
小姐姐也表明了自己的渴望：盼望找到爱的人，一直努力寻找，快快出现，期待你的出现；\
![image](https://github.com/Damon0626/My-Projects/blob/master/02-Little%20sister%20of%20Lily%20net/wc.jpg
### 最后
爱情还是需要两情相悦，看对了眼，一切真的好说，不喜欢还真是得花心思，得之则幸，不得则命。希望广大的男同胞成长为积极、阳光、自信、有责任心的男人，不管小姐姐还是小哥哥都早日找到属于自己的那个Ta。
