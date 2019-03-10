# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-3-10 下午4:18
# @Email : wwymsn@163.com
# @Software: PyCharm


def step(n):
	# 找到边界
	if n == 0:
		return 0
	if n == 1:
		return 1
	if n == 2:
		return 1

	# 状态转移方程
	x, y = 1, 1
	for i in range(3, n+1):
		x, y = y, x+y

	return y


print(step(10))