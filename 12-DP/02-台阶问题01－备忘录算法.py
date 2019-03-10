# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-3-10 下午4:14
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

	# 备忘录，从下往上解决问题
	tips = {}
	for i in range(3, n+1):
		if i in tips.keys():
			return tips[i]
		else:
			h = step(n-1) + step(n-2)
			tips[i] = h
	return h


print(step(10))
