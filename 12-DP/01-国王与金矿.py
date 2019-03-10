# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-3-10 下午3:57
# @Email : wwymsn@163.com
# @Software: PyCharm

g = [400, 500, 200, 300, 350]
p = [5, 5, 3, 4, 3]
w = 10

preResults = [0]*11
result = [0]*11

for i in range(w+1):
	if i < p[0]:
		preResults[i] = 0
	else:
		preResults[i] = g[0]

for i in range(1, 5):
	for j in range(w+1):
		if j < p[i]:
			result[j] = preResults[j]
		else:
			result[j] = max(preResults[j], preResults[j-p[i]]+g[i])
	preResults = result.copy()
	print(result)
