# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-11-25 下午4:28
# @Email : wwymsn@163.com
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt

dx = 0.05  # step
dy = 0.05

y, x = np.mgrid[slice(1, 5+dy, dy), slice(1, 5+dx, dx)]  # 横向和纵向扩展x, y
# print(x.shape)  # 81*81

z = np.sin(x)**10 + np.cos(10+y*x)*np.cos(x)
# print(z.shape)  # 81*81

z = z[:-1, :-1]  # 去掉边界
# print(z.shape)  # 80*80

plt.pcolormesh(x, y, z, cmap=plt.get_cmap('rainbow'))  # camp为热色参数, x横坐标, y纵坐标, z结果
plt.colorbar()
plt.show()