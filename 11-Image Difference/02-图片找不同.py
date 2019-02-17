# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-2-17 下午7:35
# @Email : wwymsn@163.com
# @Software: PyCharm

from skimage.measure import compare_ssim as ssim
import cv2
import numpy as np

imageA = cv2.imread('image02-0.png')
imageB = cv2.imread('image02-1.png')
grayA = cv2.cvtColor(imageA[:445], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
print(grayA.shape, grayB.shape)
score, diff = ssim(grayA, grayB, full=True)

# 如果full为True, 返回两幅图的实际图像差异,值在[-1, 1]，维度同原图像．
# print(score,'\n', diff.shape)  # 0.87, (600, 400)
diff = (diff*255).astype('uint8')
print("SSIM:{}".format(score))

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # ret, thresh = ...


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

'''
原作者代码考虑了opencv不同的版本,所以使用了cnts = imutils.grab_contours(cnts)

	# if the length the contours tuple returned by cv2.findContours
	# is '2' then we are using either OpenCV v2.4, v4-beta, or
	# v4-official
	if len(cnts) == 2:
		cnts = cnts[0]

	# if the length of the contours tuple is '3' then we are using
	# either OpenCV v3, v4-pre, or v4-alpha
	elif len(cnts) == 3:
		cnts = cnts[1]

	# otherwise OpenCV has changed their cv2.findContours return
	# signature yet again and I have no idea WTH is going on
	else:
		raise Exception(("Contours tuple must have length 2 or 3, "
			"otherwise OpenCV changed their cv2.findContours return "
			"signature yet again. Refer to OpenCV's documentation "
			"in that case"))

	# return the actual contours array
	return cnts
'''

# 找出面积最大的10个轮廓
area_index = np.argsort([cv2.contourArea(c) for c in cnts])
cnts = np.array(cnts)[area_index[-10::]]
# print(cnts)

for c in cnts:
	x, y, w, h = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x+w, y+h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('diff', diff)
cv2.imshow('thresh', thresh)
cv2.imshow('img1', imageA)
cv2.imshow('img2', imageB)
cv2.waitKey(0)
