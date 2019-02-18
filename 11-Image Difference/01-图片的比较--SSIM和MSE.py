# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-2-17 下午2:48
# @Email : wwymsn@163.com
# @Software: PyCharm


import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse_ski
import matplotlib.pyplot as plt


def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
	err /= float(imageA.shape[0] * imageB.shape[1])
	return err


def compareImages(imageA, imageB, title):
	m1 = mse(imageA, imageB)
	m2 = mse_ski(imageA, imageB)
	s = ssim(imageA, imageB)

	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, MSE With Ski: %.2f, SSIM: %.2f"%(m1, m2, s))

	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap=plt.cm.gray)
	plt.axis("off")

	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap=plt.cm.gray)
	plt.axis("off")

	plt.show()


if __name__ == "__main__":
	origine = cv2.imread('image02-0.png')
	changed = cv2.imread('image02-1.png')
	# print(origine.shape, changed.shape)
	orin_gray = cv2.cvtColor(origine, cv2.COLOR_BGR2GRAY)
	changed_gray = cv2.cvtColor(changed, cv2.COLOR_BGR2GRAY)

	compareImages(orin_gray, orin_gray, "Original vs. Original")
	compareImages(orin_gray, changed_gray, "Orinal vs. Changed")
	# image = cv2.imread('test.jpg')
	# cv2.imwrite('image01-0.jpg', image[:, :400])
	# cv2.imwrite('image01-1.jpg', image[:, 400::])