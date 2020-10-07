import cv2
import numpy as np


num_string = '1270.jpg'
pathA = './validation_model_0814ssim/'###SSIM
#validation_model_0814ssim
#validation_model_0730ssim
pathB = './validation_model_0802L1/'###L1
pathC = './validation_model_0802L2/'###L2
holeA = cv2.imread(pathA + num_string,cv2.IMREAD_GRAYSCALE)
holeB = cv2.imread(pathB + num_string,cv2.IMREAD_GRAYSCALE)
holeC = cv2.imread(pathC + num_string,cv2.IMREAD_GRAYSCALE)
###SSIM
m = holeA[:,0:256]
n = holeA[:,256:513]
cv2.imshow('A',cv2.subtract(m,n))
#L1
i = holeB[:,0:256]
j = holeB[:,256:513]

cv2.imshow('B',cv2.subtract(i,j))
#L2
p = holeC[:,0:256]
q = holeC[:,256:513]

cv2.imshow('C',cv2.subtract(p,q))


null = np.ones((256,3),dtype=None)
hmergeAB = np.hstack((cv2.subtract(m,n), cv2.subtract(i,j)))

hmergeABC = np.hstack((hmergeAB, cv2.subtract(p,q)))
cv2.imshow('hmergeABC',hmergeABC)
#cv2.imwrite('./validation_model_0730ssim/' + str(i)+str(j) + '.jpg', hmerge)

cv2.waitKey(0)

