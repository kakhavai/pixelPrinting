import cv2
import numpy as np

# irradianceGrid = cv2.imread('testProjection.png')
# img_debug = irradianceGrid.copy()


# cut = 110
# print(cut)

# img_gray = cv2.cvtColor(img_debug, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_gray,(1,1),0)
# img_thresh = cv2.threshold(img_blur, cut, 255, cv2.THRESH_BINARY)[1]
# contours = cv2.findContours(img_thresh.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# count = 0
# for contour in contours[1]:

# 	x,y,w,h = cv2.boundingRect(contour)
# 	cv2.circle(img_debug,(x,y),1,(0,255,0))
# 	#cv2.rectangle(img_debug,(x,y),(x+w,y+h),(0,255,0),2)


# #look down and right to find remaining

# print(len(contours[1]))

# cv2.imwrite('weird.png', img_debug)
# cv2.waitKey(0)
# cv2.imwrite('threshold.png',img_thresh)
# cv2.waitKey(0)
# cv2.imwrite('blur.png', img_blur)

def thresholdSweep(originalImage):

	#Iterate over quads of the image copy and apply threshold on each quad based on the median color of the quad.
	#After applying the individual threshold to each quad of the image copy, step over each pixel of the image: apply this filter to the pixels of the original image

	modifiedImage = originalImage.copy()

	for y in range(0, 1200, 60):
		for x in range(0, 1600, 50):

			#Assign quad white values to determine if its working correctly
			cut = originalImage[y:y+60, x:x + 60].mean() * 1.2
			modifiedImage[y:y+60, x:x + 60] = cv2.threshold(originalImage[y:y+60, x:x + 60], cut, 1, cv2.THRESH_BINARY)[1]



	cv2.imwrite('originalImage.png', originalImage)
	return cv2.adaptiveThreshold(originalImage * modifiedImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)


# #cv2.imwrite('test1.png',np.reshape(np.array([255] * (3600)), (60,60)))
# print(thresholdSweep(cv2.imread('blur.png', 0))[0][0])
# cv2.imwrite('test.png', thresholdSweep(cv2.imread('blur.png', 0)))
