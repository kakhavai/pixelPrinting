import cv2
from thresholdSweep import thresholdSweep

irradianceGrid = cv2.imread('blur.png')
img_debug = cv2.cvtColor(irradianceGrid.copy(), cv2.COLOR_BGR2GRAY)


cut = 110
print(cut)

# img_gray = cv2.cvtColor(img_debug, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_gray,(1,1),0)
# img_thresh = cv2.threshold(img_blur, cut, 255, cv2.THRESH_BINARY)[1]



contours = cv2.findContours(thresholdSweep(img_debug),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
count = 0
for contour in contours[1]:

	x,y,w,h = cv2.boundingRect(contour)
	if w > 3 and h > 3:
		cv2.circle(irradianceGrid,(x,y),1,(0,255,0))
	#cv2.rectangle(img_debug,(x,y),(x+w,y+h),(0,255,0),2)


#look down and right to find remaining

print(len(contours[1]))

cv2.imwrite('weird.png', irradianceGrid)
cv2.waitKey(0)
# cv2.imwrite('threshold.png',img_thresh)
# cv2.waitKey(0)
# cv2.imwrite('blur.png', img_blur)