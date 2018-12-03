import numpy as np
import cv2

projectionResolution = [1080, 1920]
pixelGrid = np.zeros(projectionResolution, dtype=int)



cp = [int(x / 2) for x in projectionResolution]

for y in range(cp[0] - 75, cp[0] + 75):
	for x in range(cp[1] - 75, cp[1] + 75):
		pixelGrid[y][x] = 255


cv2.imwrite('bigWhiteBox.png', pixelGrid)	