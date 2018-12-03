import numpy as np
import cv2
cameraResolution = (900,1600)
projectionResolution = (1080, 1920)
pixelGrid = np.zeros(projectionResolution, dtype=int)
pixelSize = .065 #.065 mm == 65 microns
cameraWidthMM = 14.5 #mm
cameraWidthPp = cameraWidthMM/ pixelSize
cameraHeightPp = cameraWidthPp * cameraResolution[0]/cameraResolution[1]
widthDivisor = 192 # 1920/10
heightDivisor = 108 # 1080/10
print('max flash area is ' + str(cameraWidthPp) + ' * ' + str(cameraHeightPp))

for offsetCountY in range(0,10):
	for offsetCountX in range(0,10):
		for y in range (0, heightDivisor):
			for x in range (0, widthDivisor):
				pixelGrid[y + heightDivisor * offsetCountY][x + widthDivisor * offsetCountX] = 255

		cv2.imwrite('irradianceGrid.' + str(offsetCountX) + '.' +  str(offsetCountY) +'.png', pixelGrid)
		print(offsetCountY)
		print(offsetCountX)
		pixelGrid = np.zeros(projectionResolution, dtype=int)
