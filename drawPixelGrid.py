import numpy as np
import cv2
cameraResolution = (900,1600)
projectionResolution = (1080, 1920)
pixelGrid = np.zeros(projectionResolution, dtype=int)
pixelGrid1 = np.zeros(projectionResolution, dtype=int)
pixelGrid2 = np.zeros(projectionResolution, dtype=int)
pixelGrid3 = np.zeros(projectionResolution, dtype=int)
fullWhite = np.zeros(projectionResolution, dtype=int)
pixelSize = .065 #.065 mm == 65 microns
cameraWidthMM = 14.5 #mm
cameraWidthPp = cameraWidthMM/ pixelSize
cameraHeightPp = cameraWidthPp * cameraResolution[0]/cameraResolution[1]
widthDivisor = 1920 # 1920/10
heightDivisor = 1080 # 1080/10
startCountRange = 200
xSpacing = 10
pixelGrids = []
for x in range (0, xSpacing):
	pixelGrids.append(np.zeros(projectionResolution, dtype=int))


for photoCount in range(0, 10):
	for y in range (0, 1080):
		if (y + photoCount) % 10 == 0:
			for x in range (0, 1920):
				for space in range(0, xSpacing):
					if x % xSpacing == space:
						pixelGrids[space][y][x] = 255
						
						# make sure im never hitting the same pixel
						if fullWhite[y][x] == 255:
							print('error')
						else:
							fullWhite[y][x] = 255


	for x in range(0, xSpacing):	
		cv2.imwrite('./images/pixelGrid.' + str(photoCount) + '.' + str(x) + '.png', pixelGrids[x])

	pixelGrids = []
	for x in range (0, xSpacing):
		pixelGrids.append(np.zeros(projectionResolution, dtype=int))

cv2.imwrite('white.png', fullWhite)