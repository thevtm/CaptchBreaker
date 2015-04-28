import sys
import math
import itertools as it
import numpy as np
import cv2
import pyocr
from PIL import Image

def CaptchaBreak(imgInput):
	## Add Alpha and White Border
	imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2BGRA)
	imgInput = cv2.copyMakeBorder(imgInput, 10, 10, 10, 10,cv2.BORDER_CONSTANT,value = [255, 255, 255])

	## Convert to Grayscale
	imgGray = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)

	## Use Threshold to convert to B&W
	BWThreshold = 230
	ret,imgBW = cv2.threshold(imgGray, BWThreshold, 255, cv2.THRESH_BINARY)

	## Detect Noise using filter 3x3 and Threshold
	kernel = np.ones((3, 3), np.float32) / 9
	dst = cv2.filter2D(imgBW, -1, kernel)

	ret, imgNoise = cv2.threshold(dst, 30, 255, cv2.THRESH_BINARY)
	imgNoiseMask = cv2.bitwise_not(imgNoise)

	## Clean image
	imgNoiseRGB = cv2.cvtColor(imgNoise, cv2.COLOR_GRAY2BGRA)
	imgNoNoise = cv2.bitwise_and(imgInput, imgInput, mask = imgNoiseMask)
	imgNoNoise += imgNoiseRGB

	## Find Contourns
	cnts, hierarchy = cv2.findContours(imgNoise.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnts = list(it.compress(cnts, (hierarchy[:, :, 3] == 0)[0])) ## Keep only the ones that have the root as parent
	cnts = filter(lambda x:cv2.contourArea(x) > 100, cnts) ## Remove the small ones
	cnts = sorted(cnts, key = lambda x: min([p[0][0] for p in x])) ## Sort by min(X)

	## Extract contours images
	imgCnts = []
	for i in range(len(cnts)):
	    ## Create contour maks
	    mask = cv2.cvtColor(np.zeros_like(imgInput), cv2.COLOR_BGRA2GRAY)
	    cv2.drawContours(mask, cnts, i, 255, -1)
    
	    ## Apply mask to Input Img
	    out = np.zeros_like(imgInput)
	    out[mask == 255] = imgNoNoise[mask == 255]
    
	    imgCnts.append(out)

	## Find minimal area rectangle arround the letter
	rects = map(cv2.minAreaRect, cnts)
	angles = [a for x, y, a in rects]
	boxs = map(cv2.cv.BoxPoints, rects)
	boxsInt = [np.int0(np.around(box)) for box in boxs]

	## Crops contours using its bounding rect
	bRects = map(cv2.boundingRect, cnts)
	imgContoursCrops = [img[y:y+h, x:x+w] for (x, y, w, h), img in zip(bRects, imgCnts)]

	## Rotate the crops to the Right Angle
	imgCropsNormAngle = []
	for crop, angle in zip(imgContoursCrops, angles):
	    img = cv2.copyMakeBorder(crop, 10, 10, 10, 10,cv2.BORDER_CONSTANT,value = [255, 255, 255, 255])
	    rows, cols, ch = img.shape

	    a = min(90 + angle, abs(angle)) * (1 if abs(angle) > 45 else -1)
    
	    M = cv2.getRotationMatrix2D((cols/2,rows/2), a, 1)
	    dst = cv2.warpAffine(img, M, (cols,rows))
    
	    imgCropsNormAngle.append(dst)

	## Threshold crops to B&W
	imgsOut = []
	for img in imgCropsNormAngle:
	    alpha = img[:, :, 3]

	    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
	    img[alpha < 250] = 255
    
	    ret,imgBW = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    
	    imgsOut.append(imgBW)

	## Apply ocr to each contour
	ocr = pyocr.tesseract
	ocrBuilder = pyocr.tesseract.builders.TextBuilder()
	ocrLang = 'eng'

	ocrBuilder.tesseract_configs[1] = '10'
	ocrBuilder.tesseract_configs.append('letters')

	imgsOutPIL = [Image.fromarray(img) for img in imgsOut]

	lettersOut = [ocr.image_to_string(img, lang = ocrLang, builder = ocrBuilder) for img in imgsOutPIL]
	lettersOut = [('?' if l == '' else l) for l in lettersOut]

	return (''.join(lettersOut)).upper()

if __name__ == "__main__":
    ## Consts/Args
    inFileName = sys.argv[1]

    ## Load img
    imgInput = cv2.imread(inFileName)

    ## Print result
    print CaptchaBreak(imgInput)
