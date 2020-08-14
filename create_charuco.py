import cv2

squaresX = 8
squaresY = 5
squareLength = 32
markerLength = 23
DICT = 14

'''
DATA
  DICT_4X4_100 = 1
  DICT_4X4_1000 = 3
  DICT_4X4_250 = 2
  DICT_4X4_50 = 0
  DICT_5X5_100 = 5
  DICT_5X5_1000 = 7
  DICT_5X5_250 = 6
  DICT_5X5_50 = 4
  DICT_6X6_100 = 9
  DICT_6X6_1000 = 11
  DICT_6X6_250 = 10
  DICT_6X6_50 = 8
  DICT_7X7_100 = 13
  DICT_7X7_1000 = 15
  DICT_7X7_250 = 14
  DICT_7X7_50 = 12
  DICT_ARUCO_ORIGINAL = 16
'''

charucoDictionary = cv2.aruco.Dictionary_get(DICT)


'''
retval	=	cv.aruco.CharucoBoard_create(	squaresX, squaresY, squareLength, markerLength, dictionary	)
---------
Parameters:
squaresX	number of chessboard squares in X direction
squaresY	number of chessboard squares in Y direction
squareLength	chessboard square side length (normally in meters)
markerLength	marker side length (same unit than squareLength)
dictionary	dictionary of markers indicating the type of markers. The first markers in the dictionary are used to fill the white chessboard squares.
'''

charucoBoard = cv2.aruco.CharucoBoard_create(squaresX, squaresY,  squareLength, markerLength, charucoDictionary)


'''
img	=	cv.aruco_CharucoBoard.draw(	outSize[, img[, marginSize[, borderBits]]]	)
---------
Parameters:
outSize	        size of the output image in pixels.
img	        output image with the board. The size of this image will be outSize and the board will be on the center, keeping the board proportions.
marginSize	minimum margins (in pixels) of the board in the output image
borderBits	width of the marker borders.
'''

img = charucoBoard.draw((4096,3072), 100, 100)
# cv2.imshow('ChaRuCo', img)
# cv2.waitKey()
cv2.imwrite('charuco.png', img)


detectorParams = cv2.aruco.DetectorParameters_create()
detectorParams.cornerRefinementMaxIterations = 500
detectorParams.cornerRefinementMinAccuracy = 0.001
detectorParams.adaptiveThreshWinSizeMin = 10
detectorParams.adaptiveThreshWinSizeMax = 10









