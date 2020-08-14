
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_PLAIN
line = cv2.LINE_AA
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(2)
_, testFrame = capL.read()
height, width, _ = testFrame.shape
print(testFrame.shape)
# width = 1280
# height = 720
# width = 640
# height = 480
ft = 1.5

if not capL.isOpened():
    print("Unable to read LEFT camera")
    capL.release()

if not capR.isOpened():
    print("Unable to read RIGHT camera")
    capR.release()

while (True):
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    ret = retL and retR
    if ret:
        prev_Lframe = frameL.copy()
        shown_text = 'LEFT'
        cv2.putText(prev_Lframe, shown_text, (50, 50), font, 2.0, (0, 0, 255), 4, line)
        prev_Rframe = frameR.copy()
        shown_text = 'RIGHT'
        cv2.putText(prev_Rframe, shown_text, (50, 50), font, 2.0, (0, 0, 255), 4, line)
        fullFrame = np.concatenate((prev_Lframe, prev_Rframe), axis=1)
        cv2.imshow('stereo view', cv2.resize(fullFrame, (0, 0), fx=ft, fy=ft))
    key = cv2.waitKey(1)
    if key == 27:
        capL.release()
        capR.release()
        cv2.destroyAllWindows()
        break
    # if key == ord('s'):
    #     fullFrame = np.concatenate((frameL, frameR), axis=1)
    #     cv2.imwrite('./scenes/scene.png', fullFrame)
