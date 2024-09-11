import cv2
import sys

img = cv2.imread('girl_laughing.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

x, y, w, h = cv2.selectROI(img)  # 관심 영역을 선택
print("Selected ROI:", x, y, w, h)

roi = img[y:y + h, x:x + w]  # 선택한 영역만 잘라냄

cv2.imwrite('roi.jpg', roi)  # 잘라낸 영역을 저장

cv2.imshow('ROI', roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
