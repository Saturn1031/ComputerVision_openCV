import cv2
import sys
import numpy as np

img = cv2.imread('coins.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray',gray)

median = cv2.medianBlur(gray, 3)
_, gray_bin = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('Binary', gray_bin)

# cnt, labels = cv2.connectedComponents(gray_bin)
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_bin)
print(cnt)
# print(labels)
# print(stats)
# print(centroids)    # 요소의 중심 값

cv2.imshow('labelling', (labels*50).astype(np.uint8))   # 객체 라벨링 1, 2, 3, 4 밝기로 구분 (배경은 0)

img[labels == 0] = [127,127,127]
img[labels == 1] = [127,0,0]    # 라벨이 1이면 파란색
img[labels == 2] = [0,127,0]    # 라벨이 2이면 초록색
img[labels == 3] = [0,0,127]    # 라벨이 3이면 빨간색
img[labels == 4] = [0,127,127]  # 라벨이 4이면 노란색

for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
    (x, y, w, h, area) = stats[i]   # 요소 를 감싼 직사각형의 좌측 상단 좌표, width와 height, 요소 픽셀 개수

    # 노이즈 제거
    if area < 20:
        continue

    cv2.rectangle(img, (x, y, w, h), (255, 0, 255), 2)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)  # 위와 같은 직사각형 그림 (표현법만 다름)

cv2.imshow('Connected components', img)

cv2.waitKey()
cv2.destroyAllWindows()