import copy

import cv2
import numpy as np

def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h/ w if h > w else w/ h # 종횡비 계산

    chk1 = 3000 < (h * w) < 12000 # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 6.5 # 번호판 종횡비 조건
    return (chk1 and chk2)

car00 = cv2.imread('cars/00.jpg')
car01 = cv2.imread('cars/01.jpg')
car02 = cv2.imread('cars/02.jpg')
car03 = cv2.imread('cars/03.jpg')
car04 = cv2.imread('cars/04.jpg')
car05 = cv2.imread('cars/05.jpg')
cars = [car00, car01, car02, car03, car04, car05]

# 1. bilateral smoothing
smoothingCars = []
for i in range(len(cars)):
    smoothingCars.append(cv2.bilateralFilter(cars[i], -1, 200, 5))


# 2. 세로 에지 검출
edgeCars = []
for i in range(len(cars)):
    grayImg = cv2.cvtColor(smoothingCars[i], cv2.COLOR_BGR2GRAY)
    gray3channel = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR)
    prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # 수직 필터
    prewitt_grad_x = cv2.filter2D(gray3channel, -1, prewitt_filter_x)  # 수직 에지
    edgeCars.append(cv2.convertScaleAbs(prewitt_grad_x))  # 양수로 변환


# 3. 임계값을 이용한 에지 분리
threshCars = []
for i in range(len(cars)):
    ret, img_binaryB = cv2.threshold(edgeCars[i], 200, 255, cv2.THRESH_BINARY)
    threshCars.append(img_binaryB)

# 4. 가로로 긴 구조 요소를 이용한 여러 번의 닫힘(close)를 통해 흰 숫자 에지를 팽창한 후 원 상태로 침식
morphologyCars = []
for i in range(len(cars)):
    se1 = np.uint8([[0, 0, 0, 0, 0],  # 구조 요소
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0]])
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k = 8  # 반복 횟수
    morphologyCars.append(cv2.erode(cv2.dilate(threshCars[i], se1, iterations=k), se2, iterations=k))   # 닫기

minRectCars = copy.deepcopy(cars)
licensePlateCars = copy.deepcopy(cars)

for i in range(len(cars)):
    canny = cv2.Canny(morphologyCars[i], 100, 255)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for j in range(len(contours)):
        contour = contours[j]
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)  # 4개의 꼭지점 좌표
        box = np.int32(box)  # 정수 좌표
        cv2.drawContours(minRectCars[i], [box], 0, (0, 255, 0), 2) # 초록 사각형

        if verify_aspect_size(rect[1]):
            cv2.drawContours(licensePlateCars[i], [box], 0, (0, 255, 0), 2)  # 초록 사각형


# 이미지 출력
for i in range(len(cars)):
    row1 = np.hstack((cars[i], morphologyCars[i]))  # 첫 번째 줄 (car00, car01)
    row2 = np.hstack((minRectCars[i], licensePlateCars[i]))  # 두 번째 줄 (car02, car03)

    carImgs = np.vstack((row1, row2))
    cv2.imshow('car0' + str(i), carImgs)

cv2.waitKey()
cv2.destroyAllWindows()