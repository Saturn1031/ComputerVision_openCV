import cv2
import sys

cap = cv2.VideoCapture('hw03-1_original.mp4')
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   # 비디오 크기 지정
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')        # 비디오 저장 방식 지정
outV = cv2.VideoWriter('./hw03-1.mp4', fourcc, 20.0, frame_size) # 비디오 저장 객체 생성

while True:  # 무한루프로
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득(frame)
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    finalFrame = frame

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 피부색 마스크
    skin_mask = cv2.inRange(hsv_img, (0, 80, 80), (255, 255, 255))  # skin color 범위

    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k = 5  # 반복 횟수
    skin_mask_opened = cv2.dilate(cv2.erode(skin_mask, se1, iterations=k), se1, iterations=k)  # 열림
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k = 10  # 반복 횟수
    skin_mask_closed = cv2.erode(cv2.dilate(skin_mask_opened, se2, iterations=k), se2, iterations=k)  # 닫힙

    canny = cv2.Canny(skin_mask_closed, 100, 255)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:      # 손이 있으면
        max = 0
        for i in range(len(contours)):
            if contours[i].shape[0] > contours[max].shape[0]:
                max = i

        contour = contours[max]     # 가장 긴 경계선

        hull = cv2.convexHull(contour, returnPoints=False)
        monoHull = cv2.sort(hull, cv2.SORT_EVERY_COLUMN)
        defects = cv2.convexityDefects(contour, monoHull)

        defectsCount = 0    # 깊이가 55000을 넘는 결함의 개수
        for j in range(defects.shape[0]):
            s, e, f, d = defects[j, 0]
            if d > 55000:
                defectsCount += 1

        cv2.drawContours(frame, contour, -1, (255, 0, 255), 2)

        if defectsCount == 0:
            finalFrame = cv2.putText(frame, "Rock", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        elif defectsCount == 2:
            finalFrame = cv2.putText(frame, "Scissors", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        elif defectsCount >= 4:
            finalFrame = cv2.putText(frame, "Paper", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    outV.write(finalFrame)  # 비디오로 프레임 저장
    cv2.imshow('hw03-1', finalFrame)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()