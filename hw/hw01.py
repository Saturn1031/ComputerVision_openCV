import cv2
import numpy as np

img = np.ones((600, 900, 3), np.uint8) * 255 	# 600*900*3*8bits 행렬, 흰색으로 초기화

BrushSiz = 5  # 붓의 크기
LColor, RColor = (255, 0, 0), (0, 0, 255)  # 파란색과 빨간색
SLColor, SRColor = (0, 255, 0), (0, 255, 255)   # 초록색과 노란색

def draw(event, x, y, flags, param):  # 마우스 콜백 함수
    global ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        if (flags & cv2.EVENT_FLAG_ALTKEY) or (flags & cv2.EVENT_FLAG_CTRLKEY):
            ix, iy = x, y
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(img, (x, y), BrushSiz, SLColor, -1)  # 쉬프트 키 + 마우스 왼쪽 버튼 클릭하면 초록색
        else:
            cv2.circle(img, (x, y), BrushSiz, LColor, -1)  # 마우스 왼쪽 버튼 클릭하면 파란색
    elif event == cv2.EVENT_LBUTTONUP:
        if (flags & cv2.EVENT_FLAG_ALTKEY) and not (flags & cv2.EVENT_FLAG_CTRLKEY):   # 마우스 왼쪽 버튼 & alt 클릭했을 때 빈 직사각형 그리기
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 255), 2)
        elif (flags & cv2.EVENT_FLAG_CTRLKEY) and not (flags & cv2.EVENT_FLAG_ALTKEY):    # 마우스 왼쪽 버튼 & ctrl 클릭했을 때 빈 원 그리기
            radius = int(np.sqrt((ix - x) ** 2 + (iy - y) ** 2))
            cv2.circle(img, (ix, iy), radius, (255, 255, 0), 2)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if (flags & cv2.EVENT_FLAG_ALTKEY) or (flags & cv2.EVENT_FLAG_CTRLKEY):
            ix, iy = x, y
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(img, (x, y), BrushSiz, SRColor, -1)  # 쉬프트 키 + 마우스 오른쪽 버튼 클릭하면 노란색
        else:
            cv2.circle(img, (x, y), BrushSiz, RColor, -1)  # 마우스 오른쪽 버튼 클릭하면 빨간색
    elif event == cv2.EVENT_RBUTTONUP:
        if (flags & cv2.EVENT_FLAG_ALTKEY) and not (flags & cv2.EVENT_FLAG_CTRLKEY):   # 마우스 오른쪽 버튼 & alt 클릭했을 때 채워진 직사각형 그리기
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 255), -1)
        elif (flags & cv2.EVENT_FLAG_CTRLKEY) and not (flags & cv2.EVENT_FLAG_ALTKEY):    # 마우스 오른쪽 버튼 & ctrl 클릭했을 때 채워진 원 그리기
            radius = int(np.sqrt((ix - x) ** 2 + (iy - y) ** 2))
            cv2.circle(img, (ix, iy), radius, (255, 255, 0), -1)

    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(img, (x, y), BrushSiz, SLColor, -1)  # 쉬프트 키 + 왼쪽 버튼 클릭하고 이동하면 초록색
        elif not (flags & cv2.EVENT_FLAG_CTRLKEY) and not (flags & cv2.EVENT_FLAG_ALTKEY):
            cv2.circle(img, (x, y), BrushSiz, LColor, -1)  # 왼쪽 버튼 클릭하고 이동하면 파란색
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(img, (x, y), BrushSiz, SRColor, -1)  # 쉬프트 키 + 오른쪽 버튼 클릭하고 이동하면 노란색
        elif not (flags & cv2.EVENT_FLAG_CTRLKEY) and not (flags & cv2.EVENT_FLAG_ALTKEY):
            cv2.circle(img, (x, y), BrushSiz, RColor, -1)  # 오른쪽 버튼 클릭하고 이동하면 빨간색

    cv2.imshow('hw01', img)

cv2.namedWindow('hw01')
cv2.imshow('hw01', img)

cv2.setMouseCallback('hw01', draw)

while (True):
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) == ord('s'):
        cv2.imwrite('hw01.png', img)  # 이미지를 저장