import os

import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Face 분류기 로드

images = []
path = '.\\faces\\'
for root, directories, files in os.walk(path):
    for file in files:
        if ('.png' in file) or ('.jpg' in file):
            src = cv2.imread(os.path.join(root, file))
            images.append(src)

face_mask = cv2.imread('mask_fire.png')
h_mask, w_mask = face_mask.shape[:2]

stickerImgs = []

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 이미지로 변환
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 얼굴 검출
    # face_cascade.detectMultiScale(이미지 변수, 스케일 요소, 얼굴 신뢰도)
    # 신뢰도 : 얼굴에 최소한 5개의 후보 경계 박스가 있어야 해당 얼굴을 검출

    ksize = 15  # 모자익 3-19 참조
    for (x, y, w, h) in faces:  # 검출된 모든 얼굴에 대해
        if h > 0 and w > 0:
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)    # 사각형으로 표시

            roi = img[y:y + h, x:x + w]  # 검출된 얼굴에 대한 사각형 영역을 관심 영역(ROI)로 설정

            face_mask_small = cv2.resize(face_mask, (w, h),
                                         interpolation=cv2.INTER_AREA)  # 가면 이미지의 크기를 검출된 얼굴의 크기와 같도록 resize

            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)  # ① 가면 마스크에서 검지 않은 부분만 통과(검은색 부분은 투명)
            ret, mask = cv2.threshold(gray_mask, 220, 255, cv2.THRESH_BINARY_INV)
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

            mask_inv = cv2.bitwise_not(mask)  # ② 검출된 얼굴에서 가면 마스크의 검은 부분만 통과(검지 않은 부분은 투명)
            masked_img = cv2.bitwise_and(roi, roi, mask=mask_inv)

            img[y:y + h, x:x + w] = cv2.add(masked_face, masked_img)  # 흰색 마스크와 마스크된 얼굴 합성 (① + ②)

    stickerImgs.append(img)

for i in range(len(stickerImgs)):
    cv2.imshow('Face detection with haarcascade' + str(i), stickerImgs[i])
    cv2.imwrite('./capture/haarcascade_img_' + str(i + 1) + '.png', stickerImgs[i])  # 이미지 캡처 및 저장

cv2.waitKey()
cv2.destroyAllWindows()
