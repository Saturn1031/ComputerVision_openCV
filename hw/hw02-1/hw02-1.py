import cv2
import sys

cap = cv2.VideoCapture('face2.mp4')
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

specialEffect = 'n'

while True:  # 무한루프로
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득(frame)
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    if specialEffect == 'n':   # original
        effectFrame = frame
        effectFrame = cv2.putText(effectFrame, "original", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    elif specialEffect == 'b':   # bilateral
        effectFrame = cv2.bilateralFilter(frame, -1, 10, 5)
        effectFrame = cv2.putText(effectFrame, "bilateral", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    elif specialEffect == 's':   # stylization
        effectFrame = cv2.stylization(frame, sigma_s=60, sigma_r=0.45)
        effectFrame = cv2.putText(effectFrame, "stylization", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    elif specialEffect == 'g':   # Gray pencilSketch
        graySketch, colorSketch = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.7, shade_factor=0.02)
        effectFrame = cv2.cvtColor(graySketch, cv2.COLOR_GRAY2BGR)
        effectFrame = cv2.putText(effectFrame, "Gray pencilSketch", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    elif specialEffect == 'c':   # Color pencilSketch
        graySketch, effectFrame = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.7, shade_factor=0.02)
        effectFrame = cv2.putText(effectFrame, "Color pencilSketch", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    elif specialEffect == 'o':   # oilPainting
        effectFrame = cv2.xphoto.oilPainting(frame, 7, 1)
        effectFrame = cv2.putText(effectFrame, "oilPainting", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('special effect', effectFrame)

    key = cv2.waitKey(1)
    if key == ord('q'):     # 영상 종료
        break
    elif key == ord('n'):   # original
        specialEffect = 'n'
    elif key == ord('b'):   # bilateral
        specialEffect = 'b'
    elif key == ord('s'):   # stylization
        specialEffect = 's'
    elif key == ord('g'):   # Gray pencilSketch
        specialEffect = 'g'
    elif key == ord('c'):   # Color pencilSketch
        specialEffect = 'c'
    elif key == ord('o'):   # oilPainting
        specialEffect = 'o'

cap.release()
cv2.destroyAllWindows()