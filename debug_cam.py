import cv2
print('OpenCV', cv2.__version__)
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    print('idx', i, 'isOpened', ok)
    if ok:
        ret, frame = cap.read()
        print('  ret', ret, 'shape', None if frame is None else frame.shape)
    cap.release()
