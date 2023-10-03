import cv2
import time
import os
from ultralytics import YOLO

projectdir = os.getcwd()
print(projectdir)

model = YOLO(projectdir + "/testfiles/best.pt")
#model.export(format="openvino", half = True)
print("Yolo Loaded")
ov_model = YOLO(projectdir + "/testfiles/best_openvino_model")
print("OV Loaded")

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.merge((frame, frame, frame))

    results = ov_model(frame)

    boxes = results[0].boxes.xyxy

    for box in boxes:
        x1, y1, x2, y2 = box

        cv2.rectangle(frame,(int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()