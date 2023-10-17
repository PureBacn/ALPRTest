import cv2
import pytesseract
import os
from ultralytics import YOLO

projectdir = os.getcwd()
print(projectdir)

ov_model = YOLO(projectdir + "/testfiles/best_openvino_model", task = "detect")
print("OV Loaded")

video = cv2.VideoCapture(-1)

while True:
    ret, src = video.read()
    if not ret:
       break
    
    x, y, _ = src.shape
    ratio = x/y
    frame = cv2.resize(src, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.merge((frame, frame, frame))

    results = ov_model(frame)

    boxes = results[0].boxes.xyxy

    for box in boxes:
        x1, y1, x2, y2 = box

        cv2.rectangle(frame,(int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        if len(boxes) == 1:
            src = cv2.resize(src, (640, 640))
            frame = src[int(y1):int(y2), int(x1):int(x2)]
            frame = cv2.resize(frame, (int(100/ratio),int(100*ratio)))

    text = pytesseract.image_to_string(frame)
    text = text.upper()
    result = ""
    for char in text:
        if char.isalnum():
           result = result + char
    frame = cv2.putText(frame, result, (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imshow("Object Detection", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()