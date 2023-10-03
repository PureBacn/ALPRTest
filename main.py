import cv2
import time
import os
from ultralytics import YOLO

projectdir = os.getcwd()
print(projectdir)

src = cv2.imread(projectdir + "/testfiles/test2.jpeg")
src = cv2.resize(src, (640, 640))
model = YOLO(projectdir + "/testfiles/best.pt")
#model.export(format="openvino", half = True, optimize = True, simplify = True)
print("Exported")
ov_model = YOLO(projectdir + "/testfiles/best_openvino_model")
print("Model Loaded")
#model = torch.load('testfiles/best.pt')
print(ov_model)


start_time = time.time()
results = model(src)
stop_time = time.time()

print("Inference Time YOLo: ", stop_time - start_time)

start_time = time.time()
results = ov_model(src)
stop_time = time.time()

print("Inference Time OV: ", stop_time - start_time)


boxes = results[0].boxes.xyxy

for box in boxes:
    print(box)
    x1, y1, x2, y2 = box

    cv2.rectangle(src,(int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

cv2.imshow("Object Detection", src)
cv2.waitKey(0)
cv2.destroyAllWindows()