"""from ultralytics import YOLO

#-m venv openvino_env

model = YOLO("best.pt")
model.export(format="openvino")"""

import cv2
import openvino as ov
import ipywidgets as widgets
import numpy as np
import time
import os

core = ov.Core()
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

projectdir = os.getcwd()
print(projectdir)

core.set_property({'CACHE_DIR': '../cache'})
model = core.read_model(projectdir + "/model/best.xml")
compiled_model = core.compile_model(model=model, device_name=device.value)

print("Model Compiled")
print(compiled_model)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

src = cv2.imread(projectdir + "/testfiles/test.jpg")
test = cv2.resize(src, (640, 640))

input_image = np.expand_dims(np.transpose(test, (2, 0, 1)), 0)


start_time = time.time()
 # Get the results.
results = compiled_model([input_image])[output_key]
stop_time = time.time()

print("Inference Time: ", stop_time - start_time)

cv2.imshow("test", src)
cv2.waitKey(0)
cv2.destroyAllWindows()