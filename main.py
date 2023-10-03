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
from ultralytics import YOLO

"""core = ov.Core()
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)



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

input_image = cv2.imread(projectdir + "/testfiles/test.jpg")
detections = detect(input_image, compiled_model)[0]
"""

import torch
import tensorflow as tf
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


projectdir = os.getcwd()
print(projectdir)



# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()
model.load_state_dict(torch.load(projectdir + "/testfiles/best.pt"))


model_int8 = torch.ao.quantization.quantize_dynamic(
    model,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)

torch.save(model_int8, projectdir + "/testfiles/bestquantized.pt")

"""
# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

device = torch.device("cuda")

src = torch.from_numpy(src)
src = torch.unsqueeze(src, 0)
model.eval()
prediction = model(src) #F.softmax(model(src), dim = 1)

print(prediction)

print(prediction.argmax())

cv2.imshow("test", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""