import numpy as np
import cv2
import time
import sqlite3
import pytesseract
from tensorflow.lite.python.interpreter import Interpreter

start = time.time()
print("Loading Interpreter...")
interpreter = Interpreter("licensemodel/detect.tflite") # load interpreter
interpreter.allocate_tensors() # allocate memory for the tensors
input_details = interpreter.get_input_details() # get details of input for model
output_details = interpreter.get_output_details() # get details of output of model
shape = (input_details[0]["shape"][2], input_details[0]["shape"][1]) # get the input shape
print("Finished Loading Interpreter ", round((time.time() - start) * 100000) / 100, " ms")

start = time.time()
print("Connecting To Database...")
conn = sqlite3.connect("database")  # connect to local database
print("Connected ", round((time.time() - start) * 100000) / 100, " ms")

cur = conn.cursor()  # get database cursor
cur.execute(
    "create table if not exists log (plate text, timestamp int, timetype int)"
)  # create table for the database

input_mean = 127.5 # input mean
input_std = 127.5 # input standard deviation
min_conf = 0.5 # minimum confidence
lastlogged = ""  # last logged license plate

input = cv2.VideoCapture("demo.mp4") # set camera input

def logdata(plate, timestamp):  # new function for logging data
    cur.execute(
        "insert into log (plate, timestamp, timetype) values (?, ?, ifnull(1 - (select timetype from log where plate=? order by timestamp desc limit 1), 0))",
        [plate, timestamp, plate],
    )  # log data
    cur.execute("select * from log")  # get all the data within the table
    for val in cur.fetchall():  # loop through stuff
        print(val)  # print it

print("Starting Program")

while(True):
    ret, frame = input.read()
    #frame = cv2.imread("test.jpg")

    src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # format image
    imH, imW, _ = frame.shape # get image bounds
    src = cv2.resize(src, shape) # resize image to input shape
    src = np.expand_dims(src, 0) # add batch axis
    src = (np.float32(src) - input_mean) / input_std # normalize pixels
        
    interpreter.set_tensor(input_details[0]["index"], src) # set input tensor
    interpreter.invoke() # run detection
    
    boxes = interpreter.get_tensor(output_details[1]["index"])[0] # Bounding box coordinates of detected objects
    scores = interpreter.get_tensor(output_details[0]["index"])[0] # Confidence of detected objects
    
    highest = 0 # set base high score
    box = [] # set base box

    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)) and scores[i] > highest: # check if score is above confidence threshold and below 1, as well as if it is the highest score from detected
            highest = scores[i] # set highest score

            # define box bounds
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            box = [ymin, xmin, ymax, xmax] # set box to current box
            #frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2) # draw box around detected

    if highest > 0: # check if any license plate has been found

        frame = frame[box[0] : box[2], box[1] : box[3]] # crop
        frame = cv2.resize(frame, fx=2, fy=2) # scale image by a factor of 2

        text = pytesseract.image_to_string(frame)  # extract text
        text = text.upper()  # convert license plate to caps
        result = ""  # result variable
        for char in text:  # for character within the result text
            if char.isalnum():  # if character isnt special
                result = result + char  # add it to the result text

        if result != "":
            if lastlogged != result:  # if plate is fresh
                logdata(result, int(time.time()))  # log the data
            lastlogged = result  # set the last logged plate
            frame = cv2.putText(
                frame,
                result,
                (5, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )  # display text on the image
        else:
            lastlogged = ""  # reset last logged plate
    else:
        lastlogged = ""  # reset last logged plate

    cv2.imshow("Camera", frame) # show image
    if cv2.waitKey(1) == ord("q"):  # if q key pressed
        break  # break from main loop

conn.commit()  # commit database changes
input.release()  # release camera
cv2.destroyAllWindows()  # kill all open image windows
    