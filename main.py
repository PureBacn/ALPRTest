import numpy as np
import cv2
import time
import sqlite3
import pytesseract
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect
 
start = time.time()
print("Loading Interpreter...")
interpreter = edgetpu.make_interpreter("licensemodel/detect.tflite") # load interpreter
interpreter.allocate_tensors() # allocate memory for the tensors
shape = common.input_size(interpreter) # get the input shape
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
 
start = time.time()
print("Connecting To Camera...")
input = cv2.VideoCapture("demo.mp4") # set input
print("Connected ", round((time.time() - start) * 100000) / 100, " ms")
 
def logdata(plate, timestamp):  # new function for logging data
    cur.execute(
        "insert into log (plate, timestamp, timetype) values (?, ?, ifnull(1 - (select timetype from log where plate=? order by timestamp desc limit 1), 0))",
        [plate, timestamp, plate],
    )  # log data
    cur.execute("select * from log")  # get all the data within the table
    for val in cur.fetchall():  # loop through stuff
        print(val)  # print it
 
print("Starting Program")
 
while(input.isOpened()):
    #start = time.time()
    ret, frame = input.read() # get input frame
    if not ret:
        break
    #frame = cv2.imread("test.jpg")
    
    frame = cv2.resize(frame, shape) # resize image to input shape
    src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # format image
    imH, imW, _ = frame.shape # get image bounds
    src = np.expand_dims(src, 0) # add batch axis
    src = (np.float32(src) - input_mean) / input_std # normalize pixels
        
    start = time.time()
    print("Starting Inference...")
    common.set_input(interpreter, src) # set input tensor
    interpreter.invoke() # run the inference
    objects = detect.get_objects(interpreter, min_conf) # get objects from inference
    print("Inference Took ", round((time.time() - start) * 100000) / 100, " ms")
    
    highest = None
 
    for object in objects:
        if (object.score <= 1.0) and (highest == None or object.score > highest.score): # check if score is above confidence threshold and below 1, as well as if it is the highest score from detected
            highest = object # set highest score

    if highest != None: # check if any license plate has been found
        bbox = highest.bbox # get bounding box
        frame = cv2.rectangle(frame, (bbox.xmin,bbox.ymin), (bbox.xmax,bbox.ymax), (0,255,0), 2) # draw box around detected
        frame = cv2.resize(frame, (frame.shape[0]*2,frame.shape[1]*2)) # scale image by a factor of 2
 
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
    
    #frame = cv2.putText(frame, str(round(1/(time.time()-start))), (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera", frame) # show image
    if cv2.waitKey(1) == ord("q"):  # if q key pressed
        break  # break from main loop
 
conn.commit()  # commit database changes
input.release()  # release camera
cv2.destroyAllWindows()  # kill all open image windows