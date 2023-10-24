# Dependencies
import cv2
import pytesseract
import time
import sqlite3
from ultralytics import YOLO

start = time.time()
print("Loading Model...")
model = YOLO("license_plate_openvino_model", task="detect")  # initialize model
print("Finished Loading Model ", round((time.time() - start) * 100000) / 100, " ms")

start = time.time()
print("Connecting To Database...")
conn = sqlite3.connect("database")  # connect to local database
print("Connected ", round((time.time() - start) * 100000) / 100, " ms")

cur = conn.cursor()  # get database cursor
cur.execute(
    "create table if not exists log (plate text, timestamp int, timetype int)"
)  # create table for the database

video = cv2.VideoCapture(0)  # get camera (-1 default)
lastlogged = ""  # last logged license plate


def logdata(plate, timestamp):  # new function for logging data
    cur.execute(
        "insert into log (plate, timestamp, timetype) values (?, ?, ifnull(1 - (select timetype from log where plate=? order by timestamp desc limit 1), 0))",
        [plate, timestamp, plate],
    )  # log data
    cur.execute("select * from log")  # get all the data within the table
    for val in cur.fetchall():  # loop through stuff
        print(val)  # print it


print("Starting Program")

while True:  # main loop
    ret, src = video.read()  # get image from camera
    if not ret:  # if image read improperly
        break  # break out of main loop

    x, y, _ = src.shape  # get dimensions of input image
    ratio = x / y  # set the ratio of dimensions of input image
    frame = cv2.resize(
        src, (640, 640)
    )  # resize the input image and store to new variable
    frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )  # convert input image to grayscale to be faster
    frame = cv2.merge(
        (frame, frame, frame)
    )  # make image into 3 channels because model made for 3 channels

    results = model(frame)  # get the result data of detected license plate

    boxes = results[0].boxes.xyxy  # get boxes from the results
    if len(boxes) > 0:  # if there are any license plates
        box = boxes[0]  # get the first license plate found
        x1, y1, x2, y2 = box  # get the bounding box of license plate

        cv2.rectangle(
            frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
        )  # draw the bounding box of the license plate
        src = cv2.resize(src, (640, 640))  # resize source to square
        frame = src[int(y1) : int(y2), int(x1) : int(x2)]  # crop image
        frame = cv2.resize(
            frame, (int(100 / ratio), int(100 * ratio))
        )  # set output image to the same aspect ratio as original source

        text = pytesseract.image_to_string(frame)  # extract text
        text = text.upper()  # convert license plate to caps
        result = ""  # result variable
        for char in text:  # for character within the result text
            if char.isalnum():  # if character isnt special
                result = result + char  # add it to the result text
        print("Result: ", result)  # print resulting license plate
        if result != "":  # make sure there is a license plate
            print(lastlogged)
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

    cv2.imshow("Object Detection", frame)  # display the image

    if cv2.waitKey(1) == ord("q"):  # if q key pressed
        break  # break from main loop

conn.commit()  # commit database changes
video.release()  # release camera
cv2.destroyAllWindows()  # kill all open image windows
