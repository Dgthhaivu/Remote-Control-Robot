# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask import request, redirect, url_for, make_response
import numpy as np
import threading
import argparse
import datetime
import imutils
import time
import cv2
import motors
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD) #set up GPIO

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
vs = VideoStream(usePiCamera=1).start()
# vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def main():
    # return the rendered template
    return render_template("main.html")

def detect_motion():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    # initialize the list of class labels MobileNet SSD was trained to

    # detect, then generate a set of bounding box colors for each class

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("/home/pi/Desktop/Sy/stream-video-browser-update/pi-object-detection/MobileNetSSD_deploy.prototxt.txt", "/home/pi/Desktop/Sy/stream-video-browser-update/pi-object-detection/MobileNetSSD_deploy.caffemodel")

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
            # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the main of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # show the output frame
                #cv2.imshow("Frame", frame)
                #key = cv2.waitKey(1) & 0xFF
        with lock:
            outputFrame = frame.copy()
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
            
@app.route('/<changepin>', methods=['POST'])
def reroute(changepin):

    changePin = int(changepin) #cast changepin to an int

    if changePin == 1:
        motors.turnLeft()
    elif changePin == 2:
        motors.forward()
    elif changePin == 3:
        motors.turnRight()
    elif changePin == 4:
        motors.backward()
    elif changePin == 5:
        motors.stop()
    elif changePin == 6:
        motors.handin() 
    elif changePin == 7:
        motors.handout()    
    elif changePin == 8:
        motors.handup() 
    elif changePin == 9:
        motors.handdown()   
    elif changePin == 10:
        motors.handleft()
    elif changePin == 11:
        motors.handright()
    elif changePin == 12:
        motors.handstop()               
        
    response = make_response(redirect(url_for('main')))
    return(response)

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
            help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion) #args=(
    #   args["frame_count"],)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
        

