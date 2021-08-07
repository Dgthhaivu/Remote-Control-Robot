# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask import request, redirect, url_for, make_response
from time import sleep
from pyzbar import pyzbar

import numpy as np
import threading
import argparse
import datetime
import imutils
import time
import cv2
import motors
import Cam_motors
import Cam_motorss
import RPi.GPIO as GPIO
import os

GPIO.setmode(GPIO.BOARD) #set up GPIO
panServo = 5 #11
tiltServo = 3 #13
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
QRframe = None
target_frame = None

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

def generate():
    
    # loop over frames from the output stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        frame = imutils.rotate(frame, angle=180)
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')

@app.route("/color_tracking")
def color_tracking():
    # return the response generated along with the specific media
    # type (mime type)
    return render_template("color_tracking.html")

#position servos 
def positionServo (servo, angle):
    os.system("python angleServoCtrl.py " + str(servo) + " " + str(angle))
    print("[INFO] Positioning servo at GPIO {0} to {1} degrees\n".format(servo, angle))
    
# position servos to present object at center of the frame
def mapServoPosition (x, y):
    global panAngle
    global tiltAngle
    if (x < 220):
        panAngle -= 10
        if panAngle > 140:
            panAngle = 140
        positionServo (panServo, panAngle)
 
    if (x > 280):
        panAngle += 10
        if panAngle < 40:
            panAngle = 40
        positionServo (panServo, panAngle)

    if (y < 160):
        tiltAngle -= 10
        if tiltAngle > 140:
            tiltAngle = 140
        positionServo (tiltServo, tiltAngle)
 
    if (y > 210):
        tiltAngle += 10
        if tiltAngle < 40:
            tiltAngle = 40
        positionServo (tiltServo, tiltAngle)

# define the lower and upper boundaries of the object
# to be tracked in the HSV color space
colorLower = (26, 100, 100)
colorUpper = (46, 255, 255)

# Initialize angle servos at 90-90 position
global panAngle
panAngle = 90
global tiltAngle
tiltAngle =90

# positioning Pan/Tilt servos at initial position
positionServo (panServo, panAngle)
positionServo (tiltServo, tiltAngle)


def color_tracking():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, colorFrame, lock
    
    # loop over the frames from the video stream
    while True:
        # grab the next frame from the video stream, Invert 180o, resize the
        # frame, and convert it to the HSV color space
        frame = vs.read()
        frame = imutils.resize(frame, width= 480)
        #frame = cv2.GaussianBlur(frame, (11, 11), 0)
        frame = cv2.flip(frame, 1)
        frame = imutils.rotate(frame, angle=180)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the object color, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the object
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    #     cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
                # position Servo at center of circle
                mapServoPosition(int(x), int(y))
           
        # show the frame to our screen
        #frame = imutils.rotate(frame,angle=180)
        cv2.imshow("Frame", frame)
        with lock:
            #frame = imutils.rotate(frame, angle=180)
            colorFrame = frame.copy()

def generate_color():
    # grab global references to the output frame and lock variables
    global colorFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if colorFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", colorFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
 #   return render_template('video_feed.html')

@app.route("/QRcode")
def video_QRcode():
    return render_template("QRcode.html")

def QRcode():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, QRframe, lock
    csv = open('QRcodes.csv', "w")
    found = set()

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it to
        # have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # find the QRcodes in the frame and decode each of the QRcodes
        QRcodes = pyzbar.decode(frame)

        # loop over the detected QRcodes
        for QRcode in QRcodes:
            # extract the bounding box location of the QRcode and draw
            # the bounding box surrounding the QRcode on the image
            (x, y, w, h) = QRcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # the QRcode data is a bytes object so if we want to draw it
            # on our output image we need to convert it to a string first
            QRcodeData = QRcode.data.decode("utf-8")
            QRcodeType = QRcode.type

            # draw the QRcode data and QRcode type on the image
            text = "{} ({})".format(QRcodeData, QRcodeType)
            cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # if the QRcode text is currently not in our CSV file, write
            # the timestamp + QRcode to disk and update the set
            if QRcodeData == 'Going Backward' :
                motors.backward()
                motors.stop()
            if QRcodeData not in found:
                csv.write("{},{}\n".format(datetime.datetime.now(),
                    QRcodeData))
                csv.flush()
                found.add(QRcodeData)
    
        with lock:
            frame = imutils.rotate(frame, angle=180)
            QRframe = frame.copy()
            
def generate_QR():
    # grab global references to the output frame and lock variables
    global QRframe, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if QRframe is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", QRframe)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')


@app.route("/object_detection")
def video_obj_detection():
    return render_template("object_detection.html")

def object_detection():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

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
        frame = imutils.rotate(frame, angle=180)
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
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (endX, endY, startX, startY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                
                # show the output frame
                #cv2.imshow("Frame", frame)
                #key = cv2.waitKey(1) & 0xFF
        with lock:
#             frame = imutils.rotate(frame, angle=180)
            outputFrame = frame.copy()
            
def generate_obj():
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


@app.route("/target_detection")
def video_tar_detection():
    return render_template("target_detection.html")

def target_detection():
    global vs, target_frame, lock

    MIN_MATCH_COUNT= 15

    detector=cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDITREE=0
    flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
    flann=cv2.FlannBasedMatcher(flannParam,{})

    trainImg=cv2.imread("TrainingData/TrainImg.jpeg",0)
    trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
    #cam = cv2.VideoCapture(0)
    
    while True:
        QueryImgBGR = vs.read()
        QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
        queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
        matches=flann.knnMatch(queryDesc,trainDesc,k=2)

        goodMatch=[]
        for m,n in matches:
            if(m.distance<0.75*n.distance):
                goodMatch.append(m)
        if(len(goodMatch)>MIN_MATCH_COUNT):
            tp=[]
            qp=[]
            
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp,qp=np.float32((tp,qp))
            H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
            h,w=trainImg.shape
            
            trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
            queryBorder=cv2.perspectiveTransform(trainBorder,H)
            cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        
#         else:
#             print ("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
          
        with lock:
            QueryImgBGR = imutils.rotate(QueryImgBGR, angle=180)
            target_frame = QueryImgBGR.copy()
            
def generate_tar():
    # grab global references to the output frame and lock variables
    global target_frame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if target_frame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", target_frame)

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
        sleep(1)
        motors.handstop()
    elif changePin == 7:
        motors.handout()
        sleep(1)
        motors.handstop()
    elif changePin == 8:
        motors.handup()
        sleep(1)
        motors.handstop()
    elif changePin == 9:
        motors.handdown()
        sleep(1)
        motors.handstop()
    elif changePin == 10:
        motors.handleft()
        sleep(0.5)
        motors.handstop()
    elif changePin == 11:
        motors.handright()
        sleep(0.5)
        motors.handstop()
    elif changePin == 12:
        motors.handstop()
        
    elif changePin == 13:
        Cam_motorss.Cam(5,60)
    elif changePin == 14:
        Cam_motorss.Cam(5,90)
    elif changePin == 15:
        Cam_motorss.Cam(5,120)
        
    elif changePin == 16:
        Cam_motorss.Cam(3,60)
    elif changePin == 17:
        Cam_motorss.Cam(3,90)
    elif changePin == 18:
        Cam_motorss.Cam(3,120)
        
    elif changePin == 19:
        motors.handleft()
        sleep(1)
        motors.handstop()
    elif changePin == 20:
        motors.handright()
        sleep(1)
        motors.handstop()
    elif changePin == 21:
        motors.speedzero()
    elif changePin == 22:
        motors.speedlow()
    elif changePin == 23:
        motors.speedhigh()

    response = make_response(redirect(url_for('main')))
    return(response)

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_obj")
def video_feed_obj():
    # return the response generated along with the specific media
    # type (mime type)
    
    return Response(generate_obj(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")
    
@app.route("/video_feed_tar")
def video_feed_tar():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate_tar(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_QR")
def video_feed_QR():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate_QR(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_color")
def video_feed_color():
    # return the response generated along with the specific media
    # type (mime type)
    
    return Response(generate_color(),
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
    o = threading.Thread(target=object_detection)
    o.daemon = True
    o.start()

    t = threading.Thread(target=target_detection)
    t.daemon = True
    t.start()

    q = threading.Thread(target=QRcode)
    q.daemon = True
    q.start()
    
    # start a thread that will perform motion detection
    c = threading.Thread(target=color_tracking)
    c.daemon = True
    c.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
