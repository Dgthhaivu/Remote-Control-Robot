import RPi.GPIO as GPIO
from time import sleep


GPIO.setmode(GPIO.BOARD)

CamTilt = 3
CamPan = 5
i = 8

#Set up all as outputs
GPIO.setup(CamPan,GPIO.OUT)
GPIO.setup(CamTilt,GPIO.OUT)
GPIO.setwarnings(False)

def CamPan_0():
    global i
    i = i + 1
    print("CamPan 0 degree")
    pan = GPIO.PWM(CamPan,50)
    pan.start(0)
    pan.ChangeDutyCycle(i)
    sleep(0.2)
    pan.stop()

def CamPan_90():
    global i
    i = 8
    print("CamPan 90 degree")
    pan = GPIO.PWM(CamPan,50)
    pan.start(0)
    pan.ChangeDutyCycle(i)
    sleep(0.2)
    pan.stop()

def CamPan_180():
    global i
    i = i - 1
    print("CamPan 180 degree")
    pan = GPIO.PWM(CamPan,50)
    pan.start(0)
    pan.ChangeDutyCycle(i)
    sleep(0.2)
    pan.stop()


def CamTilt_0():
    print("CamTilt 0 degree")
    tilt = GPIO.PWM(CamTilt,50)
    tilt.start(0)
    tilt.ChangeDutyCycle(10)
    sleep(0.2)
    tilt.stop()

def CamTilt_90():
    print("CamTilt 90 degree")
    tilt = GPIO.PWM(CamTilt,50)
    tilt.start(0)
    tilt.ChangeDutyCycle(8)
    sleep(0.2)
    tilt.stop()
    
def CamTilt_180():
    print("CamTilt 180 degree")
    tilt = GPIO.PWM(CamTilt,50)
    tilt.start(0)
    tilt.ChangeDutyCycle(5)
    sleep(0.2)
    tilt.stop()
