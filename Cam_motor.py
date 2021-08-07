import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BOARD)

CamPan  = 3
CamTilt = 5

#Set up all as outputs
GPIO.setup(CamPan,GPIO.OUT)
GPIO.setup(CamTilt,GPIO.OUT)
GPIO.setwarnings(False)



def Cam(servo, angle):
    assert angle >=30 and angle <= 150
    
    pwm = GPIO.PWM(servo, 50)
    pwm.start(8)
    dutyCycle = angle / 18. + 3.
    pwm.ChangeDutyCycle(dutyCycle)
    sleep(0.15)
    pwm.stop()