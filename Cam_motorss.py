
import os
import RPi.GPIO as GPIO

#define Servos GPIOs
panServo = 5
tiltServo = 3

# initialize LED GPIO

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)


#position servos 
def Cam (servo, angle):
    os.system("python angleServoCtrl.py " + str(servo) + " " + str(angle))
    print("[INFO] Positioning servo at GPIO {0} to {1} degrees\n".format(servo, angle))
