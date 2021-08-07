import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)

Motor1A = 21 #(GPIO 23 - Pin 16)
Motor1B = 19 #(GPIO 24 - Pin 18)
Motor1Enable = 40 #38 #(GPIO 25 - Pin 22)

Motor2A =  16 #(GPIO 9 - Pin 21)
Motor2B =  18 #(GPIO 10 - Pin 19)
Motor2Enable = 22 #(GPIO 11 - Pin 23)

HandUP = 11
HandDOWN = 13
HandLEFT = 15
HandRIGHT = 37
HandIN = 32
HandOUT = 33
duty = 0

#Set up all as outputs
GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor1Enable,GPIO.OUT)
GPIO.setup(Motor2A,GPIO.OUT)
GPIO.setup(Motor2B,GPIO.OUT)
GPIO.setup(Motor2Enable,GPIO.OUT)
GPIO.setup(HandUP,GPIO.OUT)
GPIO.setup(HandDOWN,GPIO.OUT)
GPIO.setup(HandLEFT,GPIO.OUT)
GPIO.setup(HandRIGHT,GPIO.OUT)
GPIO.setup(HandIN,GPIO.OUT)
GPIO.setup(HandOUT,GPIO.OUT)

p1 = GPIO.PWM(Motor1Enable,100)
p2 = GPIO.PWM(Motor2Enable,100)
p1.start(0)
p2.start(0)

def speedzero():
    print("Speed zero")
    p1.ChangeDutyCycle(0)
    p2.ChangeDutyCycle(0)
    duty = 0

def speedlow():
    print("Speed low")
    p1.ChangeDutyCycle(10)
    p2.ChangeDutyCycle(10)
    duty = 10

def speedhigh():
    print("Speed high")
    p1.ChangeDutyCycle(60)
    p2.ChangeDutyCycle(60)
    duty = 60

def forward():
    print("Going Forwards")
    GPIO.output(Motor1A,GPIO.HIGH)
    GPIO.output(Motor1B,GPIO.LOW)
    #GPIO.output(Motor1Enable,GPIO.HIGH) #Enable Motor 1 High
    GPIO.output(Motor2A,GPIO.HIGH)
    GPIO.output(Motor2B,GPIO.LOW)
    #GPIO.output(Motor2Enable,GPIO.HIGH) #Enable Motor 2 High
    sleep(2)


def backward():
    print("Going Backwards")
    GPIO.output(Motor1A,GPIO.LOW)
    GPIO.output(Motor1B,GPIO.HIGH)
    #GPIO.output(Motor1Enable,GPIO.HIGH)
    GPIO.output(Motor2A,GPIO.LOW)
    GPIO.output(Motor2B,GPIO.HIGH)
    #GPIO.output(Motor2Enable,GPIO.HIGH)
    sleep(2)

def turnRight():
    print("Going Right")
    p1.ChangeDutyCycle(60)
    p2.ChangeDutyCycle(60)
    GPIO.output(Motor1A,GPIO.HIGH)
    GPIO.output(Motor1B,GPIO.LOW)
    #GPIO.output(Motor1Enable,GPIO.HIGH)
    GPIO.output(Motor2A,GPIO.LOW)
    GPIO.output(Motor2B,GPIO.HIGH)
    #GPIO.output(Motor2Enable,GPIO.LOW)
    sleep(1)
    print("Hold")
    GPIO.output(Motor1A,GPIO.LOW)
    GPIO.output(Motor2B,GPIO.LOW)
    p1.ChangeDutyCycle(duty)
    p2.ChangeDutyCycle(duty)
    

def turnLeft():
    print("Going Left")
    p1.ChangeDutyCycle(60)
    p2.ChangeDutyCycle(60)
    GPIO.output(Motor1A,GPIO.LOW)
    GPIO.output(Motor1B,GPIO.HIGH)
    #GPIO.output(Motor1Enable,GPIO.LOW)
    GPIO.output(Motor2A,GPIO.HIGH)
    GPIO.output(Motor2B,GPIO.LOW)
    #GPIO.output(Motor2Enable,GPIO.HIGH)
    sleep(1)
    print("Hold")
    GPIO.output(Motor1B,GPIO.LOW)
    GPIO.output(Motor2A,GPIO.LOW)
    p1.ChangeDutyCycle(duty)
    p2.ChangeDutyCycle(duty)

def stop():
    print("Stopping")
    GPIO.output(Motor1A,GPIO.LOW)
    GPIO.output(Motor1B,GPIO.LOW)
    #GPIO.output(Motor1Enable,GPIO.LOW)
    GPIO.output(Motor2A,GPIO.LOW)
    GPIO.output(Motor2B,GPIO.LOW)
    #GPIO.output(Motor2Enable,GPIO.LOW)
    sleep(2)
    
def handin():
    print("Closing Hands")
    GPIO.output(HandIN,GPIO.HIGH)
    GPIO.output(HandOUT,GPIO.LOW)
    
def handout():
    print("Release Hands")
    GPIO.output(HandIN,GPIO.LOW)
    GPIO.output(HandOUT,GPIO.HIGH)
    
def handup():
    print("Hands Up")
    GPIO.output(HandUP,GPIO.HIGH)
    GPIO.output(HandDOWN,GPIO.LOW)
    
def handdown():
    print("Hands Down")
    GPIO.output(HandUP,GPIO.LOW)
    GPIO.output(HandDOWN,GPIO.HIGH)
    
def handleft():
    print("Hands to the left")
    GPIO.output(HandLEFT,GPIO.HIGH)
    GPIO.output(HandRIGHT,GPIO.LOW) 

def handright():
    print("Hands to the right")
    GPIO.output(HandLEFT,GPIO.LOW)
    GPIO.output(HandRIGHT,GPIO.HIGH)
    
def handstop():
    print("Stop Hands")
    GPIO.output(HandIN,GPIO.LOW)
    GPIO.output(HandOUT,GPIO.LOW)
    GPIO.output(HandUP,GPIO.LOW)
    GPIO.output(HandDOWN,GPIO.LOW)
    GPIO.output(HandLEFT,GPIO.LOW)
    GPIO.output(HandRIGHT,GPIO.LOW)
