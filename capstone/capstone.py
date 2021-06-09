
import RPi.GPIO as GPIO
import time
from smbus2 import SMBus
from mlx90614 import MLX90614
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pyrebase

def bukapintu():

    servoPIN1 = 27
    servoPIN2=22
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servoPIN1, GPIO.OUT)
    GPIO.setup(servoPIN2, GPIO.OUT)
    p = GPIO.PWM(servoPIN1, 50)
    p2= GPIO.PWM(servoPIN2, 50)
    p.start(2.5) # Initialization
    p2.start(2.5)
    p.ChangeDutyCycle(0)
    p2.ChangeDutyCycle(0)
    time.sleep(0.5)
    p.ChangeDutyCycle(2.5)
    p2.ChangeDutyCycle(8)
    time.sleep(0.5)
    p.stop()
    p2.stop()
    GPIO.cleanup()

def tutuppintu():

    servoPIN1 = 27
    servoPIN2=22
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servoPIN1, GPIO.OUT)
    GPIO.setup(servoPIN2, GPIO.OUT)
    p = GPIO.PWM(servoPIN1, 50)
    p2= GPIO.PWM(servoPIN2, 50)
    p.start(2.5) # Initialization
    p2.start(2.5)
    p.ChangeDutyCycle(0)
    p2.ChangeDutyCycle(0)
    time.sleep(0.5)
    p.ChangeDutyCycle(8)
    p2.ChangeDutyCycle(2.5)
    time.sleep(0.5)
    p.stop()
    p2.stop()
    GPIO.cleanup()

def suhu():
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    b=sensor.get_object_1()
    bus.close()
    return b 

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
kondisi_pintu=0
count=0
berhasil=0
gagal=0
sumsuhu=0
n=1

config = {     
  'apiKey': "AIzaSyABj7hffKzPSwIqdcKVIEtK7gGaLkbujCM",
  'authDomain': "smart-gate-875a2.firebaseapp.com",
  'databaseURL': "https://smart-gate-875a2-default-rtdb.firebaseio.com",
  'projectId': "smart-gate-875a2",
  'storageBucket': "smart-gate-875a2.appspot.com",
  'messagingSenderId': "852751241548",
  'appId': "1:852751241548:web:d344e84057b99547b97fe9",
  'measurementId': "G-Z0Z23SQR2K"
}

firebase = pyrebase.initialize_app(config)  

storage = firebase.storage()
database = firebase.database()

# loop over the frames from the video stream
while True:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(10,GPIO.IN)
    GPIO.setup(9,GPIO.IN)
    sensor1=GPIO.input(9)
    sensor2=GPIO.input(10)
    if sensor2==1 and kondisi_pintu==0:
        database.child("UserMenerobos").child("Ilham")
        data = {"trobos":"ada yang trobos"}
        database.update(data)   
        
    if sensor1==0:
        if kondisi_pintu==0:
            count=1
            print(count)
            tutuppintu()
            kondisi_pintu=1
        while count==1:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels

            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            
            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
            
                # determine the class label and color we'll use to draw
                # the bounding box and text
                
                if mask > withoutMask:
                    label = "Thank You. Mask On." 
                    color = (0, 255, 0)
                    for i in range(5):
                        sumsuhu=sumsuhu+suhu()
                        print (str(sumsuhu/n))
                        n=n+1
                    if sumsuhu/5<=38 and sumsuhu/5>=36:            
                        bukapintu()
                        berhasil=berhasil+1
                        suhukirim="{:.2f}".format(sumsuhu/5)
                        database.child("User").child("Ilham")
                        data = {"berhasil":str(berhasil), "suhu":suhukirim}
                        database.update(data) 
                        kondisi_pintu=0
                        sumsuhu=0
                        n=1
                        count=0
                       
                    else :
                        gagal=gagal+1
                        database.child("User").child("Ilham")
                        data = {"gagal":str(gagal)}
                        database.update(data) 
                else:
                    label = "No Face Mask Detected"
                    color = (0, 0, 255)
                    gagal=gagal+1
                    database.child("User").child("Ilham")
                    data = {"gagal":str(gagal)}
                    database.update(data)   
                    
                    
                # include the probability in the label
                label = "{}:  suhu:{:.2f} ".format(label, max(mask, withoutMask) * 100, sumsuhu)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF





