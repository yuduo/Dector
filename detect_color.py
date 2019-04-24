# USAGE
# python detect_color.py --image pokemon_games.png

# import the necessary packages
import numpy as np
import argparse
import cv2
import VideoStream
import time
import os
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
mode = GPIO.getmode()
print(mode)

GPIO.setup(12, GPIO.OUT)

IM_WIDTH = 1280
IM_HEIGHT = 1280 
FRAME_RATE = 10

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

freq = cv2.getTickFrequency()

videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,1,0).start()
time.sleep(1) # Give the camera time to warm up

cam_quit = 0 # Loop control variable

cv2.namedWindow('images',cv2.WINDOW_NORMAL)
# cv2.namedWindow('mask',cv2.WINDOW_NORMAL)

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
trainImg = cv2.imread(path+'/Card_Imgs/Diamonds1.jpg', cv2.IMREAD_GRAYSCALE)
# Begin capturing frames
while cam_quit == 0:

	# Grab frame from video stream
    image = videostream.read()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()


	# define the list of boundaries
	# boundaries = [
	# 	([17, 15, 100], [50, 56, 200]),
	# 	([86, 31, 4], [220, 88, 50]),
	# 	([25, 146, 190], [62, 174, 250]),
	# 	([103, 86, 65], [145, 133, 128])
	# ]

	# loop over the boundaries
	#for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array([10, 100, 100], dtype = "uint8")
    upper = np.array([30, 255, 255], dtype = "uint8")

    imgHSV= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(imgHSV, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)

    
    imgray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)

    # Find rank contour and bounding rectangle, isolate and find largest contour
    dummy, Qrank_cnts, hier = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)

    

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        # size = cv2.contourArea(Qrank_cnts[0])
        # peri = cv2.arcLength(Qrank_cnts[0],True)
        # approx = cv2.approxPolyDP(Qrank_cnts[0],0.01*peri,True)
        
        # #if ((size < 2) and (size > CARD_MIN_AREA)
        # #    and (hier_sort[i][3] == -1) and (len(approx) == 4)):
        # if (len(approx) != 4):
        #     continue
        x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
        Qsuit_roi = thresh[y1:y1+h1, x1:x1+w1]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)

        diff_img = cv2.absdiff(Qsuit_sized, trainImg)
        rank_diff = int(np.sum(diff_img)/255)
        cv2.putText(image,"diff:"+str(rank_diff),(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2,cv2.LINE_AA)
        if rank_diff < 1400:
            cv2.putText(image,"success detected",(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2,cv2.LINE_AA)
            GPIO.output(12, GPIO.HIGH)
        else:
            GPIO.output(12, GPIO.LOW)
           # time.sleep(1)
            
        #GPIO.clean()
        cv2.imshow("train", trainImg)
        cv2.imshow("contour", Qsuit_sized)

    # show the images
    cv2.imshow("output", output)
    cv2.imshow("images", image)

    #cv2.waitKey(0)
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    
    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()
videostream.stop()
