# import dependencies
import cv2 as cv
import numpy as np
import mediapipe as mp
import time 


import pyautogui

#setting up the mediapipe library for this 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


capture = cv.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence= 0.5,min_tracking_confidence= 0.5) as hands:
    while True:
        isTrue,frame = capture.read()

        h,w,c = frame.shape

        start = time.perf_counter()

        # this flips the image horizontally to fit the gamer (self view display)
        # and converts the image from BGR to RGB
        imgRGB = cv.cvtColor(cv.flip(frame,1),cv.COLOR_BGR2RGB)

        # improve the performance ,optimally as not writeable to 
        # pass by reference
        frame.flags.writeable = False

        results = hands.process(imgRGB)
        print(results.multi_hand_landmarks)

        frame.flags.writeable =True

        imgBGR = cv.cvtColor(imgRGB,cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(imgBGR,hand_landmarks,mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(200,255,0),thickness= 2,circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(200,0,0),thickness= 2,circle_radius=1)
                                        )
                index_finger_tip = hand_landmarks.landmark[0] 

                index_finger_tip_x = index_finger_tip.x * w
                index_finger_tip_y = index_finger_tip.y * h

                # this part has an issue otherwise the rest of the code freezes
                # if index_finger_tip_x > w/2:
                #     cv.putText(imgBGR,'Gas',(500,70),cv.FONT_HERSHEY_SIMPLEX,1.5,(112,140,0),2)
                #     pyautogui.keyDown('right')
                #     pyautogui.keyUp('left')
                # elif index_finger_tip_x < w/2:
                #     cv.putText(imgBGR,'Break',(500,70),cv.FONT_HERSHEY_SIMPLEX,1.5,(0,140,120),2)
                #     pyautogui.keyDown('left')
                #     pyautogui.keyUp('right')
                

        cv.line(imgBGR,(int(w/2),0),(int(w/2),h),(0,255,0),2)
        end = time.perf_counter()
        totalTime= end - start

        fps = 1/totalTime

        cv.putText(imgBGR,f'FPS: {int(fps)}',(20,70),cv.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)


        cv.imshow('game time',imgBGR)
        if cv.waitKey(5) & 0xFF ==ord("q"):
            break
    capture.release()
    cv.destroyAllWindows()
    