import sys
import os
import cv2
import json
import torch
import copy
import numpy as np
from pathlib import Path
import time
import mediapipe as mp
import pyrealsense2 as rs
from common import *

PATH_TXT = "/root/Projects/Maester/2025_AI_soln/human_motion.txt"
MANO_LIST = [9, 13]

class recordHandMotion():
    def __init__(self):
        
        super(recordHandMotion, self).__init__()
        
        # remove output file
        try:
            os.remove(PATH_TXT)
        except:
            pass
            
        # params
        self.sleep_s = 0.01
        
        # check for gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # sensor
        self.sensor = femtoBolt()
        
        # marker detector
        self.det = simpleAprilTagDetector(self.sensor.camera_matrix, self.sensor.distortion_coefficient, 22.)
        
        # mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # flag to record data
        self.write_flag = False    
      
    def cyclic_task(self):
        
        # loop
        while True:
            try:
                # get image and depth data
                cimg, dimg, points = self.sensor.getColorDepthImageWithPointCloud()
                
                if cimg is not None and dimg is not None: # valid data obtained
                
                    imgH, imgW, _ = cimg.shape # (1080, 1920, 3)
                    
                    # detect marker
                    cimg, T = self.det.detect(cimg)
                    # if marker is detected, get angle
                    if T is not None:
                        angle = np.arctan2(T[2][0], T[1][0])
                        print(angle*180./np.pi)
        
                    with self.mp_hands.Hands(model_complexity=0,
                                             min_detection_confidence=0.5,
                                             min_tracking_confidence=0.5) as hands:

                        # detect hand
                        cimg.flags.writeable = False
                        cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
                        results = hands.process(cimg)
                        cimg.flags.writeable = True
                        cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR)
        
                        # reset
                        found_flag = False
                        hand_cnt = 0
                        # process data if hand is detected
                        if results.multi_hand_landmarks:
                            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                handedness = results.multi_handedness[idx].classification[0]
                
                                # hand position in image coordinate (2-D)
                                # find right hand
                                if handedness.label == 'Left':
                        
                                    self.mp_drawing.draw_landmarks(cimg, hand_landmarks,
                                                                   self.mp_hands.HAND_CONNECTIONS,
                                                                   self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                                                   self.mp_drawing_styles.get_default_hand_connections_style())
                         

                                    # position --------------------------------------------------------------------------------
                                    # hand position: index = 0, 1, 2, 5, 9, 13, 17
                                    index = copy.deepcopy(MANO_LIST)
                                    x_normalized = 0
                                    y_normalized = 0
                                    for i in range(len(index)):
                                        x_normalized = x_normalized + hand_landmarks.landmark[index[i]].x
                                        y_normalized = y_normalized + hand_landmarks.landmark[index[i]].y
                                    x_normalized = x_normalized/len(index)
                                    y_normalized = y_normalized/len(index)

                                    # denormalize
                                    x_unnormalized = x_normalized * imgW # (1920)
                                    y_unnormalized = y_normalized * imgH # (1080)

                                    # draw a circle on hand position
                                    cimg = cv2.circle(cimg, (int(x_unnormalized),int(y_unnormalized)), 30, (0, 0, 255), 15)
                                
                                    # get xyz position (depth)
                                    #hand_pos_xyz = self.sensor.getXYZfromUV(int(x_unnormalized), int(y_unnormalized))
                                    hand_pos_xyz = points[int(y_unnormalized), int(x_unnormalized), :]
                                    hand_pos_x = hand_pos_xyz[0] / 1000. # [m]
                                    hand_pos_y = hand_pos_xyz[1] / 1000.
                                    hand_pos_z = hand_pos_xyz[2] / 1000.
                   
                                    # record data, if needed
                                    if self.write_flag:
                                        f = open(PATH_TXT, "a+")
                                        _line = f"{time.time()}, {hand_pos_x}, {hand_pos_y}, {hand_pos_z}\n"
                                        f.write(_line)
                                        f.close()
                                    
                                    # update cimg
                                    if self.write_flag: # saving                                        
                                        _text = 'Press [s] to save, press [q] to quit'                                        
                                        cimg = cv2.putText(cimg, _text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) , 1)                                        
                                    else: # not saving                                        
                                        _text = 'Press [r] to record, press [q] to quit'
                                        cimg = cv2.putText(cimg, _text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) , 1)
                                        
                                    # show hand pos         
                                    _text = f"{hand_pos_x:.3f}, {hand_pos_y:.3f}, {hand_pos_z:.3f}"
                                    cimg = cv2.putText(cimg, _text, (int(x_unnormalized),int(y_unnormalized)), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) , 1)     
                                    
                       
                    cv2.imshow("Hand tracking", cimg)
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == 27: # 27: ESC
                        break
                    if key == ord('r') and self.write_flag == False: # start recording
                        self.write_flag = True
                    elif key == ord('s') and self.write_flag: # stop recording
                        self.write_flag = False
            except KeyboardInterrupt:
                break
                
            #self.rate.sleep()                
            time.sleep(self.sleep_s)    
        self.sensor.pipeline.stop()

code = recordHandMotion()
code.cyclic_task()      



