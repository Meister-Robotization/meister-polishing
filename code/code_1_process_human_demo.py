#!/usr/bin/env python

"""
source python3.10/bin/activate
"""
import sys
import os
import cv2
import json
import torch
import copy
import numpy as np
from pathlib import Path
import dm_env
import h5py
import copy
import supervision as sv
import math
import matplotlib.pyplot as plt
import threading
from PIL import Image as PIL_Image
from matplotlib import cm
from typing import Union, Tuple
import time
from common import *

# DMP
PATH_TXT = "/root/Projects/Maester/2025_AI_soln/human_motion_zigzag_250918.txt"
OUT_TXT  = "/root/Projects/Maester/2025_AI_soln/dmp_motion_zigzag_250918.txt"

PATH_TXT = "/root/Projects/Maester/2025_AI_soln/human_motion_circular_250918.txt"
OUT_TXT  = "/root/Projects/Maester/2025_AI_soln/dmp_motion_circular_250918.txt"

# robot motion
ROBOT_HZ = 100.
ROBOT_DT = float(1./ROBOT_HZ)

code = DMPPathGenerator(n_dofs=3,
                        n_kernels=30,
                        human_demo_txt=PATH_TXT, 
                        camera_angle=-43.*np.pi/180., 
                        lpf_alpha=0.8, 
                        dt=ROBOT_DT)
#code.process_human_demo_plot()

code.initializeDMP(test=False)


start = np.array([582.6689268126078, 174.84252981628896, 300.0])     # x, y, z, [mm]
end   = np.array([732.6616531619053, -169.12481743925292, 300.0])  # x, y, z, [mm]



code.PathGeneration(start, end, OUT_TXT, save=True)


"""
# load data
_, x, y, z = loadtxt(OUT_TXT)

# plot and show
ax = plt.figure(1).add_subplot(projection="3d")
ax.set_box_aspect([1.0, 1.0, 1.0])
ax.plot(x, y, z, 'b')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")  

ax.plot(x[0], y[0], z[0], 'or')
#ax.plot(x[3], y[3], z[3], 'or')
#ax.plot(x[6], y[6], z[6], 'or')
ax.plot(x[-1], y[-1], z[-1], 'ok')
#ax.plot(x[-3], y[-3], z[-3], 'ok')
#ax.plot(x[-6], y[-6], z[-6], 'ok')
plt.show()
"""
