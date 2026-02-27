"""
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import copy
from dt_apriltags import Detector
import math
import time
import sys
import matplotlib.pyplot as plt
sys.path.append("/root/Projects/pyorbbecsdk/examples/")  # to import pyorbbecsdk.utils
from pyorbbecsdk import *
from utils import frame_to_bgr_image
import serial
import time

class femtoBolt():
    def __init__(self):
        # setup device
        self.pipeline = Pipeline()
        device = self.pipeline.get_device()
        device_info = device.get_device_info()
        device_pid = device_info.get_pid()
        config = Config()  
        align_mode = "HW"
        enable_sync = True
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            #color_profile = profile_list.get_default_video_stream_profile()
            color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB, 15)
            config.enable_stream(color_profile)
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            assert profile_list is not None
            #depth_profile = profile_list.get_default_video_stream_profile()
            depth_profile = profile_list.get_video_stream_profile(512, 512, OBFormat.Y16, 15)
            assert depth_profile is not None
            #print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
            #                                           color_profile.get_height(),
            #                                           color_profile.get_fps(),
            #                                           color_profile.get_format()))
            #print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
            #                                           depth_profile.get_height(),
            #                                           depth_profile.get_fps(),
            #                                           depth_profile.get_format()))
            config.enable_stream(depth_profile)   
                
        except Exception as e:
            print(e)
            return
        if align_mode == 'HW':
            if device_pid == 0x066B:
                # Femto Mega does not support hardware D2C, and it is changed to software D2C
                config.set_align_mode(OBAlignMode.SW_MODE)
            else:
                config.set_align_mode(OBAlignMode.HW_MODE)
        elif align_mode == 'SW':
            config.set_align_mode(OBAlignMode.SW_MODE)
        else:
            config.set_align_mode(OBAlignMode.DISABLE)
        if enable_sync:
            try:
                self.pipeline.enable_frame_sync()
            except Exception as e:
                print(e)      
        try:
            self.pipeline.enable_frame_sync()
            self.pipeline.start(config)
            
            # for point cloud
            self.camera_param = self.pipeline.get_camera_param()   
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            self.point_cloud_filter = PointCloudFilter()
            self.point_cloud_filter.set_camera_param(self.camera_param)

            # camera matrix
            self.camera_matrix = np.zeros((3,3))
            self.camera_matrix[0,0] = self.camera_param.rgb_intrinsic.fx
            self.camera_matrix[0,2] = self.camera_param.rgb_intrinsic.cx
            self.camera_matrix[1,1] = self.camera_param.rgb_intrinsic.fy
            self.camera_matrix[1,2] = self.camera_param.rgb_intrinsic.cx
            self.camera_matrix[2,2] = 1.0
            self.distortion_coefficient = np.zeros((5,))
            self.distortion_coefficient[0] = rgb_distortion.k1
            self.distortion_coefficient[1] = rgb_distortion.k2
            self.distortion_coefficient[2] = rgb_distortion.p1
            self.distortion_coefficient[3] = rgb_distortion.p2
            self.distortion_coefficient[4] = rgb_distortion.k3
                        
        except Exception as e:
            print(e)
            return
    def getColorDepthImages(self):
        try:
            frames: FrameSet = self.pipeline.wait_for_frames(100)
            if frames is None:
                print("frames is None")
                pass
            color_frame = frames.get_color_frame()
            if color_frame is None:
                print("color_frame is None")
                pass
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                pass
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                print("Depth_frame is None")
                pass

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            
            MIN_DEPTH = 2.*10. # default:20mm
            MAX_DEPTH = 100.*10.  # default:10000mm
            
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)


            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)   
        except:
            color_image = None
            depth_image = None             
        return color_image, depth_image

    def getColorDepthImageWithPointCloud(self):
        try:
            frames: FrameSet = self.pipeline.wait_for_frames(100)
            if frames is None:
                print("frames is None")
                pass
            color_frame = frames.get_color_frame()
            if color_frame is None:
                print("color_frame is None")
                pass
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                pass
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                print("Depth_frame is None")
                pass


            
            scale = depth_frame.get_depth_scale()
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            MIN_DEPTH = 20  # 20mm
            MAX_DEPTH = 10000  # 10000mm            
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)   
            
            
            # point cloud
            frame = self.align_filter.process(frames)
            self.point_cloud_filter.set_position_data_scaled(scale)
            self.point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT if color_frame is not None else OBFormat.POINT)
            point_cloud_frame = self.point_cloud_filter.process(frame)
            points = self.point_cloud_filter.calculate(point_cloud_frame).reshape(1080, 1920, 6) # (2073600, 6)
            
                       
        except Exception as e:
            print(e)
            color_image = None
            depth_image = None  
            points = None           
        return color_image, depth_image, points     
        
class simpleAprilTagDetector():
    def __init__(self, cameraMatrix, distCoeffs, tagSize):
        
        # setup detector
        self.detector = Detector(families='tagStandard41h12',
                                 nthreads=1,
                                 quad_decimate=1.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.tag_size = tagSize
                              
                                 
        self.camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )
        self.distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0],dtype=np.float32)      

    def detect(self, cImage):
        cImageGrayScale = cv2.cvtColor(cImage, cv2.COLOR_RGB2GRAY)
        tags = self.detector.detect(cImageGrayScale, True, self.camera_params, self.tag_size)
    
        T = None
        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(cImage, tuple(tag.corners[idx-1,:].astype(int)),tuple(tag.corners[idx,:].astype(int)),(0,255,0))
      
                # obtrain marker pose (x, y, z)
                pose = tag.pose_t
                x = pose[0][0]
                y = pose[1][0]
                z = pose[2][0]
                # obtain marker orientation (R, RPY)
                R = tag.pose_R
                roll = np.arctan2(R[1,0], R[0,0])*180./np.pi
                pitch = np.arctan2( -R[2,0], np.sqrt( R[2,1]*R[2,1] + R[2,2]*R[2,2] ) )*180./np.pi
                yaw = np.arctan2(R[2,1], R[2,2])*180./np.pi
                
                T = np.eye(4)
                T[0:3, 0:3] = R
                T[0,3] = x
                T[1,3] = y
                T[2,3] = z
        
                # show tag ID
                tag_info = str(tag.tag_id)
                cv2.putText(cImage, str(tag_info), org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8,
                            color=(0, 0, 255))
        
                cImage = cv2.drawFrameAxes(cImage, self.cameraMatrix, self.distCoeffs, tag.pose_R, tag.pose_t, self.tag_size)         
        return cImage, T
        
class myRealSense_v2():
    def __init__(self, SN, img_w=1920, img_h=1080):
        # setup realsense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
   
        self.img_w = img_w
        self.img_h = img_h
    
        self.config.enable_device(SN)
        self.cameraMatrix = np.array([ [914.3502197265625, 0, 635.7286987304688],
                                       [0, 913.3948974609375, 355.22625732421875],
                                       [0., 0., 1.] ], dtype=np.float32)
        self.distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0],dtype=np.float32)  
        self.pc = rs.pointcloud()
        self.depth = False
        
    def setupConfig_colorOnly(self):
        self.config.enable_stream(rs.stream.color, self.img_w, self.img_h, rs.format.bgr8, 30)

    def setupConfig_depthOnly(self):
        self.config.enable_stream(rs.stream.depth, self.img_w, self.img_h, rs.format.z16, 30)

    def setupConfig_depthColor(self):
        self.config.enable_stream(rs.stream.color, self.img_w, self.img_h, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.depth = True
        
        

    def startStream(self):
        self.profile = self.pipeline.start(self.config)
        print("[Realsense] Start streaming")
        
        if self.depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            self.clipping_distance_in_meters = 1 #1 meter
            self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale
    
    def getColorImageNP(self):
        frames = self.pipeline.wait_for_frames()
        cImage = np.asanyarray(frames.get_color_frame().get_data())
        return cImage
        
    def getColorDepthImageNP(self):
        frame = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frame)
        depth = aligned_frames.get_depth_frame()
        color = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        return color_image, depth_colormap

    def getXYZfromUV(self, u, v):
        # get color/depth images
        frame = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frame)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        num_rows = depth_image.shape[0]
        num_cols = depth_image.shape[1]
	
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
        u = int(u)
        v = int(v)
        #print(num_rows, num_cols)
        depth = depth_frame.get_distance(u, v)
        depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth)
        
        return depth_point_in_meters_camera_coords



def loadHumanDemoData(txt):
    # open txt
    f = open(txt, "r")
    lines = f.readlines()
    # parse data
    Time = []
    X = []
    Y = []
    Z = []
    for line in lines:
        tmp = line.split("\n")[0]
        tmp = line.split(",")    
        _t = float(tmp[0])
        _x = float(tmp[1])
        _y = float(tmp[2])
        _z = float(tmp[3])
    
        if (abs(_x) + abs(_y) + abs(_z)) > 1e-1:
    
            Time.append(_t)
            X.append(_x)
            Y.append(_y)
            Z.append(_z) 
    
    return copy.deepcopy(Time), copy.deepcopy(X), copy.deepcopy(Y), copy.deepcopy(Z)


class MovementPrimitive:

    def __init__(self, n_dofs: int, n_kernels: int, kernels_std_scaling: float = 1.5):
        self.__init_helper__(n_dofs, n_kernels, kernels_std_scaling)


    def __init_helper__(self, n_dofs: int, n_kernels: int, kernels_std_scaling: float = 1.5):
        self._config = {'n_dofs': n_dofs, 'n_kernels': n_kernels, 'kernels_std_scaling': kernels_std_scaling}
        self.n_dofs = n_dofs
        self.n_kernels = n_kernels
        self.kernels_std_scaling = kernels_std_scaling

        self.weights = np.zeros((n_dofs, n_kernels))

        self.c = np.linspace(0.0, 1.0, self.n_kernels)[:, np.newaxis]

        hi = 1.0 / (kernels_std_scaling * (self.c[1] - self.c[0])) ** 2
        self.h = np.ones((n_kernels, 1)) * hi

        zero_tol = np.nextafter(np.float32(0), np.float32(1))
        self.s_min = self.c[0] - math.sqrt(-math.log(zero_tol) / self.h[0])
        self.s_max = self.c[-1] + math.sqrt(-math.log(zero_tol) / self.h[-1])

    def get_pos(self, s: float) -> np.array:
        return np.matmul(self.weights, self.regress_vec(s))

    def get_vel(self, s: float, s_dot: float) -> np.array:
        return np.matmul(self.weights, self.regress_vec_dot(s, s_dot))

    def get_accel(self, s: float, s_dot: float, s_ddot: float) -> np.array:
        return np.matmul(self.weights, self.regress_vec_ddot(s, s_dot, s_ddot))

    def get_vel_spatial_scale(self, s: float, s_dot: float, ks: np.array) -> np.array:
        _w = copy.deepcopy(self.weights)
        _w = np.matmul(ks, _w)
        return np.matmul(_w, self.regress_vec_dot(s, s_dot))


    def get_accel_spatial_scale(self, s: float, s_dot: float, s_ddot: float, ks: np.array) -> np.array:
        _w = copy.deepcopy(self.weights)
        _w = np.matmul(ks, _w)
        return np.matmul(_w, self.regress_vec_ddot(s, s_dot, s_ddot))
        
    def train(self, s: np.array, pos_data: np.array, train_method: str = 'LS', end_points_constraints=False):

        s = np.squeeze(s)

        if pos_data.ndim == 1:
            pos_data = np.expand_dims(pos_data, 0)

        if pos_data.shape[0] != self.n_dofs:
            raise AttributeError('[MovementPrimitive::train]: The training data have wrong number of DoFs...')

        if np.any(s > 1) or np.any(s < 0):
            print('\33[1m\33[33m[MovementPrimitive::train]: The training timestamps are not normalized...\33[0m')

        H = np.hstack([self.regress_vec(s[j]) for j in range(len(s))])

        if train_method.upper() == 'LWR':
            self.weights = np.matmul(pos_data, H.transpose()) / np.sum(H, axis=1).transpose()
        elif train_method.upper() == 'LS':
            self.weights = np.linalg.lstsq(H.transpose(), pos_data.transpose(), rcond=None)[0].transpose()
        else:
            print('\33[1m\33[31m[MovementPrimitive::train]: Unsupported training method...\33[0m')

        if end_points_constraints:
            Sw = np.linalg.inv(np.matmul(H, H.transpose()))

            # enforce start and end point constraints
            A = np.hstack([self.regress_vec(0), self.regress_vec(1),
                           self.regress_vec_dot(0, 1), self.regress_vec_dot(1, 1),
                           self.regress_vec_ddot(0, 1, 0), self.regress_vec_ddot(1, 1, 0)])
            b = np.stack([pos_data[:, 0], pos_data[:, -1],
                          np.zeros((self.n_dofs,)), np.zeros((self.n_dofs,)),
                          np.zeros((self.n_dofs,)), np.zeros((self.n_dofs,))],
                         axis=1)

            R = np.diag([1e-5, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2])
            Sw_At = np.matmul(Sw, A)  # A is already transposed!
            K = np.matmul(Sw_At, np.linalg.inv(R + np.matmul(A.transpose(), Sw_At)))
            e = (b - np.matmul(self.weights, A)).transpose()
            self.weights = self.weights + np.matmul(K, e).transpose()

        err_data = np.matmul(self.weights, H) - pos_data
        train_err = np.linalg.norm(err_data, axis=1)

        return train_err

    def reconfig(self, n_kernels=None, kernels_std_scaling=None,
                 n_points=200, train_method='LS', end_points_constraints=False):

        if not n_kernels:
            n_kernels = self.n_kernels
        if not kernels_std_scaling:
            kernels_std_scaling = self.kernels_std_scaling

        s_data = np.linspace(0, 1, n_points)
        pos_data = np.hstack([self.get_pos(s) for s in s_data])
        # reconfigure MP
        self.__init_helper__(n_dofs=self.n_dofs, n_kernels=n_kernels, kernels_std_scaling=kernels_std_scaling)
        return self.train(s_data, pos_data, train_method=train_method, end_points_constraints=end_points_constraints)

    def to_state_dict(self):
        return {'weights': self.weights, 'config': self._config}

    @classmethod
    def from_state_dict(cls, state_dict):
        mp = cls(**state_dict['config'])
        mp.weights = state_dict['weights']
        return mp

    def save(self, filename):
        pickle.dump(self.to_state_dict(), open(filename, 'wb'))

    @staticmethod
    def load(filename):
        return MovementPrimitive.from_state_dict(pickle.load(open(filename, 'rb')))

    def deep_copy(self):
        return copy.deepcopy(self)

    def regress_vec(self, s: float) -> np.array:

        # take appropriate actions when x causes phi = 0 due to finite
        # numerical precision.
        if s < self.s_min:
            psi = np.zeros((self.n_kernels, 1))
            psi[0] = 1.0
        elif s > self.s_max:
            psi = np.zeros((self.n_kernels, 1))
            psi[-1] = 1.0
        else:
            psi = self._kernel_fun(s)

        phi = psi / np.sum(psi)
        return phi

    def regress_vec_dot(self, s: float, s_dot: float) -> np.array:

        if s < self.s_min or s > self.s_max:
            return np.zeros((self.n_kernels, 1))

        psi = self._kernel_fun(s)
        psi_dot = self._kernel_fun_dot(s, s_dot)
        sum_psi = np.sum(psi)
        sum_psi_dot = np.sum(psi_dot)

        phi = psi / sum_psi
        phi_dot = (psi_dot - phi * sum_psi_dot) / sum_psi
        return phi_dot

    def regress_vec_ddot(self, s: float, s_dot: float, s_ddot: float) -> np.array:

        if s < self.s_min or s > self.s_max:
            return np.zeros((self.n_kernels, 1))

        psi = self._kernel_fun(s)
        psi_dot = self._kernel_fun_dot(s, s_dot)
        psi_ddot = self._kernel_fun_ddot(s, s_dot, s_ddot)
        sum_psi = np.sum(psi)
        sum_psi_dot = np.sum(psi_dot)
        sum_psi_ddot = np.sum(psi_ddot)

        phi = psi / sum_psi
        phi_dot = (psi_dot - phi * sum_psi_dot) / sum_psi
        phi_ddot = (psi_ddot - 2 * phi_dot * sum_psi_dot - phi * sum_psi_ddot) / sum_psi
        return phi_ddot

    def _kernel_fun(self, s: float) -> np.array:
        return np.exp(-self.h * np.power(s - self.c, 2))

    def _kernel_fun_dot(self, s: float, s_dot: float) -> np.array:
        psi = self._kernel_fun(s)
        a = (s - self.c) * s_dot
        psi_dot = -2 * self.h * (psi * a)
        return psi_dot

    def _kernel_fun_ddot(self, s: float, s_dot: float, s_ddot: float) -> np.array:

        psi = self._kernel_fun(s)
        psi_dot = self._kernel_fun_dot(s, s_dot)
        a = (s - self.c) * s_dot
        a_dot = (s - self.c) * s_ddot + s_dot ** 2
        psi_ddot = -2 * self.h * (psi_dot * a + psi * a_dot)

        return psi_ddot

    def _generate_pos_traj(self, s: np.array):
       
        pos_data = np.hstack([self.get_pos(_s) for _s in s])
        
        return pos_data        
        
    def _upsample(self, dt: float, Td: float, n_kernels):
    
        sp = np.linspace(0, 1, int(Td/dt))
        y_desp = np.hstack([self.get_pos(s) for s in sp])  # up-sample
        sp = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(y_desp, axis=1), axis=0)]))
        sp = sp / sp[-1]
        #self.__init_helper__(n_dofs = self.n_dofs, n_kernels = n_kernels)
        #self.train(sp, y_desp, 'LS', end_points_constraints=False) # train again
        
    def _train_for_new_start_goal(self, s: np.array, start: np.array, goal: np.array, n_iter: int = 30, gain1: float = 30., gain2: float = 0.3, gain3: float = 0.3):
                
        # use a copy of original learned weight
        _weights = copy.deepcopy(self.weights)
        
        start = start.reshape(-1, 1)
        goal = goal.reshape(-1, 1)

        s_dot = np.hstack((0, np.diff(s)))
        s_ddot = np.hstack((0, np.diff(s_dot)))
                
        # compuate scaling matrix Ks
        H = np.hstack([self.regress_vec(s[j]) for j in range(len(s))])
        y_hat_0 = np.matmul(_weights, H[:,0]) # [n,]
        g_hat = np.matmul(_weights, H[:,-1]) # [n,]
        y0 = start.reshape(-1)
        g = goal.reshape(-1)        
        denom = g - y0
        num = g_hat - y_hat_0
        # element-wise division
        ks = denom[:,np.newaxis]/num  # [nxn]
        ks = denom / num
        Ks = np.diag(ks)

        # desired acceleration profile for optim
        # do NOT apply Ks here. Pure, recorded acc is needed.
        #y_ddot_d = np.hstack([self.get_accel_spatial_scale(_s, _s_dot, _s_ddot, Ks) for _s, _s_dot, _s_ddot in zip(s, s_dot, s_ddot)])
        y_ddot_d = np.hstack([self.get_accel(_s, _s_dot, _s_ddot) for _s, _s_dot, _s_ddot in zip(s, s_dot, s_ddot)])    
        
        # update DMP weight with Ks
        _weights = np.matmul(Ks, _weights)
        

        # traj without optm
        sz = s.shape[0]
        y1 = np.zeros((self.n_dofs, sz))
        for i in range(sz):
            _ys = np.matmul(_weights, H[:,i]) - np.matmul(Ks, y_hat_0.reshape(-1)) + y0
            y1[:,i] = _ys
            
        # weights 
        _v = np.ones((self.n_dofs))       
        A1 = np.diag(_v)
        A2 = np.diag(_v)
        A3 = np.diag(_v)
        A4 = np.diag(_v)
        A5 = np.diag(_v)
        A6 = np.diag(_v)
        A7 = np.diag(_v)

        # pre-computation
        VEC_ddot = []
        for j in range(s.shape[0]):
            VEC_ddot.append(self.regress_vec_ddot(s[j], s_dot[j], s_ddot[j]))
        s0 = self.regress_vec(0)
        s1 = self.regress_vec(1)
        Ks_y_hat_0 = np.matmul(Ks, y_hat_0.reshape(-1,1))

        t1 = time.time()

        # iteration loop
        for i in range(n_iter):

            # c1: acceleration
            _dc1_dw = []
            for j in range(s.shape[0]):
                tmp = VEC_ddot[j]
                #tmp = self.regress_vec_ddot(s[j], s_dot[j], s_ddot[j])                
                e1i = y_ddot_d[:, j].reshape(-1,1) - np.matmul(_weights, tmp)
                ttmp = -2*np.matmul(A1, e1i)
                ttmp = np.matmul(ttmp, tmp.T)
                _dc1_dw.append(ttmp)
                
            
            # c2: start pos
            #s0 = self.regress_vec(0) 
            #_y_hat_0 = np.matmul(self.weights, s0)    
            _y_hat_0 = np.matmul(_weights, s0).reshape(-1,1) - Ks_y_hat_0 + y0.reshape(-1,1)  
            e2 = (start - _y_hat_0)       
            dc2_dw = -2*np.matmul(A2, e2)
            dc2_dw = np.matmul(dc2_dw, s0.T)
            
      
            # c3: goal pos
            #s1 = self.regress_vec(1)
            #_y_hat_1 = np.matmul(self.weights, s1)
            _y_hat_1 = np.matmul(_weights, s1).reshape(-1,1) - Ks_y_hat_0 + y0.reshape(-1,1)            
            e3 = (goal - _y_hat_1)       
            dc3_dw = -2*np.matmul(A3, e3)
            dc3_dw = np.matmul(dc3_dw, s1.T)
            
            _weights = _weights - gain2*dc2_dw - gain3*dc3_dw
            for j in range(len(_dc1_dw)):
                _weights = _weights - gain1*_dc1_dw[j]
            
        t2 = time.time()
        
        print(f'Optimization took {t2-t1} sec')
        
        y_ddot = np.hstack([self.get_accel(_s, _s_dot, _s_ddot) for _s, _s_dot, _s_ddot in zip(s, s_dot, s_ddot)])        
        y_dot  = np.hstack([self.get_vel(_s, _s_dot) for _s, _s_dot, _s_ddot in zip(s, s_dot, s_ddot)])        
        
        
       
        # traj with optm
        sz = s.shape[0]
        yf = np.zeros((self.n_dofs, sz))
        for i in range(sz):
            _ys = np.matmul(_weights, H[:,i]) - np.matmul(Ks, y_hat_0.reshape(-1)) + y0
            yf[:,i] = _ys

        return y1, yf             

    # train and apply temporal scaling
    def _setupDMP(self, Td: float, dt: float, n_kernels: int, y_des: np.array):
        
        # 1. learn desired path
        data_len = y_des.shape[1]
        s = np.linspace(0, 1, data_len)
        self.train(s, y_des, 'LS')

        # 2. temporal scaling
        self._upsample(dt, Td, n_kernels=n_kernels)

    def _generateRndPath(self, new_start: np.array, new_goal: np.array, s: np.array, n_iter: int = 30, gain1: float = 30., gain2: float = 0.3, gain3: float = 0.3, noise_amp: float = 3.): 
       
        # use a copy of original learned weight
        _weights = copy.deepcopy(self.weights)
        s1, s2 = _weights.shape
        
        # apply noise to weight
        _weights = _weights + 2.*(np.random.rand(s1, s2) - 0.5) * noise_amp
        start = new_start.reshape(-1, 1)
        goal = new_goal.reshape(-1, 1)


        s_dot = np.hstack((0, np.diff(s)))
        s_ddot = np.hstack((0, np.diff(s_dot)))
        
        # compuate scaling matrix Ks
        H = np.hstack([self.regress_vec(s[j]) for j in range(len(s))])
        y_hat_0 = np.matmul(_weights, H[:,0]) # [n,]
        g_hat = np.matmul(_weights, H[:,-1]) # [n,]
        y0 = start.reshape(-1)
        g = goal.reshape(-1)        
        denom = g - y0
        num = g_hat - y_hat_0
        # element-wise division
        ks = denom[:,np.newaxis]/num  # [nxn]
        ks = denom / num
        Ks = np.diag(ks)

        # desired acceleration profile for optim
        # do NOT apply Ks here. Pure, recorded acc is needed.
        y_ddot_d = np.hstack([self.get_accel(_s, _s_dot, _s_ddot) for _s, _s_dot, _s_ddot in zip(s, s_dot, s_ddot)])    
        
        # update DMP weight with Ks
        _weights = np.matmul(Ks, _weights)

        # traj without optm
        sz = s.shape[0]
        y1 = np.zeros((self.n_dofs, sz))
        for i in range(sz):
            _ys = np.matmul(_weights, H[:,i]) - np.matmul(Ks, y_hat_0.reshape(-1)) + y0
            y1[:,i] = _ys

            
        # weights 
        _v = np.ones((self.n_dofs))       
        A1 = np.diag(_v)
        A2 = np.diag(_v)
        A3 = np.diag(_v)
        A4 = np.diag(_v)
        A5 = np.diag(_v)
        A6 = np.diag(_v)
        A7 = np.diag(_v)



        # pre-computation
        VEC_ddot = []
        for j in range(s.shape[0]):
            VEC_ddot.append(self.regress_vec_ddot(s[j], s_dot[j], s_ddot[j]))
        s0 = self.regress_vec(0)
        s1 = self.regress_vec(1)
        Ks_y_hat_0 = np.matmul(Ks, y_hat_0.reshape(-1,1))

        t1 = time.time()

        # iteration loop
        for i in range(n_iter):

            # c1: acceleration
            _dc1_dw = []
            for j in range(s.shape[0]):
                tmp = VEC_ddot[j]
                #tmp = self.regress_vec_ddot(s[j], s_dot[j], s_ddot[j])                
                e1i = y_ddot_d[:, j].reshape(-1,1) - np.matmul(_weights, tmp)
                ttmp = -2*np.matmul(A1, e1i)
                ttmp = np.matmul(ttmp, tmp.T)
                _dc1_dw.append(ttmp)
                
            
            # c2: start pos
            #s0 = self.regress_vec(0) 
            #_y_hat_0 = np.matmul(self.weights, s0)    
            _y_hat_0 = np.matmul(_weights, s0).reshape(-1,1) - Ks_y_hat_0 + y0.reshape(-1,1)  
            e2 = (start - _y_hat_0)       
            dc2_dw = -2*np.matmul(A2, e2)
            dc2_dw = np.matmul(dc2_dw, s0.T)
            
      
            # c3: goal pos
            #s1 = self.regress_vec(1)
            #_y_hat_1 = np.matmul(self.weights, s1)
            _y_hat_1 = np.matmul(_weights, s1).reshape(-1,1) - Ks_y_hat_0 + y0.reshape(-1,1)            
            e3 = (goal - _y_hat_1)       
            dc3_dw = -2*np.matmul(A3, e3)
            dc3_dw = np.matmul(dc3_dw, s1.T)
            
            _weights = _weights - gain2*dc2_dw - gain3*dc3_dw
            for j in range(len(_dc1_dw)):
                _weights = _weights - gain1*_dc1_dw[j]
            
        t2 = time.time()
        
        print(f'Optimization took {t2-t1} sec')
        
        y_ddot = np.hstack([self.get_accel(_s, _s_dot, _s_ddot) for _s, _s_dot, _s_ddot in zip(s, s_dot, s_ddot)])        
        y_dot  = np.hstack([self.get_vel(_s, _s_dot) for _s, _s_dot, _s_ddot in zip(s, s_dot, s_ddot)])        
        
        
       
        # traj with optm
        sz = s.shape[0]
        yf = np.zeros((self.n_dofs, sz))
        for i in range(sz):
            _ys = np.matmul(_weights, H[:,i]) - np.matmul(Ks, y_hat_0.reshape(-1)) + y0
            yf[:,i] = _ys

        return yf 
        
    def _showTestPlot(self, new_start: np.array, new_goal: np.array, s: np.array, n_iter: int = 30, gain1: float = 30., gain2: float = 0.3, gain3: float = 0.3, noise_amp: float = 3., n_test: int = 10): 
       
        n_dofs = self.n_dofs
       
       
       
        # with noise
        for i in range(n_test):
            path = self._generateRndPath(new_start, new_goal, s, n_iter, gain1, gain2, gain3, noise_amp) 
           
            for j in range(n_dofs):
                plt.subplot(int(f'{n_dofs}1{j+1}'))
                plt.plot(path[j,:], '--k') 

        # without noise
        path = self._generateRndPath(new_start, new_goal, s, n_iter, gain1, gain2, gain3, 0)            
        for j in range(n_dofs):
            plt.subplot(int(f'{n_dofs}1{j+1}'))
            plt.plot(path[j,:], 'r') 

           
        plt.show()  
        
class DMPPathGenerator():    

    def __init__(self, n_dofs: int, n_kernels: int,
                 human_demo_txt: str, camera_angle: float, lpf_alpha: float, dt: float):

        # copy params
        self.dt = dt
        self.camera_angle = camera_angle
        self.txt = human_demo_txt
        self.n_dofs = n_dofs
        self.n_kernels = n_kernels
        self.lpf_alpha = lpf_alpha
        
            
        # DMPs
        self.dmp = MovementPrimitive(n_dofs=3, n_kernels = 30) 
               
    # functions for inside-use ----------------------------------------------------------
    def loadtxt(self):
        # open txt
        f = open(self.txt, "r")
        lines = f.readlines()
        # parse data
        Time = []
        X = []
        Y = []
        Z = []
        for line in lines:
            tmp = line.split("\n")[0]
            tmp = line.split(",")    
            _t = float(tmp[0])
            _x = float(tmp[1])
            _y = float(tmp[2])
            _z = float(tmp[3])
            
            if (abs(_x) + abs(_y) + abs(_z)) > 1e-1:
            
                Time.append(_t)
                X.append(_x)
                Y.append(_y)
                Z.append(_z) 
            
        return copy.deepcopy(Time), copy.deepcopy(X), copy.deepcopy(Y), copy.deepcopy(Z)

    def applyLPF(self, X, Y, Z):
        x_lpf = X[0]
        y_lpf = Y[0]
        z_lpf = Z[0]
        X_lpf = []
        Y_lpf = []
        Z_lpf = []
        for i in range(len(X)):

            if i == 0:
        
                X_lpf.append(X[0])
                Y_lpf.append(Y[0])
                Z_lpf.append(Z[0])
    
            else:
        
                x_lpf = self.lpf_alpha*X[i] + (1. - self.lpf_alpha)*X_lpf[i-1]
                y_lpf = self.lpf_alpha*Y[i] + (1. - self.lpf_alpha)*Y_lpf[i-1]
                z_lpf = self.lpf_alpha*Z[i] + (1. - self.lpf_alpha)*Z_lpf[i-1]
        
                X_lpf.append(x_lpf)
                Y_lpf.append(y_lpf)
                Z_lpf.append(z_lpf)        
        return copy.deepcopy(X_lpf), copy.deepcopy(Y_lpf), copy.deepcopy(Z_lpf)    
    
    def compensateCameraAngle(self, X, Y, Z):
        T = np.array([[1., 0., 0., 0.],
                      [0., np.cos(self.camera_angle), -np.sin(self.camera_angle), 0.],
                      [0, np.sin(self.camera_angle), np.cos(self.camera_angle), 0],
                      [0., 0., 0., 1.]])
        Xp = []
        Yp = []
        Zp = []
        for i in range(len(X)):
            v = [X[i]-X[0], Y[i]-Y[0], Z[i]-Z[0], 1.]
            vp = np.matmul(T, v)
    
            Xp.append(vp[0] + X[0])
            Yp.append(vp[1] + Y[0])
            Zp.append(vp[2] + Z[0]) 
        return copy.deepcopy(Xp), copy.deepcopy(Yp), copy.deepcopy(Zp)
    
    def toRobotAxis(self, X, Y, Z):
        Xr = []
        Yr = []
        Zr = []
        
        for i in range(len(X)):
    
            Xr.append(-Z[i])
            Yr.append(X[i])
            Zr.append(-Y[i])
        return copy.deepcopy(Xr), copy.deepcopy(Yr), copy.deepcopy(Zr)
        
    def process_human_demo(self):
        
        # 1. load data
        T, X, Y, Z = self.loadtxt()
        
        # 2. apply lpf
        X_lpf, Y_lpf, Z_lpf = self.applyLPF(copy.deepcopy(X), copy.deepcopy(Y), copy.deepcopy(Z))

        # 3. compensate camera angle
        Xp, Yp, Zp = self.compensateCameraAngle(copy.deepcopy(X_lpf),copy.deepcopy(Y_lpf),copy.deepcopy(Z_lpf))

        # 4. convert axis
        Xr, Yr, Zr = self.toRobotAxis(copy.deepcopy(Xp),copy.deepcopy(Yp),copy.deepcopy(Zp))
                
        path = np.vstack((Xr, Yr, Zr))
        
        
        # get time
        Td = T[-1] - T[0]
        self.Td = float(Td)
        print("Motion duration:", self.Td)

        return path.copy()

    def process_human_demo_plot(self):
        
        # 1. load data
        _, X, Y, Z = self.loadtxt()
        
        # 2. apply lpf
        X_lpf, Y_lpf, Z_lpf = self.applyLPF(copy.deepcopy(X), copy.deepcopy(Y), copy.deepcopy(Z))

        # 3. compensate camera angle
        Xp, Yp, Zp = self.compensateCameraAngle(copy.deepcopy(X_lpf),copy.deepcopy(Y_lpf),copy.deepcopy(Z_lpf))

        # 4. convert axis
        Xr, Yr, Zr = self.toRobotAxis(copy.deepcopy(Xp),copy.deepcopy(Yp),copy.deepcopy(Zp))
                        
        # plot
        ax = plt.figure(1).add_subplot(projection="3d")
        # original demo
        ax.plot(X, Y, Z, 'k')
        # lpf
        ax.plot(X_lpf, Y_lpf, Z_lpf, '--b')
        # compensated camera angle
        ax.plot(Xp, Yp, Zp, '--m')
        # robot coordinate
        ax.plot(Xr, Yr, Zr, 'r')
        
        ax.set_box_aspect([1.0, 1.0, 1.0])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  
        
        plt.show()

    # learn original paths here
    def initializeDMP(self, test: bool = True):
    
        # load demo data
        path = self.process_human_demo() 
        
        # train DMPs
        self.dmp._setupDMP(self.Td, self.dt, n_kernels=30, y_des=path)
        
        # check trained path
        if test:
            ax = plt.figure(1).add_subplot(projection="3d")
            # original path
            ax.plot(path[0,:], path[1,:], path[2,:], 'k')
            
            # imitate original path
            s = np.linspace(0, 1, int(self.Td/self.dt))
            _path = self.dmp._generate_pos_traj(s)  
            
            # imitated path
            ax.plot(_path[0,:], _path[1,:], _path[2,:], '--r')
            
            # show
            ax.set_box_aspect([1.0, 1.0, 1.0])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")          
            plt.show()        


    def PathGeneration(self, start, goal, txt, save):
    
        # modify goal if goal and start are too close
        if abs(start[0] - goal[0]) < 3: goal[0] = goal[0] + 7.
        if abs(start[1] - goal[1]) < 3: goal[1] = goal[1] + 7.
        
        s = np.linspace(0, 1, int(self.Td/self.dt))
        path = self.dmp._generateRndPath(new_start = start, new_goal = goal, s = s, n_iter = 1000, gain1 = 10., gain2 = 0.3, gain3 = 0.3, noise_amp = 0.)
        
        
        if save:
            print("Saving DMP motion...")
            f = open(txt, "a+")
            for i in range(len(path[0,:])):
                _line = f"{time.time()}, {path[0,i]}, {path[1,i]}, {path[2,i]}\n"
                f.write(_line)
            f.close()
            print("TXT now ready")                       
        
        ax = plt.figure(1).add_subplot(projection="3d")        
        ax.plot(start[0], start[1], start[2], 'ob')
        ax.plot(goal[0], goal[1], goal[2], 'og')
        ax.plot(path[0,:], path[1,:], path[2,:], 'r')

        plt.show()
        
        


class airPressureControl():
    
    def __init__(self, port='/dev/ttyACM0'):
        
        self.ser = serial.Serial(port,
                                 baudrate=9600,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 bytesize=serial.EIGHTBITS,
                                 timeout=0)
        
        time.sleep(1)                                 
        print("Connected to arduino")
        
    def pressureON(self):
    
        self.ser.write(str("b").encode())
        time.sleep(1)
        print("ON")
    
    def pressureOFF(self):  
        
        self.ser.write(str("a").encode())
        time.sleep(1)
        print("OFF")        
        
        
        
        
        
        
        
        
def loadtxt(txt):
    # open txt
    f = open(txt, "r")
    lines = f.readlines()
    # parse data
    Time = []
    X = []
    Y = []
    Z = []
    for line in lines:
        tmp = line.split("\n")[0]
        tmp = line.split(",")    
        _t = float(tmp[0])
        _x = float(tmp[1])
        _y = float(tmp[2])
        _z = float(tmp[3])
            
        if (abs(_x) + abs(_y) + abs(_z)) > 1e-1:
        
            Time.append(_t)
            X.append(_x)
            Y.append(_y)
            Z.append(_z) 
            
    return copy.deepcopy(Time), copy.deepcopy(X), copy.deepcopy(Y), copy.deepcopy(Z)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

