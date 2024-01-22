import os
import cv2
import copy
import numpy as np

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Calibration:
    def __init__(self, path):
        # camera parameters
        f60,f120,r120= path
        cam_param_f60 = []
        cam_param_f120 = []
        cam_param_r120 = []
        
        local = os.getcwd()
        print('cali is here', local)
        with open(f60, 'r') as f:
            data = f.readlines()
            for content in data:
                content_str = content.split()
                for compo in content_str:
                    cam_param_f60.append(float(compo))
        self.camera_matrix_f60 = np.array([[cam_param_f60[0], cam_param_f60[1], cam_param_f60[2]], 
                                       [cam_param_f60[3], cam_param_f60[4], cam_param_f60[5]], 
                                       [cam_param_f60[6], cam_param_f60[7], cam_param_f60[8]]])
        self.dist_coeffs__f60 = np.array([[cam_param_f60[9]], [cam_param_f60[10]], [cam_param_f60[11]], [cam_param_f60[12]]])

        with open(f120, 'r') as f:
            data = f.readlines()
            for content in data:
                content_str = content.split()
                for compo in content_str:
                    cam_param_f120.append(float(compo))
        self.camera_matrix_f120 = np.array([[cam_param_f120[0], cam_param_f120[1], cam_param_f120[2]], 
                                       [cam_param_f120[3], cam_param_f120[4], cam_param_f120[5]], 
                                       [cam_param_f120[6], cam_param_f120[7], cam_param_f120[8]]])
        self.dist_coeffs__f120 = np.array([[cam_param_f120[9]], [cam_param_f120[10]], [cam_param_f120[11]], [cam_param_f120[12]]])

        with open(r120, 'r') as f:
            data = f.readlines()
            for content in data:
                content_str = content.split()
                for compo in content_str:
                    cam_param_r120.append(float(compo))
        self.camera_matrix_r120 = np.array([[cam_param_r120[0], cam_param_r120[1], cam_param_r120[2]], 
                                       [cam_param_r120[3], cam_param_r120[4], cam_param_r120[5]], 
                                       [cam_param_r120[6], cam_param_r120[7], cam_param_r120[8]]])
        self.dist_coeffs__r120 = np.array([[cam_param_r120[9]], [cam_param_r120[10]], [cam_param_r120[11]], [cam_param_r120[12]]])
        print('f60 is ',cam_param_f60)
        print('f120 is ',cam_param_f120)
        print('r120 is ',cam_param_r120)

        self.is_init_mapxy = False

    def init_undistort_maps(self, w, h):
        img_size = (w, h)
        ### The only one camera which is needed is FOV 190 camera
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 0)
        # result_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
       
        self.mapx_f60, self.mapy_f60 = cv2.initUndistortRectifyMap(self.camera_matrix_f60, self.dist_coeffs__f60, None, None, img_size, cv2.CV_32FC1)
        self.mapx_f120, self.mapy_f120 = cv2.initUndistortRectifyMap(self.camera_matrix_f120, self.dist_coeffs__f120, None, None, img_size, cv2.CV_32FC1)
        self.mapx_r120, self.mapy_r120 = cv2.initUndistortRectifyMap(self.camera_matrix_r120, self.dist_coeffs__r120, None, None, img_size, cv2.CV_32FC1)
        self.is_init_mapxy = True
    
    def undistort(self, img,flag):
        # w,h = (img.shape[1], img.shape[0])
        ### The only one camera which is needed is FOV 190 camera
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 0)
        # result_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, newcameramtx)

        if not self.is_init_mapxy:
            self.init_undistort_maps(*img.shape[1::-1])

        if flag == 'f60':
            # result_img = cv2.undistort(img, self.camera_matrix_f60, self.dist_coeffs__f60, None, self.camera_matrix_f60)
            result_img = cv2.remap(img, self.mapx_f60, self.mapy_f60, cv2.INTER_LINEAR)
            return result_img

        if flag == 'f120':
            # result_img = cv2.undistort(img, self.camera_matrix_f120, self.dist_coeffs__f120, None, self.camera_matrix_f120)
            result_img = cv2.remap(img, self.mapx_f120, self.mapy_f120, cv2.INTER_LINEAR)
            return result_img

        if flag == 'r120':
            # result_img = cv2.undistort(img, self.camera_matrix_r120, self.dist_coeffs__r120, None, self.camera_matrix_r120)
            result_img = cv2.remap(img, self.mapx_r120, self.mapy_r120, cv2.INTER_LINEAR)
            return result_img
        
    def IMG_f60_callback(self,msg):
        print('sub')
        if not self.get_f60_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # front_img = cv2.resize(front_img, (self.img_shape))

            #raw
            # self.cur_f120_img['img'] = front_img
            #distorted
            self.cur_f60_img['img'] = self.undistort(front_img,'f60')
            self.cur_f60_img['header'] = msg.header
            self.get_f60_new_image = True
    
    
    def main(self):
        rospy.init_node('test_node')
        self.get_f60_new_image = False
        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_f60_img = {'img':None, 'header':None}
        self.bridge = CvBridge()
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_f60_callback)
        self.pub_f60_det = rospy.Publisher('/mobinha/perception/camera/bounding_box', Image, queue_size=1)

        while not rospy.is_shutdown():
            if self.get_f60_new_image:
                self.sub_f60_img['img'] = self.cur_f60_img['img']
                orig_im_f60 = copy.copy(self.sub_f60_img['img']) 
                det_f60_msg = self.bridge.cv2_to_imgmsg(orig_im_f60, "bgr8")#color
                self.pub_f60_det.publish(det_f60_msg)


if __name__ == "__main__":
    # w, h = front_img.shape[1], front_img.shape[0]
    # img_size = (w, h)
    camera_path = [
                './calibration_data/f60.txt',
                './calibration_data/f120.txt',
                './calibration_data/r120.txt'
                ]
    cal = Calibration(camera_path) 
    cal.main()