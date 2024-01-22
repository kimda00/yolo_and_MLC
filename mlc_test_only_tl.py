import os
import time
import math
import numpy as np
import copy
import cv2
import argparse
import pycuda.driver as cuda
from PIL import Image as PLIimage
from io import BytesIO
from datetime import datetime

from detection.det_infer import Predictor
from detection.calibration import Calibration

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
###

from sort import *

class Camemra_Node:
    def __init__(self,args,day_night):
        rospy.init_node('Camemra_node')
        self.model = load_model('detection/weights/500_weights.h5')
        self.args = args
        
        # self.class_names = ['person', 'bicycle','car','bus','motorcycle','truck','tl_v' ,'tl_p','traffic_sign','traffic_light']
        
        self.rgb_day_list = [(148, 108, 138), (207, 134, 240), (138, 88, 55), (26, 99, 86), (38, 125, 80), (72, 80, 57), # 6 
                        (118, 23, 200), (117, 189, 183), (0, 128, 128), (60, 185, 90), (20, 20, 150), (13, 131, 204), 
                        (30, 200, 200), (43, 38, 105), (104, 235, 178), (135, 68, 28), (140, 202, 15), (67, 115, 220),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30)]
        
        self.class_names = ['person', 'bicycle','car','bus','motorcycle','truck', 'green', 'red', 'yellow',
                            'red_arrow', 'red_yellow', 'green_arrow','green_yellow','green_right',
                            'warn','black','tl_v', 'tl_p', 'traffic_sign', 'traffic_light']
        
        self.tls_states = ['person', 'bicycle','car','bus','motorcycle','truck', 'green', 'red', 'yellow','red_arrow', 'red_yellow', 
                           'green_arrow','green_yellow','green_right','black' ,'unknown']
        
        self.rgb_list = [(148, 108, 138), (207, 134, 240), (138, 88, 55), (26, 99, 86), (38, 125, 80), (72, 80, 57), # 6 
                        (118, 23, 200), (117, 189, 183), (0, 128, 128), (60, 185, 90), (20, 20, 150)]

        self.last_observed_light = []
        self.go_signals = [4, 9, 12, 14, 17]
        self.allowed_unrecognized_frames = 0
        self.last_observed_time = 0
        self.baseline_boxes = [960,270]
        self.detect_in_roi_only = True

        sort_max_age = 15
        sort_min_hits = 2
        sort_iou_thresh = 0.1
        self.sort_tracker_f60 = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
        local = os.getcwd()
        # print('now is here', local)
        camera_path = ['./detection/calibration_data/f60.txt',
                    './detection/calibration_data/f120.txt',
                    './detection/calibration_data/r120.txt']
        # self.calib = Calibration(camera_path)

        ###fov 60
        self.get_f60_new_image = False
        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_f60_img = {'img':None, 'header':None}
        self.bbox_f60 = PoseArray()
        
        ### Det Model initiolization
        self.day_night = day_night
        self.det_pred = Predictor(engine_path=args.det_weight , day_night=day_night)
        self.pub_od_f60 = rospy.Publisher('/mobinha/perception/camera/bounding_box', PoseArray, queue_size=1)
        rospy.Subscriber('/gmsl_camera/dev/video1/compressed', CompressedImage, self.IMG_f60_callback)
        ##########################
        self.pub_f60_det = rospy.Publisher('/det_result/f60', Image, queue_size=1)
        
        self.bridge = CvBridge()
        self.is_save =False
        self.sup = []

        self.real_cls_hist = 0



    def image_process(self,img,flag):
        if flag == 'f60' :
            box_result_f60 = self.det_pred.steam_inference(img,conf=0.1, end2end='end2end',day_night=self.day_night) 
        
            # roi_x, roi_y, roi_width, roi_height = 100, 600, 1700, 300
            # class_detected = any(box[0] in [6, 7, 9] and (roi_x <= (box[3][0] + box[3][2]) / 2 <= roi_x + roi_width) and (roi_y <= (box[3][1] + box[3][3]) / 2 <= roi_y + roi_height) for box in box_result_f60)
            # if class_detected:
            #     self.detect_in_roi_only = False
            # else:  
            #     self.detect_in_roi_only = True
            # if self.detect_in_roi_only:
            #     box_result_f60 = [box for box in box_result_f60 if (roi_x <= (box[3][0] + box[3][2]) / 2 <= roi_x + roi_width) and (roi_y <= (box[3][1] + box[3][3]) / 2 <= roi_y + roi_height)]

            original_img = img.copy()
            # 디텍션된 박스들 중에서 교통 신호등 객체만을 추출
            tl_boxes = self.get_traffic_light_objects(box_result_f60)
            if tl_boxes:
                box=tl_boxes[0][3]
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
            
                # 각 객체에 대해 MLC 실행 및 클래스 업데이트
                for i, box in enumerate(box_result_f60):
                    if box in tl_boxes:
                        cropped_img = original_img[y0:y1, x0:x1]  # 객체의 바운딩 박스로 이미지 자르기
                        mlc_class_id = self.run_testing(cropped_img)[0]  # MLC 실행
                        box_result_f60[i][0] = mlc_class_id  # 클래스 번호 업데이트
           
                box_result_f60 = self.update_tracking(box_result_f60, flag)
                # cv2.rectangle(box_result_f60, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2) # roi 박스 확인 
            
                if len(tl_boxes)>0 :
                    self.real_cls_hist = tl_boxes[0][0]

                filter_img_f60, one_tl, c1, c2 ,tl,tf= self.draw_img_filter(original_img, tl_boxes, original_img) 
            
                cv2.putText(filter_img_f60,self.tls_states[mlc_class_id], (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

                overlayed_img = self.overlay_cropped_tl(filter_img_f60, one_tl) #필터링한거에서 가져오기
                self.det_pubulissher(overlayed_img, tl_boxes,flag)
            else:
                self.det_pubulissher(original_img, tl_boxes,flag)

    def draw_img_filter(self,img, boxes,original_img):
      
        height, weight, _ = img.shape
        tl = 3 or round(0.002 * (height + weight) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        cur_img = copy.copy(img)
        cropped_img = None 

        c1, c2 = (0,0),(0,0)

        if len(boxes) > 0 :
            box = boxes[0][3]
            cls_id = boxes[0][0]
            score = boxes[0][2]

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            x0 = 0 if  x0 < 0 else x0
            y0 = 0 if  y0 < 0 else y0

            _COLORS = self.rgb_day_list

            cropped_img = original_img[y0:y1, x0:x1]

            c1, c2 = (x0,y0), (x1,y1)
            cv2.rectangle(cur_img, c1, c2, _COLORS[cls_id], thickness=tl, lineType=cv2.LINE_AA)

            tf = max(tl - 1, 1)  # font thickness
            text = '{}:{:.1f}%'.format(self.tls_states[cls_id], score * 100)
            t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
           
        img = cur_img
        return img, cropped_img, c1, c2, tl, tf 

    def get_traffic_light_objects(self,bbox_f60):
        traffic_light_obs = []

        if len(bbox_f60) > 0:
            for traffic_light in bbox_f60:
                if traffic_light[2] > 0.2:  # if probability exceed 20%
                    traffic_light_obs.append(traffic_light)
        # sorting by size
        traffic_light_obs = self.get_one_boxes(traffic_light_obs)
        return traffic_light_obs
    
    def get_one_boxes(self,traffic_light_obs):
        traffic_light_obs = self.filtered_obs(traffic_light_obs)
        if len(traffic_light_obs) >0:
            # print(f'traffic_light_obs is {traffic_light_obs}')            
            boxes = np.array(traffic_light_obs)[:,3]
            distances = [math.sqrt(((box[0] + box[2]) / 2 - self.baseline_boxes[0]) ** 2 + ((box[1] + box[3]) / 2 - self.baseline_boxes[1]) ** 2) for box in boxes]
            areas = [((box[2] - box[0]) * (box[3] - box[1])) for box in boxes]
            weights = [0.6*(distances[x] / max(distances)) + 0.4*(1 - (areas[x] / max(areas)))  for x in range(len(boxes))]
            result_box = [traffic_light_obs[weights.index(min(weights))]]
            return result_box
        else:
            result_box = traffic_light_obs
            return result_box
        
    def filtered_obs(self,traffic_light_obs):
        new_obs = []
        for obs in traffic_light_obs:
            if obs[0] == 15 and self.real_cls_hist != 0:
                obs[0] = self.real_cls_hist
            new_obs.append(obs)
        return new_obs

    def IMG_f60_callback(self,msg):
        if not self.get_f60_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.cur_f60_img['img'] = front_img
            self.cur_f60_img['header'] = msg.header
            self.get_f60_new_image = True

    def pose_set(self,bboxes,flag):
        bbox_pose = PoseArray()

        for bbox in bboxes:
            pose = Pose()
            pose.position.x = bbox[0]# box class

            pose.position.y = bbox[1]# box area
            pose.position.z = bbox[2]# box score
            pose.orientation.x = bbox[3][0]# box mid x
            pose.orientation.y = bbox[3][1]# box mid y
            pose.orientation.z = bbox[3][2]# box mid y
            pose.orientation.w = bbox[3][3]# box mid y
            bbox_pose.poses.append(pose)

        if flag == 'f60':

            self.pub_od_f60.publish(bbox_pose)


    def det_pubulissher(self,det_img,det_box,flag):
        if flag =='f60':
            det_f60_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pose_set(det_box,flag)
            self.pub_f60_det.publish(det_f60_msg)
    
    def update_tracking(self,box_result,flag):
        update_list = []
        if len(box_result)>0:
            cls_id = np.array(box_result)[:,0]
            areas = np.array(box_result)[:,1]
            scores = np.array(box_result)[:,2]
            boxes = np.array(box_result)[:,3]
            dets_to_sort = np.empty((0,6))

            for i,box in enumerate(boxes):
                x0, y0, x1, y1 = box
                cls_name = cls_id[i]
                dets_to_sort = np.vstack((dets_to_sort, 
                            np.array([x0, y0, x1, y1, scores[i], cls_name])))
            if flag == 'f60':
                tracked_dets = self.sort_tracker_f60.update(dets_to_sort)
                tracks = self.sort_tracker_f60.getTrackers()
        
            bbox_xyxy = tracked_dets[:,:4]
            categories = tracked_dets[:, 4]

            new_areas = (bbox_xyxy[:,2] - bbox_xyxy[:,0]) * (bbox_xyxy[:,3] - bbox_xyxy[:,1])
            update_list = [[int(categories[x]),new_areas[x],scores[x],bbox_xyxy[x]] for x in range(len(tracked_dets)) ]

        else:
            tracked_dets = self.sort_tracker_f60.update()

        return update_list
    
    def resize_image(self, image, width, height):
        return cv2.resize(image, (width,height))
    
    def overlay_cropped_tl(self, origin_img, cropped_img, width = 100, height = 40, crop_save_folder = '/home/da0/catkin_ws/src/kimda0_ws_test/cropped'):
        if not os.path.exists(crop_save_folder):
            os.makedirs(crop_save_folder)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.png"

        save_path = os.path.join(crop_save_folder, filename)
        cv2.imwrite(save_path, cropped_img)

        resized_cropped_tl = self.resize_image(cropped_img, width, height)
        h, w, _ = resized_cropped_tl.shape
        origin_img[0:h, 0:w] = resized_cropped_tl
        return origin_img
    
    def run_testing(self, cropped_img):
        img_width = 150
        img_height = 150

        data = pd.read_csv('500_labels.csv')
        classes = data.columns[2:]

        img_resized = cv2.resize(cropped_img, (img_width, img_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = image.img_to_array(img_rgb)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, img_width, img_height, 3)
        
        y_prob = self.model.predict(img_array)
        top3 = np.argsort(y_prob[0])[:-3:-1]
        predictions = [classes[i] for i in top3]
        percentage = [y_prob[0][i] for i in top3]
        
        over_percent = [p for p in percentage if p >= 0.5]

        # percentage가 10% 이상인 예측이 하나만 있고, 그 예측이 'red'인 경우
        if len(over_percent) == 1 and predictions[0] == 'red':
            return 7 , predictions, percentage 
        if len(over_percent) == 1 and predictions[0] == 'green':
            return 6, predictions, percentage 
        if len(over_percent) == 1 and predictions[0] == 'yellow':
            return 8, predictions, percentage 
        if len(over_percent) == 1 and predictions[0] == 'black':
            return 14, predictions, percentage 

        # percentage가 10% 이상인 예측이 두 개 있고, 가장 높은 확률의 예측이 'red'이며, 다음으로 높은 확률의 예측이 'arrow'인 경우
        if len(over_percent) == 2 and predictions[0] == 'red' and predictions[1] == 'arrow':
            return 9, predictions, percentage  
        if len(over_percent) == 2 and predictions[0] == 'green' and predictions[1] == 'arrow':
            return 11, predictions, percentage 
        if len(over_percent) == 2 and predictions[0] == 'red' and predictions[1] == 'yellow':
            return 10, predictions, percentage 
        if len(over_percent) == 2 and predictions[0] == 'green' and predictions[1] == 'yellow':
            return 12, predictions, percentage 
        
        else:
            return 15, predictions, percentage
        
    def main(self):
        rate = rospy.Rate(100)  

        while not rospy.is_shutdown():
            if self.get_f60_new_image:
                self.sub_f60_img['img'] = self.cur_f60_img['img']
                orig_im_f60 = copy.copy(self.sub_f60_img['img']) 
                self.image_process(orig_im_f60,'f60')
                self.get_f60_new_image = False
  
            rate.sleep()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--end2end", default=False, action="store_true",help="use end2end engine")
    
    day_night_list = ['day','night']
    day_night = day_night_list[0]
    if day_night == 'day':
        # print('*'*12)
        # print('*** DAY TIME ***')
        # print('*'*12)
        #parser.add_argument('--det_weight', default="./detection/weights/yolov7x_flicker_with_nms.trt")  ### end2end
        # parser.add_argument('--det_weight', default="./detection/weights/add120_1017.trt")  ### end2end 
        # parser.add_argument('--det_weight', default="./detection/weights/incheon_dataset.trt")  ### end2end 
        parser.add_argument('--det_weight', default="./detection/weights/only_tl_with_nms.trt")  ### end2end 
        # parser.add_argument('--det_weight', default="./detection/weights/withnms.trt")  ### end2end 



    if day_night == 'night':
        # print('*'*12)
        # print('*** NIGHT TIME ***')
        # print('*'*12)
        # parser.add_argument('--det_weight', default="./detection/weights/230615_night_songdo_no_nms_2.trt")  ### end2end
        parser.add_argument('--det_weight', default="./detection/weights/230615_night_songdo_no_nms_2.trt")  ### end2end

    args = parser.parse_args()
    
    camemra_node = Camemra_Node(args,day_night)
    camemra_node.main()

