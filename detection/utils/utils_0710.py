import tensorrt as trt
import copy
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from random import randint
import time
import matplotlib.pyplot as plt
###                    1        2         3         4           5       6       7       8           9           10        11        12          13              14            15
day_label_list = [ 'person', 'bicycle','car','Motorcycle','green3_h', 'bus', 'red3_h','truck','yellow3_h','green4_h', 'red4_h','yellow4_h','redgreen4_h','redyellow4_h','greenarrow4_h', 'red_v','yellow_v','green_v']
night_label_list = ['bicycle', 'bus', 'car', 'green3_h', 'green4_h', 'greenarrow4_h', 'motorcycle', 'red3_h', 'red4_h', 'redgreen4_h', 'traffic light', 'traffic sign', 'truck']

rgb_day_list = [(148, 108, 138), (207, 134, 240), (138, 88, 55), (26, 99, 86), (38, 125, 80), (72, 80, 57), # 6 
                (118, 23, 200), (117, 189, 183), (0, 128, 128), (60, 185, 90), (20, 20, 150), (13, 131, 204), 
                (30, 200, 200), (43, 38, 105), (104, 235, 178), (135, 68, 28), (140, 202, 15), (67, 115, 220)]

rgb_night_list = [(122, 115, 148), (214, 250, 183), (130, 244, 188), (54, 90, 24), (247, 92, 240), (63, 156, 148), 
                (111, 251, 203), (158, 71, 22), (97, 184, 110), (154, 220, 21), (130, 207, 119), 
                (122, 232, 221), (23, 23, 197)]

# Directory of your image files
def judge_bulb(img):
    if img is None:
        print(f"Failed to load image")
    else:
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Calculate the indices for the rows to sum
        row_start = int(hsv.shape[0] / 4)
        row_end = int(hsv.shape[0] * (3 / 4))
        row_indices = np.arange(row_start, row_end)

        # Sum the values of the selected rows
        summed_row = np.sum(hsv[row_indices], axis=0) / len(row_indices)

        # Calculate the average value of the V channel in the calculation range
        average_v = np.mean(hsv[row_start:row_end, :, 2])

        exceed_indices = np.where(summed_row[:,2] > average_v)[0]

        diff = np.diff(exceed_indices)
        temp = np.where(diff > 1)

        intervals = np.split(exceed_indices, temp[0]+1)

        num_intervals = len(intervals)
        total_width = summed_row.shape[0]
        interval_lengths = [len(interval) for interval in intervals]
        interval_proportions = [length / total_width for length in interval_lengths]
        return logic_gate_judge(hsv,intervals,interval_proportions,interval_thresh=0.26,inter_pro_min_thresh=35,inter_pro_max_thresh=100)

def filt_interval(hsv,bulbs,intervals,flag):
    average_list = [np.mean(sublist) for sublist in intervals]
    indices_list = []
    new_flag=False
    if flag == 'four':
        first_indices = [i for i, avg in enumerate(average_list) if avg <= bulbs[0]]
        second_indices = [i for i, avg in enumerate(average_list) if bulbs[0] < avg <= bulbs[1]]
        third_indices = [i for i, avg in enumerate(average_list) if bulbs[1] < avg <= bulbs[2]]
        fourth_indices = [i for i, avg in enumerate(average_list) if bulbs[2] < avg ]
        if len(first_indices) >0:
            indices_list.append(first_indices)
            new_flag=True
        if len(second_indices) >0:
            indices_list.append(second_indices)
            new_flag=True
        if len(third_indices) >0:
            indices_list.append(third_indices)
            new_flag=True
        if len(fourth_indices) >0:
            indices_list.append(fourth_indices)
            new_flag=True
        if new_flag:
            intervals_new = []
            for indices in indices_list:
                arrays_in_range = [intervals[i] for i in indices]
                intervals_new.append(max(arrays_in_range,key=len))
        else:
            intervals_new = intervals
        return intervals_new

    if flag == 'three':
        first_indices = [i for i, avg in enumerate(average_list) if avg <= bulbs[0]]
        second_indices = [i for i, avg in enumerate(average_list) if bulbs[0] < avg <= bulbs[1]]
        third_indices = [i for i, avg in enumerate(average_list) if bulbs[1] > avg]
        if len(first_indices) >0:
            indices_list.append(first_indices)
            new_flag=True
        if len(second_indices) >0:
            indices_list.append(second_indices)
            new_flag=True
        if len(third_indices) >0:
            indices_list.append(third_indices)
            new_flag=True

        if new_flag:
            intervals_new = []
            for indices in indices_list:
                arrays_in_range = [intervals[i] for i in indices]
                intervals_new.append(max(arrays_in_range,key=len))
        else:
            intervals_new = intervals
        return intervals_new

def logic_gate_judge(hsv,intervals,interval_proportions,interval_thresh=0.26,inter_pro_min_thresh=35,inter_pro_max_thresh=100):
        flag_list = ['three','four']
        ### number of bulbs
        if max(interval_proportions)<=interval_thresh:
            logic_gate = [0,0,0,0]
            flag= flag_list[1]
            four_bulbs = [int(hsv.shape[1]/4),int(hsv.shape[1]*2/4),int(hsv.shape[1]*3/4)]
            filt_interval(hsv,four_bulbs,intervals,flag)
        if max(interval_proportions)>interval_thresh:
            logic_gate = [0,0,0]
            flag= flag_list[0]
            three_bulbs = [int(hsv.shape[1]/3),int(hsv.shape[1]*2/3)]
            filt_interval(hsv,three_bulbs,intervals,flag)

        ### histogram with continue
        continue_histo = 5
        if inter_pro_min_thresh <= hsv.shape[1] < inter_pro_max_thresh:
            continue_histo = 4
        elif hsv.shape[1] < inter_pro_min_thresh:
            continue_histo = 2
            
        
        for interval in intervals:
            mid_index = np.mean(interval)
            if flag == 'four' and len(interval) > continue_histo:
                if four_bulbs[2] <= int(mid_index) < hsv.shape[1]:
                    logic_gate[3] = 1
                elif four_bulbs[1] <= int(mid_index) < four_bulbs[2] :
                    logic_gate[2] = 1
                elif four_bulbs[0] <= int(mid_index) < four_bulbs[1] :
                    logic_gate[1] = 1
                elif int(mid_index) < four_bulbs[0] :
                    logic_gate[0] = 1
                else:
                    pass
            if flag == 'three' and len(interval) > continue_histo:
                if three_bulbs[1] <= int(mid_index) < hsv.shape[1]:
                    logic_gate[2] = 1
                elif three_bulbs[0] <= int(mid_index) < three_bulbs[1] :
                    logic_gate[1] = 1
                elif int(mid_index) < three_bulbs[0] :
                    logic_gate[0] = 1
                else:
                    pass

        return logic_gate

def hsv_judge(img,da0):
    new_class = -1  # 초기값을 -1로 설정
    logic_gate = judge_bulb(img)
    new_class = name_mapping(logic_gate,da0)

    return new_class


def name_mapping(logic_gate,da0):
    ### day time
    if da0 == 'day':
        logic_gate_list = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0],[0, 1, 0], [1, 0, 0], [0, 0, 1], [0,0,1,1]]
        name_list = ['red4_h', 'yellow4_h', 'green4_h', 'redyellow4_h', 'redgreen4_h','yellow3_h', 'red3_h', 'green3_h','greenarrow4_h']
    
    ### night time
    if da0 == 'night':
        logic_gate_list = [[1, 0, 0, 0], [0, 0, 0, 1],  [1, 0, 1, 0], [1, 0, 0], [0, 0, 1], [0,0,1,1]]
        name_list = ['red4_h',  'green4_h',  'redgreen4_h', 'red3_h', 'green3_h','greenarrow4_h']
    
    if logic_gate in logic_gate_list:
        index = logic_gate_list.index(logic_gate)
        new_class = name2eption(name_list[index],da0)  # 해당 index의 이름을 문자열로 설정        

        return new_class

def name2eption(class_name,da0):
    if da0 == 'day':
        index = day_label_list.index(class_name)
    if da0 == 'night':
        index = night_label_list.index(class_name)
    return index

class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})


    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    ### image steam
    def steam_inference(self, origin_img, conf=0.5, end2end=False,day_night='day'):
        print('DETECTION...')
        t1 =time.time()
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        print('Pre-process is :',round((time.time()-t1)*1000,2),' ms')
        t3 = time.time()
        data = self.infer(img)
        print('Model is :',round((time.time()-t3)*1000,2),' ms')
        t2 = time.time()
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            # box_result = zero_vis(origin_img, final_boxes, final_scores, final_cls_inds,
            origin_img, box_result = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names,da0=day_night)
        print('Post-process is :',round((time.time()-t2)*1000,2),' ms')
        
        return origin_img,box_result

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def iou_yolo(box1, box2):
    ### if wanna run this file then use below two

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    union_area = area_box1 + area_box2 - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area
    
    return iou,box2

def jurdge_boxes(boxes,cls_id):
    for i, item1 in enumerate(boxes):
        for j, item2 in enumerate(boxes):
            if i != j:
                iou_source, box = iou_yolo(item1,item2)
                if iou_source > 0.5:
                    cls_id[i] = max(cls_id[i],cls_id[j]) 
    return cls_id

def filter_boxes(boxes,scores, cls_ids, fix_flag, conf=0.5):
    if fix_flag:
        fix_boxes=[]
        fix_cls=[]
        fix_scores=[]
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            if cls_id in [4,6,8,9,10,11,12,13,14]:
                score = scores[i]
                if score < conf:
                    continue
                fix_boxes.append(box)
                fix_cls.append(cls_id)
                fix_scores.append(score)
        return fix_boxes, fix_cls, fix_scores
    else:
        return boxes,cls_ids,scores


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None,da0="day"):
    box_result = []
    height, weight, _ = img.shape
    tl = 3 or round(0.002 * (height + weight) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness

    #### Parames
    cur_img = copy.copy(img)
    fix_list = [True,False]
    fix = fix_list[0]

    #### Parames

    fix_boxes, fix_cls, fix_scores = filter_boxes(boxes, scores, cls_ids, fix_flag = fix,conf=0.2)
    fix_cls = jurdge_boxes(fix_boxes,fix_cls)
    
    widodod = 40
    chang_line = widodod + len(fix_boxes)*25
    count = 0
    for i in range(len(fix_boxes)):
        
        box = fix_boxes[i]
        cls_id = int(fix_cls[i])
        
        score = fix_scores[i]

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        if  x0 < 0 :
            x0 = 0
        if  y0 < 0 :
            y0 = 0


        cls_id_wo_hsv = class_names[cls_id]
        ### hsv ###

        if da0 == 'day':
            traffic_4 = [9,10,11,12,13,14] # day
            traffic_3 = [4,6,8] # day
            _COLORS = rgb_day_list

        if da0 == 'night':
            traffic_4 = [4,5,8,9] # night
            traffic_3 = [3,7] # night
            _COLORS = rgb_night_list

        if cls_id in traffic_4 or cls_id in traffic_3:
            bbox_image = cur_img[y0:y1, x0:x1]           
            # if cls_id == 12:
            #     FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) 
            #     cv2.imwrite(f'/home/cvlab-swlee/Desktop/temp/raw/{FOLDER_NAME}_{count}.jpg', bbox_image)
            #     count+=1
            #### HSV judge
            if cls_id in traffic_4 : 
                # print(bbox_image.shape)
                cls_c = hsv_judge(bbox_image,da0)
                if cls_c == None:
                    pass
                else:
                    cls_id= cls_c
            if cls_id in traffic_3:
                cls_c = hsv_judge(bbox_image,da0)
                if cls_c == None:
                  FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) 
                  cv2.imwrite(f'/home/inha/ros_vision/src/log_data/{cls_id}_{FOLDER_NAME}_{count}.jpg', bbox_image)
                  count+=1
                else:
                    cls_id= cls_c

        ### hsv ###
        box_area = (x1-x0)*(y1-y0)
        box_result.append([cls_id,box_area,score,[x0,y0,x1,y1]])
        
        c1, c2 = (x0,y0), (x1,y1)
        cv2.rectangle(cur_img, c1, c2, _COLORS[cls_id], thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(cur_img, c1, c2, _COLORS[cls_id], -1, cv2.LINE_AA)  # filled
        cv2.putText(cur_img, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        cls_hsv = class_names[cls_id]
        gab = i*25
        cv2.putText(cur_img, f'wo_hsv is :{cls_id_wo_hsv}', (10, widodod + gab), 0, tl / 3, [0, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(cur_img, f'w_hsv is :{cls_hsv}', (10, chang_line + gab), 0, tl / 3, [225, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
        
        img = cur_img

    return img,box_result
