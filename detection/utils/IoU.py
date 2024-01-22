def iou_yolo(box1, box2, img_width, img_height):
    ### if wanna run this file then use below two
    box1 = yolo_to_pixel(box1, img_width, img_height)
    box2 = yolo_to_pixel(box2, img_width, img_height)

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


def yolo_to_pixel(box, img_width, img_height):
    x_center, y_center, width, height = box
    x1 = int((x_center - width/2) * img_width)
    y1 = int((y_center - height/2) * img_height)
    x2 = int((x_center + width/2) * img_width)
    y2 = int((y_center + height/2) * img_height)
    return (x1, y1, x2, y2)


if __name__ == '__main__':
    box1 = (0.5, 0.5, 0.2, 0.2)
    box2 = (0.5, 0.5, 0.2, 0.2)
    # box2 = (0.6, 0.6, 0.3, 0.3)
    img_width = 1280
    img_height = 720
    iou_score = iou_yolo(box1, box2,img_width, img_height)
    print("IoU score:", iou_score)

