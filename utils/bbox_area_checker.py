
def bbox_area_checker(bboxes):
    flag=0
    for bbox in bboxes:
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        if area<0:
            flag = 1
    return flag
