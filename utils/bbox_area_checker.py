
def bbox_area_checker(bboxes):
    flag=0
    bbxidx = -9999
    for idx, bbox in enumerate(bboxes):
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        if area<0 or area ==0:
            flag = 1
            bbxidx = idx
    return flag,bbxidx
