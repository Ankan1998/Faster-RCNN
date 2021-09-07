from torch_snippets import *
from utils.preprocess_bbox_from_output import decode_bbox_output

def image_inference(model,image,indices2label):
    model.eval()
    images = [image]
    outputs = model(images)
    for ix, output in enumerate(outputs):
        bbs, confs, labels = decode_bbox_output(output,indices2label)
        show(images[ix].cpu().permute(1, 2, 0), bbs=bbs, texts=labels, sz=5)