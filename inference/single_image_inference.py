from torch_snippets import *

from data_processing.data_loader import dataloader
from utils.label_to_indices import label_idx_converter
from utils.preprocess_bbox_from_output import decode_bbox_output
from model.model import get_model
from utils.preprocess import preprocess_image

def image_inference(model,img,indices2label):
    img = cv2.imread(img_path, 1)[..., ::-1]
    img = cv2.resize(img, (224, 224)) / 255.
    img = preprocess_image(img)
    model.eval()
    images = [img]
    outputs = model(images)
    for ix, output in enumerate(outputs):
        bbs, confs, labels = decode_bbox_output(output,indices2label)
        show(images[ix].cpu().permute(1, 2, 0), bbs=bbs, texts=labels, sz=5)

if __name__=="__main__":
    img_path = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train\DJI_0273_JPG.rf.2919634e5e938fb3f955e4c0ceac437a.jpg'
    csv_file = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train\_annotations.csv'
    df = pd.read_csv(csv_file)
    model = get_model(len(label_idx_converter(df['class'].tolist(), 0))).to(device)
    i2l = label_idx_converter(df['class'].tolist(), 1)
    image_inference(model,img_path,i2l)

