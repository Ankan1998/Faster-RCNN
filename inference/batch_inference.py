from torch_snippets import *

from data_processing.data_loader import dataloader
from utils.label_to_indices import label_idx_converter
from utils.preprocess_bbox_from_output import decode_bbox_output
from model.model import get_model
from utils.label_to_indices import *

def batch_inference(model,test_dataloader,indices2label):
    model.eval()
    for idx, (images, targets) in enumerate(test_dataloader):
        images = [im for im in images]
        outputs = model(images)
        for ix, output in enumerate(outputs):
            bbs, confs, labels = decode_bbox_output(output,indices2label)
            show(images[ix].cpu().permute(1, 2, 0), bbs=bbs, texts=labels, sz=5)

if __name__=="__main__":
    dir = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train'
    csv_file = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train\_annotations.csv'
    df = pd.read_csv(csv_file)
    model = get_model(len(label_idx_converter(df['class'].tolist(), 0))).to(device)
    i2l = label_idx_converter(df['class'].tolist(), 1)
    train_dl, val_dl = dataloader(dir, df, 2)

    batch_inference(model,val_dl,i2l)