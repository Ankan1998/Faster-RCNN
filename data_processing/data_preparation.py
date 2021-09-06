import os
from torch.utils.data import Dataset, DataLoader
from torch_snippets import *
import cv2
import pandas as pd
import warnings

from utils.label_to_indices import label_idx_converter
from utils.preprocess import preprocess_image
from utils.bbox_area_checker import bbox_area_checker
warnings.filterwarnings("ignore")


class AerialMaritimeDataset(Dataset):
    def __init__(self, data_dir, dataframe, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.df = dataframe
        self.label2idx = label_idx_converter(self.df['class'].tolist(), ctype=0)
        self.unique_imgs = self.df['filename'].unique()
        self.w = 224
        self.h = 224

    def __len__(self):
        return len(self.unique_imgs)

    def __getitem__(self, idx):
        image_id = self.unique_imgs[idx]
        image_path = f'{self.data_dir}/{image_id}'
        # Convert BGR to RGB
        img = cv2.imread(image_path, 1)[..., ::-1]
        scale_w = self.w / img.shape[0]
        scale_h = self.h / img.shape[1]
        img = cv2.resize(img,(self.w, self.h)) / 255.
        data = self.df[self.df['filename'] == image_id]
        labels = data['class']
        boxes = data['xmin,ymin,xmax,ymax'.split(',')].values
        boxes[:,[0,2]] = boxes[:,[0,2]]*scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
        boxes = boxes.astype(np.uint16).tolist()
        flag,bbxidx = bbox_area_checker(boxes)
        if flag !=0:
            print(bbxidx)
            print(boxes)
            print(image_id)
        target = {}
        target["boxes"] = torch.Tensor(boxes).float()
        target["labels"] = torch.Tensor([self.label2idx[i] for i in labels]).long()
        img = preprocess_image(img)
        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))

if __name__=="__main__":
    dir = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train'
    csv_file = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train\_annotations.csv'
    df = pd.read_csv(csv_file)
    ds = AerialMaritimeDataset(dir,df)

    for i in range(44):
        im, target = ds[i]
