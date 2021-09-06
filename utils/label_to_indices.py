'''
0 --> label to indices
1 --> indices to label
'''
import pandas as pd
from torch_snippets import *

def label_idx_converter(cls_lst, ctype=0):
    df = pd.DataFrame(cls_lst, columns=['label'])
    label2idx = {l: t for t, l in enumerate(df['label'].unique())}
    if ctype == 0:
        return label2idx
    else:
        idx2label = {t: l for l, t in label2idx.items()}
        return idx2label

if __name__=="__main__":
    csv_file = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train\_annotations.csv'
    df = pd.read_csv(csv_file)
    l2i = label_idx_converter(df['class'].tolist(),0)
    print(len(l2i))