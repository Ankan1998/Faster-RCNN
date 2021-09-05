from sklearn.model_selection import train_test_split
import pandas as pd
from data_processing.data_preparation import AerialMaritimeDataset
from torch.utils.data import DataLoader
'''
0 --> train
else --> validation
'''
def dataloader(data_dir, csv_file, type=0):

    df = pd.read_csv(csv_file)
    train_ids, val_ids = train_test_split(df['filename'].unique(),test_size=0.1)
    train_df, val_df = df[df['filename'].isin(train_ids)], df[df['filename'].isin(val_ids)]

    train_ds = AerialMaritimeDataset(data_dir,train_df)
    val_ds = AerialMaritimeDataset(data_dir,val_df)
    if type == 0:
        return DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn)
    else:
        return DataLoader(val_ds, batch_size=4, collate_fn=val_ds.collate_fn)
