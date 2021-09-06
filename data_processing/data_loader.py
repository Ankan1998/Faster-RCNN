from sklearn.model_selection import train_test_split
import pandas as pd
from data_processing.data_preparation import AerialMaritimeDataset
from torch.utils.data import DataLoader

def dataloader(data_dir, csv_file, batch_size=4):

    df = pd.read_csv(csv_file)
    train_ids, val_ids = train_test_split(df['filename'].unique(),test_size=0.1)
    train_df, val_df = df[df['filename'].isin(train_ids)], df[df['filename'].isin(val_ids)]

    train_ds = AerialMaritimeDataset(data_dir,train_df)
    val_ds = AerialMaritimeDataset(data_dir,val_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn)

    val_loader =  DataLoader(val_ds, batch_size=batch_size, collate_fn=val_ds.collate_fn)
    return train_loader, val_loader

if __name__=="__main__":

    dir = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train'
    csv_file = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train\_annotations.csv'
    train_dl, val_dl = dataloader(dir,csv_file,1)
    for i in train_dl:
        print(i)
        break

