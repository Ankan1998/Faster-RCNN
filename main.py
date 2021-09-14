import torch
from model.model import get_model
from torch_snippets import *
from data_processing.data_loader import dataloader
from training.train_func import train_batch
from training.val_func import validate_batch
from utils.label_to_indices import label_idx_converter
from tqdm import tqdm

def main_training(n_epochs,data_dir,csv_file):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(csv_file)
    model = get_model(len(label_idx_converter(df['class'].tolist(),0))).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    train_loader,val_loader = dataloader(data_dir,df,4)
    for epoch in tqdm(range(n_epochs)):
        loss=0
        val_loss=0
        _t = len(train_loader)
        for idx, inputs in enumerate(tqdm(train_loader)):
            loss, losses = train_batch(inputs,model, optimizer,device)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
                [losses[k] for k in ['loss_classifier', 'loss_box_reg', 'loss_objectness','loss_rpn_box_reg']]

        print("---------------Validation---------------------")
        _v = len(val_loader)
        for idx,inputs in enumerate(tqdm(val_loader)):
            val_loss, val_losses = validate_batch(inputs, model, optimizer,device)
            val_loc_loss, val_regr_loss, val_loss_objectness, val_loss_rpn_box_reg = \
                [val_losses[k] for k in ['loss_classifier', 'loss_box_reg', 'loss_objectness','loss_rpn_box_reg']]


        if (epoch + 1) % 2 == 0:
            print("training loss {}, validation_loss{}".format(loss.item(),val_loss.item()))


if __name__=="__main__":
    dir = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train'
    csv_file = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train\_annotations.csv'
    main_training(10,dir,csv_file)
