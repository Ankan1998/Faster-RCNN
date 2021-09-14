import torch
from model.model import get_model
from torch_snippets import *
from data_processing.data_loader import dataloader
from training.train_func import train_batch
from training.val_func import validate_batch
from utils.label_to_indices import label_idx_converter
from tqdm import tqdm
from saving_loading_model.saving import save_checkpoint
# from saving_loading_model.loading import save_checkpoint

def main_training(n_epochs,data_dir,csv_file, checkpoint_path, best_model_path):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(csv_file)
    model = get_model(len(label_idx_converter(df['class'].tolist(),0))).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    train_loader,val_loader = dataloader(data_dir,df,4)
    for epoch in tqdm(range(n_epochs)):
        total_loss = 0
        total_val_loss = 0
        best_val_loss = 9999999
        is_best = False
        _t = len(train_loader)
        for idx, inputs in enumerate(tqdm(train_loader)):
            loss, losses = train_batch(inputs,model, optimizer,device)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
                [losses[k] for k in ['loss_classifier', 'loss_box_reg', 'loss_objectness','loss_rpn_box_reg']]
            total_loss += loss
        total_avg_training_loss = total_loss/_t
        print("---------------Validation---------------------")
        _v = len(val_loader)
        for idx,inputs in enumerate(tqdm(val_loader)):
            val_loss, val_losses = validate_batch(inputs, model, optimizer,device)
            val_loc_loss, val_regr_loss, val_loss_objectness, val_loss_rpn_box_reg = \
                [val_losses[k] for k in ['loss_classifier', 'loss_box_reg', 'loss_objectness','loss_rpn_box_reg']]
            total_val_loss += val_loss
        total_avg_val_loss = total_val_loss / _v

        if (epoch + 1) % 2 == 0:
            print("training loss {}, validation_loss{}".format(total_avg_training_loss,total_avg_val_loss))

        if total_avg_val_loss < best_val_loss:
            best_val_loss = total_avg_val_loss
            is_best = True
        print(best_val_loss)
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': total_avg_val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, is_best, checkpoint_path, best_model_path)


if __name__=="__main__":
    dir = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train'
    csv_file = r'C:\Users\Ankan\Downloads\Aerial Maritime.v14-black_pad_one_pixel.tensorflow\train\_annotations.csv'
    ckpt_path = r'C:\Users\Ankan\Desktop\Github\Faster-RCNN\ckpt_best_path\latest.pt'
    best_model_path = r'C:\Users\Ankan\Desktop\Github\Faster-RCNN\ckpt_best_path\best.pt'
    main_training(10,dir,csv_file,ckpt_path,best_model_path)
