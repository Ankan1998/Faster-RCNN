import torch

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss.item()