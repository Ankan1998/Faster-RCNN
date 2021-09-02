import torch
import torchvision.transforms as transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    return img.to(device).float()