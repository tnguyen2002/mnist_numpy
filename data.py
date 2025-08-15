from torchvision import transforms, datasets
import torch 
from torch.utils.data import Dataset, random_split



# class MNISTNumpyDataset(Dataset):
#     def __init__(self, img_dir, transforms=None):
#         self.img_labels = 

torch.manual_seed(12)
def get_dataset():
    preprocess_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset, test_dataset = random_split(datasets.MNIST("./data", transform=preprocess_transforms, download=True), [.9, .1])
    return train_dataset, test_dataset
    
    

if __name__ == "__main__":
    get_dataset()
