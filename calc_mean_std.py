import numpy as np
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from skimage import io
import os

tfs = transforms.Compose([
    iaa.Sequential([
                    iaa.Sometimes(0.5, iaa.Fliplr(1.0)),
                    iaa.Sometimes(0.5, iaa.MultiplyBrightness((0.3, 1.3))),
                    iaa.Sometimes(0.2, iaa.ChangeColorTemperature((4300, 6000))),
                    #iaa.Sometimes(0.9, iaa.Affine(rotate=(-180, 180), shear=(-6, 6))),
    ]).augment_image,
    transforms.ToTensor()
])


class CustomDataset(Dataset):
    def __init__(self, n_images, n_classes=15, transform=None):
        self.images = []
        
        self.transform = transform

        test_path = "/content/gdrive/My Drive/Arirang/data/test/images"
        file_list = os.listdir(test_path)
        file_list_png = [file for file in file_list if file.endswith(".png")]
        
        for idx, filename in enumerate(file_list_png):
            self.images.append( os.path.join(test_path, filename)) 

        train_path = "/content/gdrive/My Drive/Arirang/data/train/coco_all/train2017"
        file_list = os.listdir(train_path)
        file_list_png = [file for file in file_list if file.endswith(".png")]
        
        for idx, filename in enumerate(file_list_png):
            #self.images.append( io.imread(os.path.join(train_path, filename)) )
            self.images.append( os.path.join(train_path, filename)) 

        self.targets = np.random.randn(len(self.images)-1, n_classes)

    def __getitem__(self, item):

        image = io.imread( self.images[item] )
        target = self.targets[item]

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.images)


custom_ds = CustomDataset(n_images=50, n_classes=10, transform=tfs)
loader = DataLoader(custom_ds, batch_size=64,
                       num_workers=1, pin_memory=True)



print(custom_ds[3])


#############

len(custom_ds.images)

#############


mean = 0.0
for _ in range(10):
    for images, _ in loader:
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
mean = mean / len(loader.dataset)

var = 0.0
for _ in range(10):
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(loader.dataset)*224*224))

#############

 mean * 255, std *255
