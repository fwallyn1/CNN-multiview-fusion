import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import random
data_config={'ROOT_DIR' : '/home/onyxia/work/data/Images_ROI/Images_ROI_Typ_for_multiview_echant/',
'VALID_SPLIT' : 0.2,
'TEST_SPLIT' : 0.2,
'IMAGE_SIZE' : 224, # Image size of resize when applying transforms.
'BATCH_SIZE' : 32,
'NUM_WORKERS' : 0}

torch.manual_seed(0)
torch.cuda.manual_seed(0)
class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = sorted(os.listdir(main_dir))
        
    def __len__(self):
        return len(self.all_imgs)
    
    def get_img_path(self,idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        return img_loc
    
    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image

class MultiInputLoader():
    """
    Class for loading the two views data
    """
    def __init__(self,data_config=data_config,pretrained=True):
        self.root_dir = data_config["ROOT_DIR"]
        self.valid_split = data_config["VALID_SPLIT"]
        self.test_split = data_config["TEST_SPLIT"]
        self.image_size = data_config["IMAGE_SIZE"]
        self.batch_size = data_config["BATCH_SIZE"]
        self.num_workers = data_config["NUM_WORKERS"]
        self.pretrained=pretrained

    def get_train_transform(self):
        train_transform = transforms.Compose([
        transforms.Resize((self.image_size, self.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        self.normalize_transform()
        ])
        return train_transform

    # Validation transforms
    def get_valid_transform(self):
        valid_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            self.normalize_transform()
        ])
        return valid_transform

    # Test transforms
    def get_test_transform(self):
        test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            self.normalize_transform()
        ])
        return test_transform

    # Image normalization transforms.
    def normalize_transform(self):
        if self.pretrained: # Normalization for pre-trained weights.
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        else: # Normalization when training from scratch.
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        return normalize


    def get_datasets(self):
        """
        Returns datasets for each view for purpose : train/validation/test
        """
        if self.test_split + self.valid_split == 0:
            data = []
            for view in os.listdir(self.root_dir):
                data.append(CustomDataSet(
                    self.root_dir + 'RECTO', 
                    transform=(self.get_test_transform())
                ))
            return data
        data = []
        data_train = []
        data_valid = []
        data_test = []
        for view in os.listdir(self.root_dir):
            data.append(datasets.ImageFolder(
                self.root_dir + view, 
                transform=(self.get_train_transform())
            ))

        # Radomize the data indices.
        train_val_idx, test_idx = train_test_split(np.arange(len(data[0])),
                                             test_size=self.test_split,
                                             random_state=0,
                                             shuffle=True,
                                             stratify=data[0].targets)

        train_idx, val_idx = train_test_split(train_val_idx,
                                             test_size=self.valid_split,
                                             random_state=0,
                                             shuffle=True,
                                             stratify= np.array(data[0].targets)[train_val_idx])
        # Training and validation sets.
        for i,_ in enumerate(os.listdir(self.root_dir)):
            data_train.append(Subset(data[i], train_idx))
            data_valid.append(Subset(data[i],val_idx))
            data_test.append(Subset(data[i],test_idx))

        return data_train, data_valid, data_test, data[0].classes


    def get_data_loaders(self):
        """
        Prepares the training and validation data loaders.
        :param dataset_train: The training dataset.
        :param dataset_valid: The validation dataset.
        Returns the training and validation data loaders.
        """
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        if self.test_split + self.valid_split == 0:
            data = self.get_datasets()
            data_loaders = []
            for d in data:
                data_loaders.append(DataLoader(
                d, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers,
                    worker_init_fn=seed_worker,generator=g
                ))
            return data_loaders
        
        data_train, data_valid, data_test, dataset_class = self.get_datasets()

        data_train_loaders = []
        data_valid_loaders = []
        data_test_loaders = []
        for d_train, d_valid, d_test in zip(data_train, data_valid, data_test):
            data_train_loaders.append(DataLoader(
                d_train, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers,
                worker_init_fn=seed_worker,generator=g
            ))
            data_valid_loaders.append(DataLoader(
                d_valid, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers,
                worker_init_fn=seed_worker,generator=g
            ))
            data_test_loaders.append(DataLoader(
                d_test, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers,
                worker_init_fn=seed_worker,generator=g
            ))
        
        return data_train_loaders, data_valid_loaders, data_test_loaders
