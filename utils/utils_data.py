import torch, os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from einops import rearrange

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = f'{self.img_dir}/{img}'
        img = Image.open(file).convert('RGB')
        return img

def imagenet_transform(phase):

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
       
    elif phase == 'test':
        transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()
        ])

    return transform

class Dataset_embedding(data.Dataset):
    def __init__(self, cfg_data, phase='train'):
        self.transform = imagenet_transform(phase)
        self.type_name = cfg_data.type_name
        self.phase = phase
        
        # Create consistent index mapping
        self.type2idx = {name: idx for idx, name in enumerate(self.type_name)}
        
        base_dir = cfg_data.train_dir if phase == 'train' else cfg_data.test_dir
        self.loader = ImageLoader(base_dir)
        
        print(f"\nPhase: {phase}")
        print("Type to index mapping:")
        for type_name, idx in self.type2idx.items():
            print(f"{type_name}: {idx}")
        
        # Build data list
        self.data = []
        if phase == 'train':
            types_to_use = self.type_name
        else:
            # Skip 'clear' type for testing
            types_to_use = [t for t in self.type_name if t != 'clear']
        
        # Get reference image names
        first_type_dir = os.path.join(base_dir, self.type_name[0])
        name = [n for n in os.listdir(first_type_dir) 
               if n.endswith(('.jpg', '.png', '.jpeg', '.PNG'))]
        
        # Build data list
        for type_name in types_to_use:
            for img_name in name:
                self.data.append([type_name, img_name])
        
        print(f'\nInitialized {phase} dataset:')
        print(f'Total samples: {len(self.data)}')
        print(f'Number of classes: {len(self.type_name)}')
        print(f'Types used: {types_to_use}')

    def __getitem__(self, index):
        try:
            type_name, image_name = self.data[index]
            scene = self.type2idx[type_name]
            
            image = self.loader(f'{type_name}/{image_name}')
            if image is None:
                raise ValueError(f"Failed to load image: {type_name}/{image_name}")
            
            image = self.transform(image)
            return (scene, image)
            
        except Exception as e:
            print(f"Error loading item {index}: {str(e)}")
            return (0, torch.zeros((3, 224, 224)))

    def __len__(self):
        return len(self.data)

def init_embedding_data(cfg_em, phase):
    """Initialize data loaders with validation"""
    try:
        if phase == 'train':
            # Initialize datasets
            train_dataset = Dataset_embedding(cfg_em, 'train')
            test_dataset = Dataset_embedding(cfg_em, 'test')

            # Create data loaders
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=cfg_em.batch,
                shuffle=True,
                num_workers=cfg_em.num_workers,
                pin_memory=True
            )
            
            test_loader = data.DataLoader(
                test_dataset,
                batch_size=cfg_em.batch,
                shuffle=False,
                num_workers=cfg_em.num_workers,
                pin_memory=True
            )
            
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Test dataset size: {len(test_dataset)}")
            
            
        elif phase == 'inference':
            test_dataset = Dataset_embedding(cfg_em, 'test')
            test_loader = data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfg_em.num_workers,
                pin_memory=True
            )
        return train_loader, test_loader
            
    except Exception as e:
        print(f"Error initializing data loaders: {str(e)}")
        raise