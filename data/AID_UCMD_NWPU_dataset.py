import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import autoaugment
from torch.utils.data import Dataset

class Source_Dataset(Dataset):
    def __init__(self, source_list, transforms):
        self.source_list = source_list
        self.transforms = transforms

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, index):
        source_path, source_label = self.source_list[index]
        source_image = Image.open(source_path).convert("RGB")
        source_image = self.transforms(source_image)
        return source_image, source_label

class Target_Dataset(Dataset):
    def __init__(self, target_list, transforms):
        self.target_list = target_list
        self.transforms = transforms

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        target_path, _ = self.target_list[index]
        target_image = Image.open(target_path).convert("RGB")
        target_image = self.transforms(target_image)
        return target_image, index  

class Test_Dataset(Dataset):
    def __init__(self, data_list, class_num, transforms):
        self.data_list = data_list
        self.class_num = class_num
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index][0]
        data_label = int(self.data_list[index][1])

        image = Image.open(data_path).convert("RGB")
        image = self.transforms(image)

        one_hot_label = np.eye(self.class_num, dtype=np.uint8)[data_label]
        
        return image, data_label, one_hot_label, data_path

def read_text(args, separate_targets=False):
    source_domain, target_domain ,target_domain2= args.dataset_mode.split("_")
    
    source_train_txt_path = os.path.join(args.root, source_domain, "train.txt")

    target1_train_txt_path = os.path.join(args.root, target_domain, "train.txt")
    target1_val_txt_path = os.path.join(args.root, target_domain, "val.txt")
    target1_test_txt_path = os.path.join(args.root, target_domain, "test.txt")

    target2_train_txt_path = os.path.join(args.root, target_domain2, "train.txt")
    target2_val_txt_path = os.path.join(args.root, target_domain2, "val.txt")
    target2_test_txt_path = os.path.join(args.root, target_domain2, "test.txt")
    
    source_classes = args.source_classes

    source_train_data_list = []
    target1_train_data_list = []
    target1_val_data_list = []
    target1_test_data_list = []
    target2_train_data_list = []
    target2_val_data_list = []
    target2_test_data_list = []

    with open(source_train_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            temp_label = int(temp_label)
            if temp_label in source_classes:
                source_train_data_list.append([temp_path, temp_label])

    with open(target1_train_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            temp_label = int(temp_label)
            target1_train_data_list.append([temp_path, temp_label])

    with open(target1_val_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            temp_label = int(temp_label)
            if temp_label in args.source_classes:
                target1_val_data_list.append([temp_path, temp_label])
            else:
                target1_val_data_list.append([temp_path, args.unknown_class_index])

    with open(target1_test_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            temp_label = int(temp_label)
            if temp_label in args.source_classes:
                target1_test_data_list.append([temp_path, temp_label])
            else:
                target1_test_data_list.append([temp_path, args.unknown_class_index])

    with open(target2_train_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            temp_label = int(temp_label)
            target2_train_data_list.append([temp_path, temp_label])

    with open(target2_val_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            temp_label = int(temp_label)
            if temp_label in args.source_classes:
                target2_val_data_list.append([temp_path, temp_label])
            else:
                target2_val_data_list.append([temp_path, args.unknown_class_index])

    with open(target2_test_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            temp_label = int(temp_label)
            if temp_label in args.source_classes:
                target2_test_data_list.append([temp_path, temp_label])
            else:
                target2_test_data_list.append([temp_path, args.unknown_class_index])

    combined_target_train_data_list = target1_train_data_list + target2_train_data_list
    combined_target_val_data_list = target1_val_data_list + target2_val_data_list
    combined_target_test_data_list = target1_test_data_list + target2_test_data_list

   
    return (source_train_data_list, target1_train_data_list, target2_train_data_list,
                target1_val_data_list , target2_val_data_list,  
                target1_test_data_list, target2_test_data_list)
    
def get_data(args):

    (source_train_list, target1_train_list, target2_train_list, target1_val_list,target2_val_list,  
                target1_test_list, target2_test_list)  = read_text(args, separate_targets=True)

    if args.training_phase == "phase1":
        target_train_list = target1_train_list
        target_val_list = target1_val_list
        target_test_list = target1_test_list
        
    elif args.training_phase == "phase2":
        target_train_list = target2_train_list
        target_val_list = target2_val_list
        target_test_list = target2_test_list

    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        autoaugment.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=sigma),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=sigma)
    ])

    source_train_dataset = Source_Dataset(source_train_list, train_transform)
    target_train_dataset = Target_Dataset(target_train_list, train_transform)
    val_dataset = Test_Dataset(target_val_list, args.class_num, test_transform)
    test_dataset = Test_Dataset(target_test_list,args.class_num, test_transform)

    return source_train_dataset, target_train_dataset, val_dataset, test_dataset