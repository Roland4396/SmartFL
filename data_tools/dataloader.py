import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torchvision.datasets as tvdatasets
import os
import urllib.request
import zipfile
import shutil
from PIL import Image
from data_tools.sampling import *

# Optional NLP dependencies are not needed for the vision training paths used in this repo.
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import pandas as pd
except ImportError:
    load_dataset = None
    AutoTokenizer = None
    pd = None


class TinyImageNetDataset(Dataset):
    """Custom Dataset for TinyImageNet that properly handles validation set"""
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        # First create unified class mapping from training set
        train_dir = os.path.join(root_dir, 'train')
        self.classes = sorted([d for d in os.listdir(train_dir)
                              if os.path.isdir(os.path.join(train_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        if train:
            self.data_dir = os.path.join(root_dir, 'train')
            self._load_train_data()
        else:
            self.data_dir = os.path.join(root_dir, 'val')
            self._load_val_data()

    def _load_train_data(self):
        """Load training data - organized by class folders using os.walk like FlexFL"""
        self.images = []
        self.targets = []  # Add targets list for compatibility

        # Use os.walk to recursively find images in class_name/images/ subdirectories
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for root, _, files in sorted(os.walk(class_dir)):
                    for fname in sorted(files):
                        if fname.endswith('.JPEG'):
                            img_path = os.path.join(root, fname)
                            label = self.class_to_idx[class_name]
                            self.images.append((img_path, label))
                            self.targets.append(label)

    def _load_val_data(self):
        """Load validation data - requires parsing val_annotations.txt"""
        self.images = []
        self.targets = []  # Add targets list for compatibility

        # Read validation annotations
        val_annotations_file = os.path.join(self.data_dir, 'val_annotations.txt')
        self.val_img_to_class = {}
        classes_set = set()

        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_name = parts[1]
                    self.val_img_to_class[img_name] = class_name
                    classes_set.add(class_name)

        # Validate that all validation classes exist in training set
        missing_classes = classes_set - set(self.classes)
        if missing_classes:
            print(f"Warning: Validation set contains classes not in training set: {missing_classes}")

        # Load image paths and labels
        val_images_dir = os.path.join(self.data_dir, 'images')
        for img_name in os.listdir(val_images_dir):
            if img_name.endswith('.JPEG') and img_name in self.val_img_to_class:
                img_path = os.path.join(val_images_dir, img_name)
                class_name = self.val_img_to_class[img_name]
                if class_name in self.class_to_idx:
                    label = self.class_to_idx[class_name]
                    self.images.append((img_path, label))
                    self.targets.append(label)
                else:
                    print(f"Warning: Skipping image {img_name} with unknown class {class_name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]

        # Load image
        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def download_tiny_imagenet(data_root):
    """Download and extract Tiny ImageNet dataset if not exists"""
    tiny_imagenet_dir = os.path.join(data_root, 'tiny-imagenet-200')
    train_dir = os.path.join(tiny_imagenet_dir, 'train')
    val_dir = os.path.join(tiny_imagenet_dir, 'val')

    # Check if already exists
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        return tiny_imagenet_dir

    print("Downloading Tiny ImageNet dataset...")
    os.makedirs(data_root, exist_ok=True)

    # Download URL
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")

    # Download
    urllib.request.urlretrieve(url, zip_path)
    print("Download completed. Extracting...")

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_root)

    # Remove zip file
    os.remove(zip_path)
    print("Tiny ImageNet dataset ready!")

    return tiny_imagenet_dir


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_datasets(args):
    val_set = None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.25, 0.25, 0.25])
        train_set = tvdatasets.CIFAR10(args.data_root, train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize
                                       ]))
        test_set = tvdatasets.CIFAR10(args.data_root, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize
                                      ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.25, 0.25, 0.25])
        train_set = tvdatasets.CIFAR100(args.data_root, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize
                                        ]))
        test_set = tvdatasets.CIFAR100(args.data_root, train=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           normalize
                                       ]))
    elif args.data == 'tiny_imagenet':
        # Tiny ImageNet (200 classes, 64x64 images)
        # Auto-download if not exists
        tiny_imagenet_dir = download_tiny_imagenet(args.data_root)

        # 训练集（有数据增强）
        trans_imagenet_train = transforms.Compose([transforms.RandomCrop(64),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                                                        std=[0.2770, 0.2691, 0.2821])])

        # 验证集（无数据增强）
        trans_imagenet_val = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                                                      std=[0.2770, 0.2691, 0.2821])])

        # 使用自定义数据集类正确处理 TinyImageNet
        train_set = TinyImageNetDataset(tiny_imagenet_dir, train=True, transform=trans_imagenet_train)
        test_set = TinyImageNetDataset(tiny_imagenet_dir, train=False, transform=trans_imagenet_val)
    else:
        raise NotImplementedError

    return train_set, val_set, test_set


def get_user_groups(train_set, val_set, test_set, args):
    train_user_groups, val_user_groups, test_user_groups = create_noniid_users(train_set, val_set, test_set, args, args.alpha)
    return train_user_groups, val_user_groups, test_user_groups


def get_dataloaders(args, batch_size, dataset):
    train_loader, val_loader, test_loader = None, None, None
    train_set, val_set, test_set = dataset

    if args.use_valid:
        if val_set is None:
            train_set_index = torch.randperm(len(train_set))
            if os.path.exists(os.path.join(args.save_path, 'index.pth')):
                train_set_index = torch.load(os.path.join(args.save_path, 'index.pth'))
            else:
                torch.save(train_set_index, os.path.join(args.save_path, 'index.pth'))
            if args.data.startswith('cifar'):
                num_sample_valid = 0
            elif args.data == 'tiny_imagenet':
                num_sample_valid = 0
            else:
                raise NotImplementedError

            train_indices = train_set_index[:-num_sample_valid]
            val_indices = train_set_index[-num_sample_valid:]
            val_set = train_set
        else:
            train_indices = torch.arange(len(train_set))
            val_indices = torch.arange(len(val_set))

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_indices),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    val_indices),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    if 'train' not in args.splits:
        if len(val_loader.dataset.transform.transforms) > 2:
            val_loader.dataset.transform.transforms = val_loader.dataset.transform.transforms[-2:]

    if 'bert' in args.arch:
        train_loader.collate_fn = collate_fn
        val_loader.collate_fn = collate_fn
        test_loader.collate_fn = collate_fn

    return train_loader, val_loader, test_loader


def get_client_dataloader(dataset, idxs, args, batch_size):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    if 'bert' in args.arch:
        return torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(idxs)),
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs),
                                           num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(idxs)),
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs),
                                           num_workers=args.workers, pin_memory=True)


def collate_fn(data):
    return (pad_sequence([torch.tensor(d['input_ids']) for d in data], batch_first=True, padding_value=0),
            torch.tensor([d['label'] for d in data]))
