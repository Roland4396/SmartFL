
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from models.searchable_resnet import SearchableResNet
from models.searchable_vgg import searchable_vgg16
from models.searchable_mobilenet import searchable_mobilenet_v2
from data_tools.dataloader import get_dataloaders, get_datasets
from args import arg_parser, modify_args

def train_supernet(args):
    """
    Trains a Supernet, which is the largest possible searchable network,
    and saves its weights. This network's weights will be used as a
    backbone for evaluating all sub-networks.
    """
    print("======================================================")
    print(" S T A R T I N G   S U P E R N E T   T R A I N I N G ")
    print("======================================================")

    # 1. Setup device, dataset, and dataloader
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu_idx}" if args.gpu_idx else "cuda")
    else:
        device = torch.device("cpu")
    print(f"--> Using device: {device}")

    print(f"--> Loading dataset: {args.data}")
    train_set, val_set, test_set = get_datasets(args)
    # Use supernet-specific batch size
    batch_size = getattr(args, 'supernet_batch_size', args.batch_size if hasattr(args, 'batch_size') else 128)
    train_loader, _, _ = get_dataloaders(args, batch_size, (train_set, val_set, test_set))

    # 2. Define the Supernet instance based on model type
    model_type = getattr(args, 'model', 'resnet').lower()
    print(f"--> Initializing Supernet model ({model_type})...")

    if model_type == 'resnet':
        supernet = SearchableResNet(
            num_blocks=[18, 18, 18],  # Max blocks for resnet110
            num_classes=args.num_classes,
            width_multipliers=[1.0, 1.0, 1.0]  # Max width for all stages
        ).to(device)
    elif model_type == 'vgg':
        supernet = searchable_vgg16(
            num_classes=args.num_classes,
            width_multipliers=[1.0] * 15,  # Max width for all 15 layers (13 conv + 2 fc)
            num_channels=3
        ).to(device)
    elif model_type == 'mobilenet':
        supernet = searchable_mobilenet_v2(
            num_classes=args.num_classes,
            width_multipliers=[1.0] * 10,  # Max width for all 10 stages [32,16,24,32,64,96,160,160,160,320]
            num_channels=3
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: 'resnet', 'vgg', 'mobilenet'")
    
    # 3. Define optimizer, scheduler, and loss function
    optimizer = optim.SGD(supernet.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"--> Starting training for {args.epochs} epochs...")

    # 4. Training loop
    for epoch in range(args.epochs):
        supernet.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = supernet(inputs)
            
            # The model might return multiple outputs from early exits,
            # but for supernet training, we only care about the final exit.
            if isinstance(outputs, list):
                final_output = outputs[-1]
            else:
                final_output = outputs

            loss = criterion(final_output, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"    Epoch [{epoch+1}/{args.epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"--> Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        scheduler.step()

    # 5. Save the trained weights
    print("======================================================")
    print("--> Training finished.")
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(supernet.state_dict(), args.save_path)
    print(f"--> Supernet weights saved to: {args.save_path}")
    print("======================================================")


if __name__ == "__main__":
    # Add arguments specific to this script
    arg_parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for SGD')
    arg_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the supernet')
    
    args = arg_parser.parse_args()
    
    # Set a default GPU index if one isn't provided.
    if args.gpu_idx is None:
        args.gpu_idx = '0'

    # Use existing config modifiers, but override a few for this specific task
    args = modify_args(args)
    args.data = 'cifar100'
    args.batch_size = 128 # A larger batch size is common for supernet training
    
    # Set a default save path if not provided via command line
    if args.save_path is None or args.save_path == "outputs/自学习阶段":
        args.save_path = 'supernet.pth'

    train_supernet(args)