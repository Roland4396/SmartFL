"""
Debug version with detailed progress output
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
import sys
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting script...")
sys.stdout.flush()

from models.searchable_resnet import SearchableResNet
print(f"[{datetime.now().strftime('%H:%M:%S')}] Imported SearchableResNet")
sys.stdout.flush()

from data_tools.dataloader import get_dataloaders, get_datasets
print(f"[{datetime.now().strftime('%H:%M:%S')}] Imported dataloader")
sys.stdout.flush()

from args import arg_parser, modify_args
print(f"[{datetime.now().strftime('%H:%M:%S')}] Imported args")
sys.stdout.flush()

from utils.utils import AverageMeter, accuracy
print(f"[{datetime.now().strftime('%H:%M:%S')}] Imported utils")
sys.stdout.flush()

def train_model_simple(model, train_loader, val_loader, device, epochs=5):
    """Simple training with progress output"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {epochs} epochs")
    sys.stdout.flush()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{epochs} - Training...")
        sys.stdout.flush()

        model.train()
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Batch {batch_idx}/{len(train_loader)}")
                sys.stdout.flush()

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, list):
                outputs = outputs[-1]

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_acc = 100. * train_correct / train_total
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1} - Train Acc: {train_acc:.2f}%")
        sys.stdout.flush()

    # Validation
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Validating...")
    sys.stdout.flush()

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if batch_idx % 20 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Val batch {batch_idx}/{len(val_loader)}")
                sys.stdout.flush()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if isinstance(outputs, list):
                outputs = outputs[-1]

            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Final Val Acc: {val_acc:.2f}%")
    sys.stdout.flush()

    return val_acc


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ===== STARTING DEBUG RUN =====")
    sys.stdout.flush()

    # Parse arguments
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Parsing arguments...")
    sys.stdout.flush()

    import argparse
    parser = argparse.ArgumentParser(description='Debug proxy correlation', parents=[arg_parser], conflict_handler='resolve')
    parser.add_argument('--num_samples', type=int, default=2, help='Number of architectures (debug=2)')
    parser.add_argument('--train_epochs', type=int, default=2, help='Training epochs (debug=2)')

    args = parser.parse_args()
    args = modify_args(args)

    # Set defaults
    args.data = 'cifar100'
    args.batch_size = 64  # Smaller batch for debug
    args.use_gpu = torch.cuda.is_available()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Config: samples={2}, epochs={2}, batch_size={args.batch_size}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU available: {args.use_gpu}")
    sys.stdout.flush()

    # Set seed
    if not hasattr(args, 'seed'):
        args.seed = 42
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using device: {device}")
    sys.stdout.flush()

    # Load architecture library
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading architecture library...")
    sys.stdout.flush()

    library_path = 'resnet_cifar100_architecture_library.json'
    with open(library_path, 'r') as f:
        data = json.load(f)
    configs = data['configurations']

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(configs)} configurations")
    sys.stdout.flush()

    # Sample 2 architectures
    sampled_configs = random.sample(configs, 2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sampled 2 architectures")
    sys.stdout.flush()

    # Load dataset
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading dataset: {args.data}...")
    sys.stdout.flush()

    train_set, val_set, test_set = get_datasets(args)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loaded: train={len(train_set)}, test={len(test_set)}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]   val_set is None, will use test_set for validation")
    sys.stdout.flush()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating dataloaders...")
    sys.stdout.flush()

    # Set args for dataloader
    args.use_valid = False  # Don't split train set
    args.splits = ['train', 'val', 'test']
    if not hasattr(args, 'workers'):
        args.workers = 2
    if not hasattr(args, 'save_path'):
        args.save_path = './outputs/debug'
        os.makedirs(args.save_path, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(args, args.batch_size, (train_set, val_set, test_set))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataloaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches")
    sys.stdout.flush()

    # Train architectures
    results = []

    for i, config in enumerate(sampled_configs):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ========== Architecture {i+1}/2 ==========")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Nuclear Norm: {config['total_conv_nuclear_norm']:.2f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Width: {config['width_multipliers']}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Exit: {config['early_exit_location']}")
        sys.stdout.flush()

        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating model...")
            sys.stdout.flush()

            model = SearchableResNet(
                num_blocks=[18, 18, 18],
                num_classes=args.num_classes,
                width_multipliers=config['width_multipliers'],
                early_exit_location=config['early_exit_location']
            )

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Model created")
            sys.stdout.flush()

            val_acc = train_model_simple(model, train_loader, val_loader, device, epochs=2)

            results.append({
                'nuclear_norm': config['total_conv_nuclear_norm'],
                'accuracy': val_acc
            })

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Result: Norm={config['total_conv_nuclear_norm']:.2f}, Acc={val_acc:.2f}%")
            sys.stdout.flush()

            del model
            if args.use_gpu:
                torch.cuda.empty_cache()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU cache cleared")
                sys.stdout.flush()

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ===== DEBUG RUN COMPLETE =====")
    print(f"Results:")
    for i, r in enumerate(results):
        print(f"  Arch {i+1}: Norm={r['nuclear_norm']:.2f}, Acc={r['accuracy']:.2f}%")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
