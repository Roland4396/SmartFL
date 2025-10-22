"""
AlignFL: Evaluate correlation between Nuclear Norm (proxy) and True Accuracy
Uses real architecture configurations from PPO-generated library
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
import argparse
import os
from tqdm import tqdm

from models.searchable_resnet import SearchableResNet
from data_tools.dataloader import get_dataloaders, get_datasets
from args import arg_parser, modify_args
from utils.utils import AverageMeter, accuracy

# ============================================================================
# Helper Functions
# ============================================================================

def load_architecture_library(library_path):
    """Load architecture configurations from JSON"""
    with open(library_path, 'r') as f:
        data = json.load(f)
    return data['metadata'], data['configurations']

def create_model_from_config(config, num_classes):
    """Create SearchableResNet from configuration"""
    model = SearchableResNet(
        num_blocks=[18, 18, 18],  # ResNet110 structure
        num_classes=num_classes,
        width_multipliers=config['width_multipliers'],
        early_exit_location=config['early_exit_location']
    )
    return model


def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.1):
    """Train model for fixed epochs and return final validation accuracy"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            # Handle early exit - use final output
            if isinstance(outputs, list):
                outputs = outputs[-1]

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

    # Final validation
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if isinstance(outputs, list):
                outputs = outputs[-1]

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    return top1.avg

# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate proxy correlation', parents=[arg_parser], conflict_handler='resolve')
    parser.add_argument('--library_path', type=str,
                        default='resnet_cifar100_architecture_library.json',
                        help='Path to architecture library')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of architectures to sample and evaluate')
    parser.add_argument('--train_epochs', type=int, default=30,
                        help='Training epochs for each architecture (from scratch, recommend 20-40 for CIFAR-100)')

    args = parser.parse_args()
    args = modify_args(args)

    # Set defaults for this experiment
    args.data = 'cifar100'
    args.batch_size = 128
    args.use_gpu = torch.cuda.is_available()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    print(f"Using device: {device}")

    # Load architecture library
    print(f"\nLoading architecture library from: {args.library_path}")
    metadata, configs = load_architecture_library(args.library_path)
    print(f"Total configurations available: {len(configs)}")
    print(f"Model type: {metadata['model_type']}")
    print(f"Dataset: {metadata['dataset']}")

    # Sample architectures
    num_samples = min(args.num_samples, len(configs))
    sampled_configs = random.sample(configs, num_samples)
    print(f"\nRandomly sampled {num_samples} architectures for evaluation")

    # Load dataset
    print(f"\nLoading dataset: {args.data}")
    train_set, val_set, test_set = get_datasets(args)
    train_loader, val_loader, test_loader = get_dataloaders(
        args, args.batch_size, (train_set, val_set, test_set)
    )

    # Evaluate each architecture
    print(f"\n{'='*70}")
    print(f"Training and evaluating {num_samples} architectures FROM SCRATCH")
    print(f"Training epochs: {args.train_epochs}")
    print(f"{'='*70}\n")

    # Check for existing results (resume capability)
    results_path = 'proxy_correlation_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Found existing results with {len(results)} architectures evaluated")
        print(f"Resuming from architecture {len(results)}")
    else:
        results = []

    for i, config in enumerate(tqdm(sampled_configs, desc="Evaluating architectures")):
        # Skip if already evaluated
        if i < len(results):
            continue
        try:
            # Create model with random initialization
            model = create_model_from_config(config, args.num_classes)

            # Train from scratch (no pretrained weights)
            # Note: We only need relative ranking for correlation analysis
            val_accuracy = train_model(
                model, train_loader, val_loader, device,
                epochs=args.train_epochs, lr=0.1
            )

            # Record results (核范数already computed in库)
            results.append({
                'config_idx': i,
                'nuclear_norm': config['total_conv_nuclear_norm'],  # Already in library!
                'accuracy': val_accuracy,
                'flops_m': config['flops_m'],
                'params': config['num_params'],
                'width_multipliers': config['width_multipliers'],
                'early_exit': config['early_exit_location']
            })

            print(f"\n[{i+1}/{num_samples}] "
                  f"Nuclear Norm: {config['total_conv_nuclear_norm']:.2f}, "
                  f"Accuracy: {val_accuracy:.2f}%, "
                  f"FLOPs: {config['flops_m']:.1f}M, "
                  f"Exit: {config['early_exit_location']}")

            # Save after each evaluation (for resume capability)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)

            # Clean up GPU memory
            del model
            if args.use_gpu:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError evaluating architecture {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final save
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n\nAll results saved to: {results_path}")

    # ========================================================================
    # Statistical Analysis and Visualization
    # ========================================================================

    nuclear_norms = np.array([r['nuclear_norm'] for r in results])
    accuracies = np.array([r['accuracy'] for r in results])

    print(f"\n{'='*70}")
    print("Statistical Analysis")
    print(f"{'='*70}")
    print(f"Number of evaluated architectures: {len(results)}")
    print(f"Nuclear Norm range: [{nuclear_norms.min():.2f}, {nuclear_norms.max():.2f}]")
    print(f"Accuracy range: [{accuracies.min():.2f}%, {accuracies.max():.2f}%]")

    # Kendall's Tau
    tau, p_value = kendalltau(nuclear_norms, accuracies)
    print(f"\nKendall's τ = {tau:.4f}")
    print(f"p-value = {p_value:.2e}")

    if p_value < 0.001:
        print("Result: Highly significant correlation (p < 0.001)")
    elif p_value < 0.05:
        print("Result: Significant correlation (p < 0.05)")
    else:
        print("Result: Not significant (p ≥ 0.05)")

    # Linear regression
    X = nuclear_norms.reshape(-1, 1)
    y = accuracies
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    y_pred = reg_model.predict(X)
    r_squared = reg_model.score(X, y)
    slope = reg_model.coef_[0]
    intercept = reg_model.intercept_

    print(f"\nLinear Regression:")
    print(f"Equation: Accuracy = {slope:.4f} × Nuclear_Norm + {intercept:.2f}")
    print(f"R² = {r_squared:.4f}")

    # ========================================================================
    # Visualization
    # ========================================================================

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot
    scatter = ax.scatter(nuclear_norms, accuracies,
                         c=accuracies, cmap='viridis',
                         s=80, alpha=0.6, edgecolors='black',
                         linewidth=0.5, label='Architectures')

    # Regression line
    ax.plot(nuclear_norms, y_pred,
            color='red', linewidth=2.5, linestyle='--',
            label=f'Linear fit: y = {slope:.4f}x + {intercept:.1f}\n$R^2$ = {r_squared:.3f}',
            zorder=5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Test Accuracy (%)', rotation=270, labelpad=20)

    # Labels
    ax.set_xlabel('Proxy Score (Total Convolutional Nuclear Norm)', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title('AlignFL: Nuclear Norm vs. Model Accuracy Correlation\n'
                 f'(ResNet on CIFAR-100, {len(results)} sampled architectures)',
                 fontweight='bold', pad=20)

    # Statistical annotation
    textstr = f"Kendall's τ = {tau:.3f}\np < 0.001" if p_value < 0.001 else f"Kendall's τ = {tau:.3f}\np = {p_value:.3f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()

    # Save figure
    fig_path = 'alignfl_proxy_correlation_real.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")

    fig_pdf = 'alignfl_proxy_correlation_real.pdf'
    plt.savefig(fig_pdf, format='pdf', bbox_inches='tight')
    print(f"Figure saved to: {fig_pdf}")

    plt.show()

    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
