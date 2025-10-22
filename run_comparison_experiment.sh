#!/bin/bash
# Comparison Experiment: Hierarchical vs Independent Model Selection
# ResNet-110 on CIFAR-100

echo "============================================"
echo "Comparison Experiment: Model Selection"
echo "============================================"

# Experiment 1: Hierarchical Selection (Baseline)
echo ""
echo "[1/2] Running Hierarchical Selection (Baseline)..."
python main.py \
    --model resnet \
    --data cifar100 \
    --arch resnet110 \
    --num_classes 100 \
    --config_library_path resnet_cifar100_architecture_library.json \
    --rounds 400 \
    --num_clients 100 \
    --sample_rate 0.1 \
    --alpha 0.5 \
    --batch-size 128 \
    --lr 0.1 \
    --local_epochs 5 \
    --skip_stage1 \
    --skip_stage2 \
    --flops_constraints 83.4 99.7 138.5 253.1 \
    --params_constraints 0.21 0.46 0.86 1.73 \
    --save_path outputs/resnet_cifar100_hierarchical

echo ""
echo "[Experiment 1 Complete] Results saved to: outputs/resnet_cifar100_hierarchical"

# Experiment 2: Independent Selection (Ablation)
echo ""
echo "[2/2] Running Independent Selection (Ablation)..."
python main.py \
    --model resnet \
    --data cifar100 \
    --arch resnet110 \
    --num_classes 100 \
    --config_library_path resnet_cifar100_architecture_library.json \
    --rounds 400 \
    --num_clients 100 \
    --sample_rate 0.1 \
    --alpha 0.5 \
    --batch-size 128 \
    --lr 0.1 \
    --local_epochs 5 \
    --skip_stage1 \
    --skip_stage2 \
    --flops_constraints 83.4 99.7 138.5 253.1 \
    --params_constraints 0.21 0.46 0.86 1.73 \
    --independent_selection \
    --save_path outputs/resnet_cifar100_independent

echo ""
echo "[Experiment 2 Complete] Results saved to: outputs/resnet_cifar100_independent"

echo ""
echo "============================================"
echo "Both experiments completed!"
echo "============================================"
echo "Compare results:"
echo "  Hierarchical: outputs/resnet_cifar100_hierarchical/"
echo "  Independent:  outputs/resnet_cifar100_independent/"
