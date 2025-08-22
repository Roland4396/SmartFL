# SmartFL: 通过动态模型缩放实现资源感知的联邦学习框架

## 概述

联邦学习（FL）的一个核心挑战是如何有效处理大量具有不同计算和存储能力的（即“异构”的）客户端设备。标准联邦学习为所有设备使用相同大小的模型，这会导致“掉队者效应”——系统整体效率受限于最慢的设备。

**SmartFL** 是一个先进的联邦学习框架，旨在解决这一挑战。它没有采用固定的模型，而是实现了一套复杂的**动态模型缩放**机制。在每一轮训练中，框架会根据客户端的资源约束，**实时地、动态地**为每个参与的客户端生成一个“量身定制”的、具有最优宽度和深度的子网络。这使得强大的设备可以使用更大的模型以贡献更多，而资源受限的设备则使用较小的模型以避免掉队，从而显著提升整个联邦网络的训练效率和模型性能。

## 核心特性

- **动态在线架构选择**: 在每轮训练开始时，根据客户端的资源约束（如FLOPs）和预先计算的性能数据，为本轮动态选择最优的模型架构组合。
- **混合模型缩放**: 同时支持模型的**垂直缩放**（通过调整卷积层的通道数来改变**宽度**）和**水平缩放**（通过使用网络不同深度的“早退出口”来改变**深度**）。
- **多出口知识蒸馏**: 在客户端本地训练时，采用一种先进的训练策略。它不仅使用真实标签，还迫使较浅的“早退出口”学习并模仿网络最终出口的行为，极大地提升了小模型的性能。
- **离线架构搜索**: 提供了一个“暴力搜索”脚本，用于预先遍历数千种架构组合，计算它们的性能成本（FLOPs），并生成一个JSON文件作为后续动态决策的“大脑”。

## 项目工作流与使用方法

#### 步骤 1: (可选) 生成架构成本JSON文件

如果您想修改搜索空间或重新生成决策文件，首先需要训练一个“超网”（Supernet），然后运行搜索脚本。

```bash
# 1. 训练超网 (具体参数请参考 train_supernet.py)
python train_supernet.py --save_path supernet.pth

# 2. 运行暴力搜索
python brute_force_search.py --supernet-path supernet.pth --results-path brute_force_results.json
```
*注意：项目中已提供预计算的 `brute_force_results.json`，因此该步骤对于直接运行联邦学习不是必需的。*

#### 步骤 2: 运行联邦学习训练

这是项目的主要入口。通过 `main.py` 启动联邦学习。

```bash
python main.py \
    --arch resnet110_4 \
    --data cifar100 \
    --num_clients 100 \
    --num_rounds 400 \
    --client_split_ratios 0.25 0.25 0.25 0.25 \
    --flops_constraints 83.4 99.7 138.5 253.1 \
    --use_gpu 1 \
    --gpu_idx 0 \
    --save_path outputs/my_smartfl_run
```

**关键参数说明:**
- `--client_split_ratios`: 定义了4个客户端复杂度级别的比例。`0.25 0.25 0.25 0.25` 表示各占25%。
- `--flops_constraints`: 为这4个级别分别设置FLOPs（百万次）上限。这是动态架构选择的核心依据。
- 其他参数如 `--num_clients`, `--num_rounds` 等控制联邦学习的基本设定。

#### 步骤 3: 评估已训练的模型

使用 `--evalmode` 和 `--evaluate_from` 参数来评估已保存的模型。

```bash
# 评估全局模型在测试集上的平均性能
python main.py --evalmode global --evaluate_from outputs/my_smartfl_run/model_best.pth.tar

# 评估每个客户端（及其对应模型复杂度）的性能
python main.py --evalmode local --evaluate_from outputs/my_smartfl_run/model_best.pth.tar
```

## 安装

1.  克隆本仓库。
2.  强烈建议使用Python虚拟环境。
3.  安装核心依赖：
    ```bash
    pip install torch torchvision numpy tqdm
    ```

## 项目结构

```
SmartFL/
├── main.py             # 主程序入口，启动联邦学习
├── fed.py              # 核心联邦学习逻辑 (Federator类)
├── train.py            # 客户端本地训练逻辑，包含知识蒸馏损失
├── brute_force_search.py # 离线暴力搜索脚本，生成 a-c-json
├── train_supernet.py   # 用于训练超网的脚本
├── hierarchical_model_selector.py # 在线动态决策算法
├── models/
│   ├── resnet.py       # 核心：可动态缩放、带早退出口的ResNet模型
│   └── searchable_resnet.py # 用于暴力搜索的模型定义
├── outputs/            # 存放实验结果、日志和模型
└── brute_force_results.json # 预计算的架构-成本查找表
```