# SmartFL: 智能三阶段联邦学习框架

## 概述

SmartFL现在集成了完整的三阶段智能流水线：

1. **阶段1：超网预训练** - 训练大型超网获得预训练权重
2. **阶段2：PPO智能架构搜索** - 使用强化学习智能生成高质量架构配置库
3. **阶段3：联邦学习训练** - 基于智能配置库进行分层联邦学习

## 核心优势

- 🎯 **一键执行**：单个main.py完成全流程自动化
- 🧠 **智能搜索**：PPO强化学习替代暴力搜索，大幅提升效率和质量
- ⚙️ **参数统一**：所有阶段参数通过命令行集中管理
- 🛡️ **智能检测**：自动检查文件依赖，支持跳过已完成阶段
- 🚀 **质量优化**：最大化核范数+多样性探索，生成更优架构

## 使用方法

### 完整三阶段自动执行
```bash
python main.py --arch resnet110_4 --data cifar100 --num_clients 100 --num_rounds 400
# 自动执行: 超网训练 → PPO配置生成 → 联邦学习
```

### 控制特定阶段

**跳过已完成阶段：**
```bash
python main.py --skip_stage1 --arch resnet110_4 --num_clients 100
# 跳过超网训练，执行: PPO配置生成 → 联邦学习
```

**只执行特定阶段：**
```bash
python main.py --stages_only "2" --num_architectures 1000
# 只执行PPO配置生成阶段
```

**自定义阶段参数：**
```bash
python main.py --supernet_epochs 200 --num_architectures 1000 --episodes_per_batch 150 --num_rounds 500
# 所有阶段使用自定义参数
```

## 参数说明

### 阶段控制参数
- `--skip_stage1`: 跳过阶段1（超网训练）
- `--skip_stage2`: 跳过阶段2（PPO配置生成）  
- `--skip_stage3`: 跳过阶段3（联邦学习训练）
- `--stages_only`: 只运行指定阶段（如 "1,2" 或 "3"）

### 阶段1：超网训练参数
- `--supernet_epochs`: 超网训练轮数（默认：200）
- `--supernet_lr`: 超网学习率（默认：0.1）
- `--supernet_save_path`: 超网保存路径（默认：supernet.pth）

### 阶段2：PPO配置生成参数  
- `--num_architectures`: 生成架构数量（默认：500）
- `--episodes_per_batch`: PPO每批次训练轮数（默认：100）
- `--ppo_learning_rate`: PPO学习率（默认：0.001）
- `--config_library_path`: 配置库保存路径（默认：ppo_architecture_library.json）

### 阶段3：联邦学习参数
保持所有原有参数：`--num_clients`, `--num_rounds`, `--flops_constraints` 等

## 智能特性

### 自动依赖检查
系统会自动检查每阶段所需文件：
- 阶段2需要 `supernet.pth`
- 阶段3需要 `ppo_architecture_library.json`

如果文件不存在会给出明确错误提示。

### 自动阶段跳过
如果检测到输出文件已存在，会自动跳过对应阶段：
- 存在 `supernet.pth` → 跳过阶段1
- 存在 `ppo_architecture_library.json` → 跳过阶段2

### PPO智能搜索
相比传统暴力搜索：
- ✅ **效率提升**：智能探索vs全遍历  
- ✅ **质量优化**：最大化核范数+多样性奖励
- ✅ **速度优势**：生成1000+配置时间显著缩短

## 输出文件

- `supernet.pth`: 阶段1输出的预训练超网权重
- `ppo_architecture_library.json`: 阶段2输出的PPO智能配置库
- `outputs/`: 阶段3输出的联邦学习训练结果

## 示例场景

### 首次完整运行
```bash
python main.py --data cifar100 --supernet_epochs 100 --num_architectures 200 --num_rounds 200
```

### 增量配置生成
```bash  
python main.py --stages_only "2" --num_architectures 1000 --episodes_per_batch 200
```

### 仅联邦训练
```bash
python main.py --skip_stage1 --skip_stage2 --num_rounds 500 --num_clients 200
```

## 注意事项

1. **首次运行**会执行完整三阶段流程，耗时较长
2. **增量运行**会自动跳过已完成阶段，快速启动
3. **GPU推荐**用于阶段1和3，CPU可用于阶段2
4. **配置兼容**完全向后兼容原有brute_force_results.json格式