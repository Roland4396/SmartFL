"""
AlignFL 第一阶段探索策略消融实验自动化脚本

实验配置：
- 模型: ResNet110_4
- 数据集: CIFAR100
- 客户端数: 100
- 训练轮数: 400
- Seed: 0
- FLOPs约束: [83.4, 99.7, 138.5, 253.1]
- Params约束: [0.21, 0.46, 0.86, 1.73]

消融组：
1. Random Search (independent selection)
2. w/o Diversity Reward
3. w/o Region Search
4. w/o Adaptive Strength
5. w/o Partial Reset
(Full AlignFL已有结果)
"""

import os
import sys
import shutil
import subprocess
import time
import json
import csv
from datetime import datetime
from pathlib import Path

# 实验配置
EXPERIMENT_CONFIG = {
    'arch': 'resnet110_4',
    'data': 'cifar100',
    'num_clients': 100,
    'num_rounds': 400,
    'seed': 0,
    'flops_constraints': [83.4, 99.7, 138.5, 253.1],
    'params_constraints': [0.21, 0.46, 0.86, 1.73],
    'num_architectures': 5000,
    'episodes_per_batch': 100,
}

# 消融实验配置
ABLATION_EXPERIMENTS = [
    # {
    #     'name': 'Random Search',
    #     'code': 'random_search',
    #     'description': 'Pure random search baseline (uniform sampling, no optimization)',
    #     'method': 'use_random_search',  # 使用真正的随机搜索
    #     'library_suffix': 'random'
    # },
    # {
    #     'name': 'w/o Diversity Reward',
    #     'code': 'no_diversity',
    #     'description': 'Remove diversity bonus and exploration bonus from reward',
    #     'method': 'modify_ppo',
    #     'library_suffix': 'no_diversity',
    #     'modifications': {
    #         '_calculate_reward': {
    #             'remove_lines': ['diversity_bonus = ', 'exploration_bonus = '],
    #             'add_line': 'diversity_bonus = 0.0\n    exploration_bonus = 0.0'
    #         }
    #     }
    # },  # 已完成，prec@1: 52.461%
    # {
    #     'name': 'w/o Region Search',
    #     'code': 'no_region',
    #     'description': 'No region constraint (search full exit_location space)',
    #     'method': 'modify_ppo',
    #     'library_suffix': 'no_region',
    #     'modifications': {
    #         'choose_search_region': 'replace_return',
    #         'return_value': 'return None  # Search full space'
    #     }
    # },  # 不需要了
    # {
    #     'name': 'w/o Adaptive Strength',
    #     'code': 'no_adaptive',
    #     'description': 'Fixed exploration strength (no TCP-AIMD)',
    #     'method': 'modify_ppo',
    #     'library_suffix': 'no_adaptive',
    #     'modifications': {
    #         'update_exploration_strength': 'replace_return',
    #         'return_value': 'return 0.5  # Fixed strength'
    #     }
    # },
    {
        'name': 'w/o Partial Reset',
        'code': 'no_reset',
        'description': 'Disable periodic parameter reset',
        'method': 'modify_ppo',
        'library_suffix': 'no_reset',
        'modifications': {
            'periodic_reset': 'comment_out'
        }
    }
]


class AblationExperimentRunner:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.results_dir = self.base_dir / 'ablation_results'
        self.results_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.results_dir / f'ablation_log_{self.timestamp}.txt'
        self.results_csv = self.results_dir / f'ablation_results_{self.timestamp}.csv'

        # 初始化结果CSV
        with open(self.results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Experiment', 'Code', 'Library Path', 'Start Time', 'End Time',
                'Duration (hours)', 'Final Test Acc (%)', 'Status', 'Notes'
            ])

    def log(self, message):
        """记录日志到文件和控制台"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def create_modified_ppo_generator(self, experiment):
        """为消融实验创建修改版的PPO生成器"""
        if experiment['method'] == 'use_random_search':
            # Random Search使用命令行flag，不需要修改代码
            return None

        original_file = self.base_dir / 'ppo_architecture_generator.py'
        modified_file = self.base_dir / f"ppo_architecture_generator_{experiment['code']}.py"

        self.log(f"Creating modified PPO generator: {modified_file.name}")

        # 读取原始文件
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 应用修改
        modifications = experiment.get('modifications', {})

        if experiment['code'] == 'no_diversity':
            # 移除diversity reward
            content = content.replace(
                '    width_variance = np.var(config.width_multipliers)\n    diversity_bonus = width_variance * 0.3',
                '    width_variance = np.var(config.width_multipliers)\n    diversity_bonus = 0.0  # ABLATION: removed'
            )
            content = content.replace(
                '    exploration_bonus = (max_width - min_width) * 0.2',
                '    exploration_bonus = 0.0  # ABLATION: removed'
            )

        elif experiment['code'] == 'no_region':
            # 修改region_boundaries，让所有region都是全范围
            content = content.replace(
                'self.region_boundaries = [(28, 35), (36, 44), (45, 53)]',
                'self.region_boundaries = [(28, 53), (28, 53), (28, 53)]  # ABLATION: all regions = full space'
            )
            # 固定使用region 0（实际是全范围）
            content = content.replace(
                'def choose_search_region(self, batch_num: int) -> int:',
                'def choose_search_region(self, batch_num: int) -> int:\n        return 0  # ABLATION: always region 0 (which is full space)\n        # Original implementation below (disabled):'
            )

        elif experiment['code'] == 'no_adaptive':
            # 固定exploration strength
            content = content.replace(
                'def update_exploration_strength(self, batch_unique_ratio: float):',
                'def update_exploration_strength(self, batch_unique_ratio: float):\n        return 0.5  # ABLATION: fixed strength\n        # Original implementation below (disabled):'
            )

        elif experiment['code'] == 'no_reset':
            # 禁用periodic reset
            content = content.replace(
                'def periodic_reset(self, batch_num):',
                'def periodic_reset(self, batch_num):\n        return  # ABLATION: disabled\n        # Original implementation below (disabled):'
            )

        # 写入修改后的文件
        with open(modified_file, 'w', encoding='utf-8') as f:
            f.write(content)

        self.log(f"  Created: {modified_file.name}")
        return modified_file

    def prepare_experiment_environment(self, experiment):
        """准备实验环境"""
        self.log(f"\n{'='*80}")
        self.log(f"Preparing experiment: {experiment['name']}")
        self.log(f"Description: {experiment['description']}")
        self.log(f"{'='*80}")

        # 创建修改版的PPO生成器（如果需要）
        modified_ppo = None
        if experiment['method'] == 'modify_ppo':
            modified_ppo = self.create_modified_ppo_generator(experiment)

            # 临时替换原始文件
            original_file = self.base_dir / 'ppo_architecture_generator.py'
            backup_file = self.base_dir / 'ppo_architecture_generator_backup.py'

            self.log("Backing up original PPO generator...")
            shutil.copy2(original_file, backup_file)

            self.log("Replacing with modified version...")
            shutil.copy2(modified_ppo, original_file)

        return modified_ppo

    def restore_original_ppo(self):
        """恢复原始PPO生成器"""
        original_file = self.base_dir / 'ppo_architecture_generator.py'
        backup_file = self.base_dir / 'ppo_architecture_generator_backup.py'

        if backup_file.exists():
            self.log("Restoring original PPO generator...")
            shutil.copy2(backup_file, original_file)
            backup_file.unlink()

    def build_command(self, experiment):
        """构建实验命令"""
        config = EXPERIMENT_CONFIG

        # 生成配置库路径
        library_name = f"{config['arch']}_{config['data']}_{experiment['library_suffix']}_architecture_library.json"
        library_path = self.results_dir / library_name

        # Random Search使用500次尝试，其他实验使用5000
        num_archs = 500 if experiment['code'] == 'random_search' else config['num_architectures']

        # 构建命令
        cmd = [
            'python', 'main.py',
            '--arch', config['arch'],
            '--data', config['data'],
            '--num_clients', str(config['num_clients']),
            '--num_rounds', str(config['num_rounds']),
            '--seed', str(config['seed']),
            '--num_architectures', str(num_archs),
            '--episodes_per_batch', str(config['episodes_per_batch']),
            '--config_library_path', str(library_path),
            '--flops_constraints'] + [str(x) for x in config['flops_constraints']] + [
            '--params_constraints'] + [str(x) for x in config['params_constraints']]

        # Random Search使用use_random_search flag (真正的随机搜索)
        if experiment['method'] == 'use_random_search':
            cmd.append('--use_random_search')
            # Random search使用层次化选择

        # 跳过Stage 1和Stage 2，只运行Stage 3
        cmd.append('--skip_stage1')
        cmd.append('--skip_stage2')
        cmd.append('--stages_only')
        cmd.append('3')

        return cmd, library_path

    def run_experiment(self, experiment):
        """运行单个实验"""
        start_time = datetime.now()
        self.log(f"\n{'#'*80}")
        self.log(f"# STARTING EXPERIMENT: {experiment['name']}")
        self.log(f"# Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"{'#'*80}\n")

        # 准备环境
        modified_ppo = self.prepare_experiment_environment(experiment)

        # 构建命令
        cmd, library_path = self.build_command(experiment)

        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"Library path: {library_path}")
        self.log("\nStarting training...\n")

        # 运行实验
        try:
            # 切换到项目目录
            os.chdir(self.base_dir)

            # 运行命令
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # 实时输出日志
            exp_log_file = self.results_dir / f"{experiment['code']}_output.log"
            with open(exp_log_file, 'w', encoding='utf-8') as log_f:
                for line in process.stdout:
                    print(line, end='')
                    log_f.write(line)
                    log_f.flush()

            # 等待完成
            return_code = process.wait()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 3600  # hours

            if return_code == 0:
                self.log(f"\n{'='*80}")
                self.log(f"EXPERIMENT COMPLETED: {experiment['name']}")
                self.log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.log(f"Duration: {duration:.2f} hours")
                self.log(f"{'='*80}\n")

                # 提取结果
                final_acc = self.extract_final_accuracy(exp_log_file)
                status = 'SUCCESS'
                notes = ''
            else:
                self.log(f"\n{'='*80}")
                self.log(f"EXPERIMENT FAILED: {experiment['name']}")
                self.log(f"Return code: {return_code}")
                self.log(f"{'='*80}\n")

                final_acc = 'N/A'
                status = 'FAILED'
                notes = f'Return code: {return_code}'

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 3600

            self.log(f"\n{'='*80}")
            self.log(f"EXPERIMENT ERROR: {experiment['name']}")
            self.log(f"Error: {str(e)}")
            self.log(f"{'='*80}\n")

            final_acc = 'N/A'
            status = 'ERROR'
            notes = str(e)

        finally:
            # 恢复原始PPO生成器
            if modified_ppo:
                self.restore_original_ppo()

        # 记录结果
        self.record_result(
            experiment, library_path,
            start_time, end_time, duration,
            final_acc, status, notes
        )

        return status == 'SUCCESS'

    def extract_final_accuracy(self, log_file):
        """从日志文件中提取最终测试准确率"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找最终测试准确率
            # 通常格式: "Final Test Accuracy: XX.XX%"
            import re
            matches = re.findall(r'Test.*?Accuracy.*?:\s*(\d+\.\d+)', content, re.IGNORECASE)
            if matches:
                return float(matches[-1])  # 取最后一个

            # 尝试其他格式
            matches = re.findall(r'test_acc.*?(\d+\.\d+)', content, re.IGNORECASE)
            if matches:
                return float(matches[-1])

            return 'N/A'
        except Exception as e:
            self.log(f"Warning: Could not extract accuracy: {e}")
            return 'N/A'

    def record_result(self, experiment, library_path, start_time, end_time,
                     duration, final_acc, status, notes):
        """记录实验结果到CSV"""
        with open(self.results_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                experiment['name'],
                experiment['code'],
                str(library_path),
                start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_time.strftime('%Y-%m-%d %H:%M:%S'),
                f'{duration:.2f}',
                final_acc,
                status,
                notes
            ])

    def run_all_experiments(self):
        """运行所有消融实验"""
        self.log(f"\n{'#'*80}")
        self.log(f"# AlignFL ABLATION STUDY")
        self.log(f"# Total experiments: {len(ABLATION_EXPERIMENTS)}")
        self.log(f"# Configuration:")
        for key, value in EXPERIMENT_CONFIG.items():
            self.log(f"#   {key}: {value}")
        self.log(f"{'#'*80}\n")

        success_count = 0
        total_start = datetime.now()

        for i, experiment in enumerate(ABLATION_EXPERIMENTS, 1):
            self.log(f"\n{'='*80}")
            self.log(f"EXPERIMENT {i}/{len(ABLATION_EXPERIMENTS)}: {experiment['name']}")
            self.log(f"{'='*80}")

            success = self.run_experiment(experiment)
            if success:
                success_count += 1

            # 短暂休息
            if i < len(ABLATION_EXPERIMENTS):
                self.log("\nWaiting 10 seconds before next experiment...\n")
                time.sleep(10)

        total_end = datetime.now()
        total_duration = (total_end - total_start).total_seconds() / 3600

        # 最终总结
        self.log(f"\n{'#'*80}")
        self.log(f"# ABLATION STUDY COMPLETED")
        self.log(f"# Total time: {total_duration:.2f} hours")
        self.log(f"# Successful: {success_count}/{len(ABLATION_EXPERIMENTS)}")
        self.log(f"# Results saved to: {self.results_csv}")
        self.log(f"{'#'*80}\n")

        # 显示结果表格
        self.display_results_table()

    def display_results_table(self):
        """显示结果表格"""
        self.log("\n" + "="*80)
        self.log("RESULTS SUMMARY")
        self.log("="*80)

        try:
            with open(self.results_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)

            # 打印表格
            self.log(f"\n{'Experiment':<30} {'Final Acc':<15} {'Duration':<15} {'Status':<10}")
            self.log("-"*80)
            for row in rows:
                exp_name = row[0]
                final_acc = row[6]
                duration = row[5]
                status = row[7]
                self.log(f"{exp_name:<30} {final_acc:<15} {duration:<15} {status:<10}")

        except Exception as e:
            self.log(f"Could not display results table: {e}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("AlignFL Stage 1 Ablation Study - Automated Experiment Runner")
    print("="*80)
    print("\nThis script will run 5 ablation experiments:")
    for i, exp in enumerate(ABLATION_EXPERIMENTS, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")

    print(f"\nConfiguration:")
    print(f"  Model: {EXPERIMENT_CONFIG['arch']}")
    print(f"  Dataset: {EXPERIMENT_CONFIG['data']}")
    print(f"  Clients: {EXPERIMENT_CONFIG['num_clients']}")
    print(f"  Rounds: {EXPERIMENT_CONFIG['num_rounds']}")
    print(f"  FLOPs constraints: {EXPERIMENT_CONFIG['flops_constraints']}")
    print(f"  Params constraints: {EXPERIMENT_CONFIG['params_constraints']}")

    print("\nEstimated total time: 60-90 hours (2.5-3.75 days)")

    response = input("\nDo you want to proceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return

    # 运行实验
    runner = AblationExperimentRunner()
    runner.run_all_experiments()

    print(f"\nAll experiments completed!")
    print(f"Results saved to: {runner.results_csv}")
    print(f"Log file: {runner.log_file}")


if __name__ == '__main__':
    main()
