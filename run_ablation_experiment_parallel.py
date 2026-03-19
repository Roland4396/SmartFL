"""
AlignFL 第一阶段探索策略消融实验 - 并行版本（2个实验同时运行）

针对单GPU（3080 10G）优化：
- 每次并行运行2个实验
- 5个实验分为3批：[实验1,2] -> [实验3,4] -> [实验5]
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
import threading
from queue import Queue

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
    # },  # 已完成，prec@1: 52.461%
    # {
    #     'name': 'w/o Region Search',
    #     'code': 'no_region',
    #     'description': 'No region constraint (search full exit_location space)',
    #     'method': 'modify_ppo',
    #     'library_suffix': 'no_region',
    # },  # 不需要了
    # {
    #     'name': 'w/o Adaptive Strength',
    #     'code': 'no_adaptive',
    #     'description': 'Fixed exploration strength (no TCP-AIMD)',
    #     'method': 'modify_ppo',
    #     'library_suffix': 'no_adaptive',
    # },
    {
        'name': 'w/o Partial Reset',
        'code': 'no_reset',
        'description': 'Disable periodic parameter reset',
        'method': 'modify_ppo',
        'library_suffix': 'no_reset',
    }
]


class ParallelAblationRunner:
    def __init__(self, base_dir=None, num_parallel=2):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.num_parallel = num_parallel
        self.results_dir = self.base_dir / 'ablation_results'
        self.results_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.results_dir / f'ablation_log_{self.timestamp}.txt'
        self.results_csv = self.results_dir / f'ablation_results_{self.timestamp}.csv'

        # 线程安全的日志锁
        self.log_lock = threading.Lock()

        # 初始化结果CSV
        with open(self.results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Experiment', 'Code', 'Library Path', 'Start Time', 'End Time',
                'Duration (hours)', 'Final Test Acc (%)', 'Status', 'Notes'
            ])

    def log(self, message, prefix=""):
        """线程安全的日志记录"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}]{prefix} {message}"

        with self.log_lock:
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

        # 读取原始文件
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 应用修改
        if experiment['code'] == 'no_diversity':
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
            content = content.replace(
                'def update_exploration_strength(self, batch_unique_ratio: float):',
                'def update_exploration_strength(self, batch_unique_ratio: float):\n        return 0.5  # ABLATION: fixed strength\n        # Original implementation below (disabled):'
            )

        elif experiment['code'] == 'no_reset':
            content = content.replace(
                'def periodic_reset(self, batch_num):',
                'def periodic_reset(self, batch_num):\n        return  # ABLATION: disabled\n        # Original implementation below (disabled):'
            )

        # 写入修改后的文件
        with open(modified_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return modified_file

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

    def run_single_experiment(self, experiment):
        """在独立线程中运行单个实验"""
        exp_prefix = f"[{experiment['code']}]"
        start_time = datetime.now()

        self.log(f"Starting experiment: {experiment['name']}", exp_prefix)

        # 准备环境
        modified_ppo = None
        backup_file = None

        if experiment['method'] == 'modify_ppo':
            modified_ppo = self.create_modified_ppo_generator(experiment)
            original_file = self.base_dir / 'ppo_architecture_generator.py'
            backup_file = self.base_dir / f"ppo_architecture_generator_backup_{experiment['code']}.py"

            # 备份原始文件
            shutil.copy2(original_file, backup_file)
            self.log(f"Backed up PPO generator", exp_prefix)

            # 替换为修改版本
            shutil.copy2(modified_ppo, original_file)
            self.log(f"Applied modifications", exp_prefix)

        # 构建命令
        cmd, library_path = self.build_command(experiment)
        self.log(f"Command: {' '.join(cmd[:15])}...", exp_prefix)

        # 运行实验
        exp_log_file = self.results_dir / f"{experiment['code']}_output.log"

        try:
            os.chdir(self.base_dir)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # 实时输出日志
            with open(exp_log_file, 'w', encoding='utf-8') as log_f:
                for line in process.stdout:
                    log_f.write(line)
                    log_f.flush()
                    # 可选：打印重要信息
                    if 'Epoch' in line or 'Round' in line or 'Accuracy' in line:
                        self.log(line.strip(), exp_prefix)

            return_code = process.wait()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 3600

            if return_code == 0:
                self.log(f"COMPLETED in {duration:.2f}h", exp_prefix)
                final_acc = self.extract_final_accuracy(exp_log_file)
                status = 'SUCCESS'
                notes = ''
            else:
                self.log(f"FAILED with code {return_code}", exp_prefix)
                final_acc = 'N/A'
                status = 'FAILED'
                notes = f'Return code: {return_code}'

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 3600
            self.log(f"ERROR: {str(e)}", exp_prefix)

            final_acc = 'N/A'
            status = 'ERROR'
            notes = str(e)

        finally:
            # 恢复原始PPO生成器
            if backup_file and backup_file.exists():
                original_file = self.base_dir / 'ppo_architecture_generator.py'
                shutil.copy2(backup_file, original_file)
                backup_file.unlink()
                self.log(f"Restored original PPO generator", exp_prefix)

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

            import re
            matches = re.findall(r'Test.*?Accuracy.*?:\s*(\d+\.\d+)', content, re.IGNORECASE)
            if matches:
                return float(matches[-1])

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
        with self.log_lock:
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

    def run_experiments_batch(self, experiments):
        """并行运行一批实验（使用线程）"""
        threads = []

        for experiment in experiments:
            thread = threading.Thread(
                target=self.run_single_experiment,
                args=(experiment,)
            )
            thread.start()
            threads.append(thread)

        # 等待所有线程完成
        for thread in threads:
            thread.join()

    def run_all_experiments(self):
        """分批并行运行所有实验"""
        self.log(f"\n{'#'*80}")
        self.log(f"# AlignFL ABLATION STUDY - PARALLEL MODE")
        self.log(f"# Total experiments: {len(ABLATION_EXPERIMENTS)}")
        self.log(f"# Parallel workers: {self.num_parallel}")
        self.log(f"# Configuration:")
        for key, value in EXPERIMENT_CONFIG.items():
            self.log(f"#   {key}: {value}")
        self.log(f"{'#'*80}\n")

        total_start = datetime.now()

        # 将实验分批，每批最多num_parallel个
        batches = []
        for i in range(0, len(ABLATION_EXPERIMENTS), self.num_parallel):
            batch = ABLATION_EXPERIMENTS[i:i + self.num_parallel]
            batches.append(batch)

        self.log(f"Split into {len(batches)} batches:")
        for i, batch in enumerate(batches, 1):
            batch_names = ', '.join([exp['name'] for exp in batch])
            self.log(f"  Batch {i}: {batch_names}")

        # 依次运行每批
        for batch_idx, batch in enumerate(batches, 1):
            self.log(f"\n{'='*80}")
            self.log(f"BATCH {batch_idx}/{len(batches)}: Running {len(batch)} experiments in parallel")
            self.log(f"{'='*80}")

            batch_start = datetime.now()
            self.run_experiments_batch(batch)
            batch_end = datetime.now()

            batch_duration = (batch_end - batch_start).total_seconds() / 3600
            self.log(f"\nBatch {batch_idx} completed in {batch_duration:.2f}h")

            # 批次之间休息
            if batch_idx < len(batches):
                self.log("\nWaiting 30 seconds before next batch...\n")
                time.sleep(30)

        total_end = datetime.now()
        total_duration = (total_end - total_start).total_seconds() / 3600

        # 最终总结
        self.log(f"\n{'#'*80}")
        self.log(f"# ABLATION STUDY COMPLETED")
        self.log(f"# Total time: {total_duration:.2f} hours")
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
    print("AlignFL Stage 1 Ablation Study - PARALLEL MODE (2 experiments at once)")
    print("="*80)
    print("\nThis script will run 5 ablation experiments in parallel:")
    print("  Batch 1: Random Search + w/o Diversity Reward")
    print("  Batch 2: w/o Region Search + w/o Adaptive Strength")
    print("  Batch 3: w/o Partial Reset")

    print(f"\nConfiguration:")
    print(f"  Model: {EXPERIMENT_CONFIG['arch']}")
    print(f"  Dataset: {EXPERIMENT_CONFIG['data']}")
    print(f"  FLOPs constraints: {EXPERIMENT_CONFIG['flops_constraints']}")
    print(f"  Params constraints: {EXPERIMENT_CONFIG['params_constraints']}")
    print(f"\nGPU: Single 3080 10G (shared by 2 experiments)")

    print("\nEstimated total time: 30-45 hours (1.25-1.9 days)")
    print("  - Batch 1&2: ~12-18h each")
    print("  - Batch 3: ~12-18h")

    response = input("\nDo you want to proceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return

    # 运行实验
    runner = ParallelAblationRunner(num_parallel=2)
    runner.run_all_experiments()

    print(f"\nAll experiments completed!")
    print(f"Results saved to: {runner.results_csv}")
    print(f"Log file: {runner.log_file}")


if __name__ == '__main__':
    main()
