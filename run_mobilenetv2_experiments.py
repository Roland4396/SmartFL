"""
MobileNetV2 批量实验脚本

测试MobileNetV2在3个数据集、2个alpha值下的性能
- 数据集: CIFAR10, CIFAR100, TinyImageNet
- Alpha: 1 (高度异构), 100 (接近IID)
- 并行度: 3个实验同时运行
"""

import os
import sys
import subprocess
import time
import csv
from datetime import datetime
from pathlib import Path
import threading
from queue import Queue

# 实验配置
BASE_CONFIG = {
    'arch': 'mobilenetv2',
    'num_clients': 100,
    'num_rounds': 400,
    'seed': 0,
    'num_architectures': 5000,
    'episodes_per_batch': 100,
}

# MobileNetV2 4个级别的约束
MOBILENETV2_CONSTRAINTS = {
    'flops_constraints': [10.69, 16.47, 22.47, 27.82],
    'params_constraints': [0.393, 0.663, 1.234, 2.255],
}

# 实验列表
EXPERIMENTS = [
    {'dataset': 'cifar10', 'alpha': 1},
    {'dataset': 'cifar10', 'alpha': 100},
    {'dataset': 'cifar100', 'alpha': 1},
    {'dataset': 'cifar100', 'alpha': 100},
    {'dataset': 'tiny_imagenet', 'alpha': 1},
    {'dataset': 'tiny_imagenet', 'alpha': 100},
]


class MobileNetV2ExperimentRunner:
    def __init__(self, base_dir=None, num_parallel=3):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.num_parallel = num_parallel
        self.results_dir = self.base_dir / 'mobilenetv2_results'
        self.results_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.results_dir / f'mobilenetv2_log_{self.timestamp}.txt'
        self.results_csv = self.results_dir / f'mobilenetv2_results_{self.timestamp}.csv'

        # 线程安全的日志锁
        self.log_lock = threading.Lock()

        # 初始化结果CSV
        with open(self.results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Dataset', 'Alpha', 'Start Time', 'End Time',
                'Duration (hours)', 'Model 0 Acc (%)', 'Model 1 Acc (%)',
                'Model 2 Acc (%)', 'Model 3 Acc (%)', 'Avg Acc (%)',
                'Status', 'Notes'
            ])

    def log(self, message, prefix=""):
        """线程安全的日志记录"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}]{prefix} {message}"

        with self.log_lock:
            print(log_message)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')

    def build_command(self, experiment):
        """构建实验命令"""
        dataset = experiment['dataset']
        alpha = experiment['alpha']

        # 生成配置库路径
        library_name = f"mobilenetv2_{dataset}_alpha{alpha}_architecture_library.json"
        library_path = self.results_dir / library_name

        # 构建命令
        cmd = [
            'python', 'main.py',
            '--arch', BASE_CONFIG['arch'],
            '--data', dataset,
            '--num_clients', str(BASE_CONFIG['num_clients']),
            '--num_rounds', str(BASE_CONFIG['num_rounds']),
            '--seed', str(BASE_CONFIG['seed']),
            '--alpha', str(alpha),
            '--num_architectures', str(BASE_CONFIG['num_architectures']),
            '--episodes_per_batch', str(BASE_CONFIG['episodes_per_batch']),
            '--config_library_path', str(library_path),
            '--flops_constraints'] + [str(x) for x in MOBILENETV2_CONSTRAINTS['flops_constraints']] + [
            '--params_constraints'] + [str(x) for x in MOBILENETV2_CONSTRAINTS['params_constraints']]

        return cmd, library_path

    def extract_accuracies(self, log_file):
        """从日志中提取4个级别的测试准确率"""
        accuracies = [None, None, None, None]

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找最终测试结果
            # 假设格式类似：Level 0/4 * prec@1 XX.XXX
            import re
            for level in range(4):
                pattern = rf'Level {level}/4.*?prec@1\s+([\d.]+)'
                match = re.search(pattern, content)
                if match:
                    accuracies[level] = float(match.group(1))

        except Exception as e:
            self.log(f"Error extracting accuracies: {e}")

        return accuracies

    def run_single_experiment(self, experiment):
        """在独立线程中运行单个实验"""
        dataset = experiment['dataset']
        alpha = experiment['alpha']
        exp_prefix = f"[{dataset}-α{alpha}]"
        start_time = datetime.now()

        self.log(f"Starting experiment: {dataset} with alpha={alpha}", exp_prefix)

        # 构建命令
        cmd, library_path = self.build_command(experiment)
        self.log(f"Command: {' '.join(cmd[:15])}...", exp_prefix)

        # 运行实验
        exp_log_file = self.results_dir / f"{dataset}_alpha{alpha}_output.log"

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
                    # 打印重要信息
                    if 'Round' in line or 'prec@1' in line or 'STAGE' in line:
                        self.log(line.strip(), exp_prefix)

            return_code = process.wait()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 3600

            if return_code == 0:
                self.log(f"COMPLETED in {duration:.2f}h", exp_prefix)
                accuracies = self.extract_accuracies(exp_log_file)
                avg_acc = sum([a for a in accuracies if a is not None]) / len([a for a in accuracies if a is not None]) if any(accuracies) else None
                status = 'SUCCESS'
                notes = ''
            else:
                self.log(f"FAILED with code {return_code}", exp_prefix)
                accuracies = [None, None, None, None]
                avg_acc = None
                status = 'FAILED'
                notes = f'Return code: {return_code}'

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 3600
            self.log(f"ERROR: {str(e)}", exp_prefix)

            accuracies = [None, None, None, None]
            avg_acc = None
            status = 'ERROR'
            notes = str(e)

        # 记录结果
        with self.log_lock:
            with open(self.results_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset,
                    alpha,
                    start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    f"{duration:.2f}",
                    accuracies[0] if accuracies[0] else 'N/A',
                    accuracies[1] if accuracies[1] else 'N/A',
                    accuracies[2] if accuracies[2] else 'N/A',
                    accuracies[3] if accuracies[3] else 'N/A',
                    f"{avg_acc:.2f}" if avg_acc else 'N/A',
                    status,
                    notes
                ])

        return status == 'SUCCESS'

    def run_experiments_parallel(self):
        """并行运行实验（每批3个）"""
        total_experiments = len(EXPERIMENTS)
        self.log(f"\n{'#'*80}")
        self.log(f"# MobileNetV2 Batch Experiments")
        self.log(f"# Total experiments: {total_experiments}")
        self.log(f"# Parallel workers: {self.num_parallel}")
        self.log(f"# Batches: {(total_experiments + self.num_parallel - 1) // self.num_parallel}")
        self.log(f"{'#'*80}\n")

        success_count = 0
        total_start = datetime.now()

        # 分批运行
        for batch_idx in range(0, total_experiments, self.num_parallel):
            batch_experiments = EXPERIMENTS[batch_idx:batch_idx + self.num_parallel]

            self.log(f"\n{'='*80}")
            self.log(f"BATCH {batch_idx//self.num_parallel + 1}: Running {len(batch_experiments)} experiments in parallel")
            self.log(f"{'='*80}")

            # 创建线程
            threads = []
            for exp in batch_experiments:
                thread = threading.Thread(
                    target=self.run_single_experiment,
                    args=(exp,)
                )
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            self.log(f"\nBatch {batch_idx//self.num_parallel + 1} completed")
            time.sleep(5)  # 短暂休息

        total_end = datetime.now()
        total_duration = (total_end - total_start).total_seconds() / 3600

        # 最终总结
        self.log(f"\n{'#'*80}")
        self.log(f"# ALL EXPERIMENTS COMPLETED")
        self.log(f"# Total time: {total_duration:.2f} hours")
        self.log(f"# Results saved to: {self.results_csv}")
        self.log(f"{'#'*80}\n")


if __name__ == "__main__":
    runner = MobileNetV2ExperimentRunner(num_parallel=3)
    runner.run_experiments_parallel()
