#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ArchSpec:
    model: str
    artifact_prefix: str
    flops_constraints: tuple[float, ...]
    params_constraints: tuple[float, ...]


ARCH_SPECS: dict[str, ArchSpec] = {
    # Values are taken from the repo's existing args.txt files and experiment scripts.
    "resnet110_4": ArchSpec(
        model="resnet",
        artifact_prefix="resnet",
        flops_constraints=(83.4, 99.7, 138.5, 253.1),
        params_constraints=(0.21, 0.46, 0.86, 1.73),
    ),
    "vgg16_4": ArchSpec(
        model="vgg",
        artifact_prefix="vgg",
        flops_constraints=(285.9, 443.4, 513.4, 532.4),
        params_constraints=(5.806, 9.834, 18.424, 33.647),
    ),
    "mobilenet_v2_4": ArchSpec(
        model="mobilenet",
        artifact_prefix="mobilenetv2",
        flops_constraints=(10.6, 16.4, 22.4, 27.8),
        params_constraints=(0.390, 0.677, 1.235, 2.255),
    ),
}

DEFAULT_ARCHS = list(ARCH_SPECS.keys())
DEFAULT_DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
DEFAULT_ALPHAS = [100]
FULL_SWEEP_ALPHAS = [1, 100]
DEFAULT_TDD_MODES = ["on"]
DEFAULT_CLIENT_SPLIT_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
COMPARE_SWEEP_CLIENT_SPLIT_WEIGHTS = [4.0, 3.0, 2.0, 1.0]
COMPARE_SWEEP_TDD_MODES = ["on", "off"]
RESNET_COMPARE_ARCHS = ["resnet110_4"]
RESNET_COMPARE_TDD_MODES = ["on", "off"]
RESNET_COMPARE_CLIENT_SPLIT_WEIGHTS = [4.0, 3.0, 2.0, 1.0]


@dataclass(frozen=True)
class Experiment:
    arch: str
    dataset: str
    alpha: int
    tdd_mode: str


@dataclass
class RunningExperiment:
    exp: Experiment
    process: subprocess.Popen
    run_log_handle: object
    supernet_path: Path
    library_path: Path
    save_path: Path
    stage_plan: str
    start_time: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for TDD training on SmartFL."
    )
    parser.add_argument(
        "--full_sweep",
        action="store_true",
        help=(
            "Run the full 4-block paper sweep: all supported backbones, all datasets, "
            "and alphas 1 and 100."
        ),
    )
    parser.add_argument(
        "--resnet_compare_sweep",
        action="store_true",
        help=(
            "Run the resnet110_4 sweep only: 3 datasets x 2 alphas (1,100) x "
            "TDD on/off, using client split weights 4:3:2:1 by default."
        ),
    )
    parser.add_argument(
        "--compare_sweep",
        action="store_true",
        help=(
            "Run the selected backbones with the comparison setup: all datasets, "
            "alphas 1 and 100, TDD on/off, and client split weights 4:3:2:1."
        ),
    )
    parser.add_argument("--archs", nargs="+", default=None, choices=DEFAULT_ARCHS)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=DEFAULT_DATASETS,
    )
    parser.add_argument("--alphas", nargs="+", type=int, default=None)
    parser.add_argument(
        "--tdd_modes",
        nargs="+",
        default=None,
        choices=["on", "off"],
        help="Choose whether to run TDD-enabled jobs, TDD-disabled jobs, or both.",
    )
    parser.add_argument(
        "--client_split_ratios",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Client split weights for the four complexity levels. Values are "
            "normalized automatically, so both '4 3 2 1' and '0.4 0.3 0.2 0.1' work."
        ),
    )
    parser.add_argument("--num_rounds", type=int, default=400)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument(
        "--validate_every",
        type=int,
        default=1,
        help="Run local validation every N rounds. 1 means every round.",
    )
    parser.add_argument(
        "--cpu_threads_per_job",
        type=int,
        default=0,
        help=(
            "Limit CPU/OpenMP threads used by each training subprocess. "
            "0 means auto-tune from host CPU count and max_parallel."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override main.py batch size (-b).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override DataLoader workers passed to main.py (-j).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional suffix added to the output save_path.",
    )
    parser.add_argument("--gpu_idx", type=str, default="0")
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--num_architectures", type=int, default=50000)
    parser.add_argument("--episodes_per_batch", type=int, default=100)
    parser.add_argument("--rotation_period", type=int, default=10)
    parser.add_argument("--tdd_growth_ratio", type=float, default=0.5)
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Maximum number of experiments to run concurrently on the same machine.",
    )
    parser.add_argument(
        "--existing_only",
        action="store_true",
        help="Only run combinations whose architecture library already exists.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun experiments even if test_scores.tsv already exists.",
    )
    parser.add_argument(
        "--no_auto_resume",
        action="store_true",
        help="Do not append --resume when an unfinished run already has a checkpoint.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="tdd_batch_results",
        help="Directory for batch logs and CSV summaries.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--phase_timing",
        action="store_true",
        help="Enable per-round timing breakdown logs inside each training run.",
    )
    args = parser.parse_args()

    selected_sweeps = [
        args.full_sweep,
        args.resnet_compare_sweep,
        args.compare_sweep,
    ]
    if sum(bool(flag) for flag in selected_sweeps) > 1:
        parser.error(
            "--full_sweep, --resnet_compare_sweep, and --compare_sweep are mutually exclusive."
        )

    if args.full_sweep:
        args.archs = args.archs or DEFAULT_ARCHS.copy()
        args.datasets = args.datasets or DEFAULT_DATASETS.copy()
        args.alphas = args.alphas or FULL_SWEEP_ALPHAS.copy()
        args.tdd_modes = args.tdd_modes or DEFAULT_TDD_MODES.copy()
        args.client_split_ratios = args.client_split_ratios or DEFAULT_CLIENT_SPLIT_WEIGHTS.copy()
    elif args.resnet_compare_sweep:
        args.archs = args.archs or RESNET_COMPARE_ARCHS.copy()
        args.datasets = args.datasets or DEFAULT_DATASETS.copy()
        args.alphas = args.alphas or FULL_SWEEP_ALPHAS.copy()
        args.tdd_modes = args.tdd_modes or RESNET_COMPARE_TDD_MODES.copy()
        args.client_split_ratios = (
            args.client_split_ratios or RESNET_COMPARE_CLIENT_SPLIT_WEIGHTS.copy()
        )
    elif args.compare_sweep:
        args.archs = args.archs or DEFAULT_ARCHS.copy()
        args.datasets = args.datasets or DEFAULT_DATASETS.copy()
        args.alphas = args.alphas or FULL_SWEEP_ALPHAS.copy()
        args.tdd_modes = args.tdd_modes or COMPARE_SWEEP_TDD_MODES.copy()
        args.client_split_ratios = (
            args.client_split_ratios or COMPARE_SWEEP_CLIENT_SPLIT_WEIGHTS.copy()
        )
    else:
        args.archs = args.archs or DEFAULT_ARCHS.copy()
        args.datasets = args.datasets or DEFAULT_DATASETS.copy()
        args.alphas = args.alphas or DEFAULT_ALPHAS.copy()
        args.tdd_modes = args.tdd_modes or DEFAULT_TDD_MODES.copy()
        args.client_split_ratios = args.client_split_ratios or DEFAULT_CLIENT_SPLIT_WEIGHTS.copy()

    expected_level_counts = {
        len(ARCH_SPECS[arch].flops_constraints)
        for arch in args.archs
    }
    if len(expected_level_counts) != 1:
        parser.error("Selected architectures must have the same number of FL levels.")

    expected_levels = expected_level_counts.pop()
    if len(args.client_split_ratios) != expected_levels:
        parser.error(
            f"--client_split_ratios expects {expected_levels} values for the selected architectures."
        )

    if any(value <= 0 for value in args.client_split_ratios):
        parser.error("--client_split_ratios values must all be positive.")

    ratio_sum = sum(args.client_split_ratios)
    args.client_split_tag = "-".join(format_ratio_tag(value) for value in args.client_split_ratios)
    args.client_split_ratios = tuple(value / ratio_sum for value in args.client_split_ratios)
    args.tdd_modes = list(dict.fromkeys(args.tdd_modes))

    return args


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def tdd_enabled(exp: Experiment) -> bool:
    return exp.tdd_mode == "on"


def mode_prefix(exp: Experiment) -> str:
    return "tdd" if tdd_enabled(exp) else "notdd"


def format_ratio_text(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def client_split_text(args: argparse.Namespace) -> str:
    return ":".join(format_ratio_text(value) for value in args.client_split_ratios)


def get_supernet_path(root: Path, exp: Experiment) -> Path:
    prefix = ARCH_SPECS[exp.arch].artifact_prefix
    return root / f"{prefix}_{exp.dataset}_supernet.pth"


def get_library_path(root: Path, exp: Experiment) -> Path:
    prefix = ARCH_SPECS[exp.arch].artifact_prefix
    return root / f"{prefix}_{exp.dataset}_architecture_library.json"


def format_ratio_tag(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def get_save_path(root: Path, exp: Experiment, args: argparse.Namespace) -> Path:
    save_name = (
        f"{mode_prefix(exp)}_{exp.arch}_{exp.dataset}_a{exp.alpha}_s{args.seed}"
        f"_cs{args.client_split_tag}"
    )
    if tdd_enabled(exp):
        growth_tag = format_ratio_tag(args.tdd_growth_ratio)
        save_name += f"_rp{args.rotation_period}_g{growth_tag}"
    if args.run_tag:
        save_name = f"{save_name}_{args.run_tag}"
    return root / "outputs" / save_name


def existing_checkpoint(save_path: Path) -> bool:
    model_dir = save_path / "save_models"
    return model_dir.exists() and any(model_dir.glob("checkpoint_*.pth.tar"))


def latest_checkpoint_round(save_path: Path) -> int | None:
    model_dir = save_path / "save_models"
    if not model_dir.exists():
        return None

    latest_round: int | None = None
    for checkpoint in model_dir.glob("checkpoint_*.pth.tar"):
        match = re.match(r"checkpoint_(\d+)\.pth\.tar$", checkpoint.name)
        if not match:
            continue
        round_idx = int(match.group(1))
        if latest_round is None or round_idx > latest_round:
            latest_round = round_idx
    return latest_round


def latest_score_round(save_path: Path) -> int | None:
    score_path = save_path / "scores.tsv"
    if not score_path.exists():
        return None

    lines = score_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        first_field = line.split("\t", 1)[0]
        if first_field.isdigit():
            return int(first_field)
    return None


def completed_run(save_path: Path, num_rounds: int) -> bool:
    expected_last_round = max(0, num_rounds - 1)

    checkpoint_round = latest_checkpoint_round(save_path)
    if checkpoint_round is not None and checkpoint_round >= expected_last_round:
        return True

    score_round = latest_score_round(save_path)
    if score_round is not None and score_round >= expected_last_round:
        return True

    return False


def determine_stage_plan(supernet_path: Path, library_path: Path) -> tuple[str, list[str]]:
    if library_path.exists():
        return "3", ["--skip_stage1", "--skip_stage2", "--stages_only", "3"]
    if supernet_path.exists():
        return "2,3", ["--skip_stage1", "--stages_only", "2,3"]
    return "1,2,3", ["--stages_only", "1,2,3"]


def build_command(
    root: Path,
    exp: Experiment,
    args: argparse.Namespace,
) -> tuple[list[str], Path, Path, Path, str]:
    spec = ARCH_SPECS[exp.arch]
    supernet_path = get_supernet_path(root, exp)
    library_path = get_library_path(root, exp)
    save_path = get_save_path(root, exp, args)
    stage_plan, stage_args = determine_stage_plan(supernet_path, library_path)

    cmd = [
        sys.executable,
        "main.py",
        "--arch",
        exp.arch,
        "--model",
        spec.model,
        "--data",
        exp.dataset,
        "--alpha",
        str(exp.alpha),
        "--seed",
        str(args.seed),
        "--gpu_idx",
        args.gpu_idx,
        "--use_gpu",
        str(args.use_gpu),
        "--num_clients",
        str(args.num_clients),
        "--num_rounds",
        str(args.num_rounds),
        "--sample_rate",
        str(args.sample_rate),
        "--validate_every",
        str(args.validate_every),
        "--client_split_ratios",
        *[str(v) for v in args.client_split_ratios],
        "--num_architectures",
        str(args.num_architectures),
        "--episodes_per_batch",
        str(args.episodes_per_batch),
        "--supernet_save_path",
        str(supernet_path),
        "--config_library_path",
        str(library_path),
        "--save_path",
        str(save_path),
        "--flops_constraints",
        *[str(v) for v in spec.flops_constraints],
        "--params_constraints",
        *[str(v) for v in spec.params_constraints],
        *stage_args,
    ]

    if tdd_enabled(exp):
        cmd.extend(
            [
                "--enable_tdd",
                "1",
                "--rotation_period",
                str(args.rotation_period),
                "--tdd_growth_ratio",
                str(args.tdd_growth_ratio),
            ]
        )
    else:
        cmd.extend(["--enable_tdd", "0"])

    if args.batch_size is not None:
        cmd.extend(["-b", str(args.batch_size)])

    if args.workers is not None:
        cmd.extend(["-j", str(args.workers)])

    if args.phase_timing:
        cmd.append("--phase_timing")

    if (
        save_path.exists()
        and existing_checkpoint(save_path)
        and not args.no_auto_resume
        and not args.rerun
    ):
        cmd.append("--resume")

    return cmd, supernet_path, library_path, save_path, stage_plan


def parse_test_score(test_score_path: Path) -> str:
    if not test_score_path.exists():
        return ""

    pattern = re.compile(r"prec@1\s+([0-9.]+)")
    last_match = ""
    for line in test_score_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            last_match = match.group(1)
    return last_match


def print_command(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def get_effective_cpu_threads_per_job(args: argparse.Namespace, num_experiments: int) -> int:
    if args.cpu_threads_per_job > 0:
        return args.cpu_threads_per_job

    cpu_total = os.cpu_count() or 1
    target_parallel = max(1, min(args.max_parallel, max(1, num_experiments)))
    # Clamp to a small per-job thread count; this workload oversubscribes CPU badly
    # when multiple PyTorch jobs each spawn many intra-op threads.
    return max(1, min(4, cpu_total // target_parallel))


def build_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    thread_count = str(args.effective_cpu_threads_per_job)

    env["OMP_NUM_THREADS"] = thread_count
    env["OMP_THREAD_LIMIT"] = thread_count
    env["MKL_NUM_THREADS"] = thread_count
    env["OPENBLAS_NUM_THREADS"] = thread_count
    env["NUMEXPR_NUM_THREADS"] = thread_count
    env["VECLIB_MAXIMUM_THREADS"] = thread_count
    env["BLIS_NUM_THREADS"] = thread_count
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    env.setdefault("KMP_BLOCKTIME", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    return env


def get_run_log_path(csv_path: Path, exp: Experiment, args: argparse.Namespace) -> Path:
    log_name = (
        f"{mode_prefix(exp)}_{exp.arch}_{exp.dataset}_a{exp.alpha}_"
        f"cs{args.client_split_tag}"
    )
    if tdd_enabled(exp):
        log_name += (
            f"_rp{args.rotation_period}_g{format_ratio_tag(args.tdd_growth_ratio)}"
        )
    log_name += ".log"
    return csv_path.parent / log_name


def collect_experiments(root: Path, args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    for arch in args.archs:
        for dataset in args.datasets:
            for alpha in args.alphas:
                for tdd_mode in args.tdd_modes:
                    exp = Experiment(
                        arch=arch,
                        dataset=dataset,
                        alpha=alpha,
                        tdd_mode=tdd_mode,
                    )
                    if args.existing_only and not get_library_path(root, exp).exists():
                        continue
                    experiments.append(exp)
    return experiments


def ensure_results_dir(root: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"tdd_batch_{timestamp}.csv"
    log_path = results_dir / f"tdd_batch_{timestamp}.log"
    return results_dir, csv_path, log_path


def write_csv_header(csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "arch",
                "dataset",
                "alpha",
                "tdd_mode",
                "client_split_ratios",
                "stage_plan",
                "status",
                "start_time",
                "end_time",
                "duration_hours",
                "final_prec1",
                "supernet_path",
                "config_library_path",
                "save_path",
                "notes",
            ]
        )


def append_csv_row(csv_path: Path, row: list[str]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def log_line(log_path: Path, text: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {text}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_experiment(
    root: Path,
    exp: Experiment,
    args: argparse.Namespace,
    csv_path: Path,
    log_path: Path,
) -> bool:
    cmd, supernet_path, library_path, save_path, stage_plan = build_command(root, exp, args)
    test_score_path = save_path / "test_scores.tsv"

    if completed_run(save_path, args.num_rounds) and not args.rerun:
        final_prec1 = parse_test_score(test_score_path)
        log_line(
            log_path,
            f"SKIP {exp.arch} {exp.dataset} alpha={exp.alpha} "
            f"tdd={exp.tdd_mode} "
            f"(existing result: {save_path})",
        )
        append_csv_row(
            csv_path,
            [
                exp.arch,
                exp.dataset,
                str(exp.alpha),
                exp.tdd_mode,
                client_split_text(args),
                stage_plan,
                "SKIPPED",
                "",
                "",
                "0.00",
                final_prec1,
                str(supernet_path),
                str(library_path),
                str(save_path),
                "test_scores.tsv already exists",
            ],
        )
        return True

    cmd_text = print_command(cmd)
    run_log_path = get_run_log_path(csv_path, exp, args)

    log_line(
        log_path,
        f"START {exp.arch} {exp.dataset} alpha={exp.alpha} "
        f"tdd={exp.tdd_mode} stages={stage_plan}",
    )
    log_line(log_path, f"CMD {cmd_text}")

    if args.dry_run:
        append_csv_row(
            csv_path,
            [
                exp.arch,
                exp.dataset,
                str(exp.alpha),
                exp.tdd_mode,
                client_split_text(args),
                stage_plan,
                "DRY_RUN",
                "",
                "",
                "0.00",
                "",
                str(supernet_path),
                str(library_path),
                str(save_path),
                cmd_text,
            ],
        )
        return True

    start_time = datetime.now()
    status = "SUCCESS"
    notes = ""

    with run_log_path.open("w", encoding="utf-8") as run_log:
        process = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=build_subprocess_env(args),
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            run_log.write(line)
            run_log.flush()

        return_code = process.wait()
        if return_code != 0:
            status = "FAILED"
            notes = f"return code {return_code}"

    end_time = datetime.now()
    duration_hours = (end_time - start_time).total_seconds() / 3600.0
    final_prec1 = parse_test_score(test_score_path)

    log_line(
        log_path,
        f"{status} {exp.arch} {exp.dataset} alpha={exp.alpha} "
        f"tdd={exp.tdd_mode} "
        f"prec1={final_prec1 or 'N/A'} duration={duration_hours:.2f}h",
    )
    append_csv_row(
        csv_path,
        [
            exp.arch,
            exp.dataset,
            str(exp.alpha),
            exp.tdd_mode,
            client_split_text(args),
            stage_plan,
            status,
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{duration_hours:.2f}",
            final_prec1,
            str(supernet_path),
            str(library_path),
            str(save_path),
            notes,
        ],
    )
    return status == "SUCCESS"


def start_experiment(
    root: Path,
    exp: Experiment,
    args: argparse.Namespace,
    csv_path: Path,
    log_path: Path,
) -> RunningExperiment | None:
    cmd, supernet_path, library_path, save_path, stage_plan = build_command(root, exp, args)
    test_score_path = save_path / "test_scores.tsv"

    if completed_run(save_path, args.num_rounds) and not args.rerun:
        final_prec1 = parse_test_score(test_score_path)
        log_line(
            log_path,
            f"SKIP {exp.arch} {exp.dataset} alpha={exp.alpha} "
            f"tdd={exp.tdd_mode} "
            f"(existing result: {save_path})",
        )
        append_csv_row(
            csv_path,
            [
                exp.arch,
                exp.dataset,
                str(exp.alpha),
                exp.tdd_mode,
                client_split_text(args),
                stage_plan,
                "SKIPPED",
                "",
                "",
                "0.00",
                final_prec1,
                str(supernet_path),
                str(library_path),
                str(save_path),
                "test_scores.tsv already exists",
            ],
        )
        return None

    cmd_text = print_command(cmd)
    run_log_path = get_run_log_path(csv_path, exp, args)

    log_line(
        log_path,
        f"START {exp.arch} {exp.dataset} alpha={exp.alpha} "
        f"tdd={exp.tdd_mode} stages={stage_plan}",
    )
    log_line(log_path, f"CMD {cmd_text}")

    if args.dry_run:
        append_csv_row(
            csv_path,
            [
                exp.arch,
                exp.dataset,
                str(exp.alpha),
                exp.tdd_mode,
                client_split_text(args),
                stage_plan,
                "DRY_RUN",
                "",
                "",
                "0.00",
                "",
                str(supernet_path),
                str(library_path),
                str(save_path),
                cmd_text,
            ],
        )
        return None

    run_log_handle = run_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=root,
        stdout=run_log_handle,
        stderr=subprocess.STDOUT,
        env=build_subprocess_env(args),
        text=True,
        bufsize=1,
    )
    return RunningExperiment(
        exp=exp,
        process=process,
        run_log_handle=run_log_handle,
        supernet_path=supernet_path,
        library_path=library_path,
        save_path=save_path,
        stage_plan=stage_plan,
        start_time=datetime.now(),
    )


def finalize_experiment(
    running: RunningExperiment,
    args: argparse.Namespace,
    csv_path: Path,
    log_path: Path,
) -> bool:
    return_code = running.process.wait()
    running.run_log_handle.close()

    end_time = datetime.now()
    duration_hours = (end_time - running.start_time).total_seconds() / 3600.0
    final_prec1 = parse_test_score(running.save_path / "test_scores.tsv")
    status = "SUCCESS" if return_code == 0 else "FAILED"
    notes = "" if return_code == 0 else f"return code {return_code}"

    log_line(
        log_path,
        f"{status} {running.exp.arch} {running.exp.dataset} alpha={running.exp.alpha} "
        f"tdd={running.exp.tdd_mode} "
        f"prec1={final_prec1 or 'N/A'} duration={duration_hours:.2f}h",
    )
    append_csv_row(
        csv_path,
        [
            running.exp.arch,
            running.exp.dataset,
            str(running.exp.alpha),
            running.exp.tdd_mode,
            client_split_text(args),
            running.stage_plan,
            status,
            running.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{duration_hours:.2f}",
            final_prec1,
            str(running.supernet_path),
            str(running.library_path),
            str(running.save_path),
            notes,
        ],
    )
    return status == "SUCCESS"


def run_experiments_parallel(
    root: Path,
    experiments: list[Experiment],
    args: argparse.Namespace,
    csv_path: Path,
    log_path: Path,
) -> int:
    failures = 0
    running: list[RunningExperiment] = []
    next_index = 0

    while next_index < len(experiments) or running:
        while next_index < len(experiments) and len(running) < args.max_parallel:
            exp = experiments[next_index]
            next_index += 1
            log_line(log_path, f"QUEUE [{next_index}/{len(experiments)}] {exp}")
            launched = start_experiment(root, exp, args, csv_path, log_path)
            if launched is not None:
                running.append(launched)
                if len(running) < args.max_parallel and next_index < len(experiments):
                    time.sleep(2)

        if not running:
            continue

        time.sleep(5)
        still_running: list[RunningExperiment] = []
        for item in running:
            if item.process.poll() is None:
                still_running.append(item)
                continue

            ok = finalize_experiment(item, args, csv_path, log_path)
            if not ok:
                failures += 1
        running = still_running

    return failures


def main() -> int:
    args = parse_args()
    root = repo_root()
    experiments = collect_experiments(root, args)
    args.effective_cpu_threads_per_job = get_effective_cpu_threads_per_job(args, len(experiments))

    if not experiments:
        print("No experiments selected.")
        return 0

    _, csv_path, log_path = ensure_results_dir(root, args)
    write_csv_header(csv_path)

    log_line(log_path, f"Repo root: {root}")
    log_line(log_path, f"Selected experiments: {len(experiments)}")
    log_line(
        log_path,
        f"TDD modes: {','.join(args.tdd_modes)} | client_split={client_split_text(args)}",
    )
    log_line(
        log_path,
        (
            f"Host CPUs: {os.cpu_count() or 'unknown'} | "
            f"Effective CPU/OpenMP threads per job: {args.effective_cpu_threads_per_job}"
        ),
    )
    for exp in experiments:
        _, supernet_path, library_path, save_path, stage_plan = build_command(root, exp, args)
        log_line(
            log_path,
            f"PLAN {exp.arch} {exp.dataset} alpha={exp.alpha} "
            f"tdd={exp.tdd_mode} "
            f"stages={stage_plan} "
            f"client_split={client_split_text(args)} "
            f"supernet={'Y' if supernet_path.exists() else 'N'} "
            f"library={'Y' if library_path.exists() else 'N'} "
            f"save={save_path}",
        )

    if args.max_parallel <= 1:
        failures = 0
        for index, exp in enumerate(experiments, start=1):
            log_line(log_path, f"QUEUE [{index}/{len(experiments)}] {exp}")
            ok = run_experiment(root, exp, args, csv_path, log_path)
            if not ok:
                failures += 1
            if not args.dry_run and index < len(experiments):
                time.sleep(2)
    else:
        log_line(log_path, f"Parallel mode enabled: max_parallel={args.max_parallel}")
        failures = run_experiments_parallel(root, experiments, args, csv_path, log_path)

    log_line(log_path, f"Finished with {failures} failure(s). CSV: {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
