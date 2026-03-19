#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
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
        flops_constraints=(285.9737, 443.4474, 513.4869, 532.48),
        params_constraints=(5.7838, 10.154, 18.5467, 34.0154),
    ),
    "mobilenet_v2_4": ArchSpec(
        model="mobilenet",
        artifact_prefix="mobilenetv2",
        flops_constraints=(10.69, 16.47, 22.47, 27.82),
        params_constraints=(0.393, 0.663, 1.234, 2.255),
    ),
}

DEFAULT_ARCHS = list(ARCH_SPECS.keys())
DEFAULT_DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
DEFAULT_ALPHAS = [100]


@dataclass(frozen=True)
class Experiment:
    arch: str
    dataset: str
    alpha: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for TDD training on SmartFL."
    )
    parser.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS, choices=DEFAULT_ARCHS)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
    )
    parser.add_argument("--alphas", nargs="+", type=int, default=DEFAULT_ALPHAS)
    parser.add_argument("--num_rounds", type=int, default=400)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--sample_rate", type=float, default=0.1)
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
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent


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
    growth_tag = format_ratio_tag(args.tdd_growth_ratio)
    save_name = (
        f"tdd_{exp.arch}_{exp.dataset}_a{exp.alpha}_s{args.seed}"
        f"_rp{args.rotation_period}_g{growth_tag}"
    )
    if args.run_tag:
        save_name = f"{save_name}_{args.run_tag}"
    return root / "outputs" / save_name


def existing_checkpoint(save_path: Path) -> bool:
    model_dir = save_path / "save_models"
    return model_dir.exists() and any(model_dir.glob("checkpoint_*.pth.tar"))


def completed_run(save_path: Path) -> bool:
    return (save_path / "test_scores.tsv").exists()


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
        "--num_architectures",
        str(args.num_architectures),
        "--episodes_per_batch",
        str(args.episodes_per_batch),
        "--enable_tdd",
        "1",
        "--rotation_period",
        str(args.rotation_period),
        "--tdd_growth_ratio",
        str(args.tdd_growth_ratio),
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


def collect_experiments(root: Path, args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    for arch in args.archs:
        for dataset in args.datasets:
            for alpha in args.alphas:
                exp = Experiment(arch=arch, dataset=dataset, alpha=alpha)
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

    if completed_run(save_path) and not args.rerun:
        final_prec1 = parse_test_score(test_score_path)
        log_line(
            log_path,
            f"SKIP {exp.arch} {exp.dataset} alpha={exp.alpha} "
            f"(existing result: {save_path})",
        )
        append_csv_row(
            csv_path,
            [
                exp.arch,
                exp.dataset,
                str(exp.alpha),
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
    log_name = (
        f"{exp.arch}_{exp.dataset}_a{exp.alpha}_"
        f"rp{args.rotation_period}_g{format_ratio_tag(args.tdd_growth_ratio)}.log"
    )
    run_log_path = csv_path.parent / log_name

    log_line(
        log_path,
        f"START {exp.arch} {exp.dataset} alpha={exp.alpha} "
        f"stages={stage_plan}",
    )
    log_line(log_path, f"CMD {cmd_text}")

    if args.dry_run:
        append_csv_row(
            csv_path,
            [
                exp.arch,
                exp.dataset,
                str(exp.alpha),
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
        f"prec1={final_prec1 or 'N/A'} duration={duration_hours:.2f}h",
    )
    append_csv_row(
        csv_path,
        [
            exp.arch,
            exp.dataset,
            str(exp.alpha),
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


def main() -> int:
    args = parse_args()
    root = repo_root()
    experiments = collect_experiments(root, args)

    if not experiments:
        print("No experiments selected.")
        return 0

    _, csv_path, log_path = ensure_results_dir(root, args)
    write_csv_header(csv_path)

    log_line(log_path, f"Repo root: {root}")
    log_line(log_path, f"Selected experiments: {len(experiments)}")
    for exp in experiments:
        _, supernet_path, library_path, save_path, stage_plan = build_command(root, exp, args)
        log_line(
            log_path,
            f"PLAN {exp.arch} {exp.dataset} alpha={exp.alpha} "
            f"stages={stage_plan} "
            f"supernet={'Y' if supernet_path.exists() else 'N'} "
            f"library={'Y' if library_path.exists() else 'N'} "
            f"save={save_path}",
        )

    failures = 0
    for index, exp in enumerate(experiments, start=1):
        log_line(log_path, f"QUEUE [{index}/{len(experiments)}] {exp}")
        ok = run_experiment(root, exp, args, csv_path, log_path)
        if not ok:
            failures += 1
        if not args.dry_run and index < len(experiments):
            time.sleep(2)

    log_line(log_path, f"Finished with {failures} failure(s). CSV: {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
