#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_command(command):
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:  # pragma: no cover
        return "", str(exc), 1
    return completed.stdout, completed.stderr, completed.returncode


def pid_exists(pid):
    return Path(f"/proc/{pid}").exists()


def write_header_if_missing(path, header):
    if path.exists():
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)


def append_rows(path, rows):
    if not rows:
        return
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows(rows)


def resolve_pyspy_path():
    pyspy_path = shutil.which("py-spy")
    if pyspy_path:
        return pyspy_path
    fallback_path = Path.home() / ".local/bin/py-spy"
    if fallback_path.exists():
        return str(fallback_path)
    return None


def query_gpu():
    stdout, stderr, returncode = run_command(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    if returncode != 0:
        return rows, stderr.strip()
    for raw_line in stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 9:
            continue
        rows.append([utc_now(), *parts])
    return rows, ""


def query_pmon():
    stdout, stderr, returncode = run_command(["nvidia-smi", "pmon", "-s", "um", "-c", "1"])
    rows = []
    if returncode != 0:
        return rows, stderr.strip()
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        gpu, pid, type_name, sm, mem, enc, dec, command = parts[:8]
        rows.append([utc_now(), gpu, pid, type_name, sm, mem, enc, dec, command])
    return rows, ""


def query_processes(pids):
    if not pids:
        return [], ""
    pid_arg = ",".join(str(pid) for pid in pids)
    stdout, stderr, returncode = run_command(
        [
            "ps",
            "-p",
            pid_arg,
            "-o",
            "pid=,ppid=,pcpu=,pmem=,nlwp=,stat=,etime=,rss=,psr=,comm=",
        ]
    )
    rows = []
    if returncode != 0:
        return rows, stderr.strip()
    for raw_line in stdout.splitlines():
        parts = raw_line.split(None, 8)
        if len(parts) != 9:
            continue
        rows.append([utc_now(), *parts])
    return rows, ""


def query_child_processes(parent_pids):
    if not parent_pids:
        return [], ""
    pid_arg = ",".join(str(pid) for pid in parent_pids)
    stdout, stderr, returncode = run_command(
        [
            "ps",
            "--ppid",
            pid_arg,
            "-o",
            "pid=,ppid=,pcpu=,pmem=,nlwp=,stat=,etime=,rss=,psr=,comm=,args=",
        ]
    )
    rows = []
    if returncode != 0:
        return rows, stderr.strip()
    for raw_line in stdout.splitlines():
        parts = raw_line.split(None, 10)
        if len(parts) != 11:
            continue
        rows.append([utc_now(), *parts])
    return rows, ""


def dump_pyspy(pyspy_path, pid, target_dir):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dump_path = target_dir / f"pyspy_pid{pid}_{timestamp}.txt"
    stdout, stderr, returncode = run_command([pyspy_path, "dump", "--pid", str(pid)])
    with dump_path.open("w") as handle:
        handle.write(stdout)
        if stderr:
            handle.write("\n[stderr]\n")
            handle.write(stderr)
        handle.write(f"\n[returncode]\n{returncode}\n")
    return dump_path, returncode


def parse_gpu_util(rows, preferred_index):
    for row in rows:
        try:
            gpu_index = int(row[2])
        except ValueError:
            continue
        if gpu_index != preferred_index:
            continue
        try:
            return float(row[4])
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Low-overhead observer for SmartFL training.")
    parser.add_argument("--pids", nargs="+", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--low-util-threshold", type=float, default=20.0)
    parser.add_argument("--low-util-samples", type=int, default=4)
    parser.add_argument("--pyspy-cooldown", type=float, default=900.0)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    log_dir = args.log_dir
    pyspy_dir = log_dir / "pyspy"
    log_dir.mkdir(parents=True, exist_ok=True)
    pyspy_dir.mkdir(parents=True, exist_ok=True)

    meta_path = log_dir / "observer_meta.log"
    gpu_tsv = log_dir / "gpu_samples.tsv"
    pmon_tsv = log_dir / "pmon_samples.tsv"
    proc_tsv = log_dir / "process_samples.tsv"
    child_proc_tsv = log_dir / "child_process_samples.tsv"
    events_log = log_dir / "observer_events.log"

    write_header_if_missing(
        gpu_tsv,
        [
            "observer_ts",
            "gpu_reported_ts",
            "gpu_index",
            "gpu_name",
            "util_gpu",
            "util_mem",
            "mem_used_mib",
            "mem_total_mib",
            "power_w",
            "temp_c",
        ],
    )
    write_header_if_missing(
        pmon_tsv,
        ["observer_ts", "gpu", "pid", "type", "sm", "mem", "enc", "dec", "command"],
    )
    write_header_if_missing(
        proc_tsv,
        ["observer_ts", "pid", "ppid", "pcpu", "pmem", "nlwp", "stat", "etime", "rss_kib", "psr", "comm"],
    )
    write_header_if_missing(
        child_proc_tsv,
        ["observer_ts", "pid", "ppid", "pcpu", "pmem", "nlwp", "stat", "etime", "rss_kib", "psr", "comm", "args"],
    )

    pyspy_path = resolve_pyspy_path()
    with meta_path.open("a") as handle:
        handle.write(f"{utc_now()} start pids={args.pids} gpu_index={args.gpu_index} interval={args.interval}\n")
        handle.write(f"{utc_now()} py_spy={pyspy_path or 'missing'}\n")
    with events_log.open("a") as handle:
        handle.write(f"{utc_now()} observer_started live_pids={args.pids} py_spy={pyspy_path or 'missing'}\n")

    low_util_streak = 0
    last_pyspy_ts = 0.0

    while True:
        live_pids = [pid for pid in args.pids if pid_exists(pid)]
        if not live_pids:
            with events_log.open("a") as handle:
                handle.write(f"{utc_now()} all target pids exited; observer stopping\n")
            return 0

        gpu_rows, gpu_err = query_gpu()
        pmon_rows, pmon_err = query_pmon()
        proc_rows, proc_err = query_processes(live_pids)
        child_proc_rows, child_proc_err = query_child_processes(live_pids)

        append_rows(gpu_tsv, gpu_rows)
        append_rows(pmon_tsv, pmon_rows)
        append_rows(proc_tsv, proc_rows)
        append_rows(child_proc_tsv, child_proc_rows)

        util_gpu = parse_gpu_util(gpu_rows, args.gpu_index)
        if util_gpu is not None and util_gpu < args.low_util_threshold:
            low_util_streak += 1
        else:
            low_util_streak = 0

        if gpu_err or pmon_err or proc_err or child_proc_err:
            with events_log.open("a") as handle:
                if gpu_err:
                    handle.write(f"{utc_now()} gpu_query_error {gpu_err}\n")
                if pmon_err:
                    handle.write(f"{utc_now()} pmon_query_error {pmon_err}\n")
                if proc_err:
                    handle.write(f"{utc_now()} proc_query_error {proc_err}\n")
                if child_proc_err:
                    handle.write(f"{utc_now()} child_proc_query_error {child_proc_err}\n")
        with events_log.open("a") as handle:
            handle.write(
                f"{utc_now()} sample live_pids={live_pids} util_gpu={util_gpu} low_util_streak={low_util_streak}\n"
            )

        now_monotonic = time.monotonic()
        if (
            pyspy_path
            and low_util_streak >= args.low_util_samples
            and now_monotonic - last_pyspy_ts >= args.pyspy_cooldown
        ):
            with events_log.open("a") as handle:
                handle.write(
                    f"{utc_now()} low_util_trigger util_gpu={util_gpu} streak={low_util_streak} live_pids={live_pids}\n"
                )
            for pid in live_pids:
                dump_path, returncode = dump_pyspy(pyspy_path, pid, pyspy_dir)
                with events_log.open("a") as handle:
                    handle.write(
                        f"{utc_now()} pyspy_dump pid={pid} returncode={returncode} path={dump_path}\n"
                    )
            last_pyspy_ts = now_monotonic

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
