import os


def phase_timing_enabled(args):
    return getattr(args, "phase_timing", False) or os.environ.get("SMARTFL_PHASE_TIMING", "0") == "1"


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def append_phase_timing_rows(save_path, filename, header, rows):
    if not rows:
        return

    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, filename)
    write_header = not os.path.exists(path)

    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(_format_value(value) for value in row) + "\n")

