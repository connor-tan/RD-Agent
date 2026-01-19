#!/usr/bin/env python
"""Read RD-Agent trace logs and emit structured JSON.

This is a standalone reader that reconstructs the UI-like structure
(data / llm_data / token_costs) from pickled logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pickle
import re

from rdagent.log.storage import FileStorage
from rdagent.log.utils import extract_evoid, extract_loopid_func_name
from rdagent.log.ui.utils import get_sota_exp_stat, load_times_info


def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


def load_data_like_ui(log_path: Path):
    """Rebuild the same structure as log/ui/ds_trace.py:load_data."""
    data = defaultdict(lambda: defaultdict(dict))
    llm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    token_costs = defaultdict(list)

    for msg in FileStorage(log_path).iter_msg():
        if not msg.tag:
            continue
        li, fn = extract_loopid_func_name(msg.tag)
        ei = extract_evoid(msg.tag)
        if li is not None:
            li = int(li)
        if ei is not None:
            ei = int(ei)

        if "debug_" in msg.tag:
            if ei is not None:
                llm_data[li][fn][ei].append({"tag": msg.tag, "obj": msg.content})
            else:
                llm_data[li][fn]["no_tag"].append({"tag": msg.tag, "obj": msg.content})
        elif "token_cost" in msg.tag:
            token_costs[li].append(msg)
        elif "llm" not in msg.tag and "session" not in msg.tag and "batch embedding" not in msg.tag:
            if msg.tag == "competition":
                data["competition"] = msg.content
                continue
            if "SETTINGS" in msg.tag:
                data["settings"][msg.tag] = msg.content
                continue

            msg_tag = re.sub(r"\.evo_loop_\d+", "", msg.tag)
            msg_tag = re.sub(r"Loop_\d+\.[^.]+\.?", "", msg_tag)
            msg_tag = msg_tag.strip()

            if ei is not None:
                if ei not in data[li][fn]:
                    data[li][fn][ei] = {}
                data[li][fn][ei][msg_tag] = msg.content
            else:
                if msg_tag:
                    data[li][fn][msg_tag] = msg.content
                else:
                    if not isinstance(msg.content, str):
                        data[li][fn]["no_tag"] = msg.content

    # Legacy compatibility: debug_llm.pkl (same as UI)
    llm_log_p = log_path / "debug_llm.pkl"
    if llm_log_p.exists():
        try:
            rd = pickle.loads(llm_log_p.read_bytes())
        except Exception:
            rd = []
        for d in rd:
            t = d.get("tag", "")
            if "debug_exp_gen" in t:
                continue
            if "debug_tpl" in t and "filter_" in d.get("obj", {}).get("uri", ""):
                continue
            lid, fn = extract_loopid_func_name(t)
            ei = extract_evoid(t)
            if lid is not None:
                lid = int(lid)
            if ei is not None:
                ei = int(ei)

            if ei is not None:
                llm_data[lid][fn][ei].append(d)
            else:
                llm_data[lid][fn]["no_tag"].append(d)

    return (
        convert_defaultdict_to_dict(data),
        convert_defaultdict_to_dict(llm_data),
        convert_defaultdict_to_dict(token_costs),
    )


def is_empty_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)) and len(value) == 0:
        return True
    if hasattr(value, "empty"):
        try:
            return bool(value.empty)
        except Exception:
            return False
    return False


def compact_value(value):
    if isinstance(value, dict):
        return compact_dict(value)
    if isinstance(value, list):
        cleaned = [compact_value(v) for v in value]
        return [v for v in cleaned if not is_empty_value(v)]
    return value


def compact_dict(d):
    cleaned = {}
    for k, v in d.items():
        v = compact_value(v)
        if is_empty_value(v):
            continue
        cleaned[k] = v
    return cleaned


def parse_evo_loop_id(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("evo_loop_"):
            try:
                return int(part.split("_", 2)[2])
            except Exception:
                return None
    return None


def parse_timestamp_from_name(path: Path) -> datetime | None:
    # filenames like 2026-01-12_10-48-44-168065.pkl
    try:
        return datetime.strptime(path.stem, "%Y-%m-%d_%H-%M-%S-%f")
    except Exception:
        return None


def load_loop_feedback(log_path: Path) -> dict[int, dict | None]:
    """
    Extract per-loop evolving feedback. For each loop:
    - pick the highest evo_loop id
    - within that, pick the latest timestamped pkl
    """
    feedback_by_loop: dict[int, dict | None] = {}
    for loop_dir in sorted(log_path.glob("Loop_*")):
        if not loop_dir.is_dir():
            continue
        try:
            loop_id = int(loop_dir.name.split("_", 1)[1])
        except Exception:
            continue
        feedback_pkls = list(loop_dir.rglob("coding/evo_loop_*/evolving feedback/*/*.pkl"))
        if not feedback_pkls:
            feedback_by_loop[loop_id] = {
                "evo_loop_id": None,
                "source_pkl": None,
                "feedbacks": [],
            }
            continue

        # group by evo_loop id
        by_evo: dict[int, list[Path]] = {}
        for p in feedback_pkls:
            evo_id = parse_evo_loop_id(p)
            if evo_id is None:
                continue
            by_evo.setdefault(evo_id, []).append(p)
        if not by_evo:
            feedback_by_loop[loop_id] = {
                "evo_loop_id": None,
                "source_pkl": None,
                "feedbacks": [],
            }
            continue

        max_evo = max(by_evo.keys())
        candidates = by_evo[max_evo]
        # choose latest timestamp among candidates
        def sort_key(p: Path):
            ts = parse_timestamp_from_name(p)
            return ts or datetime.min

        chosen = sorted(candidates, key=sort_key)[-1]
        try:
            obj = pickle.load(open(chosen, "rb"))
        except Exception:
            feedback_by_loop[loop_id] = {
                "evo_loop_id": None,
                "source_pkl": None,
                "feedbacks": [],
            }
            continue

        feedback_items = []
        feedback_list = getattr(obj, "feedback_list", None)
        if isinstance(feedback_list, list):
            for idx, fb in enumerate(feedback_list):
                fb_dict = {}
                for key in [
                    "execution_feedback",
                    "code_feedback",
                    "value_feedback",
                    "final_feedback",
                    "final_decision",
                    "final_decision_based_on_gt",
                    "shape_feedback",
                    "value_generated_flag",
                ]:
                    if hasattr(fb, key):
                        fb_dict[key] = getattr(fb, key)
                if fb_dict:
                    fb_dict["feedback_index"] = idx
                    feedback_items.append(fb_dict)

        feedback_by_loop[loop_id] = {
            "evo_loop_id": max_evo,
            "source_pkl": str(chosen),
            "feedbacks": feedback_items,
        }

    return feedback_by_loop


def load_loop_evolving_code(log_path: Path) -> dict[int, dict]:
    """
    Extract per-loop evolving code. For each loop:
    - pick the highest evo_loop id
    - within that, pick the latest timestamped pkl
    """
    code_by_loop: dict[int, dict] = {}
    for loop_dir in sorted(log_path.glob("Loop_*")):
        if not loop_dir.is_dir():
            continue
        try:
            loop_id = int(loop_dir.name.split("_", 1)[1])
        except Exception:
            continue
        code_pkls = list(loop_dir.rglob("coding/evo_loop_*/evolving code/*/*.pkl"))
        if not code_pkls:
            code_by_loop[loop_id] = {
                "evo_loop_id": None,
                "source_pkl": None,
                "workspaces": [],
            }
            continue

        by_evo: dict[int, list[Path]] = {}
        for p in code_pkls:
            evo_id = parse_evo_loop_id(p)
            if evo_id is None:
                continue
            by_evo.setdefault(evo_id, []).append(p)
        if not by_evo:
            code_by_loop[loop_id] = {
                "evo_loop_id": None,
                "source_pkl": None,
                "workspaces": [],
            }
            continue

        max_evo = max(by_evo.keys())
        candidates = by_evo[max_evo]

        def sort_key(p: Path):
            ts = parse_timestamp_from_name(p)
            return ts or datetime.min

        chosen = sorted(candidates, key=sort_key)[-1]
        try:
            obj = pickle.load(open(chosen, "rb"))
        except Exception:
            code_by_loop[loop_id] = {
                "evo_loop_id": max_evo,
                "source_pkl": str(chosen),
                "workspaces": [],
            }
            continue

        workspaces = []
        if isinstance(obj, list):
            for ws in obj:
                file_dict = getattr(ws, "file_dict", None)
                workspace_path = getattr(ws, "workspace_path", None)
                target_task = getattr(ws, "target_task", None)
                if file_dict:
                    workspaces.append(
                        {
                            "workspace_path": workspace_path,
                            "target_task": target_task,
                            "files": file_dict,
                        }
                    )

        code_by_loop[loop_id] = {
            "evo_loop_id": max_evo,
            "source_pkl": str(chosen),
            "workspaces": workspaces,
        }

    return code_by_loop


def build_ds_trace_summary(data: dict, log_path: Path) -> dict:
    feedback_by_loop = load_loop_feedback(log_path)
    code_by_loop = load_loop_evolving_code(log_path)
    loops = []
    for key in sorted([k for k in data.keys() if isinstance(k, int)]):
        loop_data = data.get(key, {})
        direct = loop_data.get("direct_exp_gen", {}) or {}
        coding = loop_data.get("coding", {}) or {}
        running = loop_data.get("running", {}) or {}
        feedback = loop_data.get("feedback", {}) or {}

        loop_entry = {
            "loop_id": key,
            "hypothesis": direct.get("hypothesis generation"),
            "experiment_generation": direct.get("experiment generation"),
            "feedback": feedback.get("feedback"),
            "code": coding.get("coder result") or coding.get("evolving code"),
            "run_result": running.get("runner result"),
            "execute_log": running.get("Qlib_execute_log"),
            "backtest_chart": running.get("Quantitative Backtesting Chart"),
            "time_info": {
                "direct_exp_gen": direct.get("time_info"),
                "coding": coding.get("time_info"),
                "running": running.get("time_info"),
                "feedback": feedback.get("time_info"),
            },
            "evolving_feedback": feedback_by_loop.get(key),
            "evolving_code": code_by_loop.get(key),
        }
        loops.append(compact_dict(loop_entry))

    return compact_dict(
        {
            "competition": data.get("competition"),
            "settings": data.get("settings"),
            "loops": loops,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Read RD-Agent logs and emit structured JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--log_name",
        help="Folder name under ./log or ./app/log (not a specific .pkl file).",
    )
    group.add_argument(
        "--log_dir",
        help="Absolute or relative path to a log folder.",
    )
    parser.add_argument(
        "--out",
        help="Write JSON output to this file instead of stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation.",
    )
    return parser.parse_args()


def resolve_log_path(log_name: str | None, log_dir: str | None) -> Path | None:
    if log_dir:
        input_path = Path(log_dir)
        candidates = [input_path]
    else:
        input_path = Path(log_name or "")
        if input_path.is_absolute() or input_path.as_posix().startswith("."):
            candidates = [input_path]
        else:
            candidates = [
                Path.cwd() / "log" / (log_name or ""),
                Path.cwd() / "app" / "log" / (log_name or ""),
            ]

    return next((p for p in candidates if p.exists() and p.is_dir()), None)


def main() -> int:
    args = parse_args()
    log_path = resolve_log_path(args.log_name, args.log_dir)
    if log_path is None:
        sys.stderr.write("Log folder not found.\n")
        return 2

    data, _llm_data, _token_costs = load_data_like_ui(log_path)

    trace = build_ds_trace_summary(data, log_path)

    summary: dict[str, object] = {}
    try:
        summary["times_info"] = load_times_info(log_path)
    except Exception:
        summary["times_info"] = None
    try:
        summary["sota_info"] = get_sota_exp_stat(log_path, selector="auto")
    except Exception:
        summary["sota_info"] = None

    payload: dict[str, object] = {
        "log_path": str(log_path),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "trace": trace,
        "summary": compact_dict(summary),
    }

    if args.pretty:
        out_json = json.dumps(payload, ensure_ascii=True, default=str, indent=2)
    else:
        out_json = json.dumps(payload, ensure_ascii=True, default=str)

    if args.out:
        Path(args.out).write_text(out_json)
    else:
        try:
            print(out_json)
        except BrokenPipeError:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
