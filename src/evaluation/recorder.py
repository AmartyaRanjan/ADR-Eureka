"""Detailed evaluation and experiment logging.

Every prompt, LLM response, reward function, PPO update, metric, exception,
and checkpoint is written as timestamped JSONL plus CSV summaries.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EvaluationRecorder:
    def __init__(self, output_dir: str, run_name: Optional[str] = None):
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = run_name or f"leap_eureka_{stamp}"
        self.output_dir = Path(output_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.metrics_csv = self.output_dir / "training_metrics.csv"
        self.eval_csv = self.output_dir / "evaluation_metrics.csv"
        self.llm_dir = self.output_dir / "llm"
        self.rewards_dir = self.output_dir / "reward_functions"
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.llm_dir.mkdir(exist_ok=True)
        self.rewards_dir.mkdir(exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        self._metric_fields = [
            "timestamp",
            "eureka_iteration",
            "ppo_step",
            "episode_reward_mean",
            "episode_reward_std",
            "oracle_reward_mean",
            "success_rate",
            "drop_frac",
            "yaw_angvel_mean",
            "episode_yaw_mean",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "learning_rate",
            "explained_variance",
            "fps",
            "step_time_s",
        ]
        self._eval_fields = [
            "timestamp",
            "eureka_iteration",
            "num_episodes",
            "mean_return",
            "std_return",
            "mean_oracle_return",
            "success_rate",
            "drop_rate",
            "mean_episode_length",
            "mean_episode_yaw",
            "mean_yaw_angvel",
            "crashes",
            "is_best",
        ]
        self._write_csv_header(self.metrics_csv, self._metric_fields)
        self._write_csv_header(self.eval_csv, self._eval_fields)
        self.events: List[Dict[str, Any]] = []
        self.best_eval: Optional[Dict[str, Any]] = None
        self.started_at = _now_iso()
        self.log_event("run_started", {"run_name": self.run_name, "output_dir": str(self.output_dir)})

    def log_host_metadata(self, extra: Optional[Dict[str, Any]] = None) -> None:
        info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "cwd": os.getcwd(),
        }
        if extra:
            info.update(extra)
        self.log_event("host_metadata", info)

    def log_config(self, config: Dict[str, Any]) -> None:
        path = self.output_dir / "config_snapshot.json"
        path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        self.log_event("config", config)

    def log_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        record = {
            "timestamp": _now_iso(),
            "event_type": event_type,
            "payload": payload or {},
        }
        self.events.append(record)
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            print(f"[eval] failed to write event log: {exc}")

    def log_llm_exchange(
        self,
        eureka_iteration: int,
        user_prompt: str,
        assistant_prompt: Optional[str],
        raw_outputs: Iterable[str],
        reward_strings: Iterable[str],
        latency_s: float,
        model: str,
        error: Optional[str] = None,
    ) -> None:
        raw_list = list(raw_outputs)
        code_list = list(reward_strings)
        exchange = {
            "eureka_iteration": eureka_iteration,
            "model": model,
            "latency_s": latency_s,
            "user_prompt": user_prompt,
            "assistant_prompt": assistant_prompt,
            "raw_outputs": raw_list,
            "reward_strings": code_list,
            "error": error,
        }
        (self.llm_dir / f"iter_{eureka_iteration:03d}.json").write_text(
            json.dumps(exchange, indent=2), encoding="utf-8"
        )
        for idx, code in enumerate(code_list):
            (self.rewards_dir / f"iter_{eureka_iteration:03d}_cand_{idx}.py").write_text(code, encoding="utf-8")
        self.log_event("llm_exchange", exchange)

    def log_reward_validation(
        self,
        eureka_iteration: int,
        code: str,
        is_valid: bool,
        error: Optional[str] = None,
        traceback_text: Optional[str] = None,
    ) -> None:
        self.log_event(
            "reward_validation",
            {
                "eureka_iteration": eureka_iteration,
                "is_valid": is_valid,
                "code": code,
                "error": error,
                "traceback": traceback_text,
            },
        )

    def log_training_step(self, metrics: Dict[str, Any]) -> None:
        row = {field: metrics.get(field, "") for field in self._metric_fields}
        row["timestamp"] = _now_iso()
        self._append_csv(self.metrics_csv, self._metric_fields, row)
        self.log_event("training_step", metrics)

    def log_evaluation(self, metrics: Dict[str, Any]) -> None:
        row = {field: metrics.get(field, "") for field in self._eval_fields}
        row["timestamp"] = _now_iso()
        self._append_csv(self.eval_csv, self._eval_fields, row)
        if metrics.get("is_best"):
            self.best_eval = metrics
        self.log_event("evaluation", metrics)

    def log_exception(self, where: str, exc: BaseException) -> None:
        self.log_event(
            "exception",
            {
                "where": where,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

    def log_checkpoint(self, path: str, iteration: int, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {"path": path, "iteration": iteration}
        if extra:
            payload.update(extra)
        self.log_event("checkpoint", payload)

    def generate_summary(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        summary = {
            "run_name": self.run_name,
            "started_at": self.started_at,
            "finished_at": _now_iso(),
            "output_dir": str(self.output_dir),
            "num_events": len(self.events),
            "best_eval": self.best_eval,
        }
        if extra:
            summary.update(extra)
        path = self.output_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (self.output_dir / "summary.md").write_text(self._markdown(summary), encoding="utf-8")
        self.log_event("run_finished", summary)
        return summary

    def _markdown(self, summary: Dict[str, Any]) -> str:
        best = summary.get("best_eval") or {}
        lines = [
            f"# {summary['run_name']}",
            "",
            f"- started: {summary['started_at']}",
            f"- finished: {summary['finished_at']}",
            f"- events: {summary['num_events']}",
            f"- best mean return: {best.get('mean_return')}",
            f"- best success rate: {best.get('success_rate')}",
            f"- best mean yaw (rad): {best.get('mean_episode_yaw')}",
            "",
            "See `events.jsonl`, `training_metrics.csv`, `evaluation_metrics.csv`, and `llm/`.",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _write_csv_header(path: Path, fields: List[str]) -> None:
        if path.exists():
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fields).writeheader()

    @staticmethod
    def _append_csv(path: Path, fields: List[str], row: Dict[str, Any]) -> None:
        try:
            with path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fields).writerow(row)
        except OSError as exc:
            print(f"[eval] failed to write CSV {path}: {exc}")
