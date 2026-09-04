"""Promotion history & audit trail (Step 49).

Persists promotion gate decisions to a structured history directory for
audit, traceability, and governance. Provides query capabilities to
list, filter, and summarize promotion decisions over time.

What this module does
---------------------
1. Saves each promotion gate decision to a JSON file in a configurable
   history directory (default: ``ml/promotion_history/``).
2. Provides a CLI to list, filter, and summarize promotion history.
3. Integrates with the promotion gate to auto-save decisions after each
   run (configurable via ``PROMO_HISTORY_DIR`` env var).

What this module never does
----------------------------
* It never modifies the production model, manifest, threshold, or
  runtime state.
* It never activates or promotes a model automatically.
* It never stores raw transactions, customer IDs, labels, predictions,
  secrets, or filesystem paths in history records.

Fail-safe behaviour
-------------------
* History write failures are logged but do not affect the promotion
  gate decision (the gate still returns APPROVED/REJECTED).
* History queries are read-only and never mutate production state.
* Storage is bounded: max 1000 files, each < 32 KB.

Usage::

    # List recent promotion decisions
    python -m ml.evaluation.promotion_history

    # Filter by decision
    python -m ml.evaluation.promotion_history --decision APPROVED

    # Show summary statistics
    python -m ml.evaluation.promotion_history --summary

    # Limit output
    python -m ml.evaluation.promotion_history --limit 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PromotionHistory",
    "PromotionHistoryError",
    "DEFAULT_HISTORY_DIR",
    "MAX_HISTORY_FILES",
    "MAX_HISTORY_FILE_SIZE",
    "save_decision",
    "list_decisions",
    "load_decision",
    "summarize_history",
    "main",
]


# ── Constants ─────────────────────────────────────────────────────────


class PromotionHistoryError(Exception):
    """Promotion history operation failed."""


DEFAULT_HISTORY_DIR: str = "ml/promotion_history"
MAX_HISTORY_FILES: int = 1000
MAX_HISTORY_FILE_SIZE: int = 32768  # 32 KB


# ── History persistence ───────────────────────────────────────────────


def _resolve_history_dir(history_dir: str | Path | None) -> Path | None:
    """Resolve the history directory from argument or environment.

    Returns ``None`` if history is disabled (empty string or explicit
    ``none``/``off``).
    """
    if history_dir is not None:
        dir_str = str(history_dir).strip()
    else:
        dir_str = os.environ.get("PROMO_HISTORY_DIR", "").strip()
    if not dir_str or dir_str.lower() in ("none", "off"):
        return None
    return Path(dir_str)


def _timestamp_to_filename(timestamp: str) -> str:
    """Convert an ISO 8601 timestamp to a filesystem-safe filename.

    Replaces colons and dots with hyphens for Windows compatibility.
    """
    safe = timestamp.replace(":", "-").replace(".", "-")
    # Remove timezone suffix for filename (keep it in the content)
    safe = safe.replace("+00:00", "Z").replace("Z", "")
    return f"{safe}.json"


def save_decision(
    decision_dict: dict[str, Any],
    history_dir: str | Path | None = None,
) -> Path | None:
    """Save a promotion decision to the history directory.

    Args:
        decision_dict: The promotion decision as a dict (from
            ``PromotionDecision.to_dict()``).
        history_dir: Override the history directory (default: from
            ``PROMO_HISTORY_DIR`` env var or ``DEFAULT_HISTORY_DIR``).

    Returns:
        The path to the saved file, or ``None`` if history is disabled
        or the save failed.

    Fail-safe: write failures are logged but do not raise. The promotion
    gate decision is unaffected.
    """
    try:
        resolved = _resolve_history_dir(history_dir)
        if resolved is None:
            return None

        # Validate decision dict
        if not isinstance(decision_dict, dict):
            logger.warning("Cannot save non-dict decision to history")
            return None
        if "decision" not in decision_dict or "gate_timestamp" not in decision_dict:
            logger.warning("Cannot save decision: missing required fields")
            return None

        # Enforce bounded storage
        payload = json.dumps(decision_dict, indent=2, sort_keys=True)
        if len(payload) > MAX_HISTORY_FILE_SIZE:
            logger.warning(
                "Decision payload too large (%d bytes > %d); not saving",
                len(payload),
                MAX_HISTORY_FILE_SIZE,
            )
            return None

        # Create directory if needed
        resolved.mkdir(parents=True, exist_ok=True)

        # Enforce max files (oldest first)
        existing = sorted(resolved.glob("*.json"))
        if len(existing) >= MAX_HISTORY_FILES:
            # Remove oldest file(s) to make room
            to_remove = len(existing) - MAX_HISTORY_FILES + 1
            for old_file in existing[:to_remove]:
                try:
                    old_file.unlink()
                    logger.info("Removed old history file: %s", old_file.name)
                except OSError as exc:
                    logger.warning("Could not remove old history file: %s", exc)

        # Write the file
        timestamp = decision_dict["gate_timestamp"]
        filename = _timestamp_to_filename(timestamp)
        file_path = resolved / filename

        # Avoid overwriting (add microseconds if collision)
        counter = 1
        while file_path.exists():
            filename = _timestamp_to_filename(timestamp)
            filename = filename.replace(".json", f"-{counter}.json")
            file_path = resolved / filename
            counter += 1

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")

        logger.info("Saved promotion decision to history: %s", file_path.name)
        return file_path

    except Exception as exc:
        # Fail-safe: log but don't raise
        logger.warning("Failed to save promotion decision to history: %s", exc)
        return None


def list_decisions(
    history_dir: str | Path | None = None,
    *,
    decision_filter: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List promotion decisions from the history directory.

    Args:
        history_dir: Override the history directory.
        decision_filter: Filter by decision (``"APPROVED"`` or
            ``"REJECTED"``).
        limit: Maximum number of decisions to return (most recent first).

    Returns:
        List of decision dicts, most recent first.

    Raises:
        PromotionHistoryError: If the history directory is invalid or
            unreadable.
    """
    resolved = _resolve_history_dir(history_dir)
    if resolved is None:
        return []

    if not resolved.exists():
        return []
    if not resolved.is_dir():
        raise PromotionHistoryError(
            f"History path exists but is not a directory: {resolved}"
        )

    try:
        files = sorted(resolved.glob("*.json"), reverse=True)
    except OSError as exc:
        raise PromotionHistoryError(
            f"Cannot read history directory: {exc}"
        ) from exc

    decisions: list[dict[str, Any]] = []
    for file_path in files:
        if limit is not None and len(decisions) >= limit:
            break
        try:
            decision = load_decision(file_path)
            if decision_filter is not None:
                if decision.get("decision") != decision_filter:
                    continue
            decisions.append(decision)
        except Exception as exc:
            logger.warning("Skipping invalid history file %s: %s", file_path.name, exc)

    return decisions


def load_decision(file_path: str | Path) -> dict[str, Any]:
    """Load a single promotion decision from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        The decision dict.

    Raises:
        PromotionHistoryError: If the file is invalid or unreadable.
    """
    path = Path(file_path)
    if not path.exists():
        raise PromotionHistoryError(f"History file does not exist: {path}")
    if not path.is_file():
        raise PromotionHistoryError(f"History path is not a file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise PromotionHistoryError(
            f"Cannot read history file: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise PromotionHistoryError("History file does not contain a dict")
    if "decision" not in data:
        raise PromotionHistoryError("History file missing 'decision' field")

    return data


def summarize_history(
    history_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize promotion history statistics.

    Args:
        history_dir: Override the history directory.

    Returns:
        Dict with summary statistics: total count, approved count,
        rejected count, first/last timestamps.
    """
    decisions = list_decisions(history_dir)
    if not decisions:
        return {
            "total": 0,
            "approved": 0,
            "rejected": 0,
            "first_timestamp": None,
            "last_timestamp": None,
        }

    approved = sum(1 for d in decisions if d.get("decision") == "APPROVED")
    rejected = sum(1 for d in decisions if d.get("decision") == "REJECTED")

    # Decisions are most-recent-first, so last is oldest
    timestamps = [d.get("gate_timestamp") for d in decisions if d.get("gate_timestamp")]
    first_ts = timestamps[-1] if timestamps else None
    last_ts = timestamps[0] if timestamps else None

    return {
        "total": len(decisions),
        "approved": approved,
        "rejected": rejected,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
    }


# ── CLI ───────────────────────────────────────────────────────────────


def _print_decision_summary(decision: dict[str, Any]) -> None:
    """Print a concise one-line summary of a decision."""
    ts = decision.get("gate_timestamp", "unknown")
    dec = decision.get("decision", "unknown")
    cand = decision.get("candidate_identity", {})
    cand_ver = cand.get("model_version", "unknown") if cand else "unknown"
    prod = decision.get("production_identity", {})
    prod_ver = prod.get("model_version", "unknown") if prod else "unknown"
    stage = decision.get("failure_stage")
    stage_str = f" (stage: {stage})" if stage else ""
    print(f"  {ts}  {dec:<10}  candidate={cand_ver}  production={prod_ver}{stage_str}")


def main() -> None:
    """CLI entry point: query promotion history.

    Exit codes: ``0`` success; ``1`` error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.promotion_history",
        description=(
            "Query and summarize promotion gate decision history. "
            "Lists decisions from the configured history directory."
        ),
    )
    parser.add_argument(
        "--history-dir",
        default=None,
        help=(
            f"History directory (default: PROMO_HISTORY_DIR env var or "
            f"{DEFAULT_HISTORY_DIR})."
        ),
    )
    parser.add_argument(
        "--decision",
        choices=["APPROVED", "REJECTED"],
        default=None,
        help="Filter by decision (APPROVED or REJECTED).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of decisions to display.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics instead of listing decisions.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if args.summary:
            summary = summarize_history(args.history_dir)
            print("=" * 70)
            print("PROMOTION HISTORY SUMMARY")
            print("=" * 70)
            print(f"  Total decisions:  {summary['total']}")
            print(f"  Approved:         {summary['approved']}")
            print(f"  Rejected:         {summary['rejected']}")
            if summary["first_timestamp"]:
                print(f"  First decision:   {summary['first_timestamp']}")
                print(f"  Latest decision:  {summary['last_timestamp']}")
            else:
                print("  No decisions recorded.")
            print("=" * 70)
        else:
            decisions = list_decisions(
                args.history_dir,
                decision_filter=args.decision,
                limit=args.limit,
            )
            if not decisions:
                print("No promotion decisions found in history.")
            else:
                print("=" * 70)
                print(f"PROMOTION HISTORY ({len(decisions)} decisions)")
                print("=" * 70)
                for decision in decisions:
                    _print_decision_summary(decision)
                print("=" * 70)

    except PromotionHistoryError as exc:
        print(f"Promotion history error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.error("Promotion history query failed: %s", type(exc).__name__)
        print(f"Promotion history error: {type(exc).__name__}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
