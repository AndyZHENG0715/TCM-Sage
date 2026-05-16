"""Arena vote statistics helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from config import PROJECT_ROOT

VOTES_PATH = PROJECT_ROOT / "data" / "feedback" / "arena_votes.jsonl"


def _load_votes(votes_path: Path) -> list[dict[str, Any]]:
    votes: list[dict[str, Any]] = []
    with open(votes_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                votes.append(json.loads(line))
    return votes


def compute_arena_stats(votes_path: Path = VOTES_PATH) -> dict[str, Any]:
    """Compute arena evaluation statistics from persisted JSONL votes."""

    if not votes_path.exists():
        return {"total_votes": 0, "votes": [], "statistics": None}

    votes = _load_votes(votes_path)
    if not votes:
        return {"total_votes": 0, "votes": [], "statistics": None}

    rag_wins = 0
    plain_wins = 0
    ties = 0
    rag_scores: list[float] = []

    for vote_record in votes:
        mapping = vote_record.get("position_mapping", {})
        vote = vote_record.get("vote", "")

        rag_side = None
        for panel, role in mapping.items():
            if role == "rag":
                rag_side = panel
                break

        if not rag_side:
            continue

        if vote == "tie":
            ties += 1
            rag_scores.append(0.5)
        elif vote == rag_side:
            rag_wins += 1
            rag_scores.append(1.0)
        else:
            plain_wins += 1
            rag_scores.append(0.0)

    total = rag_wins + plain_wins + ties

    t_test = None
    if len(rag_scores) >= 3:
        plain_scores = [1.0 - score for score in rag_scores]
        t_stat, p_value = scipy_stats.ttest_rel(rag_scores, plain_scores)
        diffs = np.array(rag_scores) - np.array(plain_scores)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        mean_score = float(np.mean(rag_scores))

        t_stat_float = float(t_stat[0] if isinstance(t_stat, tuple) else t_stat)
        p_value_float = float(p_value[0] if isinstance(p_value, tuple) else p_value)
        cohens_d_float = float(cohens_d[0] if isinstance(cohens_d, tuple) else cohens_d)

        t_test = {
            "t_statistic": round(t_stat_float, 4),
            "p_value": round(p_value_float, 6),
            "cohens_d": round(cohens_d_float, 4),
            "mean_rag_score": round(mean_score, 4),
            "sample_size": len(rag_scores),
            "significant": bool(p_value_float < 0.05),
            "interpretation": (
                "Statistically significant preference for RAG" if p_value_float < 0.05 and mean_score > 0.5
                else "Statistically significant preference for Plain LLM" if p_value_float < 0.05 and mean_score < 0.5
                else "No statistically significant difference detected"
            ),
        }

    query_results = []
    for vote_record in votes:
        mapping = vote_record.get("position_mapping", {})
        vote = vote_record.get("vote", "")
        rag_side = None
        for panel, role in mapping.items():
            if role == "rag":
                rag_side = panel
                break

        winner = "tie" if vote == "tie" else ("rag" if vote == rag_side else "plain")
        query_results.append({
            "query": vote_record.get("query", ""),
            "winner": winner,
            "model": vote_record.get("model_name", ""),
            "timestamp": vote_record.get("timestamp", ""),
            "session_id": vote_record.get("session_id", ""),
        })

    return {
        "total_votes": total,
        "rag_wins": rag_wins,
        "plain_wins": plain_wins,
        "ties": ties,
        "rag_win_rate": round(rag_wins / total * 100, 1) if total > 0 else 0,
        "plain_win_rate": round(plain_wins / total * 100, 1) if total > 0 else 0,
        "tie_rate": round(ties / total * 100, 1) if total > 0 else 0,
        "t_test": t_test,
        "query_results": query_results,
    }
