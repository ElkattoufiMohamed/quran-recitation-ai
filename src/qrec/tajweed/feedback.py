from __future__ import annotations

from typing import Dict

from .rules import TajweedRule


def format_rule_feedback(rule: TajweedRule, result: Dict) -> str:
    """
    Build a short, user-facing feedback message for a single rule.
    """
    if result.get("is_correct"):
        return f"✅ {rule.value}: correct"

    error_type = result.get("error_type") or "needs attention"
    confidence = result.get("confidence")
    if confidence is not None:
        return f"⚠️ {rule.value}: {error_type} (confidence={confidence:.2f})"
    return f"⚠️ {rule.value}: {error_type}"


def format_tajweed_feedback(results: Dict[TajweedRule, Dict]) -> str:
    """
    Format a multi-rule feedback report for user display.
    """
    lines = ["Tajweed feedback:"]
    for rule, result in results.items():
        lines.append(format_rule_feedback(rule, result))
    return "\n".join(lines)
