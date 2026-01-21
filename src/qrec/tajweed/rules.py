from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class TajweedRule(str, Enum):
    AL_MAD = "al_mad"
    GHUNNAH = "ghunnah"
    QALQALAH = "qalqalah"
    IDGHAM = "idgham"
    IKHFAA = "ikhfaa"


@dataclass(frozen=True)
class TajweedRuleDefinition:
    rule: TajweedRule
    arabic_name: str
    description: str
    expected_counts: Optional[List[int]] = None
    expected_duration_s: Optional[List[float]] = None


_RULE_DEFINITIONS: List[TajweedRuleDefinition] = [
    TajweedRuleDefinition(
        rule=TajweedRule.AL_MAD,
        arabic_name="المد",
        description="Prolongation of vowels based on Mad type.",
        expected_counts=[2, 4, 5],
        expected_duration_s=[0.4, 0.8, 1.0],
    ),
    TajweedRuleDefinition(
        rule=TajweedRule.GHUNNAH,
        arabic_name="الغنة",
        description="Nasalization with 2-count duration and nasal resonance.",
        expected_counts=[2],
        expected_duration_s=[0.4],
    ),
    TajweedRuleDefinition(
        rule=TajweedRule.QALQALAH,
        arabic_name="القلقلة",
        description="Echo/bounce on ق ط ب ج د with a short burst.",
        expected_counts=[1],
        expected_duration_s=[0.1],
    ),
    TajweedRuleDefinition(
        rule=TajweedRule.IDGHAM,
        arabic_name="الإدغام",
        description="Merging of specific letters with nasal/non-nasal variants.",
    ),
    TajweedRuleDefinition(
        rule=TajweedRule.IKHFAA,
        arabic_name="الإخفاء",
        description="Concealment with a nasalized transition.",
    ),
]


RULE_DEFINITIONS: Dict[TajweedRule, TajweedRuleDefinition] = {
    definition.rule: definition for definition in _RULE_DEFINITIONS
}


def get_rule_definition(rule: TajweedRule) -> TajweedRuleDefinition:
    return RULE_DEFINITIONS[rule]
