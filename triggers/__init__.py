"""
triggers package
─────────────────
Memory trigger system components.

Public interface:
- MemoryTriggerEngine: Unified entry point for all memory triggers
- MemoryCycleReport: Report of what happened in a trigger cycle
- RuleID: Enum of rule identifiers (for observability)
"""

from triggers.memory_trigger_engine import MemoryTriggerEngine, MemoryCycleReport
from triggers.system_rules import RuleID

__all__ = [
    "MemoryTriggerEngine",
    "MemoryCycleReport",
    "RuleID",
]