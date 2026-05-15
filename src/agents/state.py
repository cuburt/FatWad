"""Shared state dict that flows between LangGraph nodes.

`messages` accumulates across turns via operator.add — that's how the
checkpointer rebuilds multi-turn history. Everything else is overwritten
per turn."""

import operator
from typing import Annotated, Any, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
    session_id: Optional[str]

    # Server-resolved snapshot of the user's wealth, dropped into the
    # summarizer prompt. Reset each turn.
    snapshot: Optional[dict]
    snapshot_text: Optional[str]

    # Working memory that's reset per turn.
    next_step: Optional[str]
    tool_results: List[Any]
    citations: List[str]
    scratchpad: str
