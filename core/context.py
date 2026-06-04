"""
core/context.py

Token queue bridge for two-layer streaming.

Problem solved
--------------
The LangGraph report nodes (generate_report, handle_comparison,
handle_discovery) run inside the agent layer (src/).
The WebSocket handler runs inside the API layer (api/).
They run in the same async event loop but cannot share state directly
because LangGraph node functions return a plain dict — they cannot
yield tokens to the outside world.

Solution: contextvars.ContextVar
---------------------------------
A ContextVar is a per-task variable — each asyncio Task (i.e. each
WebSocket request) gets its own isolated value. The WebSocket handler
creates an asyncio.Queue, stores it in token_queue_var before invoking
the graph, and the report nodes retrieve it via token_queue_var.get().

This approach:
  - Keeps AgentState as pure data (no live objects in TypedDict)
  - Is fully type-safe: ContextVar[asyncio.Queue | None]
  - Has zero external dependencies
  - Is safe under concurrent requests — each request has its own queue
  - Is the standard Python pattern used by FastAPI and SQLAlchemy
    internally for per-request context

Layer architecture
------------------
  api/          imports from core/      (sets token_queue_var)
  src/agent/    imports from core/      (reads token_queue_var)
  core/         imports nothing from api/ or src/

This keeps the dependency direction clean:
  api/ → core/ ← src/agent/
"""

import asyncio
from contextvars import ContextVar

# One queue per active WebSocket request.
# Default is None so nodes can safely call .get() and check before using.
token_queue_var: ContextVar[asyncio.Queue | None] = ContextVar(
    "token_queue", default=None
)


sub_progress_queue_var: ContextVar[asyncio.Queue | None] = ContextVar(
    "sub_progress_queue_var", default=None
)