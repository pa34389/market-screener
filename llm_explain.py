"""Optional OpenAI explanation from deterministic `setup` JSON only (no chart guessing)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

SCOUT_SYSTEM = """You are a trading setup scout.

Explain only the structured JSON from the Python screener.
Do not invent missing conditions.
Do not say buy now or sell now.
Do not give financial advice.
Do not override the setup state.

Answer with:
1. Symbol and setup direction.
2. Current setup state.
3. HTF context.
4. POI context.
5. LTF confirmation status.
6. Invalidation, target, and RR if available.
7. Missing conditions.
8. Final instruction: watch only, wait for confirmation, valid for human review, or invalid.
"""


def explain_setup(setup: Dict[str, Any]) -> Optional[str]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai not installed; skip explain_setup")
        return None
    client = OpenAI()
    payload = json.dumps(setup, indent=2, default=str)
    try:
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SCOUT_SYSTEM},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return completion.choices[0].message.content
    except Exception:
        log.exception("explain_setup failed")
        return None
