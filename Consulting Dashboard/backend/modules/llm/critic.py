"""Critic pass for Paper2Alpha chat answers.

After the primary model finishes its draft, a second — deliberately
different — OpenRouter model reviews it against the same context and either
approves it or flags concrete issues, which the primary model then gets one
chance to fix. Using the same model to grade its own answer tends to
rubber-stamp its own mistakes, so the critic model is fixed and independent
of whatever the user picked for generation. See
`Consulting Dashboard/CLAUDE.md` for the design rationale.
"""
import json
from .openrouter_provider import OpenRouterProvider

# Free-tier frontier reasoning model on OpenRouter as of 2026-07 (1M context,
# reasoning enabled by default) — chosen specifically for the judge role over
# the coding-agent-flavored free models (Laguna/Cohere North), which are
# tuned for writing code rather than auditing someone else's answer. Fixed
# independently of the user's generation model choice. If this model is
# deprecated or the free tier's rate limit (50 req/day as of this writing)
# becomes a problem, swap it here — nothing else needs to change.
CRITIC_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Caps how much of the generation context (system prompt + any injected
# paper text, which can run to ~120k chars for full-paper grounding) gets
# re-sent to the critic. Full grounding would be nice but isn't worth the
# extra latency/cost on every single turn.
_CONTEXT_CHAR_CAP = 24_000

_CRITIC_SYSTEM = """You are a strict technical reviewer auditing another AI assistant's answer before it reaches the user. You are not answering the question yourself — you are grading the draft answer below.

Check for:
1. Does the draft actually answer what was asked, or does it dodge/generalize?
2. Any claim, number, formula, or citation in the draft that is NOT supported by the CONTEXT provided (possible hallucination)?
3. Internal inconsistency (e.g. contradicts itself, or contradicts the CONTEXT).
4. Vague hand-waving where the CONTEXT would have supported a specific, concrete answer.
5. If the CONTEXT specifies a required output format (e.g. exact section headers, a required order, a required structure like "one table per method" or "one card per option"), does the draft actually follow it? Flag any required section that is missing, renamed, reordered, or merged with another, and any structural requirement (tables, per-item cards, etc.) that was ignored.

Do NOT penalize style, length, or tone. Only flag substantive correctness/grounding/format-compliance issues.

Respond with ONLY a JSON object and nothing else — no markdown fences, no commentary:
{"verdict": "pass", "issues": []}
or
{"verdict": "revise", "issues": ["<specific, actionable issue>", ...]}
"""


async def critique(context: str, draft: str, api_key: str) -> dict:
    """Review `draft` (the primary model's answer) against `context` (the
    system prompt it was given, including any injected paper text).

    Returns {"verdict": "pass" | "revise", "issues": [str, ...]}. Never
    raises on a malformed critic response — falls back to "pass" so a
    critic-side parsing bug can't block the user's actual answer.
    """
    # reasoning_effort="low": the critic model runs with reasoning enabled
    # by default, which was the main source of "verifying takes forever" —
    # a compliance/grounding check doesn't need deep multi-step reasoning,
    # and cutting the hidden thinking-token budget is the single biggest
    # lever on critique latency.
    provider = OpenRouterProvider(
        api_key=api_key, model=CRITIC_MODEL, temperature=0.0, reasoning_effort="low",
    )
    user_msg = (
        f"CONTEXT the assistant was given:\n{context[:_CONTEXT_CHAR_CAP]}\n\n"
        f"DRAFT ANSWER to review:\n{draft}"
    )
    raw_parts: list[str] = []
    async for chunk in provider.stream([{"role": "user", "content": user_msg}], system=_CRITIC_SYSTEM):
        if chunk:
            raw_parts.append(chunk)
    return _parse_verdict("".join(raw_parts))


def _parse_verdict(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        # Strip a markdown code fence some models wrap JSON in despite instructions.
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        obj = json.loads(text)
        verdict = obj.get("verdict")
        if verdict not in ("pass", "revise"):
            raise ValueError(f"unexpected verdict: {verdict!r}")
        issues = obj.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        return {"verdict": verdict, "issues": [str(i) for i in issues]}
    except Exception:
        # Malformed critic output must never block the user's actual answer —
        # fail open (treat as pass) rather than silently hanging the chat.
        return {"verdict": "pass", "issues": []}
