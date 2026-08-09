"""Provider smoke test. Run before P1-02 and whenever credentials change.

Checks 1-2 gate everything else: without any key there is nothing left to test,
so check 1 fails fast. Check 2 is advisory, not fatal -- a key that doesn't
start with 'sk-' will usually also fail check 3, which reports the real error;
2 exists to make the *obvious* mistake (pasting the model name into the key
field) diagnosable in one line instead of a stack trace.

Everything from check 3 on is collected rather than fail-fast: one crashing
check should not hide whether the others work. A smoke test that stops at the
first surprise tells you less than one that reports everything it found.
"""

from __future__ import annotations

import os
import sys

from pydantic import BaseModel

from isc.llm import Message, get_chat_model, get_embedding_model
from isc.llm.structured import parse_structured

_PING = [Message.system("Reply with exactly one word: pong"), Message.user("ping")]

_fatal = 0


class _Ping(BaseModel):
    reply: str


def _check(status: str, name: str, detail: str = "") -> None:
    global _fatal
    line = f"[{status}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)
    if status == "FAIL":
        _fatal += 1


def main() -> int:
    # 1. key present -- fails fast, nothing downstream can work without it
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        _check("FAIL", "OPENAI_API_KEY present in os.environ")
        print("\nno key in os.environ; check .env and isc.common.config's load_dotenv call.")
        return 1
    _check("PASS", "OPENAI_API_KEY present in os.environ")

    # 2. shape -- warn only; the obvious mistake is putting the model name here
    if key.startswith("sk-"):
        _check("PASS", "OPENAI_API_KEY looks like an OpenAI key")
    else:
        _check("WARN", "OPENAI_API_KEY looks like an OpenAI key",
               "does not start with 'sk-' -- check you didn't put the model name "
               "in the key field")

    chat = None
    try:
        chat = get_chat_model()
    except Exception as exc:
        _check("FAIL", "chat model constructs", f"{type(exc).__name__}: {exc}")

    # 3. chat completion
    result = None
    if chat is not None:
        try:
            result = chat.complete(_PING, max_tokens=5)
        except Exception as exc:
            _check("FAIL", "chat completion", f"{type(exc).__name__}: {exc}")
        else:
            _check("PASS", "chat completion", f"model={result.model} text={result.text!r}")
    else:
        _check("FAIL", "chat completion", "skipped: no chat model")

    # 4. usage reported
    if result is not None:
        if result.usage.total > 0:
            _check("PASS", "token usage reported", f"{result.usage.total} tokens")
        else:
            _check("FAIL", "token usage reported", "usage.total == 0")
    else:
        _check("FAIL", "token usage reported", "skipped: chat completion did not succeed")

    # 5. mean_logprob -- watch this one. If it's None, model_confidence() falls
    # back to a constant 0.75 and every extracted field gets identical MODEL
    # confidence, silently, because nothing raises.
    if result is not None:
        if result.mean_logprob is not None:
            _check("PASS", "mean_logprob present",
                   f"mean_logprob={result.mean_logprob:.4f} "
                   f"model_confidence={result.model_confidence():.4f}")
        else:
            _check("FAIL", "mean_logprob present",
                   "provider omitted logprobs -- model_confidence() will return a constant "
                   "0.75 for every field")
    else:
        _check("FAIL", "mean_logprob present", "skipped: chat completion did not succeed")

    # 6. structured output
    if chat is not None:
        try:
            instance, confidence, _raw = parse_structured(
                chat,
                [Message.system("Return JSON only, matching the schema exactly."),
                 Message.user('Return {"reply": "ok"}')],
                _Ping,
            )
        except Exception as exc:
            _check("FAIL", "structured output validates", f"{type(exc).__name__}: {exc}")
        else:
            if len(confidence.factors) >= 2:
                _check("PASS", "structured output validates",
                       f"reply={instance.reply!r} factors={len(confidence.factors)}")
            else:
                _check("FAIL", "structured output validates",
                       f"confidence carries {len(confidence.factors)} factor(s), expected >= 2")
    else:
        _check("FAIL", "structured output validates", "skipped: no chat model")

    # 7. cache -- non-fatal. Same messages/params as check 3, so this should be
    # a cache hit rather than a second real call.
    if chat is not None:
        try:
            repeat = chat.complete(_PING, max_tokens=5)
        except Exception as exc:
            _check("WARN", "repeated call served from cache", f"repeat call raised: {exc}")
        else:
            if repeat.from_cache:
                _check("PASS", "repeated call served from cache")
            else:
                _check("WARN", "repeated call served from cache",
                       "from_cache=False -- eval reruns will cost money without this")
    else:
        _check("WARN", "repeated call served from cache", "skipped: no chat model")

    # 8. embeddings
    try:
        embed = get_embedding_model()
        vectors = embed.embed(["ping"])
    except Exception as exc:
        _check("FAIL", "embeddings return expected dimensionality", f"{type(exc).__name__}: {exc}")
    else:
        got = len(vectors[0]) if vectors and vectors[0] else 0
        if got == embed.dimensions:
            _check("PASS", "embeddings return expected dimensionality", f"{got} dims")
        else:
            _check("FAIL", "embeddings return expected dimensionality",
                   f"got {got}, expected {embed.dimensions}")

    print()
    if _fatal:
        print(f"{_fatal} fatal check(s) failed.")
        return 1
    print("ready for P1-02")
    return 0


if __name__ == "__main__":
    sys.exit(main())
