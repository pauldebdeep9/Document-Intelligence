"""Typed failure modes. Stages catch these; nothing else swallows exceptions."""


class IscError(Exception):
    """Base for everything raised deliberately by this package."""


class ConfigError(IscError):
    pass


class ParseError(IscError):
    """Document could not be turned into blocks by any parser in the chain."""


class ExtractionError(IscError):
    """Model output could not be coerced into the target schema after repair."""


class SchemaRepairExhausted(ExtractionError):
    def __init__(self, attempts: int, last_error: str) -> None:
        super().__init__(f"schema repair failed after {attempts} attempts: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


class OutputTruncated(ExtractionError):
    """The provider stopped generating because it hit max_tokens
    (finish_reason == "length"), not because it finished. A different problem
    from SchemaRepairExhausted, and needs a different response: this is a
    config problem (the cap is too low for this document), not a prompt or
    model problem, and retrying is guaranteed to truncate at the identical
    point every time -- the repair loop must never be entered for this.

    completion_tokens is reported as both "how much came back" and "what the
    ceiling was": when finish_reason == "length", the API's own contract
    guarantees they are the same number -- generation stopped exactly at the
    requested cap.
    """

    def __init__(self, completion_tokens: int) -> None:
        super().__init__(
            f"response truncated: completion_tokens={completion_tokens} hit the "
            "configured max_tokens cap before the model finished; increase "
            "llm.max_tokens rather than retrying, which will truncate identically"
        )
        self.completion_tokens = completion_tokens


class AclViolation(IscError):
    """A chunk without ACL terms reached the index, or a filter was bypassed.

    This is never caught and downgraded. It is a bug, not a condition.
    """


class ProviderError(IscError):
    """Upstream model provider failed in a way retry did not fix."""
