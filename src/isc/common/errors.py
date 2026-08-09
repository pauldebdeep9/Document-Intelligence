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


class AclViolation(IscError):
    """A chunk without ACL terms reached the index, or a filter was bypassed.

    This is never caught and downgraded. It is a bug, not a condition.
    """


class ProviderError(IscError):
    """Upstream model provider failed in a way retry did not fix."""
