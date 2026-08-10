from pydantic import BaseModel, ConfigDict

__all__ = ["ExtensionConfig"]


class ExtensionConfig(BaseModel):
    """The configuration of a PEtab extension."""

    #: The extension's semantic version.
    version: str
    #: Whether the extension is required for the mathematical
    #: interpretation of the problem.
    required: bool

    model_config = ConfigDict(extra="allow", validate_assignment=True)
