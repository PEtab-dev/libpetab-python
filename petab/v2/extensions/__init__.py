from pydantic import BaseModel, ConfigDict

__all__ = ["ExtensionConfig", "parse_extension_config"]


class ExtensionConfig(BaseModel):
    """The configuration of a PEtab extension."""

    #: The extension's semantic version.
    version: str
    #: Whether the extension is required for the mathematical
    #: interpretation of the problem.
    required: bool

    model_config = ConfigDict(extra="allow", validate_assignment=True)


def _extension_config_classes() -> dict[str, type[ExtensionConfig]]:
    """Registry of extension ID to its specific :class:`ExtensionConfig`
    subclass, if any.

    Imported lazily (rather than built at module level) to avoid a
    circular import: extension submodules (e.g. ``sciml``) import
    :class:`ExtensionConfig` from this package.
    """
    from .. import C
    from .sciml import SciMLConfig

    return {C.EXT_ID_SCIML: SciMLConfig}


def parse_extension_config(
    ext_id: str, config: dict | ExtensionConfig
) -> ExtensionConfig:
    """Parse a single extension's configuration.

    Converts ``config`` to the extension-specific :class:`ExtensionConfig`
    subclass registered for ``ext_id``, or to the generic
    :class:`ExtensionConfig` if no specific subclass is registered.

    :param ext_id: The extension ID.
    :param config: The extension's configuration, as a dict or an already
        parsed :class:`ExtensionConfig` (sub)instance.
    """
    cls = _extension_config_classes().get(ext_id, ExtensionConfig)
    return config if isinstance(config, cls) else cls(**config)
