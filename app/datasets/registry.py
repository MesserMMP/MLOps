from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class DatasetMeta:
    """Lightweight descriptor for a registered dataset."""

    name: str
    path: str
    description: str = ""


class DatasetRegistry:
    """In-memory registry that maps dataset names to their metadata.

    All access is via class-methods so no module-level mutable state is
    exposed to other packages.
    """

    _registry: Dict[str, DatasetMeta] = {}

    @classmethod
    def register(cls, name: str, path: str, description: str = "") -> DatasetMeta:
        """Add or update a dataset entry and return its metadata."""
        meta = DatasetMeta(name=name, path=path, description=description)
        cls._registry[name] = meta
        return meta

    @classmethod
    def get(cls, name: str) -> Optional[DatasetMeta]:
        """Return dataset metadata by name, or *None* if not found."""
        return cls._registry.get(name)

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a dataset from the registry.  Returns True if it existed."""
        return cls._registry.pop(name, None) is not None

    @classmethod
    def list_all(cls) -> List[DatasetMeta]:
        """Return all registered datasets as a list."""
        return list(cls._registry.values())

    @classmethod
    def as_dicts(cls) -> List[dict]:
        """Return all datasets serialised as plain dicts."""
        return [asdict(m) for m in cls._registry.values()]
