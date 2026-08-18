"""Per-instance, config-aware cache manager with LRU eviction.

Replaces the ad-hoc dict caches scattered across ``_BLPBase`` subclasses and
``ProductionV2Model`` with a single manager.  Each ``CacheManager`` owns one or
more namespaces.  Every key is transparently prefixed with a hash of the model
config, so a configuration change naturally misses stale entries without
requiring explicit cache invalidation.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterator, MutableMapping
from typing import Any


class _NamespaceCache(MutableMapping[Hashable, Any]):
    """LRU-bounded namespace used by ``CacheManager``."""

    def __init__(self, name: str, config_hash: str, maxsize: int = 128) -> None:
        self._name = name
        self._config_hash = config_hash
        self._maxsize = maxsize
        self._data: OrderedDict[tuple[str, Hashable], Any] = OrderedDict()

    def _full_key(self, key: Hashable) -> tuple[str, Hashable]:
        return (self._config_hash, key)

    def __getitem__(self, key: Hashable) -> Any:
        full = self._full_key(key)
        value = self._data[full]
        self._data.move_to_end(full)
        return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        full = self._full_key(key)
        if full in self._data:
            self._data.move_to_end(full)
        self._data[full] = value
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __delitem__(self, key: Hashable) -> None:
        del self._data[self._full_key(key)]

    def __contains__(self, key: Hashable) -> bool:
        return self._full_key(key) in self._data

    def __iter__(self) -> Iterator[Hashable]:
        for _, key in self._data:
            yield key

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


class CacheManager:
    """Config-aware cache manager with LRU eviction.

    Typical usage inside a model constructor::

        self._cache_manager = CacheManager(config_hash=self._config_hash(), maxsize=256)
        self._raw_pca_cache = self._cache_manager.namespace("raw_pca")
        self._residual_pca_cache = self._cache_manager.namespace("residual_pca")

    All namespace views behave like normal ``dict`` instances and can be passed
    to helpers such as ``download_macro_prices``.
    """

    def __init__(
        self,
        config_hash: str,
        maxsize: int = 128,
    ) -> None:
        self._config_hash = config_hash
        self._maxsize = maxsize
        self._namespaces: dict[str, _NamespaceCache] = {}

    @staticmethod
    def config_hash_from_pydantic(cfg: Any) -> str:
        """Return a stable hash string for a Pydantic (or dataclass) config.

        Falls back to ``id(cfg)`` for objects that cannot be serialized.
        """
        if hasattr(cfg, "model_dump_json"):
            return str(hash(cfg.model_dump_json()))
        if hasattr(cfg, "model_dump"):
            import json

            try:
                return str(hash(json.dumps(cfg.model_dump(), sort_keys=True, default=str)))
            except (TypeError, ValueError):
                pass
        return str(id(cfg))

    def namespace(self, name: str, maxsize: int | None = None) -> _NamespaceCache:
        """Return the namespace view with *name*, creating it if necessary."""
        if name not in self._namespaces:
            self._namespaces[name] = _NamespaceCache(
                name, self._config_hash, maxsize or self._maxsize
            )
        return self._namespaces[name]

    def clear(self) -> None:
        """Clear all namespaces."""
        for ns in self._namespaces.values():
            ns.clear()
