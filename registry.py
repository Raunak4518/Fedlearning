"""
registry.py

A tiny generic registry pattern shared by datasets/, generators/, and
targetNetModels/. This is the mechanism that makes the codebase
dataset-agnostic and architecture-agnostic: every dataset, generator, and
target network is looked up by a string name (the same name the user
passes on the command line via --dataset / --gen_model / --target_models),
never referenced directly by import elsewhere. Adding support for a new
dataset or a new target-network architecture never requires touching
args.py, utils/setup.py, or any training loop -- only adding one
`@register(...)`-decorated entry to the relevant module.
"""
from typing import Callable, Dict, Type


class Registry:
    def __init__(self, kind: str):
        self._kind = kind
        self._store: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        def _decorator(cls_or_fn):
            key = name.lower()
            if key in self._store:
                raise KeyError(f"{self._kind} '{name}' is already registered")
            self._store[key] = cls_or_fn
            return cls_or_fn
        return _decorator

    def get(self, name: str):
        key = name.lower()
        if key not in self._store:
            raise KeyError(
                f"Unknown {self._kind} '{name}'. Registered options: {sorted(self._store)}"
            )
        return self._store[key]

    def names(self):
        return sorted(self._store)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._store
