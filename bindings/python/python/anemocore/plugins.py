from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class PluginSpec:
    name: str; kind: str; fn: Callable[..., Any]; version: str = "0.3.4"

class PluginRegistry:
    def __init__(self): self._s: dict[str, PluginSpec] = {}
    def register(self, name: str, fn: Callable[..., Any], *, kind: str = "custom") -> PluginSpec:
        sp = PluginSpec(name=name, kind=kind, fn=fn); self._s[name] = sp; return sp
    def get(self, name: str) -> Callable[..., Any]: return self._s[name].fn
    def list(self, kind: str | None = None) -> list[PluginSpec]:
        return [v for v in self._s.values() if kind is None or v.kind == kind]
    def clear(self) -> None: self._s.clear()

_REG = PluginRegistry()
def register_plugin(name: str, *, kind: str = "custom"):
    def _dec(fn): _REG.register(name, fn, kind=kind); return fn
    return _dec
def get_plugin(name: str): return _REG.get(name)
def list_plugins(kind: str | None = None): return _REG.list(kind)
def clear_plugins(): _REG.clear()
