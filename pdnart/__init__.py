"""pdnart: compose layered art in code (or via Claude) and open it in paint.net."""

from .render import composite, render_layers
from .scene import Layer, Scene, SceneError

__all__ = ["Layer", "Scene", "SceneError", "composite", "render_layers"]
__version__ = "0.1.0"
