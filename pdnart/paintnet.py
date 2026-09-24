"""Drive a running paint.net (Windows only) to build a real, layered document.

paint.net has no scripting API, so this uses two things it does support well:

* the clipboard, which paint.net reads as PNG (keeping transparency), and
* its keyboard shortcuts:
    Ctrl+Alt+V   Edit > Paste into New Image
    Ctrl+Shift+V Edit > Paste into New Layer
    F4           Layer Properties (used to name each layer and set its blend mode)
    Ctrl+Shift+S Save As

Each scene layer is rendered at full canvas size and pasted as its own paint.net
layer, so the result is an ordinary .pdn you can keep editing by hand.

Requires: pip install "pdnart[windows]"  (pywin32 + pywinauto)
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from .render import apply_opacity, background_image, render_layers
from .scene import Scene

TITLE_RE = r".*paint\.net.*"

# Our blend mode names -> the names in paint.net's Layer Properties dropdown.
PDN_BLEND_NAMES = {"multiply": "Multiply", "screen": "Screen", "overlay": "Overlay", "darken": "Darken",
                   "lighten": "Lighten", "additive": "Additive", "difference": "Difference"}

CANDIDATE_EXES = [
    os.environ.get("PDN_EXE", ""),
    r"C:\Program Files\paint.net\paintdotnet.exe",
    r"C:\Program Files (x86)\paint.net\paintdotnet.exe",
    # Microsoft Store install exposes an execution alias here.
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\paintdotnet.exe"),
]


class PaintDotNetError(RuntimeError):
    pass


def find_exe() -> str:
    for p in CANDIDATE_EXES:
        if p and Path(p).exists():
            return p
    found = shutil.which("paintdotnet")
    if found:
        return found
    raise PaintDotNetError("paint.net not found; install it or set PDN_EXE to paintdotnet.exe")


def _require_windows() -> None:
    if sys.platform != "win32":
        raise PaintDotNetError(
            "driving paint.net needs Windows (paint.net is Windows-only). "
            "Use `pdnart render` to export PNG layers on this machine instead."
        )


def copy_image_to_clipboard(img: Image.Image) -> None:
    """Put an RGBA image on the Windows clipboard as PNG (with alpha) plus a DIB fallback."""
    import win32clipboard  # type: ignore[import-not-found]

    png = io.BytesIO()
    img.save(png, "PNG")
    bmp = io.BytesIO()
    img.convert("RGBA").save(bmp, "BMP")
    dib = bmp.getvalue()[14:]  # strip BITMAPFILEHEADER

    png_format = win32clipboard.RegisterClipboardFormat("PNG")
    for attempt in range(10):  # another app may be holding the clipboard open
        try:
            win32clipboard.OpenClipboard()
            break
        except Exception:
            time.sleep(0.1 * (attempt + 1))
    else:
        raise PaintDotNetError("could not open the clipboard")
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(png_format, png.getvalue())
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, dib)
    finally:
        win32clipboard.CloseClipboard()


class PaintDotNet:
    """A connection to one paint.net window."""

    def __init__(self, exe: str | None = None, delay: float = 0.6, launch_timeout: float = 30):
        _require_windows()
        from pywinauto import Application  # type: ignore[import-not-found]

        self.delay = delay
        try:
            self.app = Application(backend="uia").connect(title_re=TITLE_RE, timeout=1)
        except Exception:
            subprocess.Popen([exe or find_exe()])
            self.app = Application(backend="uia").connect(title_re=TITLE_RE, timeout=launch_timeout)
        self.window = self.app.window(title_re=TITLE_RE, top_level_only=True)
        self.window.wait("visible ready", timeout=launch_timeout)

    # -- low level -------------------------------------------------------
    def keys(self, keys: str) -> None:
        from pywinauto.keyboard import send_keys  # type: ignore[import-not-found]

        self.window.set_focus()
        send_keys(keys)
        time.sleep(self.delay)

    def _commit_paste(self) -> None:
        # A paste leaves the pixels floating under the Move Selected tool;
        # Enter finishes the move and Ctrl+D drops the selection.
        self.keys("{ENTER}")
        self.keys("^d")

    # -- operations ------------------------------------------------------
    def new_image_from(self, img: Image.Image) -> None:
        copy_image_to_clipboard(img)
        self.keys("^%v")  # Paste into New Image
        self._commit_paste()
        self.keys("^b")  # Zoom to window, so later pastes land at (0, 0)

    def paste_layer(self, img: Image.Image, name: str | None = None, blend: str = "normal") -> bool:
        copy_image_to_clipboard(img)
        self.keys("^+v")  # Paste into New Layer
        self._commit_paste()
        return self.set_layer_properties(name, blend)

    def set_layer_properties(self, name: str | None = None, blend: str = "normal") -> bool:
        """Set the current layer's name and blend mode via Layer Properties (F4).

        Returns False if the blend mode could not be set (the name is best-effort either way).
        """
        if not name and blend == "normal":
            return True
        self.keys("{F4}")
        blend_ok = blend == "normal"
        try:
            dlg = self.app.window(title_re=r".*Layer Properties.*")
            dlg.wait("visible", timeout=5)
            if name:
                try:
                    dlg.child_window(control_type="Edit", found_index=0).set_edit_text(name)
                except Exception:
                    pass
            if blend != "normal":
                try:
                    dlg.child_window(control_type="ComboBox", found_index=0).select(PDN_BLEND_NAMES[blend])
                    blend_ok = True
                except Exception:
                    pass
            dlg.child_window(title="OK", control_type="Button").click()
        except Exception:
            self.keys("{ESC}")  # leave the layer as is rather than typing into the wrong field
        time.sleep(self.delay)
        return blend_ok

    def rename_current_layer(self, name: str) -> None:
        self.set_layer_properties(name)

    def save_as(self, path: str | Path) -> None:
        """Save the document as a layered .pdn file."""
        if Path(path).suffix.lower() != ".pdn":
            # Flat formats pop extra option/flatten dialogs; export those with `pdnart render`.
            raise PaintDotNetError("save_as only supports .pdn files")
        path = str(Path(path).resolve())
        self.keys("^+s")
        dlg = self.app.window(title_re=r"Save.*")
        dlg.wait("visible", timeout=10)
        try:
            dlg.child_window(title="File name:", control_type="Edit").set_edit_text(path)
            time.sleep(self.delay)
            dlg.child_window(title="Save", control_type="Button").click()
        except Exception:
            from pywinauto.keyboard import send_keys  # type: ignore[import-not-found]

            escaped = "".join("{%s}" % c if c in "+^%~(){}[]" else c for c in path)
            send_keys(escaped + "{ENTER}", with_spaces=True)
        time.sleep(self.delay * 3)

    def send_scene(self, scene: Scene, save_path: str | Path | None = None) -> list[str]:
        """Recreate `scene` in paint.net: a Background layer plus one layer per visible scene layer.

        Layer opacity is baked into the pasted pixels; blend modes are set in Layer Properties.
        Hidden layers are skipped. Returns warnings (e.g. a blend mode that could not be set).
        """
        warnings = []
        self.new_image_from(background_image(scene))
        self.rename_current_layer("Background")
        for layer, img in render_layers(scene):
            if layer.visible and not self.paste_layer(apply_opacity(img, layer.opacity), layer.name, layer.blend):
                warnings.append(f"could not set blend mode {layer.blend!r} on layer {layer.name!r}; "
                                "set it in Layer Properties (F4)")
        if save_path:
            self.save_as(save_path)
        return warnings
