from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from irisu_blackbox.config import Rect, ResetMacroStep

try:
    import mss
    import pyautogui
    import pygetwindow as gw
except Exception as exc:  # pragma: no cover - depends on host OS
    mss = None
    pyautogui = None
    gw = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from irisu_blackbox.backends.base import GameBackend


@dataclass(slots=True)
class WindowBinding:
    title_regex: str
    window_index: int = 0
    capture_region: Rect | None = None
    focus_before_step: bool = False


class WindowsGameBackend(GameBackend):
    def __init__(
        self,
        binding: WindowBinding,
        reset_macro: list[ResetMacroStep] | None = None,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "Windows backend dependencies are unavailable. Install dependencies on Windows."
            ) from _IMPORT_ERROR

        self.binding = binding
        self.reset_macro = reset_macro or []
        self._sct = mss.mss()
        self._window = self._resolve_window()
        self._capture_region = self.binding.capture_region
        if self._capture_region is None:
            self._capture_region = self._window_region(self._window)

        # Lower fail-safe odds while still allowing corner abort.
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0

    def _resolve_window(self):
        pattern = re.compile(self.binding.title_regex)
        matches = []
        for title in gw.getAllTitles():
            if title and pattern.search(title):
                window = gw.getWindowsWithTitle(title)
                if window:
                    matches.append(window[0])

        if not matches:
            raise RuntimeError(f"No window title matched regex: {self.binding.title_regex!r}")

        if self.binding.window_index >= len(matches):
            raise RuntimeError(
                f"window_index {self.binding.window_index} is out of range for {len(matches)} matches"
            )

        return matches[self.binding.window_index]

    @staticmethod
    def _window_region(window) -> Rect:
        return Rect(
            left=int(window.left),
            top=int(window.top),
            width=int(window.width),
            height=int(window.height),
        )

    def _refresh_window_and_region(self) -> None:
        if self.binding.capture_region is not None:
            return
        self._window = self._resolve_window()
        self._capture_region = self._window_region(self._window)

    def _focus_window(self) -> None:
        try:
            self._window.activate()
            time.sleep(0.02)
        except Exception:
            # Some windows deny focus requests; continue without focus hard-fail.
            pass

    def _run_macro(self) -> None:
        for step in self.reset_macro:
            kind = step.kind.lower()
            if kind == "sleep":
                time.sleep(step.duration_s)
                continue

            if kind == "click":
                if step.x is None or step.y is None:
                    raise ValueError("Reset macro click step requires x/y")
                pyautogui.mouseDown(x=step.x, y=step.y, button=step.button)
                if step.duration_s > 0:
                    time.sleep(step.duration_s)
                pyautogui.mouseUp(x=step.x, y=step.y, button=step.button)
                continue

            if kind == "key":
                if not step.key:
                    raise ValueError("Reset macro key step requires key")
                pyautogui.press(step.key)
                if step.duration_s > 0:
                    time.sleep(step.duration_s)
                continue

            raise ValueError(f"Unsupported reset macro step kind: {step.kind!r}")

    def capture_frame(self) -> np.ndarray:
        self._refresh_window_and_region()
        if self.binding.focus_before_step:
            self._focus_window()

        frame = np.array(self._sct.grab(self._capture_region.as_mss_monitor()), dtype=np.uint8)
        return frame[:, :, :3]

    def click(self, x: int, y: int, button: str = "left", hold_s: float = 0.01) -> None:
        if self.binding.focus_before_step:
            self._focus_window()
        pyautogui.mouseDown(x=x, y=y, button=button)
        if hold_s > 0:
            time.sleep(hold_s)
        pyautogui.mouseUp(x=x, y=y, button=button)

    def reset(self) -> None:
        if self.binding.focus_before_step:
            self._focus_window()
        self._run_macro()

    def close(self) -> None:
        self._sct.close()
