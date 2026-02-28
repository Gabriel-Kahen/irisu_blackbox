from __future__ import annotations

import ctypes
import re
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

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
    relaunch_on_missing_window: bool = True
    relaunch_timeout_s: float = 15.0
    relaunch_poll_s: float = 0.25
    launch_executable: str | None = None
    launch_args: tuple[str, ...] = ()
    launch_workdir: str | None = None


@dataclass(slots=True)
class LaunchSpec:
    executable: str
    args: tuple[str, ...] = ()
    workdir: str | None = None


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
        self._explicit_launch_spec = self._binding_launch_spec()
        self._detected_launch_spec: LaunchSpec | None = None
        self._window, _ = self._resolve_window(
            allow_relaunch=self.binding.relaunch_on_missing_window
        )
        self._capture_region = self.binding.capture_region
        if self._capture_region is None:
            self._capture_region = self._window_region(self._window)
        self._remember_launch_spec_from_window(self._window)

        # Lower fail-safe odds while still allowing corner abort.
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0

    def _binding_launch_spec(self) -> LaunchSpec | None:
        if not self.binding.launch_executable:
            return None
        return LaunchSpec(
            executable=str(self.binding.launch_executable),
            args=tuple(self.binding.launch_args),
            workdir=str(self.binding.launch_workdir) if self.binding.launch_workdir else None,
        )

    @staticmethod
    def _window_identity(window) -> int:
        handle = getattr(window, "_hWnd", None)
        if handle is not None:
            return int(handle)
        return id(window)

    def _matching_windows(self) -> list:
        pattern = re.compile(self.binding.title_regex)
        matches = []
        seen: set[int] = set()
        for title in gw.getAllTitles():
            if not title or not pattern.search(title):
                continue
            for window in gw.getWindowsWithTitle(title):
                ident = self._window_identity(window)
                if ident in seen:
                    continue
                seen.add(ident)
                matches.append(window)
        return matches

    def _select_window(self, matches: list):
        if not matches:
            raise RuntimeError(f"No window title matched regex: {self.binding.title_regex!r}")
        if self.binding.window_index >= len(matches):
            raise RuntimeError(
                f"window_index {self.binding.window_index} is out of range for {len(matches)} matches"
            )
        return matches[self.binding.window_index]

    def _resolve_window(self, *, allow_relaunch: bool) -> tuple[object, bool]:
        matches = self._matching_windows()
        if matches:
            window = self._select_window(matches)
            self._remember_launch_spec_from_window(window)
            return window, False

        if not allow_relaunch or not self.binding.relaunch_on_missing_window:
            raise RuntimeError(f"No window title matched regex: {self.binding.title_regex!r}")

        self._launch_game()
        window = self._wait_for_window_after_launch()
        return window, True

    @staticmethod
    def _window_region(window) -> Rect:
        return Rect(
            left=int(window.left),
            top=int(window.top),
            width=int(window.width),
            height=int(window.height),
        )

    def _active_launch_spec(self) -> LaunchSpec | None:
        return self._explicit_launch_spec or self._detected_launch_spec

    def _remember_launch_spec_from_window(self, window) -> None:
        if self._explicit_launch_spec is not None:
            return
        inferred = self._infer_launch_spec_from_window(window)
        if inferred is not None:
            self._detected_launch_spec = inferred

    @staticmethod
    def _infer_launch_spec_from_window(window) -> LaunchSpec | None:
        hwnd = getattr(window, "_hWnd", None)
        if hwnd is None:
            return None

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            user32 = ctypes.WinDLL("user32", use_last_error=True)
        except Exception:
            return None

        pid = wintypes.DWORD()
        if user32.GetWindowThreadProcessId(wintypes.HWND(int(hwnd)), ctypes.byref(pid)) == 0:
            return None
        if pid.value <= 0:
            return None

        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid.value)
        if not handle:
            return None

        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            ok = kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size))
            if not ok or not buffer.value:
                return None
            executable = Path(buffer.value)
            return LaunchSpec(executable=str(executable), workdir=str(executable.parent))
        finally:
            kernel32.CloseHandle(handle)

    def _launch_game(self) -> None:
        launch_spec = self._active_launch_spec()
        if launch_spec is None:
            raise RuntimeError(
                "Game window disappeared, but no relaunch command is configured and the "
                "backend could not infer the original executable path."
            )

        cmd = [launch_spec.executable, *launch_spec.args]
        try:
            subprocess.Popen(cmd, cwd=launch_spec.workdir or None)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to relaunch game with command: {cmd!r}"
            ) from exc

    def _wait_for_window_after_launch(self):
        deadline = time.monotonic() + max(0.0, float(self.binding.relaunch_timeout_s))
        poll_s = max(0.0, float(self.binding.relaunch_poll_s))

        while True:
            matches = self._matching_windows()
            if matches:
                window = self._select_window(matches)
                self._remember_launch_spec_from_window(window)
                return window
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Game relaunch was attempted, but no matching window appeared before the "
                    f"{self.binding.relaunch_timeout_s:.1f}s timeout expired."
                )
            if poll_s > 0:
                time.sleep(poll_s)

    def _refresh_window_and_region(self, *, allow_relaunch: bool = True) -> bool:
        window, relaunched = self._resolve_window(allow_relaunch=allow_relaunch)
        self._window = window
        if self.binding.capture_region is None:
            self._capture_region = self._window_region(window)
        elif self._capture_region is None:
            self._capture_region = self.binding.capture_region
        return relaunched

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
                x = step.x
                y = step.y
                if step.relative_to_capture:
                    x = self._capture_region.left + x
                    y = self._capture_region.top + y
                pyautogui.mouseDown(x=x, y=y, button=step.button)
                if step.duration_s > 0:
                    time.sleep(step.duration_s)
                pyautogui.mouseUp(x=x, y=y, button=step.button)
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
        self._refresh_window_and_region(allow_relaunch=True)
        if self.binding.focus_before_step:
            self._focus_window()

        frame = np.array(self._sct.grab(self._capture_region.as_mss_monitor()), dtype=np.uint8)
        return frame[:, :, :3]

    def click(self, x: int, y: int, button: str = "left", hold_s: float = 0.01) -> None:
        relaunched = self._refresh_window_and_region(allow_relaunch=True)
        if relaunched:
            return
        if self.binding.focus_before_step:
            self._focus_window()
        pyautogui.mouseDown(x=x, y=y, button=button)
        if hold_s > 0:
            time.sleep(hold_s)
        pyautogui.mouseUp(x=x, y=y, button=button)

    def reset(self) -> None:
        self._refresh_window_and_region(allow_relaunch=True)
        if self.binding.focus_before_step:
            self._focus_window()
        self._run_macro()

    def run_macro(self, steps: list[ResetMacroStep]) -> None:
        if not steps:
            return
        self._refresh_window_and_region(allow_relaunch=True)
        if self.binding.focus_before_step:
            self._focus_window()

        original = self.reset_macro
        try:
            self.reset_macro = steps
            self._run_macro()
        finally:
            self.reset_macro = original

    def close(self) -> None:
        self._sct.close()
