import numpy as np

from irisu_blackbox.config import Rect
from irisu_blackbox.backends import windows as winmod
from irisu_blackbox.backends.windows import LaunchSpec, WindowBinding, WindowsGameBackend


class FakeWindow:
    def __init__(
        self,
        title: str,
        *,
        hwnd: int,
        left: int = 100,
        top: int = 200,
        width: int = 320,
        height: int = 240,
    ) -> None:
        self.title = title
        self._hWnd = hwnd
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.activate_calls = 0

    def activate(self) -> None:
        self.activate_calls += 1


class FakeWindowAPI:
    def __init__(self, state: dict[str, list[FakeWindow]]) -> None:
        self.state = state

    def getAllTitles(self) -> list[str]:
        return [window.title for window in self.state["windows"]]

    def getWindowsWithTitle(self, title: str) -> list[FakeWindow]:
        return [window for window in self.state["windows"] if window.title == title]


class FakePyAutoGUI:
    def __init__(self) -> None:
        self.FAILSAFE = False
        self.PAUSE = 0.0
        self.mouse_down_calls: list[tuple[int, int, str]] = []
        self.mouse_up_calls: list[tuple[int, int, str]] = []
        self.key_presses: list[str] = []

    def mouseDown(self, x: int, y: int, button: str) -> None:
        self.mouse_down_calls.append((x, y, button))

    def mouseUp(self, x: int, y: int, button: str) -> None:
        self.mouse_up_calls.append((x, y, button))

    def press(self, key: str) -> None:
        self.key_presses.append(key)


class FakeScreenCapture:
    def __init__(self) -> None:
        self.monitors: list[dict[str, int]] = []
        self.closed = False

    def grab(self, monitor: dict[str, int]) -> np.ndarray:
        self.monitors.append(monitor)
        return np.zeros((monitor["height"], monitor["width"], 4), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeMSSModule:
    def __init__(self) -> None:
        self.instances: list[FakeScreenCapture] = []

    def mss(self) -> FakeScreenCapture:
        instance = FakeScreenCapture()
        self.instances.append(instance)
        return instance


def _patch_windows_dependencies(monkeypatch, state: dict[str, list[FakeWindow]]) -> FakePyAutoGUI:
    fake_pyautogui = FakePyAutoGUI()
    monkeypatch.setattr(winmod, "_IMPORT_ERROR", None)
    monkeypatch.setattr(winmod, "gw", FakeWindowAPI(state))
    monkeypatch.setattr(winmod, "pyautogui", fake_pyautogui)
    monkeypatch.setattr(winmod, "mss", FakeMSSModule())
    monkeypatch.setattr(winmod.time, "sleep", lambda _value: None)
    return fake_pyautogui


def test_windows_backend_relaunches_missing_window_on_capture(monkeypatch):
    original = FakeWindow("Irisu syndrome", hwnd=1)
    relaunched = FakeWindow("Irisu syndrome", hwnd=2)
    state = {"windows": [original]}
    _patch_windows_dependencies(monkeypatch, state)

    launches: list[tuple[list[str], str | None]] = []

    def fake_popen(cmd: list[str], cwd: str | None = None):
        launches.append((cmd, cwd))
        state["windows"] = [relaunched]
        return object()

    monkeypatch.setattr(winmod.subprocess, "Popen", fake_popen)

    backend = WindowsGameBackend(
        binding=WindowBinding(
            title_regex="Irisu syndrome",
            capture_region=Rect(left=10, top=20, width=30, height=40),
            relaunch_on_missing_window=True,
            launch_executable="C:/Games/Irisu.exe",
            launch_workdir="C:/Games",
        )
    )
    try:
        state["windows"] = []
        frame = backend.capture_frame()
        assert launches == [(["C:/Games/Irisu.exe"], "C:/Games")]
        assert frame.shape == (40, 30, 3)
        assert backend._window is relaunched
    finally:
        backend.close()


def test_windows_backend_relaunches_missing_window_during_init(monkeypatch):
    relaunched = FakeWindow("Irisu syndrome", hwnd=2)
    state = {"windows": []}
    _patch_windows_dependencies(monkeypatch, state)

    launches: list[tuple[list[str], str | None]] = []

    def fake_popen(cmd: list[str], cwd: str | None = None):
        launches.append((cmd, cwd))
        state["windows"] = [relaunched]
        return object()

    monkeypatch.setattr(winmod.subprocess, "Popen", fake_popen)

    backend = WindowsGameBackend(
        binding=WindowBinding(
            title_regex="Irisu syndrome",
            capture_region=Rect(left=10, top=20, width=30, height=40),
            relaunch_on_missing_window=True,
            launch_executable="C:/Users/gabek/Desktop/irisu.exe",
            launch_workdir="C:/Users/gabek/Desktop",
        )
    )
    try:
        assert launches == [(["C:/Users/gabek/Desktop/irisu.exe"], "C:/Users/gabek/Desktop")]
        assert backend._window is relaunched
    finally:
        backend.close()


def test_windows_backend_skips_click_that_triggered_relaunch(monkeypatch):
    original = FakeWindow("Irisu syndrome", hwnd=1)
    relaunched = FakeWindow("Irisu syndrome", hwnd=2)
    state = {"windows": [original]}
    fake_pyautogui = _patch_windows_dependencies(monkeypatch, state)

    launches: list[tuple[list[str], str | None]] = []

    def fake_popen(cmd: list[str], cwd: str | None = None):
        launches.append((cmd, cwd))
        state["windows"] = [relaunched]
        return object()

    monkeypatch.setattr(winmod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        winmod.WindowsGameBackend,
        "_infer_launch_spec_from_window",
        staticmethod(lambda _window: LaunchSpec("C:/Games/Irisu.exe", workdir="C:/Games")),
    )

    backend = WindowsGameBackend(
        binding=WindowBinding(
            title_regex="Irisu syndrome",
            capture_region=Rect(left=10, top=20, width=30, height=40),
            relaunch_on_missing_window=True,
        )
    )
    try:
        state["windows"] = []
        backend.click(123, 456)
        assert launches == [(["C:/Games/Irisu.exe"], "C:/Games")]
        assert fake_pyautogui.mouse_down_calls == []
        assert fake_pyautogui.mouse_up_calls == []
    finally:
        backend.close()
