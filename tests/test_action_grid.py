from irisu_blackbox.action_grid import NOOP_ACTION, decode_action
from irisu_blackbox.config import ActionGridConfig


def test_noop_returns_none():
    cfg = ActionGridConfig(rows=2, cols=2, left=0, top=0, right=100, bottom=100)
    assert decode_action(NOOP_ACTION, cfg) is None


def test_left_click_first_cell():
    cfg = ActionGridConfig(rows=2, cols=2, left=0, top=0, right=100, bottom=100)
    cmd = decode_action(1, cfg)
    assert cmd is not None
    assert cmd.button == "left"
    assert cmd.x == 25
    assert cmd.y == 25


def test_right_click_first_cell():
    cfg = ActionGridConfig(rows=2, cols=2, left=0, top=0, right=100, bottom=100)
    cmd = decode_action(5, cfg)
    assert cmd is not None
    assert cmd.button == "right"
    assert cmd.x == 25
    assert cmd.y == 25
