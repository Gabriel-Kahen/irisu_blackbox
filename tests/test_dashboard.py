from irisu_blackbox.dashboard import SessionTracker, _format_duration
from irisu_blackbox.hud import HUDState


def test_format_duration():
    assert _format_duration(0) == "00:00:00"
    assert _format_duration(3661) == "01:01:01"


def test_session_tracker_counts_rounds_and_resets():
    tracker = SessionTracker(patience=2)

    stats = tracker.update(HUDState(score=0, health_percent=None, health_visible=False), now=10.0)
    assert stats.state == "WAITING"
    assert stats.episode_count == 0

    stats = tracker.update(HUDState(score=25, health_percent=0.8, health_visible=True), now=11.0)
    assert stats.state == "LIVE"
    assert stats.episode_count == 1
    assert stats.round_best_score == 25

    stats = tracker.update(HUDState(score=100, health_percent=0.6, health_visible=True), now=12.0)
    assert stats.state == "LIVE"
    assert stats.session_best_score == 100
    assert stats.round_best_score == 100

    stats = tracker.update(HUDState(score=None, health_percent=None, health_visible=False), now=13.0)
    assert stats.state == "TRANSITION"
    assert stats.reset_count == 0

    stats = tracker.update(HUDState(score=None, health_percent=None, health_visible=False), now=14.0)
    assert stats.state == "RESETTING"
    assert stats.reset_count == 1

    stats = tracker.update(HUDState(score=5, health_percent=0.9, health_visible=True), now=15.0)
    assert stats.state == "LIVE"
    assert stats.episode_count == 2
    assert stats.round_best_score == 5
