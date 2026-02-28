import numpy as np

from irisu_blackbox.observation import FrameProcessor


def test_frame_processor_outputs_stacked_rgb_channels():
    processor = FrameProcessor(width=4, height=3, stack=2)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # BGR

    reward_frame = processor.preprocess(frame)
    obs_frame = processor.preprocess_observation(frame)
    stacked = processor.reset(obs_frame)

    assert reward_frame.shape == (3, 3, 4)
    assert reward_frame.dtype == np.float32
    assert np.isclose(reward_frame[0, 0, 0], 30.0 / 255.0)
    assert np.isclose(reward_frame[1, 0, 0], 20.0 / 255.0)
    assert np.isclose(reward_frame[2, 0, 0], 10.0 / 255.0)

    assert obs_frame.shape == (3, 3, 4)
    assert obs_frame.dtype == np.uint8
    assert int(obs_frame[0, 0, 0]) == 30
    assert int(obs_frame[1, 0, 0]) == 20
    assert int(obs_frame[2, 0, 0]) == 10

    assert stacked.shape == (6, 3, 4)
    assert stacked.dtype == np.uint8
