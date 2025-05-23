from tag.buffer import HindsightExperienceReplayBuffer, ReplayBuffer


def test_replay_buffer_add_and_sample():
    buf = ReplayBuffer(capacity=4)
    buf.add([0, 0, 0], [1, 1], 1.0, [0, 0, 0], False)
    sample = buf.sample(1)
    assert len(sample["obs"]) == 1
    assert len(buf) == 1


def test_her_buffer_inherits():
    her = HindsightExperienceReplayBuffer(capacity=2)
    her.add([0], [0], 0.0, [0], False, achieved_goal=[1], desired_goal=[0])
    assert isinstance(her, ReplayBuffer)
    assert len(her) >= 1
