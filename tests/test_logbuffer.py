"""RingBuffer unit tests: ordering, eviction, and clearing."""

from vibe.controller.logbuffer import RingBuffer


def test_append_and_len():
    buffer = RingBuffer()
    assert len(buffer) == 0
    buffer.append("a")
    buffer.append("b")
    assert len(buffer) == 2


def test_recent_is_newest_first():
    buffer = RingBuffer()
    for item in range(5):
        buffer.append(item)
    assert buffer.recent(2) == [4, 3]
    assert buffer.recent(10) == [4, 3, 2, 1, 0]  # capped at the buffer length


def test_recent_zero_or_negative_is_empty():
    buffer = RingBuffer()
    buffer.append("x")
    assert buffer.recent(0) == []
    assert buffer.recent(-1) == []


def test_maxlen_evicts_oldest():
    buffer = RingBuffer(maxlen=3)
    for item in range(5):
        buffer.append(item)
    assert len(buffer) == 3
    assert buffer.recent(10) == [4, 3, 2]


def test_clear():
    buffer = RingBuffer()
    for item in range(5):
        buffer.append(item)
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.recent(10) == []
