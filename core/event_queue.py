"""
Timestamp‑ordered event queue.

This queue stores (timestamp, event) pairs in a binary min‑heap.  Typical usage:

    eq = EventQueue()
    eq.push(123, Order(...))
    for ts, ev in eq.pop_until(200):
        handle(ev)
"""

from typing import Any, Iterator, Tuple
import heapq


class EventQueue:
    """Ring‑buffer style priority queue for micro‑second events."""

    __slots__ = ("_heap",)

    def __init__(self) -> None:
        # Each element is a (timestamp, event) tuple
        self._heap: list[Tuple[int, Any]] = []

    # ────────────────────────── public ──────────────────────────
    def push(self, ts: int, event: Any) -> None:
        """Insert a new event with given timestamp."""
        heapq.heappush(self._heap, (ts, event))

    def pop_until(self, ts_bound: int) -> Iterator[Tuple[int, Any]]:
        """
        Yield and remove all events with timestamp ≤ `ts_bound`
        in ascending time order.
        """
        while self._heap and self._heap[0][0] <= ts_bound:
            yield heapq.heappop(self._heap)

    def __len__(self) -> int:  # convenience
        return len(self._heap)
