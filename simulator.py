

"""
Simple event‑driven simulation loop.

This script wires together the EventQueue, order‑book functions, and (optionally)
a trading strategy.  It is *not* a full‑featured back‑tester yet—only a minimal
driver that proves the core engine works end‑to‑end.

Usage
-----
$ python simulator.py
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Optional

from numba.typed import List  # typed list for Numba

from core.datatypes import Order, Trade, ORDER_SIDE_BUY, ORDER_SIDE_SELL
from core.event_queue import EventQueue
from core.order_book import (
    init_book,
    add_limit,
    match_incoming,
    process_market,
)

TIMESTAMP = int
Event = Order  # for now the only event type we handle


# ──────────────────────────────────────────────────────────────────────────
def run_simulation(
    events: Iterable[Tuple[TIMESTAMP, Event]],
    *,
    strategy: Optional[object] = None,
) -> List[Trade]:
    """
    Drive the event queue until empty and return the list of Trade executions.
    """
    bids_p, bids_q, asks_p, asks_q, idx_map = init_book()
    eq = EventQueue()
    trades = List.empty_list(Trade.class_type.instance_type)

    # preload events
    for ts, ev in events:
        eq.push(ts, ev)

    current_ts: TIMESTAMP = 0
    # main loop
    while len(eq):
        # pop events that are due
        for ts, ev in eq.pop_until(current_ts):
            if isinstance(ev, Order):
                if ev.price == 0.0:  # treat price==0 as a pure market order
                    process_market(ev, bids_p, bids_q, asks_p, asks_q, trades)
                else:
                    match_incoming(ev, bids_p, bids_q, asks_p, asks_q, trades)
                    if ev.quantity > 0:
                        add_limit(ev, bids_p, bids_q, asks_p, asks_q, idx_map)

            # strategy callback example (placeholder)
            if strategy is not None and hasattr(strategy, "on_event"):
                strategy.on_event(ev, ts, trades)

        current_ts += 1  # advance clock (1 µs step for demo)

    return list(trades)


# ───────────────────────── sample run ─────────────────────────
if __name__ == "__main__":
    demo_events: List[Tuple[int, Order]] = [
        # resting ask
        (1, Order(order_id=1, price=100.0, quantity=10, side=ORDER_SIDE_SELL, timestamp=1)),
        # incoming crossing bid
        (2, Order(order_id=2, price=101.0, quantity=10, side=ORDER_SIDE_BUY, timestamp=2)),
        # non‑crossing bid
        (3, Order(order_id=3, price=99.0, quantity=5, side=ORDER_SIDE_BUY, timestamp=3)),
        # market sell (price=0.0 means market)
        (4, Order(order_id=4, price=0.0, quantity=5, side=ORDER_SIDE_SELL, timestamp=4)),
    ]

    trades_out = run_simulation(demo_events)
    print("Executed trades:")
    for tr in trades_out:
        print(
            f"    ts={tr.timestamp:3d} price={tr.price:6.2f} qty={tr.quantity:3d} "
            f"(maker={tr.maker_order_id}, taker={tr.taker_order_id})"
        )