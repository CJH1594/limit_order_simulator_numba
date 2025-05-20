"""
Limit‑order book engine (Numba‑accelerated prototype).

Book depth is fixed (MAX_LEVELS).  Each side (bids / asks) is represented by two
parallel NumPy arrays:

    prices[0:MAX_LEVELS]
    qtys[0:MAX_LEVELS]

Additionally   order_id → (side, index)   매핑을 numba.typed.Dict 로 유지하여
빠른 cancel 처리를 지원한다.

Public API
----------
init_book()                       → tuple(bids_p, bids_q, asks_p, asks_q, idx_map)
add_limit(order, …)               → None
process_market(order, …, trades)  → None
cancel(order_id, …)               → bool
match_incoming(order, …, trades)  → None   (내부에서 호출)
"""

import numpy as np
from numba import njit, int64, float64
from numba.typed import Dict
from core.datatypes import (
    Order,
    Trade,
    ORDER_SIDE_BUY,
    ORDER_SIDE_SELL,
)

# ──────────────────────────────────────────────────────────────────────────
MAX_LEVELS: int = 200      # book depth
INF_PRICE: float = 1e18


@njit(cache=True)
def _shift_down(prices, qtys, start: int):
    """
    Shift [0..start-1] down by one to make room at `start`.
    """
    for j in range(MAX_LEVELS - 1, start, -1):
        prices[j] = prices[j - 1]
        qtys[j] = qtys[j - 1]


@njit(cache=True)
def _shift_up(prices, qtys):
    """
    Remove level 0 and shift up.
    """
    for i in range(MAX_LEVELS - 1):
        prices[i] = prices[i + 1]
        qtys[i] = qtys[i + 1]
    # reset last slot
    prices[MAX_LEVELS - 1] = -1.0
    qtys[MAX_LEVELS - 1] = 0


@njit(cache=True)
def init_book():
    """
    Returns
    -------
    bids_p, bids_q, asks_p, asks_q : np.ndarray
    idx_map : numba.typed.Dict[int64, int64]
        order_id → signed index
        +idx  => bid side, index = idx - 1
        -idx  => ask side, index = -idx - 1
    """
    bids_p = np.full(MAX_LEVELS, -1.0)      # descending
    bids_q = np.zeros(MAX_LEVELS, dtype=np.int64)

    asks_p = np.full(MAX_LEVELS, INF_PRICE)  # ascending
    asks_q = np.zeros(MAX_LEVELS, dtype=np.int64)

    idx_map = Dict.empty(int64, int64)
    return bids_p, bids_q, asks_p, asks_q, idx_map


# ──────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def add_limit(order: Order, bids_p, bids_q, asks_p, asks_q, idx_map):
    """
    Insert a limit order.  If it can immediately match (crossing),
    `match_incoming` will be invoked inside the caller.
    """
    if order.side == ORDER_SIDE_BUY:
        # find insertion point (prices descending)
        pos = 0
        while pos < MAX_LEVELS and order.price <= bids_p[pos]:
            pos += 1
        if pos >= MAX_LEVELS:
            return  # book full, drop
        _shift_down(bids_p, bids_q, pos)
        bids_p[pos] = order.price
        bids_q[pos] = order.quantity
        idx_map[order.order_id] = pos + 1  # + sign for bid
    else:
        pos = 0
        while pos < MAX_LEVELS and order.price >= asks_p[pos]:
            pos += 1
        if pos >= MAX_LEVELS:
            return
        _shift_down(asks_p, asks_q, pos)
        asks_p[pos] = order.price
        asks_q[pos] = order.quantity
        idx_map[order.order_id] = -(pos + 1)  # − sign for ask


@njit(cache=True)
def _best_bid(bids_p, bids_q):
    return bids_p[0], bids_q[0]


@njit(cache=True)
def _best_ask(asks_p, asks_q):
    return asks_p[0], asks_q[0]


@njit(cache=True)
def cancel(order_id: int, bids_q, asks_q, idx_map) -> bool:
    """
    Cancel an existing resting order.

    Returns
    -------
    bool : True if found & removed
    """
    if order_id in idx_map:
        idx = idx_map.pop(order_id)
        if idx > 0:          # bid
            bids_q[idx - 1] = 0
        else:                # ask
            asks_q[-idx - 1] = 0
        return True
    return False


@njit(cache=True)
def match_incoming(order: Order, bids_p, bids_q, asks_p, asks_q, trades):
    """
    Core price-time priority matching loop.
    """
    if order.side == ORDER_SIDE_BUY:
        # match vs asks
        while order.quantity > 0:
            best_price, best_qty = _best_ask(asks_p, asks_q)
            if best_qty == 0 or order.price < best_price:
                break
            traded_qty = min(order.quantity, best_qty)
            trades.append(
                Trade(-1, order.order_id, best_price, traded_qty, order.timestamp)
            )
            order.quantity -= traded_qty
            asks_q[0] -= traded_qty
            if asks_q[0] == 0:
                _shift_up(asks_p, asks_q)
    else:
        # match vs bids
        while order.quantity > 0:
            best_price, best_qty = _best_bid(bids_p, bids_q)
            if best_qty == 0 or order.price > best_price:
                break
            traded_qty = min(order.quantity, best_qty)
            trades.append(
                Trade(-1, order.order_id, best_price, traded_qty, order.timestamp)
            )
            order.quantity -= traded_qty
            bids_q[0] -= traded_qty
            if bids_q[0] == 0:
                _shift_up(bids_p, bids_q)


@njit(cache=True)
def process_market(order: Order, bids_p, bids_q, asks_p, asks_q, trades):
    """
    Process a pure market order (price ignored).
    """
    order.price = INF_PRICE if order.side == ORDER_SIDE_BUY else -INF_PRICE
    match_incoming(order, bids_p, bids_q, asks_p, asks_q, trades)
