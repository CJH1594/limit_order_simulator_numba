"""
Numba-accelerated data structures for orders, trades, and cancels.
"""

from numba import int64, float64
from numba.experimental import jitclass

ORDER_SIDE_BUY: int = 1
ORDER_SIDE_SELL: int = -1

order_spec = [
    ('order_id',   int64),
    ('price',      float64),
    ('quantity',   int64),
    ('side',       int64),   # 1 = buy, -1 = sell
    ('timestamp',  int64),
]

@jitclass(order_spec)
class Order:
    """
    Simple limit/market order structure.

    Attributes
    ----------
    order_id : int64
        Unique order identifier.
    price : float64
        Limit price (ignored for pure market orders).
    quantity : int64
        Remaining quantity.
    side : int64
        +1 for buy, -1 for sell (Numba-friendly instead of Enum).
    timestamp : int64
        Event time in micro‑seconds.
    """
    def __init__(self, order_id: int, price: float, quantity: int,
                 side: int, timestamp: int):
        self.order_id  = order_id
        self.price     = price
        self.quantity  = quantity
        self.side      = side
        self.timestamp = timestamp

    def is_filled(self) -> bool:
        return self.quantity == 0


trade_spec = [
    ('maker_order_id', int64),
    ('taker_order_id', int64),
    ('price',          float64),
    ('quantity',       int64),
    ('timestamp',      int64),
]


@jitclass(trade_spec)
class Trade:
    """
    Execution report produced by the matching engine.
    """
    def __init__(self, maker_order_id: int, taker_order_id: int,
                 price: float, quantity: int, timestamp: int):
        self.maker_order_id = maker_order_id
        self.taker_order_id = taker_order_id
        self.price          = price
        self.quantity       = quantity
        self.timestamp      = timestamp


cancel_spec = [
    ('order_id',  int64),
    ('timestamp', int64),
]


@jitclass(cancel_spec)
class Cancel:
    """
    Cancel request keyed by order_id.
    """
    def __init__(self, order_id: int, timestamp: int):
        self.order_id  = order_id
        self.timestamp = timestamp

if __name__ == "__main__":
    # Simple sanity test when running this module directly
    buy_order = Order(
        order_id=1,
        price=100.0,
        quantity=10,
        side=ORDER_SIDE_BUY,
        timestamp=0
    )
    print(
        f"Order created: id={buy_order.order_id}, price={buy_order.price}, "
        f"qty={buy_order.quantity}, side={buy_order.side}, "
        f"filled={buy_order.is_filled()}"
    )