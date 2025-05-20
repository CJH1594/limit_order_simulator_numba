limit_order_simulator_numba
===========================

A high-performance, event-driven limit order book simulator written in Python and accelerated with Numba.

Features
--------
• Numba-JIT matching engine  
• Timestamp-ordered event queue  
• Modular strategy plug-in system  
• Fast back-testing with CSV I/O  

Quick start
-----------
```bash
conda activate env
pip install -r requirements.txt
python simulator.py