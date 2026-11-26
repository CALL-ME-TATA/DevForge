"""Compatibility shim: the network now lives in policy.py.

Existing imports (`from model import Net, heads_to_buttons, load_compat, sample, evaluate, split_logits`)
in agent.py, training/train.py, training/evaluate.py and the tools keep working unchanged.
"""
from policy import *  # noqa: F401,F403
from policy import _head_stats  # noqa: F401  (private helper, re-exported for older scripts)
