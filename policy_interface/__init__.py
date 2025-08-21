from .EPO import EPOPolicyInterface
from .legged_loco import LeggedLocoPolicyInterface
from .ABS import ABSPolicyInterface
from .RT import RapidTurnPolicyInterface


def get_policy_interface(logdir, device):
    if "EPO" in logdir:
        return EPOPolicyInterface(logdir, device)
    elif "legged-loco" in logdir:
        return LeggedLocoPolicyInterface(logdir, device)
    elif "ABS" in logdir:
        return ABSPolicyInterface(logdir, device)
    elif "RT" in logdir:
        return RapidTurnPolicyInterface(logdir, device)
    else:
        raise ValueError(f"Invalid logdir: {logdir}")