from .EPO import EPOPolicyInterface
from .legged_loco import LeggedLocoPolicyInterface


def get_policy_interface(logdir, device):
    if "EPO" in logdir:
        return EPOPolicyInterface(logdir, device)
    elif "legged_loco" in logdir:
        return LeggedLocoPolicyInterface(logdir, device)
    else:
        raise ValueError(f"Invalid logdir: {logdir}")