import os
from .EPO import EPOPolicyInterface
from .legged_loco import LeggedLocoPolicyInterface, LeggedLocoVisionPolicyInterface


def get_policy_interface(logdir, device):
    if "EPO" in logdir:
        return EPOPolicyInterface(logdir, device)
    elif "legged-loco" in logdir:
        # 自动检测是否存在vision policy
        vision_policy_path = os.path.join(logdir, "vision_policy.jit")
        base_policy_path = os.path.join(logdir, "policy.jit")
        
        if os.path.exists(vision_policy_path):
            print("Loading legged-loco VISION policy interface with height map support")
            return LeggedLocoVisionPolicyInterface(logdir, device)
        elif os.path.exists(base_policy_path):
            print("Loading legged-loco BASE policy interface (no vision)")
            return LeggedLocoPolicyInterface(logdir, device)
        else:
            raise FileNotFoundError(f"No policy found in {logdir}. Expected 'policy.jit' or 'vision_policy.jit'")
    else:
        raise ValueError(f"Invalid logdir: {logdir}")