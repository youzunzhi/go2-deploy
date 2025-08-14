from typing import Optional, Tuple


class BasePolicyInterface:
    def __init__(self, logdir, device):
        self.logdir = logdir
        self.device = device

        self.handler = None
        self.policy_iter_counter = 0

        # configs for handler
        self.joint_map = None
        self.default_joint_pos = None
        self.kp = None
        self.kd = None
        self.action_scale = None
        self.clip_obs = None
        self.clip_actions = None

    def get_configs_for_handler(self) -> Tuple[list, list, float, float, float, float, Optional[float], bool, Optional[Tuple[int, int]], bool]:
        assert self.joint_map is not None, "Joint map is not set"
        assert self.default_joint_pos is not None, "Default joint pos is not set"
        assert self.kp is not None, "Kp is not set"
        assert self.kd is not None, "Kd is not set"
        assert self.action_scale is not None, "Action scale is not set"
        assert self.clip_obs is not None, "Clip obs is not set"

        # Get depth configuration
        enable_depth_capture, depth_resolution = self.get_depth_config()

        # Get translation capture configuration
        enable_translation_capture = self.get_translation_config()

        return self.joint_map, self.default_joint_pos, self.kp, self.kd, self.action_scale, self.clip_obs, self.clip_actions, enable_depth_capture, depth_resolution, enable_translation_capture
        
    def set_handler(self, handler):
        self.handler = handler
    
    def get_action(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def reset_policy_iter_counter(self):
        """Reset the policy iteration counter to 0"""
        self.policy_iter_counter = 0
    
    def get_depth_config(self) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Get depth capture configuration

        Returns:
            (enable_depth_capture, depth_resolution):
            - enable_depth_capture: Whether depth capture is needed
            - depth_resolution: (width, height) tuple if depth is needed, None otherwise
        """
        return False, None

    def get_translation_config(self) -> bool:
        """Get translation capture configuration

        Returns:
            enable_translation_capture: Whether translation capture is needed
        """
        return False
    