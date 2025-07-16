

class BasePolicyInterface:
    def __init__(self, logdir, device):
        self.logdir = logdir
        self.device = device

        self.handler = None

        # configs for handler
        self.joint_map = None
        self.default_joint_pos = None
        self.kp = None
        self.kd = None
        self.action_scale = None
        self.clip_obs = None
        self.clip_actions = None

    def get_configs_for_handler(self):
        return self.joint_map, self.default_joint_pos, self.kp, self.kd, self.action_scale, self.clip_obs, self.clip_actions
        
    def set_handler(self, handler):
        self.handler = handler
    
    def get_action(self):
        raise NotImplementedError("Subclasses must implement this method")
    