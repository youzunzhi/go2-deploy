import torch
from utils.hardware import STAND_STAGE1_DURATION, STAND_STAGE2_DURATION, STAND_TARGET_POS_STAGE1, STAND_TARGET_POS_STAGE2

ROBOT_SPORT_API_ID_BALANCESTAND = 1002
ROBOT_SPORT_API_ID_STANDUP = 1004
ROBOT_SPORT_API_ID_STANDDOWN = 1005


class ControlModeManager:
    """
    Manage the control mode of the robot.
    There are three control modes:
    1. Sport mode: The built-in sport mode of the robot provided by the manufacturer.
    2. Stand mode: Send the stand action to the robot motors to prepare the robot for locomotion policy.
    3. Locomotion mode: Send the locomotion action from the policy to the robot motors.

    Flow:
    Sport mode -(L1)-> Stand mode -(Y)-> Locomotion mode -(L2)-> Sport mode
    """
    def __init__(self, handler):
        self.handler = handler
        self.which_mode = "sport" # "sport"|"stand"|"locomotion"
        self.stand_controller = StandController(handler)
        
    def sport_mode_before_locomotion(self):
        """ Handle sport mode operations based on controller input.
        Return True if the sport mode is switched to stand policy.
        """
        if self.which_mode == "sport":
            if (self.handler.joy_stick_buffer.keys & self.handler.WirelessButtons.R1):
                self.handler.log_info("In the sport mode, R1 pressed, robot will stand up.")
                self.handler._sport_mode_command(ROBOT_SPORT_API_ID_STANDUP)
            if (self.handler.joy_stick_buffer.keys & self.handler.WirelessButtons.R2):
                self.handler.log_info("In the sport mode, R2 pressed, robot will sit down.")
                self.handler._sport_mode_command(ROBOT_SPORT_API_ID_STANDDOWN)
            if (self.handler.joy_stick_buffer.keys & self.handler.WirelessButtons.X):
                self.handler.log_info("In the sport mode, X pressed, robot will balance stand.")
                self.handler._sport_mode_command(ROBOT_SPORT_API_ID_BALANCESTAND)
            if (self.handler.joy_stick_buffer.keys & self.handler.WirelessButtons.L1):
                self.handler.log_info("Exist the sport mode. Switch to stand policy.")
                self.switch_to_stand_policy()

        if self.which_mode == "stand":
            self.stand_controller.send_stand_action()

            if (self.handler.joy_stick_buffer.keys & self.handler.WirelessButtons.Y):
                self.handler.log_info("Y pressed, use the locomotion policy")
                self.switch_to_locomotion_policy()
                return True
        return False
                    
    def switch_to_sport_mode(self):
        """Switch to sport mode from other modes"""
        self.which_mode = "sport"
        self.handler.reset_obs()
        self.handler._sport_mode_switch(1)
        self.handler._sport_mode_command(ROBOT_SPORT_API_ID_BALANCESTAND)
        
    def switch_to_stand_policy(self):
        """Switch to stand policy from sport mode"""
        self.which_mode = "stand"
        self.stand_controller.reset_stand_sequence()
        self.handler._sport_mode_switch(0)
        
    def switch_to_locomotion_policy(self):
        """Switch to locomotion policy from other modes"""
        self.which_mode = "locomotion"
        self.handler.global_counter = 0

    def sport_mode_after_locomotion(self):
        """Switch to sport mode after locomotion if L2 is pressed
        return True if the sport mode is switched to sport mode
        """
        if (self.handler.joy_stick_buffer.keys & self.handler.WirelessButtons.L2):
            self.handler.log_info("L2 pressed, stop using locomotion policy, switch to sport mode.")
            self.switch_to_sport_mode()
            return True
        return False


class StandController:
    """
    Manages the standing sequence that prepares the robot for locomotion policy.
    
    The standing sequence has two stages:
    1. Stage 1: Move from current position to intermediate standing position
    2. Stage 2: Move from intermediate position to final policy-ready position
    """
    
    def __init__(self, handler):
        self.handler = handler
        self.device = handler.device
        
        # Initialize stand configuration
        self.reset_stand_sequence()
        
        # Load target positions from hardware config
        self._target_pos_stage1 = list(STAND_TARGET_POS_STAGE1.values())
        self._target_pos_stage2 = list(STAND_TARGET_POS_STAGE2.values())
        
        # Stage durations (in control loop iterations)
        self.duration_stage1 = STAND_STAGE1_DURATION
        self.duration_stage2 = STAND_STAGE2_DURATION
    
    def reset_stand_sequence(self):
        """Reset the standing sequence to initial state"""
        self.start_pos = [0.0] * 12
        self.stand_action = [0.0] * 12
        
        # Progress tracking for each stage (0.0 to 1.0)
        self.progress_stage1 = 0.0
        self.progress_stage2 = 0.0
        
        # State flags
        self.need_capture_initial_pos = True
        self.need_log_stage1_once = True
        self.need_log_stage2_once = True
    
    def get_stand_action(self):
        """
        Compute the standing action for the current timestep.
        
        Returns:
            list: Joint positions for standing sequence [12 joints]
        """
        # Capture initial position on first run
        if self.need_capture_initial_pos:
            self._capture_initial_position()
            self.need_capture_initial_pos = False
        
        # Stage 1: Move to intermediate standing position
        if self.progress_stage1 < 1.0:
            self._execute_stage1()
        # Stage 2: Move to final policy-ready position  
        elif self.progress_stage2 < 1.0:
            self._execute_stage2()
        
        return self.stand_action
    
    def _capture_initial_position(self):
        """Record the current joint positions as starting point"""
        for i in range(12):
            self.start_pos[i] = self.handler.low_state_buffer.motor_state[i].q
    
    def _execute_stage1(self):
        """Execute stage 1: interpolate to intermediate position"""
        # Update progress (increment each timestep)
        self.progress_stage1 += 1.0 / self.duration_stage1
        self.progress_stage1 = min(self.progress_stage1, 1.0)
        
        # Log stage transition once
        if self.need_log_stage1_once:
            self.handler.log_info('Standing Stage 1: Moving to intermediate position', once=True)
            self.need_log_stage1_once = False
            self.need_log_stage2_once = True
        
        # Interpolate between start and stage 1 target positions
        for i in range(12):
            self.stand_action[i] = (
                (1.0 - self.progress_stage1) * self.start_pos[i] + 
                self.progress_stage1 * self._target_pos_stage1[i]
            )
    
    def _execute_stage2(self):
        """Execute stage 2: interpolate to final policy-ready position"""
        # Update progress (increment each timestep)
        self.progress_stage2 += 1.0 / self.duration_stage2
        self.progress_stage2 = min(self.progress_stage2, 1.0)
        
        # Log stage transition once
        if self.need_log_stage2_once:
            self.handler.log_info('Standing Stage 2: Moving to policy-ready position', once=True)
            self.need_log_stage2_once = False
        
        # Interpolate between stage 1 and stage 2 target positions
        for i in range(12):
            self.stand_action[i] = (
                (1.0 - self.progress_stage2) * self._target_pos_stage1[i] + 
                self.progress_stage2 * self._target_pos_stage2[i]
            )
    
    def is_stand_sequence_complete(self):
        """Check if the standing sequence is fully complete"""
        return self.progress_stage1 >= 1.0 and self.progress_stage2 >= 1.0
    
    def send_stand_action(self):
        """Compute and send the current stand action to robot motors"""
        stand_action = self.get_stand_action()
        actions_tensor = torch.tensor(stand_action, device=self.device).unsqueeze(0)
        self.handler._publish_legs_cmd(actions_tensor[0])

