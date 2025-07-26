import torch
from unitree_api.msg import Request
from utils.hardware_cfgs import STAND_STAGE1_DURATION, STAND_STAGE2_DURATION, STAND_TARGET_POS_STAGE1, STAND_TARGET_POS_STAGE2, WirelessButtons

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
    def __init__(self, handler, policy_interface):
        self.handler = handler
        self.policy_interface = policy_interface
        self.which_mode = "sport" # "sport"|"stand"|"locomotion"
        self.stand_controller = StandController(handler)
        
        # Button state tracking to prevent spam
        self.last_button_state = 0
        self.button_logged = {}
        
        # Show initial control prompts
        self._show_sport_mode_prompts()

    def sport_mode_before_locomotion(self):
        """ Handle sport mode operations based on controller input.
        Return True if the sport mode is switched to stand policy.
        """
        if self.which_mode == "sport":
            current_button = self.handler.joy_stick_buffer.keys
            
            if (current_button & WirelessButtons.R1):
                if not (self.last_button_state & WirelessButtons.R1):
                    self.handler.log_info("R1 pressed: Robot standing up. Press R2 to sit down, or L1 for stand policy.")
                self._sport_mode_command(ROBOT_SPORT_API_ID_STANDUP)
            if (current_button & WirelessButtons.R2):
                if not (self.last_button_state & WirelessButtons.R2):
                    self.handler.log_info("R2 pressed: Robot sitting down. Press R1 to stand up, or L1 for stand policy.")
                self._sport_mode_command(ROBOT_SPORT_API_ID_STANDDOWN)
            if (current_button & WirelessButtons.X):
                if not (self.last_button_state & WirelessButtons.X):
                    self.handler.log_info("X pressed: Robot balancing stand.")
                self._sport_mode_command(ROBOT_SPORT_API_ID_BALANCESTAND)
            if (current_button & WirelessButtons.L1):
                if not (self.last_button_state & WirelessButtons.L1):
                    self.handler.log_info("L1 pressed: Switching to stand policy.")
                self.switch_to_stand_policy()
                
            self.last_button_state = current_button

        if self.which_mode == "stand":
            self.stand_controller.send_stand_action()

            current_button = self.handler.joy_stick_buffer.keys
            if (current_button & WirelessButtons.Y):
                if not (self.last_button_state & WirelessButtons.Y):
                    self.handler.log_info("Y pressed: Activating locomotion policy...")
                self.switch_to_locomotion_policy()
            self.last_button_state = current_button
                    
    def switch_to_sport_mode(self):
        """Switch to sport mode from other modes"""
        self.which_mode = "sport"
        self.handler.reset_obs()
        self._sport_mode_switch(1)
        self._sport_mode_command(ROBOT_SPORT_API_ID_BALANCESTAND)
        self._show_sport_mode_prompts()
        
    def switch_to_stand_policy(self):
        """Switch to stand policy from sport mode"""
        self.which_mode = "stand"
        self.stand_controller.reset_stand_sequence()
        self._sport_mode_switch(0)
        self._show_stand_mode_prompts()
        
    def switch_to_locomotion_policy(self):
        """Switch to locomotion policy from other modes"""
        self.which_mode = "locomotion"
        self.policy_interface.policy_iter_counter = 0
        self._show_locomotion_mode_prompts()

    def sport_mode_after_locomotion(self):
        """Switch to sport mode after locomotion if L2 is pressed
        return True if the sport mode is switched to sport mode
        """
        current_button = self.handler.joy_stick_buffer.keys
        if (current_button & WirelessButtons.L2):
            if not (self.last_button_state & WirelessButtons.L2):
                self.handler.log_info("L2 pressed: Returning to sport mode...")
            self.switch_to_sport_mode()
        self.last_button_state = current_button

    
    def _show_sport_mode_prompts(self):
        """Show control prompts for sport mode"""
        self.handler.log_info("\n=== SPORT MODE ===\n" +
                             "Controls:\n" +
                             "  R1: Stand up\n" +
                             "  R2: Sit down\n" +
                             "  X:  Balance stand\n" +
                             "  L1: Switch to stand policy")
    
    def _show_stand_mode_prompts(self):
        """Show control prompts for stand mode"""
        self.handler.log_info("\n=== STAND POLICY MODE ===\n" +
                             "Robot is preparing for locomotion...\n" +
                             "Controls:\n" +
                             "  Y:  Switch to locomotion policy\n" +
                             "  L2: Return to sport mode")
    
    def _show_locomotion_mode_prompts(self):
        """Show control prompts for locomotion mode"""
        self.handler.log_info("\n=== LOCOMOTION POLICY MODE ===\n" +
                             "AI locomotion is active.\n" +
                             "Controls:\n" +
                             "  L2: Return to sport mode")

    def _sport_mode_command(self, api_id):
        """Send sport mode command to robot"""
        msg = Request()

        msg.header.identity.id = 0
        msg.header.identity.api_id = api_id
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        msg.parameter = ''
        msg.binary = []

        self.handler.sport_mode_pub.publish(msg)
    
    def _sport_mode_switch(self, mode):
        """Switch between sport mode and low-level control mode"""
        msg = Request()

        # Fill the header
        msg.header.identity.id = 0
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        if mode == 0:
            # Release mode (switch to low-level control mode) - use api_id 1003
            msg.header.identity.api_id = 1003
            msg.parameter = '{}'
        elif mode == 1:
            # Select sport mode - use api_id 1002
            msg.header.identity.api_id = 1002
            msg.parameter = '{"name": "mcf"}'
        
        msg.binary = []

        # Publish to motion switcher instead of robot state
        self.handler.motion_switcher_pub.publish(msg)


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
        
        # Load target positions from hardware config and convert to simulation order
        # CRITICAL: Target positions are defined in hardware order (by joint names)
        # but we need them in simulation order to match start_pos and _publish_legs_cmd
        target_pos_stage1_hw_order = list(STAND_TARGET_POS_STAGE1.values())
        target_pos_stage2_hw_order = list(STAND_TARGET_POS_STAGE2.values())
        
        # Convert from hardware order to simulation order
        self._target_pos_stage1 = [0.0] * 12
        self._target_pos_stage2 = [0.0] * 12
        for sim_idx in range(12):
            hw_idx = self.handler.joint_map[sim_idx]
            self._target_pos_stage1[sim_idx] = target_pos_stage1_hw_order[hw_idx]
            self._target_pos_stage2[sim_idx] = target_pos_stage2_hw_order[hw_idx]
                
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
        """Record the current joint positions as starting point in simulation order.
        
        CRITICAL: This captures positions in simulation order to ensure consistency:
        - start_pos: simulation order (captured here)
        - _target_pos_stage1/2: simulation order (converted during init)
        - _publish_legs_cmd: expects simulation order input
        - Interpolation happens between start_pos and targets (both sim order)
        """
        # Capture current joint positions in simulation order
        for sim_idx in range(12):
            hw_idx = self.handler.joint_map[sim_idx]
            self.start_pos[sim_idx] = self.handler.low_state_buffer.motor_state[hw_idx].q
        
        self.handler.log_info(f"Stand sequence: captured initial positions in simulation order")
    
    def _execute_stage1(self):
        """Execute stage 1: interpolate to intermediate position"""
        # Update progress (increment each timestep)
        self.progress_stage1 += 1.0 / self.duration_stage1
        self.progress_stage1 = min(self.progress_stage1, 1.0)
        
        # Log stage transition once
        if self.need_log_stage1_once:
            self.handler.log_info('Standing Stage 1: Moving to intermediate position')
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
            self.handler.log_info('Standing Stage 2: Moving to policy-ready position')
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

