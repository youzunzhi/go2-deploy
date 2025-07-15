ROBOT_SPORT_API_ID_BALANCESTAND = 1002
ROBOT_SPORT_API_ID_STANDUP = 1004
ROBOT_SPORT_API_ID_STANDDOWN = 1005


class SportModeManager:
    def __init__(self, handler):
        self.handler = handler
        self.which_mode = "sport" # "sport"|"stand"|"locomotion"
        
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
            stand_action = self.handler.get_stand_action()
            self.handler.send_stand_action(stand_action)

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