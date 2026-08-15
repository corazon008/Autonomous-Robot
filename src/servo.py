from pca9685 import PCA9685


class Servo:
    # Motor 0 is left/right, Motor 1 is up/down
    def __init__(self):
        self.pwm_frequency = 50
        self.initial_pulse = 1500
        self.pwm_channel_map = {
            "0": 8,
            "1": 9,
            "2": 10,
            "3": 11,
            "4": 12,
            "5": 13,
            "6": 14,
            "7": 15,
        }
        self.pwm_servo = PCA9685(0x40, debug=True)
        self.pwm_servo.set_pwm_freq(self.pwm_frequency)
        for channel in self.pwm_channel_map.values():
            self.pwm_servo.set_servo_pulse(channel, self.initial_pulse)

        self.servo_angle = {"0": 90, "1": 90}

        self.set_servo_pwm("0", self.servo_angle["0"])
        self.set_servo_pwm("1", self.servo_angle["1"])

    def set_servo_pwm(self, channel: str, angle: int, error: int = 10) -> None:
        angle = int(angle)
        if channel not in self.pwm_channel_map:
            raise ValueError(
                f"Invalid channel: {channel}. Valid channels are {list(self.pwm_channel_map.keys())}."
            )
        pulse = (
            2500 - int((angle + error) / 0.09)
            if channel == "0"
            else 500 + int((angle + error) / 0.09)
        )
        self.pwm_servo.set_servo_pulse(self.pwm_channel_map[channel], pulse)

    def move_left(self, angle: int) -> None:
        self.set_servo_pwm("0", angle)

    def move_left_incremental(self, angle: int = 10) -> None:
        current_angle = self.servo_angle["0"]
        new_angle = current_angle - angle
        if new_angle < 0:
            new_angle = 0
        self.set_servo_pwm("0", new_angle)
        self.servo_angle["0"] = new_angle

    def move_right(self, angle: int) -> None:
        self.set_servo_pwm("0", angle)

    def move_right_incremental(self, angle: int = 10) -> None:
        current_angle = self.servo_angle["0"]
        new_angle = current_angle + angle
        if new_angle > 180:
            new_angle = 180
        self.set_servo_pwm("0", new_angle)
        self.servo_angle["0"] = new_angle

    def move_up(self, angle: int) -> None:
        self.set_servo_pwm("1", angle)

    def move_up_incremental(self, angle: int = 10) -> None:
        current_angle = self.servo_angle["1"]
        new_angle = current_angle + angle
        if new_angle > 180:
            new_angle = 180
        self.set_servo_pwm("1", new_angle)
        self.servo_angle["1"] = new_angle

    def move_down(self, angle: int) -> None:
        self.set_servo_pwm("1", angle)

    def move_down_incremental(self, angle: int = 10) -> None:
        current_angle = self.servo_angle["1"]
        new_angle = current_angle - angle
        if new_angle < 0:
            new_angle = 0
        self.set_servo_pwm("1", new_angle)
        self.servo_angle["1"] = new_angle


# Main program logic follows:
if __name__ == "__main__":
    print("Now servos will rotate to 90 degree.")
    print("If they have already been at 90 degree, nothing will be observed.")
    print("Please keep the program running when installing the servos.")
    print("After that, you can press ctrl-C to end the program.")
    pwm_servo = Servo()
    try:
        while True:
            pwm_servo.set_servo_pwm("0", 90)
            pwm_servo.set_servo_pwm("1", 90)
    except KeyboardInterrupt:
        print("\nEnd of program")
