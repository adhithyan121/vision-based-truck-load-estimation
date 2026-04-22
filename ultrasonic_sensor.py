import time
import gpiod

class UltrasonicSystem:

    def __init__(self):
        # CHIP HANDLES
        self.chip1 = gpiod.Chip("/dev/gpiochip1")
        self.chip2 = gpiod.Chip("/dev/gpiochip2")
        self.chip3 = gpiod.Chip("/dev/gpiochip3")

        # PIN CONFIG
        self.TOP_TRIG_CHIP  = self.chip3
        self.TOP_TRIG       = 8

        self.TOP_ECHO_CHIP  = self.chip2
        self.TOP_ECHO       = 33

        self.SIDE_TRIG_CHIP = self.chip2
        self.SIDE_TRIG      = 41

        self.SIDE_ECHO_CHIP = self.chip1
        self.SIDE_ECHO      = 7

    def measure(self, trig_chip, trig_pin, echo_chip, echo_pin, timeout=0.04):

        trig_settings = gpiod.LineSettings()
        trig_settings.direction = gpiod.line.Direction.OUTPUT

        echo_settings = gpiod.LineSettings()
        echo_settings.direction = gpiod.line.Direction.INPUT

        trig_req = trig_chip.request_lines(
            consumer="ultra_trig",
            config={trig_pin: trig_settings}
        )

        echo_req = echo_chip.request_lines(
            consumer="ultra_echo",
            config={echo_pin: echo_settings}
        )

        trig_req.set_value(trig_pin, gpiod.line.Value.INACTIVE)
        time.sleep(0.05)

        trig_req.set_value(trig_pin, gpiod.line.Value.ACTIVE)
        time.sleep(0.00001)
        trig_req.set_value(trig_pin, gpiod.line.Value.INACTIVE)

        start_time = time.time()

        while echo_req.get_value(echo_pin) == gpiod.line.Value.INACTIVE:
            if time.time() - start_time > timeout:
                trig_req.release()
                echo_req.release()
                return 0.0

        pulse_start = time.time()

        while echo_req.get_value(echo_pin) == gpiod.line.Value.ACTIVE:
            if time.time() - pulse_start > timeout:
                trig_req.release()
                echo_req.release()
                return 0.0

        pulse_end = time.time()

        trig_req.release()
        echo_req.release()

        duration = pulse_end - pulse_start
        return (duration * 343.0) / 2

    def top_height(self):
        return round(self.measure(
            self.TOP_TRIG_CHIP, self.TOP_TRIG,
            self.TOP_ECHO_CHIP, self.TOP_ECHO
        ), 3)

    def side_offset(self):
        return round(self.measure(
            self.SIDE_TRIG_CHIP, self.SIDE_TRIG,
            self.SIDE_ECHO_CHIP, self.SIDE_ECHO
        ), 3)
