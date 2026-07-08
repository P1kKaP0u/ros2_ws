import time

class PID:
    def __init__(self, kp, ki, kd, axis_name=""):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.axis_name = axis_name

    def compute(self, error):
        current_time = time.time()
        dt = 0.1 if self.last_time is None else (current_time - self.last_time)
        self.last_time = current_time

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None