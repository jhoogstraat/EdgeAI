import asyncio
import time
from typing import Optional
import digitalio
import board
from adafruit_motor import stepper as Stepper
from numpy import number

class MotorController:
    A = digitalio.DigitalInOut(board.D6)
    B = digitalio.DigitalInOut(board.D19)
    C = digitalio.DigitalInOut(board.D26)
    D = digitalio.DigitalInOut(board.D13)
    
    A.switch_to_output()
    B.switch_to_output()
    C.switch_to_output()
    D.switch_to_output()

    motor = Stepper.StepperMotor(A, B, C, D, microsteps=None)

    sleep_time = 0.003
    direction = Stepper.FORWARD

    running = False

    def status(self):
        return { 'pause': self.sleep_time, 'direction': self.direction, 'running': self.running }

    def run(self, sleep):
        if not self.running:
            self.running = True
            while self.running:
                self.motor.onestep(direction=self.direction, style=Stepper.SINGLE)
                sleep(self.sleep_time)
    
    def stop(self):
        self.running = False

    def change_step_pause(self, delta: Optional[float] = None, abs: Optional[float] = None):
        if delta is not None:
            self.sleep_time = max(0.0018, self.sleep_time + delta)
        elif abs is not None:
            self.sleep_time = max(0.0018, abs)
    
    def reverse(self):
        self.direction = Stepper.BACKWARD if self.direction == Stepper.FORWARD else Stepper.FORWARD
        

