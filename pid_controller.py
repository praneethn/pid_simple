import numpy as np
import time

class PIDController:
    """
    A simple PID (Proportional-Integral-Derivative) controller implementation.
    
    Args:
        kp (float): Proportional gain
        ki (float): Integral gain
        kd (float): Derivative gain
        setpoint (float): Target value to achieve
        output_limits (tuple): Tuple of (min, max) output limits
        sample_time (float): Time between updates in seconds
    """
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0, output_limits=(-float('inf'), float('inf')), sample_time=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.sample_time = sample_time
        
        # Initialize internal state variables
        self.last_time = time.time()
        self.last_error = 0.0
        self.integral = 0.0
        
    def reset(self):
        """Reset the controller's internal state."""
        self.last_error = 0.0
        self.integral = 0.0
        
    def set_tunings(self, kp, ki, kd):
        """Update the PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
    def set_setpoint(self, setpoint):
        """Update the target setpoint."""
        self.setpoint = setpoint
        
    def compute(self, process_variable):
        """
        Compute the control output based on the process variable.
        
        Args:
            process_variable (float): Current value of the process being controlled
            
        Returns:
            float: Control output
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Only update if enough time has passed
        if dt < self.sample_time:
            return None
            
        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term (on measurement to avoid derivative kick)
        d_term = 0.0
        if dt > 0:  # Avoid division by zero
            d_term = -self.kd * (process_variable - self.last_error) / dt
            
        # Calculate total output
        output = p_term + i_term + d_term
        
        # Apply output limits if specified
        if self.output_limits[0] is not None and self.output_limits[1] is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
            
        # Update internal state
        self.last_time = current_time
        self.last_error = error
        
        return output 