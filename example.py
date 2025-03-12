import numpy as np
import matplotlib.pyplot as plt
from pid_controller import PIDController
import time

def simulate_system(current_value, control_input, dt):
    """
    Simulate a simple first-order system with some noise.
    """
    tau = 1.0  # Time constant
    noise = np.random.normal(0, 0.05)  # Reduced noise amplitude
    return current_value + (control_input - current_value) * dt / tau + noise * dt

def main():
    # Create PID controller with very aggressive tuning
    pid = PIDController(
        kp=10.0,   # Much more aggressive proportional gain
        ki=3.0,    # Much higher integral gain for rapid error elimination
        kd=0.4,    # Slightly increased derivative gain
        setpoint=1.0,
        output_limits=(-8, 8),  # Wider output limits for more aggressive control
        sample_time=0.1
    )
    
    # Simulation parameters
    duration = 15.0
    dt = 0.1
    time_points = np.arange(0, duration, dt)
    
    # Arrays to store results
    process_values = []
    setpoints = []
    control_outputs = []
    errors = []  # Track errors for visualization
    
    # Initial conditions
    current_value = 0.0
    
    # Change setpoint halfway through
    setpoint_change_time = duration / 2
    
    # Simulation loop
    for t in time_points:
        # Change setpoint halfway through simulation
        if t >= setpoint_change_time:
            pid.set_setpoint(2.0)
            
        # Compute control output
        control_output = pid.compute(current_value)
        
        if control_output is not None:
            # Simulate system
            current_value = simulate_system(current_value, control_output, dt)
            
            # Store results
            process_values.append(current_value)
            setpoints.append(pid.setpoint)
            control_outputs.append(control_output)
            errors.append(pid.setpoint - current_value)  # Track error
            
        time.sleep(dt)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Process value and setpoint
    plt.subplot(3, 1, 1)
    plt.plot(time_points[:len(process_values)], process_values, label='Process Value')
    plt.plot(time_points[:len(setpoints)], setpoints, '--', label='Setpoint')
    plt.grid(True)
    plt.ylabel('Value')
    plt.title('PID Controller Response')
    plt.legend()
    
    # Control output
    plt.subplot(3, 1, 2)
    plt.plot(time_points[:len(control_outputs)], control_outputs, label='Control Output')
    plt.grid(True)
    plt.ylabel('Control Signal')
    plt.legend()
    
    # Error plot
    plt.subplot(3, 1, 3)
    plt.plot(time_points[:len(errors)], errors, label='Error')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 