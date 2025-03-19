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
    
    # Create single plot with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot process value and setpoint on primary y-axis
    ax1.plot(time_points[:len(process_values)], process_values, 'b-', label='Process Value')
    ax1.plot(time_points[:len(setpoints)], setpoints, 'b--', label='Setpoint')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Value', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for control output
    ax2 = ax1.twinx()
    ax2.plot(time_points[:len(control_outputs)], control_outputs, 'r-', label='Control Output')
    ax2.set_ylabel('Control Signal', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Create third y-axis for error
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
    ax3.plot(time_points[:len(errors)], errors, 'g-', label='Error')
    ax3.set_ylabel('Error', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')
    
    plt.title('PID Controller Response')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 