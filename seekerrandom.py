import random

class DecayingProcess:
    def __init__(self):
        self.value = 800.0  # Starting value (like your data)
        self.optimal_input = self.value/45.0  # The secret "Sweet Spot" we don't know
        
    def step(self, input_v):
        """
        The process naturally decays (value gets smaller).
        However, the SPEED of decay depends on how close 'input_v' is to the optimal.
        """
        self.optimal_input = self.value/45.0    
        dist = abs(input_v - self.optimal_input)
        efficiency = max(0.1, 1.0 - (dist / 50.0)) 
        base_decay = self.value * 0.01 
        actual_drop = base_decay * efficiency
        noise = random.uniform(-actual_drop*0.1, actual_drop*0.1)
        self.value -= (actual_drop + noise)
        return self.value + 10, efficiency
    
import random
import math

# ==========================================
# 1. STANDARD PID CLASS (No Hacks)
# ==========================================
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        
    def compute(self, setpoint, process_variable):
        error = setpoint - process_variable
        
        # P term
        p_out = self.kp * error
        
        # I term
        self.integral += error
        i_out = self.ki * self.integral
        
        # D term
        derivative = error - self.prev_error
        d_out = self.kd * derivative
        
        self.prev_error = error
        
        return p_out + i_out + d_out

# ==========================================
# 2. THE RIGOROUS EXTREMUM SEEKER
# ==========================================
class ExtremumSeeker:
    def __init__(self, start_input):
        self.center_input = start_input # This is what the PID controls
        self.t = 0
        
        # A. The Search Signal (Dither) parameters
        self.dither_amp = 2.0   # How wide we scan
        self.dither_freq = 0.25 # Frequency (radians per step)
        
        # B. Filters (Signal Processing)
        # High Pass Filter state (removes the DC offset/absolute value)
        self.y_high_pass = 0.0
        self.y_old = 0.0
        self.hp_alpha = 0.9 # Very strong filter to kill the "Trend"
        
        # Low Pass Filter state (smooths the calculated gradient)
        self.grad_estimate = 0.0
        self.lp_alpha = 0.1
        
        # C. The Controller
        # We want the Gradient to be 0.
        # This is an INTEGRAL-heavy controller because we are controlling a rate.
        self.pid = PID(kp=0.5, ki=0.5, kd=0.0) 

    def update(self, measurement_y):
        # 1. HIGH PASS FILTER (Washout)
        # Removes the absolute value (e.g. 800 or 8000). 
        # Leaves only the wiggly bits caused by our dither + noise.
        # This solves your "Rocket Nose" objection.
        self.y_high_pass = self.hp_alpha * (self.y_high_pass + measurement_y - self.y_old)
        self.y_old = measurement_y
        
        # 2. GENERATE DITHER SIGNAL
        # A clean sine wave (or square wave)
        dither = self.dither_amp * math.sin(self.t * self.dither_freq)
        
        # 3. DEMODULATION (The Magic Math)
        # Multiply the response by the dither.
        # positive * positive = positive (Gradient is +)
        # negative * positive = negative (Gradient is -)
        raw_gradient = self.y_high_pass * dither
        
        # 4. LOW PASS FILTER
        # Smooth out the noise to get the "Average Slope"
        self.grad_estimate = (self.lp_alpha * raw_gradient) + ((1 - self.lp_alpha) * self.grad_estimate)
        
        # 5. PID CONTROL
        # Target Gradient = 0
        # Current Gradient = self.grad_estimate
        # The PID output tells us how to move the "Center Input"
        pid_output = self.pid.compute(0.0, self.grad_estimate)
        
        # Update the Center Input (The "Bias")
        # Note: We subtract because we are minimizing. 
        # If Gradient is positive, we want to reduce input.
        self.center_input += pid_output
        
        # 6. COMBINE
        # The actual input we send to the machine is Center + Dither
        final_input = self.center_input + dither
        
        self.t += 1
        return final_input, self.grad_estimate, self.center_input

# Setup
process = DecayingProcess()
seeker = ExtremumSeeker(start_input=10.0) 

print(f"{'STEP':<5} | {'VALUE':<10} | {'CENTER':<8} | {'TARGET':<8} | {'GRADIENT':<8} | {'EFFICIENCY'}")
print("-" * 75)

current_val = process.value
for t in range(200):
    # The seeker handles the math. No hacks.
    final_input, gradient, center = seeker.update(current_val)
    
    current_val, efficiency = process.step(final_input)
    
    if t % 5 == 0: 
        print(f"{t:<5} | {current_val:10.2f} | {center:8.2f} | {process.optimal_input:8.2f} | {gradient:8.2f} | {efficiency:10.2f}")