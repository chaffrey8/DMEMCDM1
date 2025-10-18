import random
import math
import time

def buffon_needle(trials, needle_length, line_distance):
    hits = 0
    
    for _ in range(trials):
        # Random angle of the needle
        angle = random.uniform(0, math.pi / 2)
        
        # Random distance from the center of the needle to the closest line
        center_distance = random.uniform(0, line_distance / 2)
        
        # Check if the needle crosses a line
        if center_distance <= (needle_length / 2) * math.sin(angle):
            hits += 1
    
    # Estimating Pi using the probability formula
    pi_estimate = (2 * needle_length * trials) / (line_distance * hits)
    
    return pi_estimate

# Example usage with execution time
trials = 1000000000
needle_length = 1
line_distance = 1

# Measure execution time
start_time = time.time()
pi_approx = buffon_needle(trials, needle_length, line_distance)
end_time = time.time()

execution_time = end_time - start_time

print(f"Estimated Pi: {pi_approx}")
print(f"Execution Time: {execution_time:.6f} seconds")
