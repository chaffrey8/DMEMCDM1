buffon_needle <- function(trials, needle_length, line_distance){
  hits <- 0
  for (i in 1:trials) {
    # Random angle of the needle
    angle <- runif(1, 0, pi/2)
    # Random distance ßfrom the center of the needle to the closest line
    center_distance <- runif(1, 0, line_distance / 2)
    # Check if the needle crosses a line
    if (center_distance <= (needle_length / 2) * sin(angle)) {
      hits <- hits + 1
    }
  }
  
  # Estimating Pi using the probability formula
  pi_estimate <- (2 * needle_length * trials) / (line_distance * hits)
  
  return(pi_estimate)
}

# Example usage with execution time
trials <- 1000000000
needle_length <- 1
line_distance <- 1

# Measure execution time
execution_time <- system.time({
  pi_approx <- buffon_needle(trials, needle_length, line_distance)
})

cat("Estimated Pi:", pi_approx, "\n")
cat("Execution Time:", execution_time, "\n")
