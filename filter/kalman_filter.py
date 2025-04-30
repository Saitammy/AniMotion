import numpy as np

class KalmanFilter:
    """
    A 1-dimensional Kalman Filter for smoothing noisy signals.

    Attributes:
        process_variance (float): Variance of the process noise.
        measurement_variance (float): Variance of the measurement noise.
        estimated_error (float): Current uncertainty in the state estimate.
        state_estimate (float): Current estimated state.
        kalman_gain (float): The computed Kalman gain.
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
        estimated_error: float = 1.0
    ) -> None:
        self.process_variance: float = process_variance
        self.measurement_variance: float = measurement_variance
        self.estimated_error: float = estimated_error
        self.state_estimate: float = 0.0
        self.kalman_gain: float = 0.0

    def update(self, measurement: float) -> float:
        """
        Update the Kalman filter with a new measurement.

        This method performs both the prediction and correction steps:
          - A prediction update increments the uncertainty,
          - A correction using the new measurement refines the state estimate.

        Args:
            measurement (float): The new measurement to update the filter with.

        Returns:
            float: The updated state estimate.
        """
        # Prediction update: Increase the estimated error by the process variance.
        self.estimated_error += self.process_variance

        # Measurement update: Compute the Kalman gain.
        self.kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_variance)
        
        # Correct the state estimate using the measurement.
        self.state_estimate += self.kalman_gain * (measurement - self.state_estimate)
        
        # Update the error estimate.
        self.estimated_error *= (1 - self.kalman_gain)
        
        return self.state_estimate

    def reset(self, initial_state: float = 0.0, initial_error: float = 1.0) -> None:
        """
        Reset the filter to a specified initial state and error.

        Args:
            initial_state (float): The new starting state. Defaults to 0.0.
            initial_error (float): The new starting uncertainty. Defaults to 1.0.
        """
        self.state_estimate = initial_state
        self.estimated_error = initial_error
        self.kalman_gain = 0.0