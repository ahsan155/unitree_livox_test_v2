import numpy as np
from numpy._typing import ArrayLike


def predict_trajectory(recorded_trajectory: ArrayLike, prediction_interval:float=10.0, number_of_predicted_points:int=11):
    """
    Estimates the current velocity of the object based on the two most recent samples and extrapolates the trajectory linearly.

    :param recorded_trajectory: The previously recorded trajectory as numpy array with dimensions [t, x, y, z].
    :param prediction_interval: The prediction time span in seconds.
    :param number_of_predicted_points: The number of samples in the prediction.
    :return: The predicted trajectory as numpy array with dimensions [t, x, y, z].
    """
    if np.shape(recorded_trajectory)[0] < 6:
        print("trajectory too short")
        return None

    estimated_velocity = (recorded_trajectory[-1, 1:] - recorded_trajectory[-6, 1:])/(recorded_trajectory[-1, 0] - recorded_trajectory[-6, 0])

    prediction_time_points = np.linspace(0, prediction_interval, number_of_predicted_points)


    predicted_trajectory = (recorded_trajectory[-1,1:])[np.newaxis] + (estimated_velocity)[np.newaxis] * (prediction_time_points)[np.newaxis].T
    predicted_trajectory = np.hstack([(prediction_time_points)[np.newaxis].T, predicted_trajectory])

    return predicted_trajectory







