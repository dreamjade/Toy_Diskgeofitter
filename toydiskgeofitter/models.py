# toydiskgeofitter_project/toydiskgeofitter/models.py
import logging
import numpy as np
from typing import Tuple, Dict, Callable, Optional

# from .geometry import calculate_elliptical_radius_squared # If models use it

logger = logging.getLogger(__name__)

# This module is intended for physical disk brightness models,
# which can be used for generating model images or for model-based fitting
# (extending beyond purely geometric fitting).

def generate_2d_gaussian_model(
    image_shape: Tuple[int, int],
    amplitude: float,
    y0: float, x0: float, # Center
    sigma_y: float, sigma_x: float, # Standard deviations
    theta_rad: float = 0.0 # Rotation angle in radians (CCW from +x)
) -> np.ndarray:
    """
    Generates a 2D Gaussian model image.

    Args:
        image_shape (Tuple[int, int]): (ny, nx) of the output image.
        amplitude (float): Peak amplitude of the Gaussian.
        y0, x0 (float): Center coordinates of the Gaussian.
        sigma_y, sigma_x (float): Standard deviations along y and x axes BEFORE rotation.
        theta_rad (float): Rotation angle in radians (counter-clockwise from +x axis).

    Returns:
        np.ndarray: The 2D Gaussian model image.
    """
    ny, nx = image_shape
    y, x = np.ogrid[0:ny, 0:nx]

    # Shift coordinates
    x_shifted = x - x0
    y_shifted = y - y0

    # Rotate coordinates
    x_rot = x_shifted * np.cos(theta_rad) + y_shifted * np.sin(theta_rad)
    y_rot = -x_shifted * np.sin(theta_rad) + y_shifted * np.cos(theta_rad)

    # Gaussian formula
    if sigma_x <=0 or sigma_y <=0:
        logger.error("sigma_x and sigma_y must be positive for Gaussian model.")
        return np.zeros(image_shape)
        
    a = 1.0 / (2 * sigma_x**2)
    b = 1.0 / (2 * sigma_y**2) # c = 0 for un-sheared Gaussian before rotation

    model_image = amplitude * np.exp(-(a * x_rot**2 + b * y_rot**2))
    return model_image

# Example placeholder for a more complex disk model function signature
def example_disk_brightness_profile(
    radius: np.ndarray, # 1D array of deprojected radii
    params: Dict[str, float] # Dictionary of profile parameters (e.g., 'amplitude', 'r_cavity', 'slope')
) -> np.ndarray:
    """
    Example of a disk brightness profile function.
    (Not fully implemented, just a signature).
    """
    logger.info(f"Calculating brightness for radii using params: {params}")
    # Replace with actual model, e.g., power law, Gaussian ring, etc.
    # Example: A simple constant brightness for demonstration
    # amplitude = params.get("amplitude", 1.0)
    # return np.full_like(radius, amplitude)

    # Example: Gaussian Ring
    amp = params.get("amplitude", 1.0)
    r0 = params.get("r0", 10.0)
    sigma_r = params.get("sigma_r", 2.0)
    if sigma_r <=0: return np.zeros_like(radius)
    return amp * np.exp(-0.5 * ((radius - r0) / sigma_r)**2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Models module example usage:")

    shape = (64, 64)
    gauss_model = generate_2d_gaussian_model(
        shape, amplitude=100, y0=31.5, x0=31.5,
        sigma_y=5, sigma_x=10, theta_rad=np.radians(30)
    )
    logger.info(f"Generated 2D Gaussian model, shape: {gauss_model.shape}, max value: {np.max(gauss_model):.2f}")

    # To visualize (optional, needs matplotlib)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(gauss_model, origin='lower', interpolation='nearest')
    # plt.colorbar(label="Amplitude")
    # plt.title("2D Gaussian Model Example")
    # plt.show()

    radii_test = np.linspace(0, 50, 100)
    profile_params_test = {"amplitude": 50.0, "r0": 25.0, "sigma_r": 5.0}
    brightness_values = example_disk_brightness_profile(radii_test, profile_params_test)
    logger.info(f"Example brightness profile calculated for {len(radii_test)} radii.")
    # plt.figure()
    # plt.plot(radii_test, brightness_values)
    # plt.xlabel("Radius")
    # plt.ylabel("Brightness")
    # plt.title("Example Disk Brightness Profile")
    # plt.show()