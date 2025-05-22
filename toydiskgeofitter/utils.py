# toydiskgeofitter_project/toydiskgeofitter/utils.py
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import astropy.units as u

logger = logging.getLogger(__name__)

def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Configures basic logging for the application.

    Args:
        level (str): Logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
        log_file (Optional[str]): Path to a file to save logs.
                                  If None, logs to console.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = []
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        handlers.append(file_handler)
    
    # Ensure console handler is always added if no file handler or if desired
    # For a library, often you don't configure the root logger directly
    # but let the application do it. For now, this is a simple setup.
    # If no handlers are configured by the application, add a default StreamHandler.
    if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
        console_handler = logging.StreamHandler()
        handlers.append(console_handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logger.info(f"Logging setup with level {level}. Output to: {'Console' if not log_file else log_file}")


def convert_radius_units(
    radii_pixels: np.ndarray,
    pixel_scale_arcsec: Optional[float] = None,
    distance_pc: Optional[float] = None
) -> Tuple[np.ndarray, str]:
    """
    Converts pixel radii to arcseconds or Astronomical Units (AU).

    Args:
        radii_pixels (np.ndarray): Array of radii in pixel units.
        pixel_scale_arcsec (Optional[float]): Pixel scale in arcseconds/pixel.
        distance_pc (Optional[float]): Distance to the object in parsecs.

    Returns:
        Tuple[np.ndarray, str]:
            - Converted radii array.
            - String describing the units of the converted radii (e.g., "pixels", "arcsec", "AU").
    """
    if pixel_scale_arcsec is not None:
        radii_arcsec = radii_pixels * pixel_scale_arcsec
        if distance_pc is not None:
            # 1 AU at 1 pc subtends 1 arcsec. So, au = arcsec * pc
            radii_au = radii_arcsec * distance_pc
            logger.debug(f"Converted radii to AU using pixel_scale={pixel_scale_arcsec} arcsec/px, distance={distance_pc} pc.")
            return radii_au, "AU"
        else:
            logger.debug(f"Converted radii to arcsec using pixel_scale={pixel_scale_arcsec} arcsec/px.")
            return radii_arcsec, "arcsec"
    else:
        logger.debug("No pixel_scale_arcsec provided; radii remain in pixels.")
        return radii_pixels, "pixels"

def get_image_center_estimate(image_data: np.ndarray, method: str = "moments") -> Tuple[float, float]:
    """
    Estimates the image center (y0, x0) using simple methods.

    Args:
        image_data (np.ndarray): The 2D image array.
        method (str): Method to use. Options: "moments", "half_shape".

    Returns:
        Tuple[float, float]: Estimated (y_center, x_center).
    """
    ny, nx = image_data.shape
    if method == "half_shape":
        return ny / 2.0 - 0.5, nx / 2.0 - 0.5
    elif method == "moments":
        try:
            from skimage.measure import moments # Optional import
            # Ensure image is positive for moment calculation, add small offset if needed
            data_for_moments = image_data - np.nanmin(image_data)
            data_for_moments[np.isnan(data_for_moments)] = 0

            if np.nansum(data_for_moments) == 0: # Avoid division by zero if image is all zeros/nans
                 logger.warning("Image sum is zero, falling back to half_shape for center estimate.")
                 return ny / 2.0 - 0.5, nx / 2.0 - 0.5

            m = moments(data_for_moments, order=1)
            y_cen = m[1, 0] / m[0, 0]
            x_cen = m[0, 1] / m[0, 0]
            if np.isnan(y_cen) or np.isnan(x_cen):
                logger.warning("NaN encountered in moment calculation, falling back to half_shape.")
                return ny / 2.0 - 0.5, nx / 2.0 - 0.5
            return y_cen, x_cen
        except ImportError:
            logger.warning("scikit-image not found. Falling back to 'half_shape' for center estimate.")
            return ny / 2.0 - 0.5, nx / 2.0 - 0.5
        except Exception as e:
            logger.error(f"Error during moment calculation for center estimate: {e}. Falling back to 'half_shape'.")
            return ny / 2.0 - 0.5, nx / 2.0 - 0.5
    else:
        raise ValueError(f"Unknown center estimation method: {method}")

def parse_parameter_config(
    param_config: Dict[str, Any],
    image_shape: Optional[Tuple[int, int]] = None
) -> Tuple[str, Any, str, List[Any], str]:
    """
    Parses a single parameter's configuration block.
    Returns name, guess, prior_type, prior_args, label.
    Handles special 'image_center_x/y' guesses.
    """
    name = param_config.get("name", "unknown_param") # Should be key from dict
    guess = param_config.get("guess")
    prior_type = param_config.get("prior_type", "uniform").lower()
    prior_args = param_config.get("prior_args", [])
    label = param_config.get("label", name)
    to_fit = param_config.get("fit", True) # is this parameter to be fitted?

    if image_shape:
        ny, nx = image_shape
        if guess == "image_center_x":
            guess = nx / 2.0 - 0.5
            if not prior_args: prior_args = [0, nx-1] # Default prior if guess was symbolic
        elif guess == "image_center_y":
            guess = ny / 2.0 - 0.5
            if not prior_args: prior_args = [0, ny-1] # Default prior

    return name, guess, prior_type, prior_args, label, to_fit


# Example usage
if __name__ == "__main__":
    # setup_logging('DEBUG') # Call this from your main script
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    radii_px = np.array([10., 20., 30.])
    print(f"Original radii: {radii_px} pixels")

    r_arcsec, u_arcsec = convert_radius_units(radii_px, pixel_scale_arcsec=0.1)
    print(f"Converted radii: {r_arcsec} {u_arcsec}")

    r_au, u_au = convert_radius_units(radii_px, pixel_scale_arcsec=0.1, distance_pc=100)
    print(f"Converted radii: {r_au} {u_au}")

    # Test center estimate
    img = np.zeros((100, 100))
    img[40:60, 45:65] = 1 # A square blob
    yc, xc = get_image_center_estimate(img, method="moments")
    print(f"Estimated center (moments): y={yc:.2f}, x={xc:.2f} (Expected ~49.5, 54.5)")
    yc_h, xc_h = get_image_center_estimate(img, method="half_shape")
    print(f"Estimated center (half_shape): y={yc_h:.2f}, x={xc_h:.2f} (Expected 49.5, 49.5)")

    # Test parameter parsing
    p_conf = {
        "x0": {"guess": "image_center_x", "prior_type": "uniform", "prior_args": [90.0, 110.0], "fit": True, "label": "x_cen (pix)"},
        "inc": {"guess": 10.0, "prior_type": "gaussian", "prior_args": [10.0, 2.0], "fit": True, "label": "Inc (deg)"},
    }
    name_x, g_x, pt_x, pa_x, l_x, fit_x = parse_parameter_config(p_conf["x0"], image_shape=(200,200))
    name_x = "x0" # Manually set as it's not in sub-dict
    print(f"Parsed x0: name={name_x}, guess={g_x}, prior={pt_x}({pa_x}), label='{l_x}', fit={fit_x}")

    name_i, g_i, pt_i, pa_i, l_i, fit_i = parse_parameter_config(p_conf["inc"])
    name_i = "inc" # Manually set
    print(f"Parsed inc: name={name_i}, guess={g_i}, prior={pt_i}({pa_i}), label='{l_i}', fit={fit_i}")