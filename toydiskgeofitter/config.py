# toydiskgeofitter_project/toydiskgeofitter/config.py
import toml
from astropy.io import fits
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

# Setup logger for this module
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a TOML configuration file.

    Args:
        config_path (str): Path to the TOML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        toml.TomlDecodeError: If the TOML file is not valid.
    """
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except toml.TomlDecodeError as e:
        logger.error(f"Error decoding TOML file {config_path}: {e}")
        raise

def generate_default_config_string() -> str:
    """
    Generates a string containing a default TOML configuration.

    Returns:
        str: A string representation of the default TOML configuration.
    """
    default_config = {
        "observations": {
            "image_paths": ["path/to/your/image1.fits", "path/to/your/image2.fits"], # List of FITS files
            "hdu_indices": [0, 0], # HDU index for each image
            "data_labels": ["Image1", "Image2"], # Labels for plotting
            "pixel_scale_arcsec": None, # arcsec/pixel, if None, try to read from FITS
            "distance_pc": None, # pc, for AU conversion
            "wavelength_um": None, # microns, for metadata
        },
        "fitting_parameters": {
            "x0": {"guess": 100.0, "prior_type": "uniform", "prior_args": [90.0, 110.0], "fit": True, "label": "x_cen (pix)"},
            "y0": {"guess": 100.0, "prior_type": "uniform", "prior_args": [90.0, 110.0], "fit": True, "label": "y_cen (pix)"},
            "inc_deg": {"guess": 10.0, "prior_type": "uniform", "prior_args": [0.0, 20.0], "fit": True, "label": "Inclination (deg)"},
            "pa_deg": {"guess": 135.0, "prior_type": "uniform", "prior_args": [90.0, 270.0], "fit": True, "label": "PA (deg)"},
            # Add other parameters you might want to fit, e.g., flux_scale for multi-image
        },
        "fitting_regions": [ # Allows defining multiple rings or regions
            {
                "name": "primary_disk",
                "r_min_pixels": 3.0,
                "r_max_pixels": 50.0,
                "annulus_width_pixels": 1.0,
                "brightness_floor_percentile": 3.0 # Percentile for noise floor estimation
            }
            # You could add another dictionary here for a second ring
        ],
        "mcmc_settings": {
            "sampler": "emcee", # 'emcee' or potentially others in future
            "nwalkers": 64,
            "nsteps": 10000,
            "nburn": 1000,
            "thin_by": 15,
            "threads": -1, # -1 for auto (all cores), 1 for no multiprocessing
            "initial_ball_scale": 1e-3, # For initializing walkers around guess
            "save_sampler": True,
            "sampler_output_path": "sampler_chain.pkl",
            "continue_from_sampler": None, # Path to a .pkl sampler to continue
        },
        "output_settings": {
            "results_directory": "fit_results/",
            "plot_chains": True,
            "plot_corner": True,
            "plot_radial_profile": True,
        },
        "profile_calculation": {
            "binning_mode": "log", # 'linear' or 'log'
            "num_log_bins": 50,
            "min_radius_px_log": 1.0, # For log binning
            "linear_bin_size_px": 3.0, # For linear binning
            "outlier_clip_fraction": 0.03, # Total fraction (0.05 means +/- 2.5 percentile)
            "use_elliptical_bins": True, # Based on best-fit geometry for profile generation
            "plot_sb_r_squared": True, # Plot SB * R^2 in the final profile plot
            "statistic": "mean", # 'mean', 'median', or 'sum' for profile calculation
            "min_pixels_per_bin": 5 # Minimum pixels in a bin to compute stats
        },
        "plotting_settings": { # New section for general plot controls
            "output_format": "png", # e.g., png, pdf, svg
            "dpi": 150,
            # Individual plot file names can be here, or constructed in the main script
            "chains_plot_filename": "mcmc_chains.{output_format}",
            "corner_plot_filename": "mcmc_corner.{output_format}",
            "profile_plot_filename": "radial_profile.{output_format}",
            "image_bunit_default": "Image Units" # Fallback BUNIT if not in FITS
        }
    }
    return toml.dumps(default_config)

def save_default_config(output_path: str = "config_template.toml") -> None:
    """
    Saves a template TOML configuration file.

    Args:
        output_path (str, optional): Path to save the template.
            Defaults to "config_template.toml".
    """
    config_str = generate_default_config_string()
    try:
        with open(output_path, 'w') as f:
            f.write(config_str)
        logger.info(f"Default configuration template saved to {output_path}")
    except IOError as e:
        logger.error(f"Could not write default config to {output_path}: {e}")
        raise

def update_config_from_fits_header(
    config: Dict[str, Any],
    header: fits.Header,
    image_shape: Tuple[int, int],
    image_index: int = 0
) -> Dict[str, Any]:
    """
    Updates configuration with information from a FITS header.
    This is a basic example; FITS headers can be very diverse.

    Args:
        config (Dict[str, Any]): The current configuration dictionary.
        header (fits.Header): The FITS header object.
        image_shape (Tuple[int, int]): The (ny, nx) shape of the image.
        image_index (int): The index of the image in the config (for multi-image setups)

    Returns:
        Dict[str, Any]: The updated configuration dictionary.
    """
    obs_config = config.get("observations", {})

    # Pixel scale (very basic example, robust WCS handling is better)
    if obs_config.get("pixel_scale_arcsec") is None:
        for key in ['CDELT2', 'CD2_2']: # Common keywords for pixel scale in Y
            if key in header:
                try:
                    # Assuming square pixels, take absolute value, convert from deg to arcsec
                    pixel_scale_deg = abs(header[key])
                    obs_config["pixel_scale_arcsec"] = pixel_scale_deg * 3600.0
                    logger.info(f"Pixel scale from header ({key}): {obs_config['pixel_scale_arcsec']:.4f} arcsec/pix")
                    break
                except (TypeError, ValueError):
                    logger.warning(f"Could not parse FITS keyword {key} for pixel scale.")
        if obs_config.get("pixel_scale_arcsec") is None:
             logger.warning("Pixel scale not found in FITS header or config. It may be needed for physical units.")

    # Wavelength (example)
    if obs_config.get("wavelength_um") is None:
        for key in ['WAVELNTH', 'RESTWAV', 'LAMBDA']: # Example keywords
            if key in header and isinstance(header[key], (int, float)):
                # Assuming header stores it in microns, or requires conversion
                obs_config["wavelength_um"] = float(header[key]) # Add unit conversion if necessary
                logger.info(f"Wavelength from header ({key}): {obs_config['wavelength_um']:.2f} um")
                break

    # Update image center guesses if they look like placeholders and not set
    fit_params_config = config.get("fitting_parameters", {})
    ny, nx = image_shape
    if "x0" in fit_params_config and fit_params_config["x0"].get("guess") == 100.0: # Default from template
        if nx > 0:
            fit_params_config["x0"]["guess"] = nx / 2.0 - 0.5
            fit_params_config["x0"]["prior_args"] = [nx / 2.0 - 0.5 - 10, nx / 2.0 - 0.5 + 10] # Example update
            logger.info(f"Updated x0 guess to image center: {fit_params_config['x0']['guess']:.2f}")
    if "y0" in fit_params_config and fit_params_config["y0"].get("guess") == 100.0: # Default from template
        if ny > 0:
            fit_params_config["y0"]["guess"] = ny / 2.0 - 0.5
            fit_params_config["y0"]["prior_args"] = [ny / 2.0 - 0.5 - 10, ny / 2.0 - 0.5 + 10] # Example update
            logger.info(f"Updated y0 guess to image center: {fit_params_config['y0']['guess']:.2f}")

    config["observations"] = obs_config
    config["fitting_parameters"] = fit_params_config
    return config

def get_parameter_attributes(config: Dict[str, Any], param_name: str) -> Dict[str, Any]:
    """
    Retrieves all attributes for a given fitting parameter.

    Args:
        config (Dict[str, Any]): The main configuration dictionary.
        param_name (str): The name of the parameter (e.g., "x0", "inc_deg").

    Returns:
        Dict[str, Any]: Attributes for the parameter (guess, prior_type, etc.).
                        Returns an empty dict if parameter not found.
    """
    params_config = config.get("fitting_parameters", {})
    if param_name not in params_config:
        logger.warning(f"Parameter '{param_name}' not found in configuration.")
        return {}
    return params_config[param_name]

# Example usage (typically called from a main script or fitting function)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate and save a template
    save_default_config("my_disk_config_template.toml")
    print("\nSaved a template config to 'my_disk_config_template.toml'")

    # Example of loading and updating (assuming a dummy FITS file for header)
    try:
        # Create a dummy FITS header for testing
        hdr = fits.Header()
        hdr['CDELT2'] = -0.00002777 # ~0.1 arcsec/pixel in degrees
        hdr['NAXIS1'] = 256
        hdr['NAXIS2'] = 256
        
        # Simulate loading a config file (e.g., the one just saved)
        loaded_cfg = load_config("my_disk_config_template.toml") # You'd use your actual config
        print(f"\nInitial x0 guess: {loaded_cfg['fitting_parameters']['x0']['guess']}")
        
        updated_cfg = update_config_from_fits_header(loaded_cfg, hdr, (256,256))
        print(f"Updated x0 guess: {updated_cfg['fitting_parameters']['x0']['guess']:.2f}")
        print(f"Pixel scale from FITS: {updated_cfg['observations']['pixel_scale_arcsec']:.4f} arcsec/pix")

        x0_attrs = get_parameter_attributes(updated_cfg, "x0")
        print(f"\nAttributes for x0: {x0_attrs}")
        inc_attrs = get_parameter_attributes(updated_cfg, "inc_deg")
        print(f"Attributes for inc_deg: {inc_attrs}")

    except Exception as e:
        print(f"An error occurred in example usage: {e}")

#print("config.py loaded")