# toydiskgeofitter_project/toydiskgeofitter/likelihoods.py
import logging
import numpy as np
from scipy.special import gammaln
from typing import Tuple, List, Dict, Any, Callable, Optional

from .geometry import calculate_elliptical_radius_squared # Relative import

logger = logging.getLogger(__name__)

def _log_chi_square_pdf(chi2_value: float, dof: float) -> float:
    """
    Calculates the log of the Chi-Square PDF.
    Helper for log_likelihood_annuli_variance.

    Args:
        chi2_value (float): The observed Chi-Square statistic. Must be positive.
        dof (float): The total degrees of freedom. Must be positive.

    Returns:
        float: The log-likelihood value. Returns -np.inf if inputs are invalid.
    """
    if chi2_value <= 0 or dof <= 0:
        return -np.inf
    
    term1 = -(dof / 2.0) * np.log(2)
    term2 = -gammaln(dof / 2.0)
    term3 = (dof / 2.0 - 1.0) * np.log(chi2_value)
    term4 = -chi2_value / 2.0
    log_pdf = term1 + term2 + term3 + term4
    
    if not np.isfinite(log_pdf):
        return -np.inf
    return log_pdf


def log_likelihood_annuli_variance(
    theta_geom_tuple: Tuple[float, ...], # Changed name to avoid conflict if used elsewhere
    param_names: List[str], 
    image_values_1d: np.ndarray,
    coords_yx_1d: Tuple[np.ndarray, np.ndarray],
    annuli_edges_r_sq: np.ndarray,
    brightness_floor: float,
    fixed_params: Dict[str, float] = None # Made fixed_params truly optional with default
) -> float:
    """
    Calculates log likelihood based on variance within elliptical annuli.

    Args:
        theta_geom_tuple (Tuple[float, ...]): Geometric parameters to fit.
        param_names (List[str]): Names of parameters in theta_geom_tuple.
        image_values_1d (np.ndarray): 1D array of pixel values from the masked region.
        coords_yx_1d (Tuple[np.ndarray, np.ndarray]): (y_coords_1d, x_coords_1d) for image_values_1d.
        annuli_edges_r_sq (np.ndarray): Array of SQUARED semi-major axis defining annulus edges.
        brightness_floor (float): Brightness floor value.
        fixed_params (Dict[str, float], optional): Dictionary of fixed geometric parameters.

    Returns:
        float: Log likelihood value. Returns -np.inf for invalid parameters.
    """
    current_params = dict(zip(param_names, theta_geom_tuple))
    if fixed_params: # Ensure fixed_params is not None before updating
        current_params.update(fixed_params)

    try:
        x0 = current_params['x0']
        y0 = current_params['y0']
        inc_deg = current_params['inc_deg']
        pa_deg = current_params['pa_deg']
    except KeyError as e:
        logger.error(f"Missing geometric parameter for likelihood: {e}. Params available: {current_params.keys()}")
        return -np.inf

    inc_rad = np.radians(inc_deg)
    pa_rad = np.radians(pa_deg)

    dist_sq_ell_1d = calculate_elliptical_radius_squared(coords_yx_1d, (y0, x0), inc_rad, pa_rad)

    if dist_sq_ell_1d is None:
        return -np.inf

    total_chi2 = 0.0
    total_dof = 0

    for i in range(len(annuli_edges_r_sq) - 1):
        r_min_sq = annuli_edges_r_sq[i]
        r_max_sq = annuli_edges_r_sq[i+1]
        
        mask_annulus_1d = (dist_sq_ell_1d >= r_min_sq) & (dist_sq_ell_1d < r_max_sq)

        if np.any(mask_annulus_1d):
            pixels_in_annulus = image_values_1d[mask_annulus_1d]
            valid_pixels = pixels_in_annulus[np.isfinite(pixels_in_annulus)]
            
            N_k = len(valid_pixels)
            if N_k <= 1:
                continue

            df_k = N_k - 1 
            mean_brightness_k = np.max([np.mean(valid_pixels), 0.0]) 
            sample_variance_k = np.var(valid_pixels, ddof=1) # Unbiased sample variance S^2
            
            expected_variance_k = mean_brightness_k + brightness_floor
            if expected_variance_k <= 1e-9:
                continue 
            
            chi2_k = df_k * sample_variance_k / expected_variance_k
            
            total_chi2 += chi2_k
            total_dof += df_k

    if total_dof <= 0:
        return -np.inf

    log_L = _log_chi_square_pdf(total_chi2, total_dof)
    return log_L


def log_prior_uniform(value: float, min_val: float, max_val: float) -> float:
    if min_val <= value <= max_val:
        return 0.0
    return -np.inf

def log_prior_gaussian(value: float, mean: float, sigma: float) -> float:
    if sigma <= 0:
        logger.error("Sigma in Gaussian prior must be positive.")
        return -np.inf
    return -0.5 * ((value - mean) / sigma)**2 - np.log(sigma * np.sqrt(2 * np.pi))


class LogPriorEvaluator:
    def __init__(self, param_names: List[str],
                 param_configs: Dict[str, Dict[str, Any]],
                 image_ny_nx: Optional[Tuple[int, int]] = None):
        self.param_names = param_names
        self.param_configs = param_configs
        self.image_ny_nx = image_ny_nx
        
        # Pre-validate param_configs structure for faster __call__
        for name in self.param_names:
            if name not in self.param_configs:
                raise ValueError(f"Configuration for parameter '{name}' missing in param_configs for LogPriorEvaluator.")
            p_config = self.param_configs[name]
            prior_type = p_config.get("prior_type", "uniform").lower()
            prior_args = p_config.get("prior_args", [])
            if prior_type == "uniform" and len(prior_args) != 2:
                raise ValueError(f"Uniform prior for {name} needs 2 arguments (min, max), got {prior_args}.")
            elif prior_type == "gaussian" and len(prior_args) != 2:
                raise ValueError(f"Gaussian prior for {name} needs 2 arguments (mean, sigma), got {prior_args}.")
            elif prior_type not in ["uniform", "gaussian"]:
                raise ValueError(f"Unknown prior type '{prior_type}' for parameter {name}.")


    def __call__(self, theta: Tuple[float, ...]) -> float:
        current_lp = 0.0
        if len(theta) != len(self.param_names):
            # This should ideally not happen if emcee is set up correctly
            logger.critical(f"Theta length ({len(theta)}) mismatch with param_names length ({len(self.param_names)}) in LogPriorEvaluator.")
            return -np.inf
            
        params = dict(zip(self.param_names, theta))

        for name, value in params.items():
            # Assumes pre-validation in __init__ has ensured p_config exists and is valid
            p_config = self.param_configs[name] 
            prior_type = p_config.get("prior_type", "uniform").lower() # Default again just in case
            prior_args = p_config.get("prior_args", [])

            lp_single = -np.inf 
            if prior_type == "uniform":
                lp_single = log_prior_uniform(value, prior_args[0], prior_args[1])
            elif prior_type == "gaussian":
                lp_single = log_prior_gaussian(value, prior_args[0], prior_args[1])
            # No else needed due to pre-validation, but good for safety if pre-val is removed
            # else: 
            #     return -np.inf # Should have been caught by __init__

            current_lp += lp_single
            if not np.isfinite(current_lp):
                return -np.inf

        if self.image_ny_nx:
            ny, nx = self.image_ny_nx
            if 'x0' in params and not (0 <= params['x0'] < nx): # x0 should be < nx, not <= nx-1 for pixel coords
                return -np.inf
            if 'y0' in params and not (0 <= params['y0'] < ny): # y0 should be < ny
                return -np.inf
                
        return current_lp

def build_log_prior_object(
    param_names: List[str],
    param_configs: Dict[str, Dict[str, Any]],
    image_ny_nx: Optional[Tuple[int, int]] = None
) -> LogPriorEvaluator:
    """
    Factory function to create a LogPriorEvaluator instance.
    """
    return LogPriorEvaluator(param_names, param_configs, image_ny_nx)


class LogProbabilityWrapper:
    def __init__(self, log_prior_evaluator: LogPriorEvaluator, # Changed to accept the evaluator instance
                 log_likelihood_func: Callable[..., float],
                 likelihood_static_args: Tuple):
        self.log_prior_evaluator = log_prior_evaluator
        self.log_likelihood_func = log_likelihood_func
        self.likelihood_static_args = likelihood_static_args

    def __call__(self, theta: Tuple[float, ...]) -> float:
        lp = self.log_prior_evaluator(theta) # Call the prior evaluator instance
        if not np.isfinite(lp):
            return -np.inf

        # The first argument to log_likelihood_func is theta (the parameters being varied)
        # The rest are static arguments.
        ll = self.log_likelihood_func(theta, *self.likelihood_static_args)
        
        if not np.isfinite(ll):
            return -np.inf
            
        return lp + ll

def make_log_probability_object(
    log_prior_evaluator: LogPriorEvaluator, # Changed to accept the evaluator instance
    log_likelihood_func: Callable[..., float], 
    likelihood_static_args: Tuple
) -> LogProbabilityWrapper:
    """
    Creates a picklable LogProbabilityWrapper object for MCMC samplers.
    """
    return LogProbabilityWrapper(log_prior_evaluator, log_likelihood_func, likelihood_static_args)


# Example Usage (conceptual, for direct script testing if needed)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Likelihoods module example usage:")

    # Dummy data for log_likelihood_annuli_variance
    dummy_theta_tuple = (50.0, 50.0, 10.0, 135.0) 
    dummy_param_names_list = ['x0', 'y0', 'inc_deg', 'pa_deg']
    
    img_size_ex = 101
    y_coords_1d_ex, x_coords_1d_ex = np.mgrid[0:img_size_ex, 0:img_size_ex].reshape(2, -1)
    coords_1d_ex = (y_coords_1d_ex, x_coords_1d_ex)
    
    r_sq_map_ex = (x_coords_1d_ex - 50)**2 + (y_coords_1d_ex - 50)**2
    image_1d_ex = np.exp(-r_sq_map_ex / (2 * 15**2)) * 100 + np.random.normal(0, 5, size=r_sq_map_ex.shape)
    
    ann_edges_r_ex = np.array([5, 10, 15, 20, 25, 30])
    ann_edges_r_sq_ex = ann_edges_r_ex**2
    b_floor_ex = 1.0
    fixed_params_ex = {} # Empty fixed params for this example

    ll_val_ex = log_likelihood_annuli_variance(
        dummy_theta_tuple, dummy_param_names_list, image_1d_ex, coords_1d_ex, 
        ann_edges_r_sq_ex, b_floor_ex, fixed_params_ex
    )
    print(f"Example log_likelihood: {ll_val_ex}")

    # Test priors
    p_configs_ex = {
        "x0": {"prior_type": "uniform", "prior_args": [0, 100]},
        "y0": {"prior_type": "uniform", "prior_args": [0, 100]},
        "inc_deg": {"prior_type": "gaussian", "prior_args": [10, 2]},
        "pa_deg": {"prior_type": "uniform", "prior_args": [0, 180]}
    }
    try:
        log_prior_obj_ex = build_log_prior_object(
            dummy_param_names_list, p_configs_ex, image_ny_nx=(img_size_ex, img_size_ex)
        )
        
        valid_theta_ex = (50, 50, 10, 90)
        invalid_theta_inc_ex = (50, 50, 30, 90) 
        invalid_theta_pos_ex = (150, 50, 10, 90)

        print(f"Log prior (valid_theta_ex): {log_prior_obj_ex(valid_theta_ex)}")
        print(f"Log prior (invalid_theta_inc_ex): {log_prior_obj_ex(invalid_theta_inc_ex)}")
        print(f"Log prior (invalid_theta_pos_ex): {log_prior_obj_ex(invalid_theta_pos_ex)}")

        # Test log_probability
        likelihood_static_args_ex = (
            dummy_param_names_list, image_1d_ex, coords_1d_ex, 
            ann_edges_r_sq_ex, b_floor_ex, fixed_params_ex
        )

        log_prob_obj_ex = make_log_probability_object(
            log_prior_obj_ex,
            log_likelihood_annuli_variance,
            likelihood_static_args_ex
        )
        
        print(f"Log probability (valid_theta_ex): {log_prob_obj_ex(valid_theta_ex)}")
        print(f"Log probability (invalid_theta_pos_ex): {log_prob_obj_ex(invalid_theta_pos_ex)}")

    except ValueError as e:
        print(f"Error during example setup (likely prior config validation): {e}")