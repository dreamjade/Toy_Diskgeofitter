# toydiskgeofitter_project/toydiskgeofitter/profiling.py
import logging
import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any, List

from .geometry import calculate_elliptical_radius_squared, define_annuli_edges, get_radial_profile_bin_centers

logger = logging.getLogger(__name__)

def generate_profile_distance_map(
    image_shape: Tuple[int, int],
    geometry_params: Dict[str, float],
    profile_type: str = "elliptical"
) -> Optional[np.ndarray]:
    """
    Creates a 2D map of squared distances from the center, either circular or elliptical.

    Args:
        image_shape (Tuple[int, int]): (ny, nx) shape of the image.
        geometry_params (Dict[str, float]): Dictionary containing geometric parameters:
            'x0', 'y0' (center), and if profile_type is 'elliptical',
            also 'inc_deg' (inclination degrees), 'pa_deg' (position angle degrees).
        profile_type (str): "elliptical" or "circular".

    Returns:
        Optional[np.ndarray]: 2D array of squared distances (radii).
                              None if essential geometry_params are missing or invalid.
    """
    ny, nx = image_shape
    y_coords, x_coords = np.ogrid[0:ny, 0:nx]
    coords_yx = (y_coords, x_coords)

    try:
        center_y, center_x = geometry_params['y0'], geometry_params['x0']
    except KeyError:
        logger.error("Center coordinates (y0, x0) missing in geometry_params for profile distance map.")
        return None

    if profile_type == "elliptical":
        try:
            inc_deg = geometry_params['inc_deg']
            pa_deg = geometry_params['pa_deg']
            inc_rad = np.radians(inc_deg)
            pa_rad = np.radians(pa_deg) # Assuming PA system consistency
            
            dist_sq_map = calculate_elliptical_radius_squared(
                coords_yx, (center_y, center_x), inc_rad, pa_rad
            )
            if dist_sq_map is None:
                logger.warning("Elliptical distance calculation failed (e.g., edge-on). Cannot generate profile map.")
                return None
            return dist_sq_map
        except KeyError as e:
            logger.error(f"Missing parameter '{e}' for elliptical profile distance map.")
            return None
    elif profile_type == "circular":
        dist_sq_map = (x_coords - center_x)**2 + (y_coords - center_y)**2
        return dist_sq_map
    else:
        logger.error(f"Unknown profile_type: {profile_type}. Choose 'elliptical' or 'circular'.")
        return None


def calculate_radial_profile(
    image_data: np.ndarray,
    distance_map_sq: np.ndarray,
    radii_edges: np.ndarray, # Actual radii, not squared
    uncertainty_map: Optional[np.ndarray] = None,
    outlier_clip_fraction: float = 0.0,
    statistic: str = "mean", # 'mean', 'median', 'sum'
    min_pixels_per_bin: int = 3 # Minimum pixels to compute std/error robustly
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates a radial profile from image data based on a distance map and bin edges.

    Args:
        image_data (np.ndarray): 2D image data.
        distance_map_sq (np.ndarray): 2D map of SQUARED distances (circular or elliptical).
        radii_edges (np.ndarray): 1D array of radii defining annulus edges (NOT squared).
        uncertainty_map (Optional[np.ndarray]): 2D map of pixel uncertainties.
                                               If provided, weighted mean and error propagation used.
        outlier_clip_fraction (float): Total fraction of outliers to clip from both ends
                                       (e.g., 0.05 clips 2.5th and 97.5th percentiles).
                                       Set to 0 to disable.
        statistic (str): Statistic to calculate per bin: 'mean', 'median', 'sum'.
        min_pixels_per_bin (int): Minimum number of valid pixels in a bin to calculate statistics.

    Returns:
        Tuple containing:
            - bin_centers (np.ndarray): Radii at the center of each bin.
            - profile_stat (np.ndarray): Calculated statistic (mean, median, sum) per bin.
            - profile_std (np.ndarray): Standard deviation of pixels in each bin (NaN if not applicable).
            - profile_err (np.ndarray): Standard error on the mean/median (NaN if not applicable).
                                        For mean: std / sqrt(N_eff). For median: ~1.253 * std / sqrt(N).
            - N_pixels (np.ndarray): Number of valid pixels per bin.
    """
    if image_data.shape != distance_map_sq.shape:
        raise ValueError("image_data and distance_map_sq must have the same shape.")
    if uncertainty_map is not None and uncertainty_map.shape != image_data.shape:
        raise ValueError("uncertainty_map must have the same shape as image_data.")

    radii_edges_sq = radii_edges**2
    # Assuming log spacing if edges suggest it, for geometric mean of bin centers.
    # A better way is to pass binning_mode ('log' or 'linear') from config.
    # For now, let's infer or default to arithmetic mean.
    is_log_spaced = False # Default to linear for bin centers unless specified
    if len(radii_edges) > 2 and np.all(radii_edges > 0): # Basic check for log spacing possibility
        ratios = radii_edges[1:] / radii_edges[:-1]
        if np.allclose(ratios, ratios[0]):
            is_log_spaced = True # Heuristic
            
    bin_centers = get_radial_profile_bin_centers(radii_edges, log_spacing=is_log_spaced)
    
    num_bins = len(bin_centers)
    profile_stat_values = np.full(num_bins, np.nan)
    profile_std_values = np.full(num_bins, np.nan)
    profile_err_values = np.full(num_bins, np.nan)
    N_pixels_values = np.zeros(num_bins, dtype=int)

    for i in range(num_bins):
        r_min_sq = radii_edges_sq[i]
        r_max_sq = radii_edges_sq[i+1]
        
        mask_annulus = (distance_map_sq >= r_min_sq) & (distance_map_sq < r_max_sq)
        
        if not np.any(mask_annulus):
            continue

        pixels_in_annulus = image_data[mask_annulus]
        valid_pixel_indices = np.isfinite(pixels_in_annulus)
        valid_pixels = pixels_in_annulus[valid_pixel_indices]
        
        N_k = len(valid_pixels)
        N_pixels_values[i] = N_k

        if N_k < min_pixels_per_bin:
            continue

        current_weights = None
        if uncertainty_map is not None:
            pixel_errors_in_annulus = uncertainty_map[mask_annulus][valid_pixel_indices]
            if np.any(pixel_errors_in_annulus <= 0) or np.any(np.isnan(pixel_errors_in_annulus)):
                logger.warning(f"Bin {i}: Invalid uncertainties (<=0 or NaN). Using unweighted stats.")
            else:
                current_weights = 1.0 / pixel_errors_in_annulus**2
        
        # Outlier clipping
        if outlier_clip_fraction > 0 and N_k > 2 / outlier_clip_fraction: # Need enough points for clipping
            lower_p = (outlier_clip_fraction / 2.0) * 100
            upper_p = (1.0 - outlier_clip_fraction / 2.0) * 100
            try:
                with warnings.catch_warnings(): # Suppress RuntimeWarning for empty slice in percentile
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # Weighted percentile is complex, so clipping is on unweighted data for now
                    vmin, vmax = np.percentile(valid_pixels, [lower_p, upper_p])
                
                clip_mask = (valid_pixels >= vmin) & (valid_pixels <= vmax)
                clipped_pixels = valid_pixels[clip_mask]
                
                if len(clipped_pixels) < min_pixels_per_bin: # Not enough points after clipping
                    logger.debug(f"Bin {i}: Too few pixels after outlier clipping. Using unclipped data.")
                    # Keep original valid_pixels and current_weights
                else:
                    valid_pixels = clipped_pixels
                    N_k = len(valid_pixels)
                    N_pixels_values[i] = N_k # Update N_k
                    if current_weights is not None:
                        current_weights = current_weights[clip_mask] # Apply clip_mask to weights
            except IndexError: # Fallback if percentile fails (e.g., all identical values)
                logger.debug(f"Bin {i}: Percentile calculation failed for clipping. Using unclipped data.")
        
        # Calculate statistic
        if N_k >= min_pixels_per_bin:
            stat_val, std_val, err_val = np.nan, np.nan, np.nan
            if statistic == "mean":
                if current_weights is not None:
                    stat_val = np.average(valid_pixels, weights=current_weights)
                    # Variance of weighted mean: 1 / sum(weights). Std dev of data: sqrt(sum(w*(x-mean)^2) / sum(w))
                    variance = np.average((valid_pixels - stat_val)**2, weights=current_weights)
                    std_val = np.sqrt(variance)
                    err_val = np.sqrt(1.0 / np.sum(current_weights)) if np.sum(current_weights) > 0 else np.nan
                else:
                    stat_val = np.mean(valid_pixels)
                    std_val = np.std(valid_pixels)
                    err_val = std_val / np.sqrt(N_k) if N_k > 0 else np.nan
            elif statistic == "median":
                # Weighted median is non-trivial, using unweighted for now
                stat_val = np.median(valid_pixels)
                std_val = np.std(valid_pixels) # For reference
                # Error on median approx 1.253 * std / sqrt(N) for Gaussian data
                err_val = 1.253 * std_val / np.sqrt(N_k) if N_k > 0 else np.nan
            elif statistic == "sum":
                stat_val = np.sum(valid_pixels)
                std_val = np.nan # Std of sum not typically used directly this way
                err_val = np.sqrt(np.sum(uncertainty_map[mask_annulus][valid_pixel_indices]**2)) if uncertainty_map is not None else np.nan
            else:
                logger.warning(f"Unknown statistic '{statistic}' requested.")

            profile_stat_values[i] = stat_val
            profile_std_values[i] = std_val
            profile_err_values[i] = err_val

    return bin_centers, profile_stat_values, profile_std_values, profile_err_values, N_pixels_values


# Example Usage (conceptual)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Profiling module example usage:")

    # Create dummy image and geometry
    img = np.random.rand(101, 101) * 10 + \
          np.exp(-((np.ogrid[0:101,0:101][1]-50)**2 + (np.ogrid[0:101,0:101][0]-50)**2) / (2*10**2)) * 100
    img_shape = img.shape
    
    geom_params_circ = {'x0': 50, 'y0': 50}
    geom_params_ellip = {'x0': 50, 'y0': 50, 'inc_deg': 30, 'pa_deg': 45}

    dist_map_circ_sq = generate_profile_distance_map(img_shape, geom_params_circ, "circular")
    dist_map_ellip_sq = generate_profile_distance_map(img_shape, geom_params_ellip, "elliptical")

    if dist_map_circ_sq is not None and dist_map_ellip_sq is not None:
        logger.info(f"Circular distance map generated, center value: {np.sqrt(dist_map_circ_sq[50,50]):.2f}")
        logger.info(f"Elliptical distance map generated, center value: {np.sqrt(dist_map_ellip_sq[50,50]):.2f}")

        # Define annuli edges (actual radii)
        # from .geometry import define_annuli_edges (if run directly as script)
        r_edges = define_annuli_edges(r_min=1, r_max=45, num_annuli=15, log_spacing=False)
        # r_edges = np.linspace(1,45,16)
        
        print(f"Radii edges: {r_edges}")

        centers, means, stds, errs, Ns = calculate_radial_profile(
            img, dist_map_ellip_sq, r_edges, outlier_clip_fraction=0.05, min_pixels_per_bin=5
        )

        print("\nRadial Profile Results (first 5 bins):")
        print(f"{'Center':>8} {'Mean':>8} {'StdDev':>8} {'StdErr':>8} {'N_pix':>8}")
        for i in range(min(5, len(centers))):
            print(f"{centers[i]:8.2f} {means[i]:8.2f} {stds[i]:8.2f} {errs[i]:8.2f} {Ns[i]:8d}")
    else:
        logger.error("Failed to generate distance maps for profile example.")