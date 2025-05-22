# toydiskgeofitter_project/toydiskgeofitter/geometry.py
import numpy as np
import logging
from typing import Tuple, Optional, Union, List

logger = logging.getLogger(__name__)

def calculate_elliptical_radius_squared(
    coords_yx: Tuple[np.ndarray, np.ndarray],
    center_yx: Tuple[float, float],
    inc_rad: float,
    pa_rad: float
) -> Optional[np.ndarray]:
    """
    Calculates the squared 'elliptical radius' for each pixel.
    Points on the same ellipse defined by center, inc, pa will have the same
    elliptical radius, which corresponds to the semi-major axis length.

    Args:
        coords_yx (Tuple[np.ndarray, np.ndarray]): Tuple of (y_coords, x_coords) numpy arrays.
                                                   These can be 2D grids or 1D arrays.
        center_yx (Tuple[float, float]): (y0, x0) center coordinates.
        inc_rad (float): Inclination in radians (0=face-on, pi/2=edge-on).
        pa_rad (float): Position angle in radians (east of North, i.e.,
                        counter-clockwise from +y axis in a typical image array if North is up).
                        Or, counter-clockwise from +x if PA is defined astronomically from North through East.
                        The original script's PA seems to be counter-clockwise from +x axis.
                        Let's stick to counter-clockwise from +x for consistency with the original.

    Returns:
        Optional[np.ndarray]: Array of squared elliptical radii.
                              Returns None if inc_rad leads to cos_inc being too small (near edge-on).
    """
    y_coords, x_coords = coords_yx
    center_y, center_x = center_yx

    # Shift coordinates to be relative to the center
    dy = y_coords - center_y
    dx = x_coords - center_x

    # Rotate coordinates by -pa_rad (clockwise rotation) to align ellipse major axis with x'
    # This aligns the disk's projected major axis with the x'-axis of the rotated frame.
    cos_pa = np.cos(-pa_rad)
    sin_pa = np.sin(-pa_rad)
    x_rot = dx * cos_pa - dy * sin_pa
    y_rot = dx * sin_pa + dy * cos_pa

    # Deproject y' coordinate based on inclination
    cos_inc = np.cos(inc_rad)

    # Handle near edge-on cases to avoid division by zero or extremely large numbers
    if np.abs(cos_inc) < 1e-6: # cos(pi/2) is 0
        logger.debug("Near edge-on inclination (cos_inc ~ 0) encountered. Returning None for radii.")
        return None

    y_deproj = y_rot / cos_inc

    # Calculate squared elliptical radius (semi-major axis squared in the deprojected plane)
    # r_ell_sq = x_rot**2 + y_deproj**2
    radius_sq = x_rot**2 + y_deproj**2
    return radius_sq

def define_annuli_edges(
    r_min: float,
    r_max: float,
    num_annuli: Optional[int] = None,
    annulus_width: Optional[float] = None,
    log_spacing: bool = False
) -> np.ndarray:
    """
    Generates an array of annulus edge radii (not squared).

    Args:
        r_min (float): Minimum radius for the annuli.
        r_max (float): Maximum radius for the annuli.
        num_annuli (Optional[int]): Number of annuli. Used if annulus_width is None.
        annulus_width (Optional[float]): Width of each annulus. Used if num_annuli is None.
                                       If both are None, raises ValueError.
                                       If both are provided, num_annuli takes precedence.
        log_spacing (bool): If True and num_annuli is used, space annuli logarithmically.
                            Otherwise, linear spacing. Ignored if annulus_width is set.

    Returns:
        np.ndarray: Array of radii defining the edges of the annuli.
                    Length will be num_annuli + 1 or determined by width.

    Raises:
        ValueError: If inputs are invalid (e.g., r_min >= r_max, or insufficient info for binning).
    """
    if r_min >= r_max:
        raise ValueError(f"r_min ({r_min}) must be less than r_max ({r_max}).")
    if r_min < 0:
        raise ValueError(f"r_min ({r_min}) must be non-negative.")

    if num_annuli is not None:
        if num_annuli <= 0:
            raise ValueError("num_annuli must be positive if specified.")
        if log_spacing:
            if r_min <= 0: # logspace needs positive start
                raise ValueError("r_min must be > 0 for log_spacing.")
            edges = np.logspace(np.log10(r_min), np.log10(r_max), num_annuli + 1)
        else:
            edges = np.linspace(r_min, r_max, num_annuli + 1)
    elif annulus_width is not None:
        if annulus_width <= 0:
            raise ValueError("annulus_width must be positive if specified.")
        # Ensure the last edge includes or is close to r_max
        num_steps = int(np.ceil((r_max - r_min) / annulus_width))
        edges = np.arange(r_min, r_min + (num_steps + 1) * annulus_width, annulus_width)
        edges = edges[edges <= r_max + annulus_width/2.0] # Ensure we don't go too far
        if not np.isclose(edges[-1], r_max) and edges[-1] < r_max : # Add r_max if it's not the last edge
             if r_max > edges[-1] + annulus_width / 2.0: # only add if significantly larger
                edges = np.append(edges,r_max)
        # Refine to make sure the last point is exactly r_max if we overshot slightly due to arange,
        # or if r_max was not perfectly divisible.
        # A cleaner way for width-based:
        edges = np.arange(r_min, r_max + annulus_width, annulus_width)
        if edges[-1] > r_max : # if last edge overshoots r_max, make it r_max
             edges[-1] = r_max
        if edges[-1] < r_max and not np.isclose(edges[-1],r_max): # if last edge is short of r_max append r_max
             edges = np.append(edges,r_max)


    else:
        raise ValueError("Either num_annuli or annulus_width must be specified.")

    return edges

def get_radial_profile_bin_centers(radii_edges: np.ndarray, log_spacing: bool = False) -> np.ndarray:
    """
    Calculates bin centers from radii_edges.

    Args:
        radii_edges (np.ndarray): Array of radii defining annulus edges.
        log_spacing (bool): If True, use geometric mean for bin centers.
                            Otherwise, use arithmetic mean.

    Returns:
        np.ndarray: Array of bin center radii.
    """
    if len(radii_edges) < 2:
        raise ValueError("radii_edges must have at least two elements.")
    
    if log_spacing:
        if np.any(radii_edges <= 0):
            raise ValueError("All radii_edges must be > 0 for log_spacing geometric mean.")
        # Geometric mean: sqrt(r1 * r2)
        centers = np.sqrt(radii_edges[:-1] * radii_edges[1:])
    else:
        # Arithmetic mean: (r1 + r2) / 2
        centers = (radii_edges[:-1] + radii_edges[1:]) / 2.0
    return centers


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test calculate_elliptical_radius_squared
    y, x = np.ogrid[0:5, 0:5] # Simple 5x5 grid
    center = (2, 2)
    inc_face_on = np.radians(0)
    inc_inclined = np.radians(30)
    inc_edge_on = np.radians(90)
    pa = np.radians(45)

    r_sq_face_on = calculate_elliptical_radius_squared((y, x), center, inc_face_on, pa)
    print("R^2 (face-on, PA=45):\n", r_sq_face_on)

    r_sq_inclined = calculate_elliptical_radius_squared((y, x), center, inc_inclined, pa)
    print("\nR^2 (inc=30, PA=45):\n", r_sq_inclined)
    
    r_sq_edge_on = calculate_elliptical_radius_squared((y, x), center, inc_edge_on, pa)
    print("\nR^2 (edge-on, PA=45):\n", r_sq_edge_on) # Should be None or handled

    # Test define_annuli_edges
    edges_lin_num = define_annuli_edges(r_min=1.0, r_max=10.0, num_annuli=9)
    print("\nAnnuli Edges (linear, num_annuli=9):\n", edges_lin_num)
    print("Bin centers (linear):\n", get_radial_profile_bin_centers(edges_lin_num, log_spacing=False))


    edges_log_num = define_annuli_edges(r_min=1.0, r_max=100.0, num_annuli=5, log_spacing=True)
    print("\nAnnuli Edges (log, num_annuli=5):\n", edges_log_num)
    print("Bin centers (log):\n", get_radial_profile_bin_centers(edges_log_num, log_spacing=True))

    edges_lin_width = define_annuli_edges(r_min=0.5, r_max=10.2, annulus_width=2.0)
    print("\nAnnuli Edges (linear, width=2.0, r_max=10.2):\n", edges_lin_width)
    print("Bin centers (linear width):\n", get_radial_profile_bin_centers(edges_lin_width, log_spacing=False))

#print("geometry.py loaded")
