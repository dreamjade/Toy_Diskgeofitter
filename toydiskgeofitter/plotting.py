# toydiskgeofitter_project/toydiskgeofitter/plotting.py
import logging
import numpy as np
import matplotlib.pyplot as plt
import corner # For corner plots
from typing import List, Dict, Any, Optional, Tuple, Union

from .utils import convert_radius_units # For unit conversion in profile plots

logger = logging.getLogger(__name__)

def plot_mcmc_chains(
    samples_chain: np.ndarray,
    param_labels: List[str],
    output_file: Optional[str] = None,
    title: Optional[str] = "MCMC Parameter Chains",
    nwalkers_to_show: int = 0, # 0 for all
    figsize: Tuple[float, float] = (12, 8),
    alpha_walker: float = 0.3
) -> None:
    """
    Plots the MCMC chains for each parameter.

    Args:
        samples_chain (np.ndarray): The MCMC chain array from sampler.get_chain(flat=False).
                                    Shape (nsteps, nwalkers, ndim).
        param_labels (List[str]): List of labels for the parameters.
        output_file (Optional[str]): Path to save the plot. If None, shows plot.
        title (Optional[str]): Title for the plot.
        nwalkers_to_show (int): Number of walkers to display. 0 for all.
        figsize (Tuple[float, float]): Figure size.
        alpha_walker (float): Alpha transparency for walker lines.
    """
    if samples_chain is None or samples_chain.ndim != 3:
        logger.error("Invalid samples_chain provided for plotting (must be 3D: steps, walkers, dims).")
        return
        
    nsteps, nwalkers_total, ndim = samples_chain.shape

    if len(param_labels) != ndim:
        logger.error(f"Number of param_labels ({len(param_labels)}) "
                     f"does not match ndim ({ndim}) from samples_chain.")
        # Fallback: use generic labels
        param_labels = [f"Param {i+1}" for i in range(ndim)]

    fig, axes = plt.subplots(ndim, figsize=figsize, sharex=True)
    if ndim == 1: # Ensure axes is always a list-like object
        axes = [axes]

    walkers_idx = slice(None) # Show all walkers
    if nwalkers_to_show > 0 and nwalkers_to_show < nwalkers_total:
        # Randomly select walkers if subsampling, or just take the first few
        walkers_idx = np.random.choice(nwalkers_total, nwalkers_to_show, replace=False)
        # walkers_idx = slice(0, nwalkers_to_show)


    for i in range(ndim):
        ax = axes[i]
        # Plot samples for each selected walker over steps
        ax.plot(samples_chain[:, walkers_idx, i], "k", alpha=alpha_walker)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(param_labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5) # Adjust label position

    axes[-1].set_xlabel("Step number")
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95 if title else 0.98]) # Adjust for suptitle

    if output_file:
        try:
            plt.savefig(output_file)
            logger.info(f"MCMC chains plot saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save MCMC chains plot to {output_file}: {e}")
        plt.close(fig)
    else:
        plt.show()


def plot_corner_mcmc(
    flat_samples: np.ndarray,
    param_labels: List[str],
    truths: Optional[List[Optional[float]]] = None, # Can have Nones for params without truth
    output_file: Optional[str] = None,
    title: Optional[str] = "MCMC Posterior Distributions",
    show_titles: bool = True,
    quantiles: Optional[List[float]] = (0.16, 0.5, 0.84),
    **corner_kwargs: Any
) -> None:
    """
    Generates a corner plot of MCMC posterior distributions.

    Args:
        flat_samples (np.ndarray): Flattened MCMC samples (n_samples, ndim).
        param_labels (List[str]): List of labels for the parameters.
        truths (Optional[List[Optional[float]]]): True values to mark on the plot.
                                       Length must match ndim. Use None for no truth line for a param.
        output_file (Optional[str]): Path to save the plot. If None, shows plot.
        title (Optional[str]): Title for the plot (suptitle).
        show_titles (bool): Whether to show titles on 1D histograms.
        quantiles (Optional[List[float]]): Quantiles to mark on 1D histograms.
        **corner_kwargs: Additional keyword arguments for corner.corner().
    """
    if flat_samples is None or flat_samples.ndim != 2:
        logger.error("Invalid flat_samples provided for corner plot (must be 2D: samples, dims).")
        return

    ndim = flat_samples.shape[1]
    if len(param_labels) != ndim:
        logger.error(f"Number of param_labels ({len(param_labels)}) "
                     f"does not match ndim ({ndim}) from flat_samples.")
        param_labels = [f"Param {i+1}" for i in range(ndim)] # Fallback

    if truths is not None and len(truths) != ndim:
        logger.warning(f"Length of truths ({len(truths)}) does not match ndim ({ndim}). Ignoring truths.")
        truths = None

    # Default corner kwargs can be set here
    default_kwargs = {
        "labels": param_labels,
        "truths": truths,
        "quantiles": quantiles,
        "show_titles": show_titles,
        "title_kwargs": {"fontsize": 10},
        "label_kwargs": {"fontsize": 10},
        "truth_color": "red",
    }
    # User-provided kwargs will override defaults
    combined_kwargs = {**default_kwargs, **corner_kwargs}

    try:
        fig = corner.corner(flat_samples, **combined_kwargs)
        
        if title:
            fig.suptitle(title, fontsize=14, y=1.0) # Adjust y to avoid overlap

        if output_file:
            try:
                plt.savefig(output_file)
                logger.info(f"Corner plot saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save corner plot to {output_file}: {e}")
            plt.close(fig)
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error generating corner plot: {e}")
        if 'fig' in locals() and fig is not None: plt.close(fig) # Ensure figure is closed on error


def plot_radial_profiles(
    profile_data_list: List[Dict[str, Any]],
    plot_config: Dict[str, Any], # Contains settings like x_scale, y_scale, labels
    obs_config: Optional[Dict[str, Any]] = None, # For pixel_scale, distance for unit conv.
    fit_geometry_params: Optional[Dict[str, float]] = None, # Best-fit geometry for textbox
    output_file: Optional[str] = None,
    title_prefix: str = "Radial Profile",
    figsize: Tuple[float, float] = (9, 6)
) -> None:
    """
    Plots one or more radial profiles with extensive customization.
    Adapted from the original script's plotting logic.

    Args:
        profile_data_list (List[Dict[str, Any]]): List of dictionaries, each containing
            profile data: 'radii' (np.ndarray), 'profile_stat' (np.ndarray),
            'profile_err' (np.ndarray), 'label' (str), 'color' (str),
            'linestyle' (str), 'fill_color' (str).
        plot_config (Dict[str, Any]): Configuration for plotting, including:
            'x_scale' ('linear' or 'log'), 'y_scale' ('linear' or 'log'),
            'radius_label_base' (e.g., "Radius" or "Semi-Major Axis"),
            'plot_sb_r_squared' (bool, if True, y-axis is Stat * R^2),
            'image_filter_name' (str, for textbox, e.g., from FITS header),
            'outlier_clip_fraction' (float, for textbox).
        obs_config (Optional[Dict[str, Any]]): Observation configuration for unit conversions.
            Needs 'pixel_scale_arcsec' and optionally 'distance_pc'.
        fit_geometry_params (Optional[Dict[str, float]]): Best-fit geometry for info text box.
            E.g., {'x0': ..., 'y0': ..., 'inc_deg': ..., 'pa_deg': ...}
        output_file (Optional[str]): Path to save the plot. If None, shows plot.
        title_prefix (str): Prefix for the plot title.
        figsize (Tuple[float, float]): Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    pixel_scale_arcsec = obs_config.get("pixel_scale_arcsec") if obs_config else None
    distance_pc = obs_config.get("distance_pc") if obs_config else None
    
    y_label_base = "Surface Brightness" # Default
    image_bunit = plot_config.get("image_bunit", "Image Units") # Get from FITS header if possible

    all_y_min, all_y_max = [], []
    max_r_plot_converted = 0.0
    final_x_unit_str = "pixels"

    for i, pdata in enumerate(profile_data_list):
        radii_pix = pdata['radii']
        profile_stat = pdata['profile_stat']
        profile_err = pdata.get('profile_err', np.zeros_like(profile_stat)) # Default error to 0 if not provided

        # Unit conversion for this profile's radii
        radii_converted, current_x_unit_str = convert_radius_units(
            radii_pix, pixel_scale_arcsec, distance_pc
        )
        if i == 0: final_x_unit_str = current_x_unit_str # Use units from first profile for axis label

        valid_mask = np.isfinite(radii_converted) & np.isfinite(profile_stat) & np.isfinite(profile_err) & (radii_converted > 0)
        
        if not np.any(valid_mask):
            logger.warning(f"No valid data points to plot for profile: {pdata.get('label', f'Profile {i}')}")
            continue

        r_plot = radii_converted[valid_mask]
        stat_plot = profile_stat[valid_mask]
        err_plot = profile_err[valid_mask]

        y_values_to_plot = stat_plot
        y_errors_to_plot = err_plot

        if plot_config.get("plot_sb_r_squared", False):
            # R should be in the *same units* as the x-axis for this multiplication
            y_values_to_plot = stat_plot * r_plot**2
            y_errors_to_plot = err_plot * r_plot**2
            # Update Y label base (unit part comes later)
            y_label_base = f"Avg. SB * {plot_config.get('radius_label_base', 'Radius')}²"
        
        ax.plot(r_plot, y_values_to_plot,
                label=pdata.get('label', f'Profile {i}'),
                color=pdata.get('color', 'blue'),
                linestyle=pdata.get('linestyle', '-'),
                linewidth=pdata.get('linewidth', 1.5),
                alpha=pdata.get('alpha', 0.9),
                zorder=10)

        ax.fill_between(r_plot,
                        y_values_to_plot - y_errors_to_plot,
                        y_values_to_plot + y_errors_to_plot,
                        color=pdata.get('fill_color', 'lightblue'),
                        alpha=pdata.get('fill_alpha', 0.3),
                        label='_nolegend_', zorder=5)
        
        if len(y_values_to_plot) > 0 :
            all_y_min.append(np.nanmin(y_values_to_plot - y_errors_to_plot))
            all_y_max.append(np.nanmax(y_values_to_plot + y_errors_to_plot))
            max_r_plot_converted = max(max_r_plot_converted, np.nanmax(r_plot))

    # --- Plot Styling ---
    ax.set_xlabel(f"{plot_config.get('radius_label_base', 'Radius')} ({final_x_unit_str})", fontsize=12)
    
    y_axis_unit_suffix = f"{final_x_unit_str}²" if plot_config.get("plot_sb_r_squared", False) else ""
    if y_axis_unit_suffix:
        y_label_full = f"{y_label_base} [{image_bunit} * {y_axis_unit_suffix}]"
    else:
        y_label_full = f"{y_label_base} [{image_bunit}]"
    ax.set_ylabel(y_label_full, fontsize=12)
    
    bin_desc = plot_config.get('binning_mode_desc', 'Binned') # e.g. "Log Elliptical Bins"
    plot_title = f"{title_prefix} - {bin_desc}"
    ax.set_title(plot_title, fontsize=14)

    # Text box with fit info
    textstr_lines = []
    if plot_config.get('image_filter_name'):
        textstr_lines.append(f"Filter: {plot_config['image_filter_name']}")
    if fit_geometry_params:
        fit_text = (f"Fit Geom: C=({fit_geometry_params.get('y0', np.nan):.1f},"
                    f"{fit_geometry_params.get('x0', np.nan):.1f}), "
                    f"Inc={fit_geometry_params.get('inc_deg', np.nan):.1f}°, "
                    f"PA={fit_geometry_params.get('pa_deg', np.nan):.1f}°")
        textstr_lines.append(fit_text)
    if pixel_scale_arcsec:
        scale_text = f"{pixel_scale_arcsec:.3f} arcsec/pix"
        if distance_pc:
            scale_text += f" ({pixel_scale_arcsec * distance_pc:.2f} AU/pix at {distance_pc} pc)"
        textstr_lines.append(f"Pixel scale: {scale_text}")
    if 'outlier_clip_fraction' in plot_config:
        clip_val = plot_config['outlier_clip_fraction'] * 100
        clip_text = f"Clip: {clip_val:.1f}%" if clip_val > 0 else "Clip: None"
        textstr_lines.append(clip_text)
    
    if textstr_lines:
        textstr = "\n".join(textstr_lines)
        props = dict(boxstyle='round', facecolor='white', alpha=0.65)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    # Set scales
    ax.set_xscale(plot_config.get('x_scale', 'linear'))
    ax.set_yscale(plot_config.get('y_scale', 'linear'))

    if all_y_min and all_y_max:
        plot_min_y = np.nanmin(all_y_min)
        plot_max_y = np.nanmax(all_y_max)
        if plot_config.get('y_scale') == 'log':
            plot_min_y = max(plot_min_y, plot_max_y * 1e-5, 1e-9) # Avoid zero or negative for log
        padding = (plot_max_y - plot_min_y) * 0.05 if (plot_max_y - plot_min_y) > 0 else abs(plot_max_y * 0.1)
        if plot_min_y < plot_max_y:
            ax.set_ylim(bottom=plot_min_y - padding, top=plot_max_y + padding)
        elif plot_config.get('y_scale') == 'log': ax.set_ylim(bottom=1e-3 * plot_max_y, top=1e1*plot_max_y) #fallback for log
        else: ax.set_ylim() # Auto

    min_plot_radius = np.inf
    for pdata in profile_data_list:
        radii_conv, _ = convert_radius_units(pdata['radii'], pixel_scale_arcsec, distance_pc)
        valid_r = radii_conv[np.isfinite(radii_conv) & (radii_conv > 0)]
        if len(valid_r)>0 : min_plot_radius = min(min_plot_radius, np.min(valid_r))
    
    if max_r_plot_converted > 0 and min_plot_radius < np.inf:
        if ax.get_xscale() == 'log':
            ax.set_xlim(left=min_plot_radius * 0.9, right=max_r_plot_converted * 1.1)
        else:
            ax.set_xlim(left=0 if plot_config.get('radius_label_base', 'Radius') != "Semi-Major Axis" else min_plot_radius*0.9,
                        right=max_r_plot_converted * 1.05) # Start from 0 for circular radii unless log
    else: ax.set_xlim() # Auto

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,4)) # Scientific notation for y-axis
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for title/textbox

    if output_file:
        try:
            plt.savefig(output_file)
            logger.info(f"Radial profile plot saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save radial profile plot to {output_file}: {e}")
        plt.close(fig)
    else:
        plt.show()


# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Plotting module example usage:")

    # --- Test MCMC Chains Plot ---
    nsteps_ex, nwalkers_ex, ndim_ex = 200, 30, 3
    dummy_chain = np.cumsum(np.random.randn(nsteps_ex, nwalkers_ex, ndim_ex), axis=0)
    param_labels_ex = ["X_cen", "Y_cen", "Inclination"]
    # plot_mcmc_chains(dummy_chain, param_labels_ex, title="Test MCMC Chains", nwalkers_to_show=10)
    # To save: plot_mcmc_chains(dummy_chain, param_labels_ex, output_file="test_chains.png")

    # --- Test Corner Plot ---
    dummy_flat_samples = dummy_chain.reshape(-1, ndim_ex) # After burn-in/thinning
    # Add some correlation for a more interesting corner plot
    dummy_flat_samples[:,1] += 0.5 * dummy_flat_samples[:,0]
    truths_ex = [np.median(dummy_flat_samples[:,i]) for i in range(ndim_ex)] # Use medians as "truths"
    # plot_corner_mcmc(dummy_flat_samples, param_labels_ex, truths=truths_ex, title="Test Corner Plot")
    # To save: plot_corner_mcmc(dummy_flat_samples, param_labels_ex, truths=truths_ex, output_file="test_corner.png")
    
    # --- Test Radial Profile Plot ---
    # Create some dummy profile data
    r_prof = np.linspace(1, 50, 20)
    sb1 = 100 * np.exp(-r_prof/10) + np.random.normal(0, 5, 20)
    err1 = np.abs(np.random.normal(0, 2, 20)) + 1
    sb2 = 80 * np.exp(-r_prof/15) + np.random.normal(0, 5, 20)
    err2 = np.abs(np.random.normal(0, 2, 20)) + 1

    profile_list_ex = [
        {'radii': r_prof, 'profile_stat': sb1, 'profile_err': err1, 
         'label': 'Data Set 1', 'color': 'crimson', 'fill_color': 'lightcoral'},
        {'radii': r_prof, 'profile_stat': sb2, 'profile_err': err2, 
         'label': 'Data Set 2 (Model)', 'color': 'dodgerblue', 'linestyle': '--', 'fill_color': 'lightskyblue'}
    ]
    plot_cfg_ex = {
        'x_scale': 'linear', 'y_scale': 'log', 
        'radius_label_base': "Radius",
        'plot_sb_r_squared': True,
        'image_filter_name': 'F160W',
        'outlier_clip_fraction': 0.05,
        'binning_mode_desc': "Linear Bins",
        'image_bunit': "MJy/sr"
    }
    obs_cfg_ex = {'pixel_scale_arcsec': 0.05, 'distance_pc': 140}
    fit_geom_ex = {'x0': 128.1, 'y0': 127.8, 'inc_deg': 25.3, 'pa_deg': 130.7}

    # plot_radial_profiles(profile_list_ex, plot_cfg_ex, obs_cfg_ex, fit_geom_ex, title_prefix="Awesome Disk Profile")
    # To save: plot_radial_profiles(profile_list_ex, plot_cfg_ex, obs_cfg_ex, fit_geom_ex, output_file="test_profile.png")
    logger.info("If plots did not show, uncomment the calls in the example section.")