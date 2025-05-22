# toydiskgeofitter_project/toydiskgeofitter/pipeline.py
import logging
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
import toml

from . import config as tdf_config
from . import io as tdf_io
from . import geometry as tdf_geometry
from . import likelihoods as tdf_likelihoods
from . import fitting as tdf_fitting
from . import plotting as tdf_plotting
from . import profiling as tdf_profiling
from . import utils as tdf_utils # For setup_logging if not already called
# Import __version__ from package __init__ as tdf_version 
from . import __version__ as tdf_version

logger = logging.getLogger(__name__) # Will be toydiskgeofitter.pipeline

def fit_disk_pipeline(
    config_file_path: str,
    image_file_path: Optional[str] = None, # Allow overriding image path from config
    output_directory_override: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    High-level API function to run the full disk fitting pipeline.

    This function loads configuration, reads image data, prepares for MCMC,
    runs the MCMC fitter, processes results, generates plots, and calculates
    radial profiles.

    Args:
        config_file_path (str): Path to the TOML configuration file.
        image_file_path (Optional[str]): If provided, overrides the image path(s)
            specified in the configuration file. For simplicity, this example
            assumes a single image override. For multiple images, config is preferred.
        output_directory_override (Optional[str]): If provided, overrides the
            output directory specified in the configuration.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing comprehensive results:
            - 'config': The configuration used.
            - 'best_fit_summary': Median and error estimates for parameters.
            - 'sampler': The emcee sampler object (if saved and loaded, or kept in memory).
            - 'flat_samples': Flattened MCMC samples after burn-in and thinning.
            - 'convergence_diagnostics': MCMC convergence metrics.
            - 'profile_results': Data from radial profile calculation.
            - 'output_paths': Dictionary of paths to saved plots and files.
            Returns None if a critical step fails.
    """
    # 0. Initial Setup (Logging, Output Directory)
    # Application should ideally set up logging, but we can do a basic one.
    if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
        tdf_utils.setup_logging(level='INFO')
    
    logger.info(f"--- Starting Toy Disk Geo Fitter Pipeline v{tdf_version} ---") # Access version from a module
    logger.info(f"Using configuration file: {config_file_path}")

    # 1. Load Configuration
    cfg = tdf_config.load_config(config_file_path)
    if not cfg:
        logger.error("Failed to load configuration. Pipeline aborted.")
        return None

    # Override output directory if specified
    if output_directory_override:
        cfg["output_settings"]["results_directory"] = output_directory_override
        logger.info(f"Output directory overridden to: {output_directory_override}")
    
    # Ensure output directory exists
    results_dir = cfg["output_settings"]["results_directory"]
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved in: {results_dir}")

    # Override image path if provided (simplistic: assumes first image)
    if image_file_path:
        if "observations" not in cfg or not cfg["observations"].get("image_paths"):
            cfg.setdefault("observations", {})["image_paths"] = [image_file_path]
            cfg["observations"].setdefault("hdu_indices", [0])
            cfg["observations"].setdefault("data_labels", [os.path.basename(image_file_path)])
        else:
            cfg["observations"]["image_paths"][0] = image_file_path
            if not cfg["observations"].get("data_labels"):
                cfg["observations"]["data_labels"] = [os.path.basename(image_file_path)]
            else:
                cfg["observations"]["data_labels"][0] = os.path.basename(image_file_path)
        logger.info(f"Image path overridden to: {image_file_path}")

    # 2. Load Image Data
    # For this example, we'll focus on the first image in the config
    obs_cfg = cfg["observations"]
    if not obs_cfg.get("image_paths"):
        logger.error("No image paths specified in the configuration. Pipeline aborted.")
        return None
    
    current_image_path = obs_cfg["image_paths"][0]
    current_hdu_idx = obs_cfg.get("hdu_indices", [0])[0]
    current_data_label = obs_cfg.get("data_labels", ["Image"])[0]

    image_data, header, wcs_obj = tdf_io.load_fits_image(
        current_image_path, hdu_index=current_hdu_idx, return_wcs=True
    )
    if image_data is None:
        logger.error(f"Failed to load image data from {current_image_path}. Pipeline aborted.")
        return None
    logger.info(f"Image '{current_image_path}' loaded. Shape: {image_data.shape}")

    # Update config from FITS header (e.g., pixel scale, guesses)
    cfg = tdf_config.update_config_from_fits_header(cfg, header, image_data.shape, image_index=0)


    # 3. Prepare Data for Fitting (as in the notebook example)
    fit_params_config = cfg["fitting_parameters"]
    param_names_to_fit = [name for name, p_cfg in fit_params_config.items() if p_cfg.get("fit", False)]
    fixed_params_dict = {name: p_cfg.get("value") for name, p_cfg in fit_params_config.items() if not p_cfg.get("fit", False)}
    ndim = len(param_names_to_fit)

    if not param_names_to_fit:
        logger.error("No parameters specified to fit in the configuration. Pipeline aborted.")
        return None
    logger.info(f"Parameters to fit ({ndim}D): {param_names_to_fit}")

    fit_region_cfg = cfg["fitting_regions"][0] # Assuming one region for now
    r_min_px = fit_region_cfg["r_min_pixels"]
    r_max_px = fit_region_cfg["r_max_pixels"]
    ann_width_px = fit_region_cfg["annulus_width_pixels"]
    fit_annuli_r_edges = tdf_geometry.define_annuli_edges(r_min_px, r_max_px, annulus_width=ann_width_px)
    fit_annuli_r_sq_edges = fit_annuli_r_edges**2

    ny, nx = image_data.shape
    initial_center_guess_x = fit_params_config["x0"]["guess"] # Assumes x0 is a fit param or has guess
    initial_center_guess_y = fit_params_config["y0"]["guess"]
    y_coords_full, x_coords_full = np.ogrid[0:ny, 0:nx]
    dist_sq_from_guess_center_2d = (x_coords_full - initial_center_guess_x)**2 + (y_coords_full - initial_center_guess_y)**2
    
    # Estimate buffer based on prior ranges for x0, y0 if they are fit, else use a fixed small buffer
    x0_prior_width = np.abs(fit_params_config["x0"]["prior_args"][1] - fit_params_config["x0"]["prior_args"][0]) if "x0" in param_names_to_fit else 5.0
    y0_prior_width = np.abs(fit_params_config["y0"]["prior_args"][1] - fit_params_config["y0"]["prior_args"][0]) if "y0" in param_names_to_fit else 5.0
    buffer_for_center_offset = max(x0_prior_width, y0_prior_width) / 2.0 # half of prior width as buffer
    max_mask_r = r_max_px + buffer_for_center_offset
    
    fitting_region_mask_2d = (dist_sq_from_guess_center_2d < max_mask_r**2) & \
                             (dist_sq_from_guess_center_2d > (max(0,r_min_px - ann_width_px*2))**2) # Use wider inner for mask

    image_to_fit_masked_values_1d = image_data[fitting_region_mask_2d]
    y_coords_masked_1d, x_coords_masked_1d = np.where(fitting_region_mask_2d)
    coords_yx_masked_1d_tuple = (y_coords_masked_1d, x_coords_masked_1d)

    if len(image_to_fit_masked_values_1d) == 0:
        logger.error("No pixels selected for fitting by initial mask. Check fitting_regions/image. Pipeline aborted.")
        return None
    logger.info(f"Selected {len(image_to_fit_masked_values_1d)} pixels for likelihood calculation.")

    bf_percentile = fit_region_cfg["brightness_floor_percentile"]
    finite_masked_pixels = image_to_fit_masked_values_1d[np.isfinite(image_to_fit_masked_values_1d)]
    brightness_floor_val = np.percentile(finite_masked_pixels, bf_percentile) if len(finite_masked_pixels) > 0 else 0.0
    logger.info(f"Brightness floor value (percentile {bf_percentile}%): {brightness_floor_val:.3e}")


    # 4. Setup Priors and Likelihood
    log_prior_obj = tdf_likelihoods.build_log_prior_object(
        param_names_to_fit, fit_params_config, image_ny_nx=(ny, nx)
    )
    likelihood_static_args = (
        param_names_to_fit, image_to_fit_masked_values_1d, coords_yx_masked_1d_tuple,
        fit_annuli_r_sq_edges, brightness_floor_val, fixed_params_dict
    )
    log_probability_obj = tdf_likelihoods.make_log_probability_object(
        log_prior_obj, tdf_likelihoods.log_likelihood_annuli_variance, likelihood_static_args
    )
    
    # Test log_prob at initial guess
    initial_guess_theta = tuple(fit_params_config[pname]["guess"] for pname in param_names_to_fit)
    test_log_prob = log_probability_obj(initial_guess_theta)
    logger.info(f"Log probability at initial guess: {test_log_prob:.2f}")
    if not np.isfinite(test_log_prob):
        logger.error("Log probability is not finite at initial parameter guesses. Check priors or initial setup. Pipeline aborted.")
        return None

    # 5. Run MCMC
    mcmc_cfg = cfg["mcmc_settings"]
    initial_walker_positions = tdf_fitting.initialize_walkers(
        nwalkers=mcmc_cfg["nwalkers"], fit_param_names=param_names_to_fit,
        param_configs=fit_params_config, ball_scale=mcmc_cfg["initial_ball_scale"]
    )
    
    sampler_to_continue_from = None
    if mcmc_cfg.get("continue_from_sampler"):
        logger.info(f"Attempting to load sampler to continue from: {mcmc_cfg['continue_from_sampler']}")
        sampler_to_continue_from = tdf_io.load_mcmc_sampler(mcmc_cfg['continue_from_sampler'])
        if sampler_to_continue_from:
             logger.info("Successfully loaded sampler to continue run.")
             # Use the last position of the loaded sampler as initial_walker_positions
             initial_walker_positions = sampler_to_continue_from.get_last_sample().coords
        else:
             logger.warning("Failed to load sampler to continue. Starting a new run.")


    use_mp = True if mcmc_cfg.get("threads", 1) != 1 else False
    sampler = tdf_fitting.run_mcmc_sampler(
        log_probability_obj, initial_walker_positions,
        nwalkers=mcmc_cfg["nwalkers"], nsteps=mcmc_cfg["nsteps"], ndim=ndim,
        mcmc_config=mcmc_cfg, sampler_to_continue=sampler_to_continue_from,
        use_multiprocessing=use_mp, num_threads=mcmc_cfg.get("threads", 1)
    )

    # 6. Process Results
    samples_chain, flat_samples, best_fit_summary, convergence_diagnostics = \
        tdf_fitting.process_sampler_output(
            sampler, param_names_to_fit, mcmc_cfg["nburn"], mcmc_cfg["thin_by"]
    )
    if best_fit_summary is None:
        logger.error("Failed to process MCMC results. Pipeline aborted.")
        return None
    
    # Save sampler if configured
    output_paths: Dict[str, str] = {}
    if mcmc_cfg.get("save_sampler", False):
        sampler_path = os.path.join(results_dir, mcmc_cfg["sampler_output_path"])
        tdf_io.save_mcmc_sampler(sampler, sampler_path)
        output_paths["sampler"] = sampler_path
        logger.info(f"MCMC sampler saved to {sampler_path}")

    # Save best-fit summary (e.g. as text or pickled dict)
    summary_file_path = os.path.join(results_dir, "fit_summary_results.txt")
    with open(summary_file_path, 'w') as f:
        f.write(f"# Fit Summary for: {current_data_label}\n")
        f.write(f"# Config: {os.path.abspath(config_file_path)}\n")
        f.write(f"# Image: {os.path.abspath(current_image_path)}\n\n")
        f.write("--- Best-Fit Parameters (median +/- 1-sigma percentile errors) ---\n")
        for pname, pvals in best_fit_summary.items():
            label = fit_params_config[pname].get('label', pname)
            f.write(f"  {label:<20s}: {pvals['median']:.4f} (+{pvals['upper_err']:.4f} / -{pvals['lower_err']:.4f})\n")
        f.write("\n--- Convergence Diagnostics ---\n")
        for key, val in convergence_diagnostics.items():
             f.write(f"  {key}: {np.round(val, 3) if isinstance(val, (list, np.ndarray)) else (f'{val:.3f}' if isinstance(val, float) else val)}\n")
    output_paths["fit_summary_text"] = summary_file_path
    logger.info(f"Fit summary text file saved to {summary_file_path}")
    
    # Save the best_fit_summary dictionary as TOML
    best_fit_toml_path = os.path.join(results_dir, "best_fit_summary.toml")
    try:
        with open(best_fit_toml_path, 'w') as f_toml:
            toml.dump(best_fit_summary, f_toml)
        output_paths["best_fit_summary_toml"] = best_fit_toml_path
        logger.info(f"Best-fit summary (dictionary) saved to TOML: {best_fit_toml_path}")
    except Exception as e:
        logger.error(f"Failed to save best_fit_summary to TOML file {best_fit_toml_path}: {e}")

    # 7. Generate Plots
    plot_cfg = cfg["plotting_settings"]
    plot_output_fmt = plot_cfg["output_format"]
    param_labels_for_plot = [fit_params_config[pname].get("label", pname) for pname in param_names_to_fit]

    if cfg["output_settings"].get("plot_chains", False) and samples_chain is not None:
        chains_fname = plot_cfg["chains_plot_filename"].format(output_format=plot_output_fmt)
        chains_fpath = os.path.join(results_dir, chains_fname)
        tdf_plotting.plot_mcmc_chains(
            samples_chain, param_labels_for_plot, output_file=chains_fpath,
            title=f"MCMC Chains: {current_data_label}"
        )
        output_paths["chains_plot"] = chains_fpath

    if cfg["output_settings"].get("plot_corner", False) and flat_samples is not None:
        corner_fname = plot_cfg["corner_plot_filename"].format(output_format=plot_output_fmt)
        corner_fpath = os.path.join(results_dir, corner_fname)
        truths_for_corner = [best_fit_summary[pname]["median"] for pname in param_names_to_fit]
        tdf_plotting.plot_corner_mcmc(
            flat_samples, param_labels_for_plot, truths=truths_for_corner,
            output_file=corner_fpath, title=f"MCMC Posteriors: {current_data_label}"
        )
        output_paths["corner_plot"] = corner_fpath

    # 8. Calculate and Plot Radial Profile
    profile_results_data = None
    if cfg["output_settings"].get("plot_radial_profile", False):
        logger.info("Calculating and plotting radial profile...")
        profile_calc_cfg = cfg["profile_calculation"]
        
        best_fit_geom_for_profile = {}
        for p_needed in ['x0', 'y0', 'inc_deg', 'pa_deg']:
            if p_needed in best_fit_summary: best_fit_geom_for_profile[p_needed] = best_fit_summary[p_needed]['median']
            elif p_needed in fixed_params_dict: best_fit_geom_for_profile[p_needed] = fixed_params_dict[p_needed]
            else:
                logger.error(f"Cannot make profile: Param '{p_needed}' missing from fit/fixed params.")
                best_fit_geom_for_profile = None # Mark as unavailable
                break
        
        if best_fit_geom_for_profile:
            profile_type = "elliptical" if profile_calc_cfg["use_elliptical_bins"] else "circular"
            dist_map_prof_sq = tdf_profiling.generate_profile_distance_map(
                image_data.shape, best_fit_geom_for_profile, profile_type
            )

            if dist_map_prof_sq is not None:
                prof_r_min = profile_calc_cfg.get("min_radius_px_log", 1.0) if profile_calc_cfg["binning_mode"] == "log" else 0.0
                # Use a reasonable max for profile, could be different from fitting r_max
                prof_r_max = min(nx, ny) / 2.0 * 0.95 # Example: 95% of image half-size
                
                if profile_calc_cfg["binning_mode"] == "log":
                    prof_edges = tdf_geometry.define_annuli_edges(prof_r_min, prof_r_max, num_annuli=profile_calc_cfg.get("num_log_bins",30), log_spacing=True)
                else:
                    prof_edges = tdf_geometry.define_annuli_edges(prof_r_min, prof_r_max, annulus_width=profile_calc_cfg["linear_bin_size_px"])

                (p_centers, p_stat, p_std, p_err, p_N) = tdf_profiling.calculate_radial_profile(
                    image_data, dist_map_prof_sq, prof_edges,
                    outlier_clip_fraction=profile_calc_cfg["outlier_clip_fraction"],
                    statistic=profile_calc_cfg["statistic"],
                    min_pixels_per_bin=profile_calc_cfg["min_pixels_per_bin"]
                )
                profile_results_data = {
                    'bin_centers': p_centers, 'statistic': p_stat, 'std_dev': p_std,
                    'std_err': p_err, 'N_pixels': p_N
                }
                
                profile_plot_fname = plot_cfg["profile_plot_filename"].format(output_format=plot_output_fmt)
                profile_fpath = os.path.join(results_dir, profile_plot_fname)
                
                plot_data_list = [{'radii': p_centers, 'profile_stat': p_stat, 'profile_err': p_err,
                                   'label': current_data_label, 'color': 'dodgerblue', 'fill_color': 'lightskyblue'}]
                
                radial_plot_cfg = {
                    'x_scale': 'log' if profile_calc_cfg["binning_mode"] == "log" else 'linear',
                    'y_scale': 'linear', # Or 'log', could be in config
                    'radius_label_base': "Semi-Major Axis" if profile_calc_cfg["use_elliptical_bins"] else "Radius",
                    'plot_sb_r_squared': profile_calc_cfg["plot_sb_r_squared"],
                    'image_filter_name': header.get("FILTER", current_data_label),
                    'outlier_clip_fraction': profile_calc_cfg["outlier_clip_fraction"],
                    'binning_mode_desc': f"{profile_calc_cfg['binning_mode'].capitalize()} {'Elliptical' if profile_calc_cfg['use_elliptical_bins'] else 'Circular'} Bins",
                    'image_bunit': header.get("BUNIT", plot_cfg.get("image_bunit_default", "Intensity"))
                }
                tdf_plotting.plot_radial_profiles(
                    plot_data_list, radial_plot_cfg, obs_cfg, best_fit_geom_for_profile,
                    output_file=profile_fpath, title_prefix=f"Radial Profile: {current_data_label}"
                )
                output_paths["radial_profile_plot"] = profile_fpath
            else:
                logger.warning("Could not generate distance map for radial profile. Skipping profile plot.")
        else:
             logger.warning("Best-fit geometry for profile incomplete. Skipping profile plot.")

    # 9. Compile final results dictionary
    final_results = {
        "config_used": cfg, # Include the (potentially modified) config
        "best_fit_summary": best_fit_summary,
        "mcmc_sampler": sampler if mcmc_cfg.get("return_sampler_object", False) else "Not returned (see saved file if enabled)", # Be careful with large objects
        "flat_samples": flat_samples if mcmc_cfg.get("return_flat_samples", True) else "Not returned",
        "convergence_diagnostics": convergence_diagnostics,
        "profile_results": profile_results_data,
        "output_paths": output_paths,
        "image_header": header, # For reference
        "image_wcs": wcs_obj # For reference
    }
    
    # Create a main results metadata TOML file that points to other files
    main_results_toml_path = os.path.join(results_dir, "pipeline_run_metadata.toml")
    metadata_to_save = {
        "pipeline_version": tdf_version,
        "original_config_file": os.path.abspath(config_file_path),
        "image_file_used": os.path.abspath(current_image_path),
        "output_directory": os.path.abspath(results_dir),
        "status": "Success", # or "Partial Success" / "Failure"
        "notes": "Summary of the fitting run.",
        "output_files": {k: os.path.abspath(v) for k,v in output_paths.items()} # Absolute paths
    }
    try:
        with open(main_results_toml_path, 'w') as f:
            toml.dump(metadata_to_save, f)
        logger.info(f"Main pipeline run metadata saved to TOML: {main_results_toml_path}")
    except Exception as e:
        logger.error(f"Failed to save main run metadata to TOML: {e}")


    logger.info("--- ToyDiskGeoFitter Pipeline Finished Successfully ---")
    return final_results