# toydiskgeofitter_project/toydiskgeofitter/__init__.py

__version__ = "0.1.0"
PACKAGE_NAME = "toydiskgeofitter"

# --- Configuration ---
from .config import (
    load_config,
    save_default_config,
    update_config_from_fits_header,
    get_parameter_attributes,
    generate_default_config_string
)

# --- Input/Output ---
from .io import (
    load_fits_image,
    save_mcmc_sampler,
    load_mcmc_sampler,
    save_fit_results
)

# --- Geometry ---
from .geometry import (
    calculate_elliptical_radius_squared,
    define_annuli_edges,
    get_radial_profile_bin_centers
)

# --- Likelihoods & Priors ---
from .likelihoods import (
    log_likelihood_annuli_variance,
    build_log_prior_object,
    LogPriorEvaluator,
    make_log_probability_object,
    LogProbabilityWrapper,
    log_prior_uniform,
    log_prior_gaussian
)

# --- Fitting ---
from .fitting import (
    initialize_walkers,
    run_mcmc_sampler,
    get_best_fit_parameters,
    check_convergence,
    process_sampler_output
)

# --- Profiling ---
from .profiling import (
    generate_profile_distance_map,
    calculate_radial_profile
)

# --- Plotting ---
from .plotting import (
    plot_mcmc_chains,
    plot_corner_mcmc,
    plot_radial_profiles
)

# --- Models (Optional) ---
from .models import (
    generate_2d_gaussian_model,
    example_disk_brightness_profile
)

# --- Utilities ---
from .utils import (
    setup_logging,
    convert_radius_units,
    get_image_center_estimate,
    parse_parameter_config # Utility for parsing individual param blocks
)

# --- Pipeline (High-Level API) ---
from .pipeline import fit_disk_pipeline

__all__ = [
    # Config
    'load_config', 'save_default_config', 'update_config_from_fits_header',
    'get_parameter_attributes', 'generate_default_config_string',
    # IO
    'load_fits_image', 'save_mcmc_sampler', 'load_mcmc_sampler', 'save_fit_results',
    # Geometry
    'calculate_elliptical_radius_squared', 'define_annuli_edges', 'get_radial_profile_bin_centers',
    # Likelihoods
    'log_likelihood_annuli_variance',
    'build_log_prior_object', 'LogPriorEvaluator',
    'make_log_probability_object', 'LogProbabilityWrapper',
    'log_prior_uniform', 'log_prior_gaussian',
    # Fitting
    'initialize_walkers', 'run_mcmc_sampler', 'get_best_fit_parameters',
    'check_convergence', 'process_sampler_output',
    # Profiling
    'generate_profile_distance_map', 'calculate_radial_profile',
    # Plotting
    'plot_mcmc_chains', 'plot_corner_mcmc', 'plot_radial_profiles',
    # Models
    'generate_2d_gaussian_model', 'example_disk_brightness_profile',
    # Utils
    'setup_logging', 'convert_radius_units', 'get_image_center_estimate', 'parse_parameter_config',
    # Pipeline
    'fit_disk_pipeline',
]

#print(f"{PACKAGE_NAME} package v{__version__} and its modules have been loaded.")