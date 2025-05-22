# toydiskgeofitter_project/toydiskgeofitter/fitting.py
import logging
import time
import numpy as np
import emcee
import multiprocessing # For multiprocessing.cpu_count()
from multiprocessing import pool as mp_pool # Import Pool specifically for isinstance check
from typing import Tuple, List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

def initialize_walkers(
    nwalkers: int,
    fit_param_names: List[str],
    param_configs: Dict[str, Dict[str, Any]],
    initialization_strategy: str = "ball",
    ball_scale: float = 1e-3
) -> np.ndarray:
    """
    Generates initial positions for MCMC walkers.

    Args:
        nwalkers (int): Number of walkers.
        fit_param_names (List[str]): List of names of parameters to be fitted.
        param_configs (Dict[str, Dict[str, Any]]): Configuration for each parameter,
            containing 'guess', 'prior_type', 'prior_args'.
        initialization_strategy (str): 'ball' (small ball around guess) or
                                       'prior' (draw from prior distribution - more complex).
                                       Currently, only 'ball' is robustly implemented.
        ball_scale (float): Scale factor for the initial ball size, relative to guess
                            or prior range if guess is not well-defined.

    Returns:
        np.ndarray: Initial positions array of shape (nwalkers, ndim).

    Raises:
        ValueError: If parameter configuration is missing or strategy is unknown.
    """
    ndim = len(fit_param_names)
    initial_positions = np.zeros((nwalkers, ndim))

    for i, param_name in enumerate(fit_param_names):
        p_config = param_configs.get(param_name)
        if not p_config:
            raise ValueError(f"Missing configuration for parameter '{param_name}' during walker initialization.")

        guess = p_config.get("guess")
        prior_type = p_config.get("prior_type", "uniform").lower()
        prior_args = p_config.get("prior_args", [])

        if guess is None: 
            if prior_type == "uniform" and len(prior_args) == 2:
                guess = (prior_args[0] + prior_args[1]) / 2.0
            else: 
                logger.warning(f"Guess for '{param_name}' is None and prior is not simple uniform. Using 0 or prior mean if possible.")
                guess = 0.0
                if prior_type == "gaussian" and len(prior_args) == 2:
                    guess = prior_args[0] 

        if initialization_strategy == "ball":
            scale = np.abs(guess * ball_scale) if guess != 0 else ball_scale
            if prior_type == "uniform" and len(prior_args) == 2: 
                width = prior_args[1] - prior_args[0]
                scale = min(scale, width * 0.1) 

            initial_positions[:, i] = guess + scale * np.random.randn(nwalkers)

            if prior_type == "uniform" and len(prior_args) == 2:
                initial_positions[:, i] = np.clip(initial_positions[:, i], prior_args[0], prior_args[1])

        elif initialization_strategy == "prior":
            logger.warning("'prior' initialization strategy is not fully implemented yet, using 'ball'.")
            scale = np.abs(guess * ball_scale) if guess != 0 else ball_scale
            initial_positions[:, i] = guess + scale * np.random.randn(nwalkers)
        else:
            raise ValueError(f"Unknown walker initialization strategy: {initialization_strategy}")
            
    return initial_positions


def run_mcmc_sampler(
    log_probability_func: Callable[[Tuple[float, ...]], float],
    initial_positions: np.ndarray,
    nwalkers: int,
    nsteps: int,
    ndim: int,
    mcmc_config: Dict[str, Any],
    sampler_to_continue: Optional[emcee.EnsembleSampler] = None,
    use_multiprocessing: bool = True,
    num_threads: int = -1
) -> emcee.EnsembleSampler:
    """
    Sets up and runs the emcee EnsembleSampler.

    Args:
        log_probability_func (Callable): The log probability function.
        initial_positions (np.ndarray): Starting positions for walkers.
        nwalkers (int): Number of walkers.
        nsteps (int): Number of steps to run.
        ndim (int): Number of parameters (dimensions).
        mcmc_config (Dict[str, Any]): MCMC settings from main config (e.g. 'backend' for emcee).
        sampler_to_continue (Optional[emcee.EnsembleSampler]): An existing sampler to continue.
        use_multiprocessing (bool): Whether to use multiprocessing.
        num_threads (int): Number of threads for multiprocessing. -1 for all available.

    Returns:
        emcee.EnsembleSampler: The executed sampler object.
    """
    backend = None 
    actual_sampler = sampler_to_continue
    
    if num_threads == -1:
        threads_to_use = multiprocessing.cpu_count()
    else:
        threads_to_use = max(1, num_threads)
    
    pool_instance_managed_here = None 

    if use_multiprocessing and threads_to_use > 1:
        logger.info(f"Using multiprocessing with {threads_to_use} threads.")
        
        current_sampler_pool = None
        if actual_sampler is not None and hasattr(actual_sampler, 'pool'):
            current_sampler_pool = actual_sampler.pool

        if current_sampler_pool is None: # No existing pool, or sampler is new
            pool_instance_managed_here = mp_pool.Pool(processes=threads_to_use)
            logger.debug(f"Created a new multiprocessing.Pool: {pool_instance_managed_here}")
            effective_pool_for_emcee = pool_instance_managed_here
        else: # Sampler already has a pool
            logger.debug(f"Using existing pool from sampler: {current_sampler_pool}")
            effective_pool_for_emcee = current_sampler_pool
            # We don't own this pool's lifecycle if it came with the sampler

        if actual_sampler is None:
            actual_sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_func,
                pool=effective_pool_for_emcee, 
                backend=backend
            )
        elif actual_sampler.pool is None and effective_pool_for_emcee is not None:
            # If sampler existed but had no pool, and we made one (or decided to use one)
            actual_sampler.pool = effective_pool_for_emcee
            logger.debug("Assigned pool to existing sampler that had none.")


        logger.info(f"Running MCMC with {nwalkers} walkers for {nsteps} steps...")
        start_time = time.time()
        # emcee's run_mcmc will use sampler.pool if it's set
        actual_sampler.run_mcmc(initial_positions, nsteps, progress=True)
        end_time = time.time()
        logger.info(f"MCMC run completed in {end_time - start_time:.2f} seconds.")

        # Only close the pool if this specific call to run_mcmc_sampler created it.
        if pool_instance_managed_here is not None and \
           isinstance(pool_instance_managed_here, mp_pool.Pool): # Check type robustly
            try:
                logger.debug(f"Attempting to close and join pool: {pool_instance_managed_here} which was managed here.")
                pool_instance_managed_here.close()
                pool_instance_managed_here.join()
                logger.info("Multiprocessing pool (managed by this function call) closed and joined.")
                # If the sampler was using the pool we just closed, set its pool attribute to None
                if hasattr(actual_sampler, 'pool') and actual_sampler.pool is pool_instance_managed_here:
                    actual_sampler.pool = None
            except Exception as e:
                logger.warning(f"Could not close the locally managed multiprocessing pool cleanly: {e}")
        elif effective_pool_for_emcee is not None:
             logger.debug("Pool was used but not managed by this function call (e.g., came with sampler_to_continue); not closing it here.")

    else: 
        logger.info("Using single thread for MCMC.")
        if actual_sampler is None:
            actual_sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_func,
                backend=backend
            )
        
        logger.info(f"Running MCMC with {nwalkers} walkers for {nsteps} steps...")
        start_time = time.time()
        actual_sampler.run_mcmc(initial_positions, nsteps, progress=True)
        end_time = time.time()
        logger.info(f"MCMC run completed in {end_time - start_time:.2f} seconds.")

    return actual_sampler


def get_best_fit_parameters(
    flat_samples: np.ndarray,
    param_names: List[str],
    percentiles: Tuple[float, float, float] = (16, 50, 84)
) -> Dict[str, Dict[str, float]]:
    """
    Calculates best-fit parameters (median) and uncertainties (percentiles)
    from the flattened MCMC samples.

    Args:
        flat_samples (np.ndarray): Flattened array of MCMC samples (n_samples, ndim).
        param_names (List[str]): List of parameter names, matching the order in flat_samples.
        percentiles (Tuple[float, float, float]): Percentiles to calculate for
                                                  lower_err, median, upper_err.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are parameter names,
            and values are dicts with 'median', 'lower_err', 'upper_err'.
    """
    results = {}
    if flat_samples.shape[1] != len(param_names):
        raise ValueError("Mismatch between flat_samples dimensions and number of param_names.")

    for i, name in enumerate(param_names):
        p_samples = flat_samples[:, i]
        p_median = np.percentile(p_samples, percentiles[1])
        p_lower = np.percentile(p_samples, percentiles[0])
        p_upper = np.percentile(p_samples, percentiles[2])
        results[name] = {
            "median": p_median,
            "lower_err": p_median - p_lower, 
            "upper_err": p_upper - p_median, 
            "16th": p_lower,
            "50th": p_median,
            "84th": p_upper
        }
        logger.info(
            f"{name}: {p_median:.3f} (+{results[name]['upper_err']:.3f} / -{results[name]['lower_err']:.3f})"
        )
    return results

def check_convergence(
    sampler: emcee.EnsembleSampler,
    nburn: int,
    thin: int,
    show_warnings: bool = True
) -> Dict[str, Any]:
    """
    Performs basic MCMC convergence checks.

    Args:
        sampler (emcee.EnsembleSampler): The executed sampler.
        nburn (int): Number of burn-in steps to discard for these checks.
        thin (int): Thinning factor.
        show_warnings (bool): If True, print warnings for potential issues.

    Returns:
        Dict[str, Any]: Dictionary containing convergence diagnostics.
                        (e.g., 'mean_acceptance_fraction', 'autocorr_time').
    """
    diagnostics = {}
    
    acceptance_fraction = np.mean(sampler.acceptance_fraction)
    diagnostics["mean_acceptance_fraction"] = acceptance_fraction
    logger.info(f"Mean acceptance fraction: {acceptance_fraction:.3f}")
    if show_warnings and not (0.1 < acceptance_fraction < 0.7): 
        logger.warning("Acceptance fraction may indicate poor sampling efficiency (ideal: ~0.2-0.5).")

    try:
        chain_for_autocorr = sampler.get_chain(discard=nburn, thin=thin, flat=False)
        if chain_for_autocorr.shape[0] > 50 : 
            autocorr_time = sampler.get_autocorr_time(discard=nburn, tol=0) 
            diagnostics["autocorrelation_time"] = autocorr_time.tolist() 
            logger.info(f"Autocorrelation time estimates: {autocorr_time}")
            
            n_samples_effective = chain_for_autocorr.shape[0] * chain_for_autocorr.shape[1] / np.mean(autocorr_time)
            diagnostics["effective_samples"] = n_samples_effective
            logger.info(f"Estimated effective number of samples: {n_samples_effective:.0f}")
            
            if show_warnings and np.any(chain_for_autocorr.shape[0] < 50 * autocorr_time): 
                logger.warning("Chain might be shorter than 50x autocorrelation time. Consider running longer.")
        else:
            logger.warning("Chain too short after burn-in/thinning for reliable autocorrelation time estimate.")
            diagnostics["autocorrelation_time"] = None
            diagnostics["effective_samples"] = None

    except emcee.autocorr.AutocorrError as e:
        logger.warning(f"Could not estimate autocorrelation time: {e}")
        diagnostics["autocorrelation_time"] = None
        diagnostics["effective_samples"] = None
    except Exception as e: 
        logger.error(f"Unexpected error during autocorrelation time calculation: {e}")
        diagnostics["autocorrelation_time"] = None
        diagnostics["effective_samples"] = None
        
    return diagnostics

def process_sampler_output(
    sampler: emcee.EnsembleSampler,
    param_names: List[str],
    nburn: int,
    thin: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Dict[str, float]]], Optional[Dict[str, Any]]]:
    """
    Processes the output of the MCMC sampler to get chains and best-fit parameters.

    Args:
        sampler (emcee.EnsembleSampler): The executed emcee sampler.
        param_names (List[str]): Names of the fitted parameters.
        nburn (int): Number of burn-in steps to discard.
        thin (int): Thinning factor for the chains.

    Returns:
        Tuple containing:
            - samples_chain (Optional[np.ndarray]): The chain of samples after burn-in and thinning.
                                                   Shape (nsteps_after_burn_thin, nwalkers, ndim).
            - flat_samples (Optional[np.ndarray]): Flattened samples after burn-in and thinning.
                                                  Shape (total_samples, ndim).
            - best_fit_summary (Optional[Dict]): Dictionary of best-fit parameters and errors.
            - convergence_diagnostics (Optional[Dict]): Dictionary of convergence metrics.
    """
    if sampler is None or not hasattr(sampler, 'get_chain'):
        logger.error("Invalid or unrun sampler provided to process_sampler_output.")
        return None, None, None, None

    try:
        logger.info(f"Processing sampler output: Discarding {nburn} burn-in steps, thinning by {thin}.")
        samples_chain = sampler.get_chain(discard=nburn, thin=thin, flat=False)
        flat_samples = sampler.get_chain(discard=nburn, thin=thin, flat=True)

        if flat_samples.shape[0] == 0:
            logger.error("No samples remaining after burn-in and thinning. Check MCMC parameters.")
            return samples_chain, flat_samples, None, None 

        best_fit_summary = get_best_fit_parameters(flat_samples, param_names)
        convergence_diagnostics = check_convergence(sampler, nburn, thin)
        
        return samples_chain, flat_samples, best_fit_summary, convergence_diagnostics

    except Exception as e:
        logger.error(f"Error processing MCMC sampler output: {e}")
        return None, None, None, None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Fitting module example usage (conceptual):")

    def dummy_log_prob(theta):
        if len(theta) != 2: return -np.inf
        return -0.5 * (theta[0]**2 + theta[1]**2)

    n_dim = 2
    n_walkers = 20
    n_steps = 500
    n_burn = 100
    n_thin = 5
    
    dummy_param_names = ["param_a", "param_b"]
    dummy_param_cfgs = {
        "param_a": {"guess": 0.1, "prior_type": "uniform", "prior_args": [-5, 5]},
        "param_b": {"guess": -0.1, "prior_type": "uniform", "prior_args": [-5, 5]},
    }

    initial_pos = initialize_walkers(n_walkers, dummy_param_names, dummy_param_cfgs)
    
    mcmc_cfg_mock = {"sampler": "emcee"} 

    sampler_obj = run_mcmc_sampler(dummy_log_prob, initial_pos, n_walkers, n_steps, n_dim, mcmc_cfg_mock, use_multiprocessing=False)
    
    if sampler_obj:
        chain, flat, best_fit, conv_diag = process_sampler_output(
            sampler_obj, dummy_param_names, n_burn, n_thin
        )
        if flat is not None:
            logger.info(f"Flat samples shape: {flat.shape}")
            logger.info("Best fit parameters:")
            for p, v in best_fit.items():
                logger.info(f"  {p}: {v['median']:.3f} +{v['upper_err']:.3f}/-{v['lower_err']:.3f}")
            logger.info(f"Convergence diagnostics: {conv_diag}")
    else:
        logger.error("MCMC run failed.")