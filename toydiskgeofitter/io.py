# toydiskgeofitter_project/toydiskgeofitter/io.py
import logging
import pickle
from typing import Tuple, Optional, Any, List, Dict

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

def load_fits_image(
    file_path: str,
    hdu_index: int = 0,
    return_wcs: bool = True
) -> Tuple[Optional[np.ndarray], Optional[fits.Header], Optional[WCS]]:
    """
    Loads FITS image data, header, and optionally the WCS object.

    Args:
        file_path (str): Path to the FITS file.
        hdu_index (int): Index of the HDU to load. Defaults to 0.
        return_wcs (bool): If True, attempts to parse and return a WCS object.

    Returns:
        Tuple[Optional[np.ndarray], Optional[fits.Header], Optional[WCS]]:
            - Image data as a NumPy array, or None on failure.
            - FITS header object, or None on failure.
            - astropy.wcs.WCS object if return_wcs is True and parsing succeeds, else None.
    """
    try:
        with fits.open(file_path) as hdul:
            if hdu_index >= len(hdul):
                logger.error(f"HDU index {hdu_index} out of range for {file_path} (found {len(hdul)} HDUs).")
                return None, None, None
            
            hdu = hdul[hdu_index]
            data = hdu.data
            header = hdu.header
            
            if data is None:
                logger.warning(f"No data found in HDU {hdu_index} of {file_path}.")
                return None, header, None

            wcs_obj = None
            if return_wcs:
                try:
                    wcs_obj = WCS(header)
                    if not wcs_obj.is_celestial:
                        logger.debug(f"WCS for {file_path} is not celestial. Still returning.")
                except Exception as e:
                    logger.warning(f"Could not parse WCS from header of {file_path}: {e}")
                    wcs_obj = None
            
            logger.info(f"Successfully loaded image data and header from {file_path}, HDU {hdu_index}.")
            return data.astype(np.float32), header, wcs_obj # Cast to float32 for consistency
    except FileNotFoundError:
        logger.error(f"FITS file not found: {file_path}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error reading FITS file {file_path}: {e}")
        return None, None, None

def save_mcmc_sampler(sampler: Any, file_path: str) -> bool:
    """
    Saves an MCMC sampler object (e.g., emcee.EnsembleSampler) using pickle.

    Args:
        sampler (Any): The sampler object to save.
        file_path (str): Path to save the sampler.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(sampler, f)
        logger.info(f"MCMC sampler saved to {file_path}")
        return True
    except pickle.PicklingError as e:
        logger.error(f"Error pickling sampler to {file_path}: {e}")
    except IOError as e:
        logger.error(f"IOError saving sampler to {file_path}: {e}")
    return False

def load_mcmc_sampler(file_path: str) -> Optional[Any]:
    """
    Loads an MCMC sampler object from a pickle file.

    Args:
        file_path (str): Path to the pickled sampler file.

    Returns:
        Optional[Any]: The loaded sampler object, or None on failure.
    """
    try:
        with open(file_path, 'rb') as f:
            sampler = pickle.load(f)
        logger.info(f"MCMC sampler loaded from {file_path}")
        return sampler
    except FileNotFoundError:
        logger.error(f"Sampler file not found: {file_path}")
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling sampler from {file_path}: {e}")
    except IOError as e:
        logger.error(f"IOError loading sampler from {file_path}: {e}")
    return None

def save_fit_results(results: Dict, file_path: str) -> bool:
    """
    Saves fit results (e.g., a dictionary) to a pickle file or JSON.
    Using pickle for now for simplicity with complex objects like astropy units.
    For broader compatibility, JSON with custom encoders/decoders for astropy objects
    or HDF5 might be better for very large datasets.

    Args:
        results (Dict): Dictionary containing fit results.
        file_path (str): Path to save the results.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(file_path, 'wb') as f: # Using pickle
            pickle.dump(results, f)
        # For JSON, you'd need to handle non-serializable types:
        # import json
        # def default_json_serializer(obj):
        #     if isinstance(obj, np.ndarray): return obj.tolist()
        #     if hasattr(obj, 'value') and hasattr(obj, 'unit'): return f"{obj.value} {obj.unit}"
        #     raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        # with open(file_path.replace('.pkl', '.json'), 'w') as f:
        #    json.dump(results, f, indent=4, default=default_json_serializer)
        logger.info(f"Fit results saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving fit results to {file_path}: {e}")
    return False

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy FITS file for testing load_fits_image
    dummy_data = np.arange(100).reshape(10, 10)
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = 8
    header['NAXIS'] = 2
    header['NAXIS1'] = 10
    header['NAXIS2'] = 10
    header['OBJECT'] = 'Test Object'
    # Add basic WCS keywords (Plate carrée projection)
    header['CTYPE1'] = 'RA---CAR'
    header['CTYPE2'] = 'DEC--CAR'
    header['CRVAL1'] = 0.0
    header['CRVAL2'] = 0.0
    header['CRPIX1'] = 5.5
    header['CRPIX2'] = 5.5
    header['CDELT1'] = -0.001
    header['CDELT2'] = 0.001
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'

    hdu = fits.PrimaryHDU(data=dummy_data, header=header)
    hdul = fits.HDUList([hdu])
    dummy_fits_path = "dummy_test_image.fits"
    hdul.writeto(dummy_fits_path, overwrite=True)
    print(f"Created dummy FITS file: {dummy_fits_path}")

    data, hdr, wcs = load_fits_image(dummy_fits_path)
    if data is not None:
        print(f"Loaded data shape: {data.shape}")
        print(f"Object from header: {hdr['OBJECT']}")
    if wcs:
        print(f"WCS celestial: {wcs.is_celestial}")
        print(f"Pixel to world (0,0): {wcs.pixel_to_world(0,0)}")

    # Test sampler save/load
    # Note: emcee is not imported here, so this is a conceptual test
    # In a real scenario, you would import emcee and create a sampler object
    class DummySampler:
        def __init__(self, name):
            self.name = name
            self.chain = np.random.rand(100, 10, 3) # steps, walkers, dim
    
    dummy_sampler_obj = DummySampler("test_emcee_sampler")
    sampler_path = "dummy_sampler.pkl"
    save_mcmc_sampler(dummy_sampler_obj, sampler_path)
    loaded_sampler = load_mcmc_sampler(sampler_path)
    if loaded_sampler:
        print(f"Loaded sampler name: {loaded_sampler.name}")
        print(f"Loaded sampler chain shape: {loaded_sampler.chain.shape}")
        
    # Clean up dummy file
    import os
    os.remove(dummy_fits_path)
    os.remove(sampler_path)
    print("Cleaned up dummy files.")