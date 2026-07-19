import numpy as np
import sys
import os
import keras
import dill as pickle



# -------------------------------------------------------------------
# -----------------------Bundle functions----------------------------
# -------------------------------------------------------------------


def list_available_bundles():
    '''
    Lists the available bundles in the cloud.

    Returns:
        list: A list of available bundle names.
    '''
    # import requests
    # from bs4 import BeautifulSoup

    # url = "https://cloud.iac.es/public.php/webdav/"
    # response = requests.get(url, auth=('H7FdjCcJcaZJSzN', ''))

    # if response.status_code == 200:
    #     soup = BeautifulSoup(response.content, 'html.parser')
    #     bundles = [a.text for a in soup.find_all('a') if a.text.endswith('.tgz')]
    #     return bundles
    # else:
    #     raise OSError("Failed to retrieve bundle list from the cloud. Status code: {}".format(response.status_code))
    return ['MW_v1.0', 'MW_v1.1', 'MW_v1.2', 'MW_v1.3', 'MW_v1.4', 'SMC_v1.0']


def install_bundle(bundle_name, bundle_path = None):
    '''
    Downloads and unpacks bundles from the cloud.

    Parameters:
        bundle_name (str): The name of the bundle to download.
        bundle_path (str, optional): The path where the bundle will be unpacked.
                                      If not provided, the bundle will be unpacked
                                      in the current directory.
    Returns:
        None: This function does not return any value. It performs the download
              and unpacking operation directly.
    Raises:
        OSError: If the curl command fails or if there are issues with the
                 network connection.
    '''

    # check that the ~/.specfann folder exists if bundle_path isn't specified
    if bundle_path == None:
        local_path = os.path.expanduser('~/.specfann')
        bundle_path = os.path.join(local_path, 'bundles')
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        if not os.path.exists(bundle_path):
            os.makedirs(bundle_path)

    # define local bundle name for convenience
    local_bundle_name=os.path.join(bundle_path, '{}.tgz'.format(bundle_name))

    # download the bundle
    print('Downloading bundle {}... This could take several minutes'.format(bundle_name))
    os.system('curl --progress-bar -u "H7FdjCcJcaZJSzN:" -H "X-Requested-With: XMLHttpRequest" "https://cloud.iac.es/public.php/webdav/{bundle_name}.tgz" -o {local_bundle_name}'.format(bundle_name=bundle_name, local_bundle_name=local_bundle_name))

    # check that the bundle exists
    if os.path.getsize(local_bundle_name) < 10000:
        os.remove(local_bundle_name)
        raise FileNotFoundError('The bundle "{bundle_name}" does not exist on the cloud.')

    # unpack the bundle
    os.system('tar -xzf {local_bundle_name} -C {bundle_path}'.format(local_bundle_name=local_bundle_name, bundle_path=bundle_path))

    # clean up
    os.remove(local_bundle_name)

    print('Bundle {} installed successfully at {}!'.format(bundle_name, bundle_path))


def set_nn_bundle_path(obj, nn_bundle_path):
    """
    Set the path to the neural net bundle files.

    Parameters:
    nn_bundle_path (str): The path to the bundle files.
    """
    try:
        sys.path.append(nn_bundle_path)
        import specfann_bundle_functions as sbf

        obj.sbf = sbf

        obj.parameters = obj.sbf.update_parameters(obj.parameters)

        obj.nn_model_string = obj.sbf.nn_model_string
        obj.nn_wavelength_string = obj.sbf.nn_wavelength_string

        obj.mean, obj.std = np.loadtxt(obj.nn_bundle_path + 'norm_array_fw.txt')
    except ImportError:
        print(f"Could not import specfann bundle functions from {nn_bundle_path}. Please check that the bundle is installed and the path to the bundle is properly set. For more information, please see setup instructions at https://github.com/MichaelAbdul-Masih/SpecFANN")



# -------------------------------------------------------------------
# -----------------------IO data functions---------------------------
# -------------------------------------------------------------------


def _calc_snr(wavelength, flux, region=[4220, 4240]):
    w = [i for i in wavelength if region[0] <= i <= region[1]]
    f = [flux[i] for i,j  in enumerate(list(wavelength)) if region[0] <= j <= region[1]]
    std = np.std(f)
    snr = np.mean(f)/std
    return snr


def load_observed_data(observed_wavelength, observed_flux, observed_error=None, snr_region=[4220, 4240]):
    """
    Load the observed data into the class.

    Parameters:
    observed_wavelength (array-like): The observed wavelengths in Angstroms.
    observed_flux (array-like): The observed flux values corresponding to the wavelengths.
    observed_error (array-like, optional): The errors in the observed flux values. If not provided, errors will be estimated based on the SNR of the data.
    snr_region (list, optional): The wavelength region to use for SNR calculation. Default is [4220, 4240].
    """

    # check that the shapes of the observed wavelength and flux are the same
    if len(observed_wavelength) != len(observed_flux):
        raise ValueError("Observed wavelength and flux must have the same length.")

    # check for nans and only keep non-nan values
    inds = np.where(~np.isnan(observed_flux) & ~np.isnan(observed_wavelength))[0]

    wavelength = observed_wavelength[inds]
    flux = observed_flux[inds]
    if observed_error is None:
        error = np.ones(len(observed_flux)) * 1/_calc_snr(observed_wavelength, observed_flux, region=snr_region)
    else:
        if len(observed_wavelength) != len(observed_flux):
            raise ValueError("Observed flux and error must have the same length.")
        error = observed_error[inds]
    
    return wavelength, flux, error


def save(obj, filename):
    '''
    Save the current state in a pickle file

    Parameters:
    filename (str): The name of the file that the bundle will be saved to
    '''
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def open_project(filename):
    """
    Open a project file and return the SpecFANN object.

    Parameters:
    filename (str): The name of the project file to open.

    Returns:
    obj (object): The SpecFANN object containing the loaded data.
    """

    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj

