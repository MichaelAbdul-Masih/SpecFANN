import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Suppress TensorFlow warnings
import numpy as np
import keras
import emcee
import matplotlib.pyplot as plt
import scipy.interpolate as si
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from scipy import stats
import corner
# import pickle
import dill as pickle
import pyGA as GA
from tqdm import trange
import importlib.util
import sys
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings (again)


def rotational_broadening(wave_spec,flux_spec,vrot,fwhm=0.25,epsilon=0.6, return_k = False):
    if fwhm>0:
        #   convert fwhm to one sigma
        fwhm /= 2.3548
        #-- make sure it's equidistant
        # wave_ = np.linspace(wave_spec[0],wave_spec[-1],len(wave_spec))
        # flux_ = np.interp(wave_,wave_spec,flux_spec)
        # dwave = wave_[1]-wave_[0]
        dwave = wave_spec[1]-wave_spec[0]
        n = int(round(2*4*fwhm/dwave, 0))
        wave_k = np.arange(n)*dwave
        wave_k-= wave_k[-1]/2.
        kernel = np.exp(- (wave_k)**2/(2*fwhm**2))
        kernel /= sum(kernel)
        # flux_conv = fftconvolve(1-flux_,kernel,mode='same')
        flux_conv = fftconvolve(1-flux_spec,kernel,mode='same')
        # flux_conv = np.convolve(1-flux_spec,kernel,mode='same')
        # flux_spec = np.interp(wave_spec+dwave/2,wave_,1-flux_conv,left=1,right=1)
        flux_spec = np.interp(wave_spec+dwave/2,wave_spec,1-flux_conv,left=1,right=1)

    if vrot>0:
        #-- convert wavelength array into velocity space, this is easier
        #   we also need to make it equidistant!
        wave_ = np.log(wave_spec)
        try:
            velo_ = np.linspace(wave_[0],wave_[-1],len(wave_))
            flux_ = np.interp(velo_,wave_,flux_spec)
            dvelo = velo_[1]-velo_[0]
            vrot_new = vrot/(299792.458)
            #-- compute the convolution kernel and normalise it
            n = int(2*vrot_new/dvelo)
            velo_k = np.arange(n)*dvelo
            velo_k -= velo_k[-1]/2.
        except:
            vrot_new = vrot/(299792.458)
            velo1 = np.linspace(wave_[0],wave_[-1],len(wave_))
            n = int((velo1[1] - velo1[0])*10/(vrot_new)) * len(wave_)
            velo_ = np.linspace(wave_[0],wave_[-1],n)
            flux_ = np.interp(velo_,wave_,flux_spec)
            dvelo = velo_[1]-velo_[0]
            n = int(2*vrot_new/dvelo)
            velo_k = np.arange(n)*dvelo
            velo_k -= velo_k[-1]/2.

        # velo_k -= velo_k[-1]/2.
        y = 1 - (velo_k/vrot_new)**2 # transformation of velocity
        K = (2*(1-epsilon)*np.sqrt(y)+np.pi*epsilon/2.*y)/(np.pi*vrot_new*(1-epsilon/3.0))  # the kernel
        K /= K.sum()
        if return_k:
            return y, K
        #-- convolve the flux with the kernel
        flux_conv = fftconvolve(1-flux_,K,mode='same')
        # flux_conv = np.convolve(1-flux_,K,mode='same')
        velo_ = np.arange(len(flux_conv))*dvelo+velo_[0]
        wave_conv = np.exp(velo_)
        return wave_spec, np.interp(wave_spec, wave_conv, 1-flux_conv, right=1, left=1)
    return wave_spec,flux_spec


def open_project(filename, bundle_path=None, bundle_name=None):
    """
    Open a project file and return the fwnnfit object.

    Parameters:
    filename (str): The name of the project file to open.
    bundle_path (str): The path to the neural net bundle files. If None, uses the default path.
    bundle_name (str): The name of the neural net bundle. If None, uses the default bundle name.

    Returns:
    fwnnfit (fwnnfit): The fwnnfit object containing the loaded data.
    """

    if bundle_path is None:
        if bundle_name is None:
            bundle_path = os.path.join(os.path.dirname(__file__), 'bundles/MW_v1.0/')
        else:
            bundle_path = os.path.join(os.path.dirname(__file__), 'bundles/%s/' % bundle_name)
    sys.path.append(bundle_path)

    with open(filename, 'rb') as inp:
        fwnnfit = pickle.load(inp)
    return fwnnfit


def import_from_path(module_name, file_path):
    """
    Import a module from a specific file path.
    
    Parameters:
    module_name (str): The name of the module to import.
    file_path (str): The path to the module file.

    Returns:
    module (module): The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class parameters(object):
    """
    Class to hold the parameters for the model
    """

    def __init__(self, teff = 40000, logg = 4.0, r = 7, he = 0.1, c = 7.5, n = 7.5, o = 7.5, si = 7.5, vrot = 0, gamma = 0):
        """
        Initialize the parameters object.
        Parameters:
        teff (float): Effective temperature in K.
        logg (float): Log of the surface gravity.
        r (float): Radius in solar radii.
        he (float): Helium abundance. (N_He/N_H)
        c (float): Carbon abundance. (log(N_x/H_H)+12)
        n (float): Nitrogen abundance. (log(N_x/H_H)+12)
        o (float): Oxygen abundance. (log(N_x/H_H)+12)
        si (float): Silicon abundance. (log(N_x/H_H)+12)
        vrot (float): Rotational velocity in km/s.
        gamma (float): Radial velocity in km/s.
        """
        
        self.teff = self.parameter('teff', teff, bounds=[15000, 60000])
        self.logg = self.parameter('logg', logg, bounds=[2.0, 4.5])
        self.r = self.parameter('r', r, bounds=[5, 30])
        self.he = self.parameter('he', he, bounds=[0.06, 0.3])
        self.c = self.parameter('c', c, bounds=[6.0, 9.0])
        self.n = self.parameter('n', n, bounds=[6.0, 9.0])
        self.o = self.parameter('o', o, bounds=[6.0, 9.0])
        self.si = self.parameter('si', si, bounds=[6.0, 9.0])
        self.vrot = self.parameter('vrot', vrot, bounds=[0, 500])
        self.gamma = self.parameter('gamma', gamma, bounds=[-500, 500])
        self.logf = self.parameter('logf', 0.0, bounds=[-10, 10])  # log of the variance scaling factor
    
    def summary(self):
        """
        Print a summary of the parameters.
        """
        print("Parameters:")
        for param in self.__dict__.values():
            if isinstance(param, self.parameter):
                print(f"{param.name}: {param.value} (fixed: {param.fixed}, bounds: {param.bounds})")

    class parameter(object):
        """
        Class to hold individual parameters
        """

        def __init__(self, name, value, bounds=None, fixed=False):
            """
            Initialize the parameter object.
            Parameters:
            name (str): The name of the parameter.
            value (float): The value of the parameter.
            bounds (list): A list containing the lower and upper bounds for the parameter.
            """
            self.name = name
            self.value = value
            self.fixed = fixed
            self.bounds = bounds if bounds is not None else [None, None]
        
        def fix(self, value = None):
            """
            Fix the parameter to a specific value.
            Parameters:
            value (float): The value to fix the parameter to.
            """
            if value is not None:
                self.value = value
            self.fixed = True
        
        def free(self):
            """
            Free the parameter from its fixed value.
            """
            self.fixed = False
        
        def set_bounds(self, bounds):
            """
            Set the bounds for the parameter.
            Parameters:
            bounds (list): A list containing the lower and upper bounds for the parameter.
            """
            self.bounds = bounds
        



class line_to_fit(object):
    """
    Class to hold information about the lines to be fitted
    """

    def __init__(self, line, nn_path='bundles/', nn_model_string = 'fluxes_$LINE$_model.keras', nn_wavelength_string = 'wnew_$LINE$.npy', fit_range=None):
        """
        Initialize the line_to_fit object.
        Parameters:
        line (str): The name of the line to be fitted.
        fit_range (list): The range of wavelengths to be fitted.
        """

        self.line_name = line
        model_filename = '/'.join([nn_path, nn_model_string.replace('$LINE$', line)])
        wavelength_filename = '/'.join([nn_path, nn_wavelength_string.replace('$LINE$', line)])
        # self.model = keras.saving.load_model('bundles/fluxes_%s_model.keras' %line)
        # self.wavelength = np.load('bundles/wnew_%s.npy'%line)
        self.model = keras.saving.load_model(model_filename)
        self.wavelength = np.load(wavelength_filename)
        if fit_range is None:
            total_range = np.max(self.wavelength) - np.min(self.wavelength)
            self.fit_range = [np.min(self.wavelength) + total_range/4, np.max(self.wavelength) - total_range/4]
        else:
            self.fit_range = fit_range


class specfann(object):
    """
    Class to fit the fluxes using a neural network
    """

    def __init__(self, bundle_path=None, bundle_name=None):

        self.observed_wavelength = None
        self.observed_flux = None
        self.observed_error = None

        self.parameters = parameters()

        if bundle_path is None:
            if bundle_name is None:
                bundle_path = os.path.join(os.path.dirname(__file__), 'bundles/MW_v1.0/')
            else:
                bundle_path = os.path.join(os.path.dirname(__file__), 'bundles/%s/' % bundle_name)
        self.set_nn_bundle_path(bundle_path)

        self.line_list = {}

        self.free_parameters = [param for param in self.parameters.__dict__ if not self.parameters.__dict__[param].fixed]

        self.n_walkers = 150
        self.n_steps = 1000

        self.object_name = None

    # ----------------------IO----------------------


    def _calc_snr(self, wavelength, flux, region=[4220, 4240]):
        w = [i for i in wavelength if region[0] <= i <= region[1]]
        f = [flux[i] for i,j  in enumerate(list(wavelength)) if region[0] <= j <= region[1]]
        std = np.std(f)
        snr = np.mean(f)/std
        return snr


    def load_observed_data(self, observed_wavelength, observed_flux, observed_error=None):
        """
        Load the observed data into the class.

        Parameters:
        observed_wavelength (array-like): The observed wavelengths in Angstroms.
        observed_flux (array-like): The observed flux values corresponding to the wavelengths.
        """

        # check that the shapes of the observed wavelength and flux are the same
        if len(observed_wavelength) != len(observed_flux):
            raise ValueError("Observed wavelength and flux must have the same length.")

        # check for nans and only keep non-nan values
        inds = np.where(~np.isnan(observed_flux) & ~np.isnan(observed_wavelength))[0]

        self.observed_wavelength = observed_wavelength[inds]
        self.observed_flux = observed_flux[inds]
        if observed_error is None:
            self.observed_error = np.ones(len(observed_flux)) * 1/self._calc_snr(observed_wavelength, observed_flux)
        else:
            self.observed_error = observed_error[inds]
    

    def set_nn_bundle_path(self, nn_bundle_path):
        """
        Set the path to the neural net bundle files.

        Parameters:
        bundle_path (str): The path to the bundle files.
        """
        self.nn_bundle_path = nn_bundle_path
        try:
            sys.path.append(nn_bundle_path)
            import specfann_bundle_functions as sbf
            # sgf = import_from_path('__main__', nn_bundle_path)
            self.sbf = sbf

            self.parameters = self.sbf.update_parameters(self.parameters)
            
            self.nn_model_string = self.sbf.nn_model_string
            self.nn_wavelength_string = self.sbf.nn_wavelength_string

            self.mean, self.std = np.loadtxt(self.nn_bundle_path + 'norm_array_fw.txt')
        except ImportError:
            if nn_bundle_path == 'bundles/MW_v1.0/':
                print("No specfann bundle found in the default relative path 'bundles/MW_v1.0/'. Please ensure you have downloaded the bundle and in the correct place.  For more information, please see setup instructions at https://github.com/MichaelAbdul-Masih/SpecFANN")
            else:
                print(f"Could not import specfann bundle functions from {nn_bundle_path}. Please check the path to the bundle is properly set.")


    def add_line(self, line, fit_range=None):
        """
        Add a line to the list of lines to be fitted.

        Parameters:
        line (str): The name of the line to be fitted.
        fit_range (list): The range of wavelengths to be fitted for the line.
        """
        if line not in self.line_list.keys():
            self.line_list[line] = line_to_fit(line, nn_path=self.nn_bundle_path, nn_model_string=self.nn_model_string, nn_wavelength_string=self.nn_wavelength_string, fit_range=fit_range)
    

    def remove_line(self, line):
        """
        Remove a line from the list of lines to be fitted.

        Parameters:
        line (str): The name of the line to be removed.
        """
        if line in self.line_list.keys():
            del self.line_list[line]
        else:
            print(f"Line {line} not found in the list of lines to be fitted.")


    def load_line_models(self):
        """
        Load the line models into the class.
        """
        self.line_models = {}
        for line in self.line_list:
            try:
                self.line_models[line] = keras.saving.load_model('bundles/fluxes_%s_model.keras' %line)
            except:
                print('Model for %s not found' %line)
                self.line_models[line] = None
    

    def save(self, filename):
        '''
        Save the current state in a pickle file

        Parameters:
        filename (str): The name of the file that the bundle will be saved to
        '''
        with open(filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    

    # -------------------Model Generation functions--------------------
    def generate_model(self, param_set):
        """
        Generate a model based on the provided parameters.

        Parameters:
        param_set (array-like): The parameters for the model.

        Returns:
        models (dict): A dictionary of models for each line.
        """

        param_set = np.array(param_set, ndmin=2)

        models = {}
        for line in self.line_list.keys():
            # Generate the model for each line
            wavelengths, fluxes = self.generate_model_per_line(line, param_set)
            models[line] = {'wavelengths': wavelengths, 'fluxes': fluxes}

        return models


    def generate_model_per_line(self, line, param_set):
        """
        Generate a model based on the provided parameters.

        Parameters:
        param_set (array-like): The parameters for the model.

        Returns:
        models (dict): A dictionary of models for each line.
        """

        vrot_ind = list(self.parameters.__dict__.keys()).index('vrot')
        gamma_ind = list(self.parameters.__dict__.keys()).index('gamma')

        if self.sbf.use_specfann_broadening:
            # Use the neural network to predict the fluxes for the line
            model_fluxes = self.sbf.predict_fluxes_from_nn(self.parameters, self.line_list, line, param_set)
            # Broaden the lines
            broadened_wavelength, broadened_fluxes = self.broaden_lines(line, model_fluxes, param_set[:, vrot_ind])
        else:
            # Use the neural network to predict the fluxes for the line and apply broadening
            broadened_wavelength, broadened_fluxes = self.sbf.predict_fluxes_from_nn(self.parameters, self.line_list, line, param_set)
        
        # Doppler shift the lines
        shifted_wavelengths = self.dopler_shift_lines(broadened_wavelength, param_set[:, gamma_ind])

        return shifted_wavelengths, broadened_fluxes


    def predict_fluxes_from_nn(self, line, param_set):
        """
        Calculate the models for each line using the neural network.

        Parameters:
        line (str): The name of the line to be fitted.
        param_set (list): The parameters for the model.

        Returns:
        models (dict): A dictionary of models for each line.
        """

        param_set = np.array(param_set, ndmin=2)

        teff_ind = list(self.parameters.__dict__.keys()).index('teff')
        logg_ind = list(self.parameters.__dict__.keys()).index('logg')
        r_ind = list(self.parameters.__dict__.keys()).index('r')
        he_ind = list(self.parameters.__dict__.keys()).index('he')
        c_ind = list(self.parameters.__dict__.keys()).index('c')
        n_ind = list(self.parameters.__dict__.keys()).index('n')
        o_ind = list(self.parameters.__dict__.keys()).index('o')
        si_ind = list(self.parameters.__dict__.keys()).index('si')

        teff = param_set[:, teff_ind]
        logg = param_set[:, logg_ind]
        r = param_set[:, r_ind]
        he = param_set[:, he_ind]
        c = param_set[:, c_ind]
        n = param_set[:, n_ind]
        o = param_set[:, o_ind]
        si = param_set[:, si_ind]

        if line.startswith('H'):
            if line in ['HGAMMA', 'HEI4121']:
                fit_array = np.array([teff, logg, r, he, si]).T
            else:
                fit_array = np.array([teff, logg, r, he, n]).T

        elif line.startswith('C'):
            fit_array = np.array([teff, logg, r, he, c]).T

        elif line.startswith('N'):
            fit_array = np.array([teff, logg, r, he, n]).T

        elif line.startswith('O'):
            fit_array = np.array([teff, logg, r, he, o]).T

        elif line.startswith('S'):
            fit_array = np.array([teff, logg, r, he, si]).T

        fit_array -= self.mean
        fit_array /= self.std

        # if len(fit_array.shape) == 1:
        #     fit_array = np.expand_dims(fit_array, axis=0)

        predicted_fluxes = self.line_list[line].model.predict(np.array(fit_array, ndmin=2), verbose=0)

        return predicted_fluxes
    

    def broaden_lines(self, line, fluxes, vrot):
        """
        Broaden the spectral lines using rotational broadening.

        Parameters:
        line (str): The name of the line to be broadened.
        fluxes (array-like): The flux values corresponding to the line.
        vrot (float): The rotational velocity to apply for broadening.

        Returns:
        broadened_wavelength (array-like): The broadened wavelength array.
        broadened_fluxes (list of array-like): The broadened flux arrays for each input flux.
        """

        wavelength = self.line_list[line].wavelength
        new_wavelength = np.arange(wavelength[0], wavelength[-1], 0.01)
        unbroadened_fluxes = si.interp1d(wavelength, fluxes, bounds_error=False, fill_value=(1.0,1.0))(new_wavelength)

        broadened_wavelengths, broadened_fluxes = [],[]
        for i in range(len(unbroadened_fluxes)):
            broadened_wavelength, broadened_flux = rotational_broadening(new_wavelength, unbroadened_fluxes[i], vrot[i])
            broadened_wavelengths.append(broadened_wavelength)
            broadened_fluxes.append(broadened_flux)

        return np.array(broadened_wavelength), np.array(broadened_fluxes)


    def dopler_shift_lines(self, wavelengths, rv):
        """
        Apply a Doppler shift to the wavelengths.

        Parameters:
        wavelengths (array-like): The wavelengths to be shifted.
        rv (array-like): The radial velocity in km/s.

        Returns:
        shifted_wavelengths (array-like): The shifted wavelengths.
        """

        c = 299792.458
        return wavelengths*c/(c-rv[:, None])
    

    def interp_model_lines_to_observed(self, observed_wavelength, model_wavelengths, model_fluxes):
        """
        Interpolate the model lines to the observed wavelengths.

        Parameters:
        observed_wavelength (array-like): The observed wavelength array.
        model_wavelengths (array-like): The wavelengths of the model lines.
        model_fluxes (array-like): The fluxes of the model lines.

        Returns:
        interpolated_fluxes (array-like): The interpolated fluxes at the observed wavelengths.
        """

        interpolated_fluxes = []
        for i in range(len(model_fluxes)):
            interpolated_fluxes.append(np.interp(observed_wavelength, model_wavelengths[i], model_fluxes[i], left=1.0, right=1.0))

        return np.array(interpolated_fluxes)


    def parse_parameter_set(self, model_args):
        """
        Parse the model arguments to return full parameter set including fixed params.

        Parameters:
        model_args (array-like): The sampled values of the free parameters.

        Returns:
        parameter_set (array-like): Full parameter set.
        """

        params = list(self.parameters.__dict__.keys())
        parameter_set = []
        for param in params:
            if param in self.free_parameters:
                parameter_set.append(np.array(model_args.T[self.free_parameters.index(param)], ndmin=1))
            else:
                parameter_set.append([self.parameters.__dict__[param].value]*len(np.array(model_args, ndmin=2)))

        return np.array(parameter_set).T
    

    # -------------------Cost functions--------------------


    def calc_log_likelihoods(self, data, error, model):
        data = np.array(data)
        error = np.array(error)
        model = np.array(model)

        log_liklihoods = np.sum(-0.5 * ((data - model)**2 / error**2 + np.log(2*np.pi * error**2)), axis=-1)
        return log_liklihoods


    def calc_chi_square(self, data, error, model):
        data = np.array(data)
        error = np.array(error)
        model = np.array(model)

        chi_squares = np.sum(((data - model)**2 / error**2), axis=-1)
        return chi_squares


    def log_likelihood(self, param_set, jitter=False):
        """
        Calculate the log likelihood of the model given the observed data.

        Parameters:
        param_set (array-like): The full parameter set including free and fixed parameters.

        Returns:
        log_likelihoods (array-like): The log likelihoods for each model.
        """

        param_set = np.array(param_set, ndmin=2)

        log_likelihoods = np.zeros(len(param_set))
        for line in self.line_list.keys():
            # Get the model wavelengths and fluxes
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, param_set)

            # Interpolate the model lines to the observed wavelengths
            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)

            # Calculate the log likelihood
            if jitter:
                logf_ind = list(self.parameters.__dict__.keys()).index('logf')
                logf = param_set[:, logf_ind]
                error = np.sqrt(self.observed_error[obs_inds] **2 + np.array(10**logf, ndmin=2).T * interpolated_fluxes**2)
                log_likelihoods += self.calc_log_likelihoods(self.observed_flux[obs_inds], error, interpolated_fluxes)
            else:
                log_likelihoods += self.calc_log_likelihoods(self.observed_flux[obs_inds], self.observed_error[obs_inds], interpolated_fluxes)

        return log_likelihoods


    def log_prior(self, param_set):
        """
        Calculate the log prior for the model parameters.

        Parameters:
        param_set (array-like): The full parameter set including free and fixed parameters.

        Returns:
        log_prior (array-like): The log prior values. Returns 0.0 if all parameters are within bounds, otherwise -inf.
        """

        param_set = np.array(param_set, ndmin=2)
        prior_array = np.zeros(np.array(param_set, ndmin=2).shape[0])

        for param in self.free_parameters:
            param_obj = self.parameters.__dict__[param]
            param_ind = list(self.parameters.__dict__.keys()).index(param)
            if not param_obj.fixed:
                prior_array += np.where(np.logical_and(param_set[:, param_ind] >= param_obj.bounds[0], 
                                                        param_set[:, param_ind] <= param_obj.bounds[1]), 0, -np.inf)


        prior_array = self.sbf.update_priors(self.parameters, param_set, prior_array)

        # # Additional priors to account for our specific bundle
        # teff_ind = list(self.parameters.__dict__.keys()).index('teff')
        # logg_ind = list(self.parameters.__dict__.keys()).index('logg')
        # r_ind = list(self.parameters.__dict__.keys()).index('r')

        # teff = param_set[:, teff_ind]
        # logg = param_set[:, logg_ind]
        # r = param_set[:, r_ind]

        # prior_array += np.where(logg >= 2/45000*teff + 4/3., 0, -np.inf)
        # prior_array += np.where(r <= -0.001*teff + 65., 0, -np.inf)

        
        return prior_array


    def log_probability(self, model_args, jitter=False):

        param_set = self.parse_parameter_set(model_args)

        lp = self.log_prior(param_set)

        return lp + self.log_likelihood(param_set, jitter=jitter)


    def reduced_chi_square(self, model_args, jitter=False):
        """
        Calculate the reduced chi-squared statistic for the model parameters.

        Parameters:
        model_args (array-like): The full parameter set including free and fixed parameters.
        jitter (bool): Whether to include jitter in the likelihood calculation.

        Returns:
        chi_squared (float): The chi-squared statistic.
        """

        param_set = self.parse_parameter_set(model_args)
        param_set = np.array(param_set, ndmin=2)

        chi_squares = np.zeros(len(param_set))
        reduced_chi_squares = np.zeros(len(param_set))
        for line in self.line_list.keys():
            # Get the model wavelengths and fluxes
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, param_set)

            # Interpolate the model lines to the observed wavelengths
            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)

            # Calculate the log likelihood
            if jitter:
                logf_ind = list(self.parameters.__dict__.keys()).index('logf')
                logf = param_set[:, logf_ind]
                error = np.sqrt(self.observed_error[obs_inds] **2 + np.array(10**logf, ndmin=2).T * interpolated_fluxes**2)
                chi_squares += self.calc_chi_square(self.observed_flux[obs_inds], error, interpolated_fluxes)
            else:
                chi_squares += self.calc_chi_square(self.observed_flux[obs_inds], self.observed_error[obs_inds], interpolated_fluxes)
            
            reduced_chi_squares += chi_squares / (len(obs_inds) - len(self.free_parameters))

        return reduced_chi_squares

    # -------------------MCMC functions--------------------


    def run_mcmc(self, initial_positions=None, n_walkers=None, n_steps=None, jitter=False, return_sampler=False):
        """
        Run the MCMC simulation to sample the parameter space.

        Parameters:
        initial_positions (array-like): Initial positions of the walkers in the parameter space.
        n_walkers (int): The number of walkers to use in the MCMC simulation.
        n_steps (int): The number of steps to run the MCMC simulation for.
        """

        if n_walkers is None:
            n_walkers = self.n_walkers
        if n_steps is None:
            n_steps = self.n_steps

        if jitter:
            print("Using jitter in the likelihood calculation.")
            self.parameters.logf.free()
        else:
            self.parameters.logf.fix(0.0)
        
        # reinitialize the free parameters array to catch any changed parameters
        self.free_parameters = [param for param in self.parameters.__dict__ if not self.parameters.__dict__[param].fixed]
        self.mcmc_free_parameters = self.free_parameters.copy()


        # Initialize the walkers if not passed to the function
        if initial_positions is None:
            initial_positions = []
            for param in self.free_parameters:
                bounds = self.parameters.__dict__[param].bounds
                initial_positions.append(np.random.uniform(bounds[0], bounds[1], n_walkers))
        
            initial_positions = np.array(initial_positions).T
        else:
            initial_positions = initial_positions + 1e-4 * np.random.randn(n_walkers, len(self.free_parameters))

        # Create the sampler
        sampler = emcee.EnsembleSampler(n_walkers, len(self.free_parameters), self.log_probability, args=(jitter,), vectorize=True)

        # Run the MCMC simulation
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        self.emcee_sampler = sampler

        if return_sampler:
            return sampler


    def plot_MCMC_results(self, sampler = None, burnin=100, thin=1):
        """
        Plot the results of the MCMC simulation.

        Parameters:
        samples (array-like): The samples from the MCMC simulation.
        """

        if sampler is None:
            if not hasattr(self, 'emcee_sampler'):
                raise ValueError("No MCMC sampler found. Run run_mcmc() first.")
            sampler = self.emcee_sampler
        
        samples = sampler.get_chain(discard=burnin)
        
        fig, axs = plt.subplots(len(self.mcmc_free_parameters), figsize=(10, 7), sharex=True)
        for i, param in enumerate(self.mcmc_free_parameters):
            axs[i].plot(samples[:, :, i], "k", alpha=0.3)
            axs[i].set_xlim(0, len(samples))
            axs[i].set_ylabel(param)
        
        plt.show()

        flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

        fig = corner.corner(flat_samples, labels=self.mcmc_free_parameters, show_titles=True)        
        plt.show()
    

    def plot_MCMC_fit(self, sampler = None, burnin=100, save_path=None):
        """
        Plot the MCMC fit results against the observed data.

        Parameters:
        samples (array-like): The samples from the MCMC simulation.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
        """

        if sampler is None:
            if not hasattr(self, 'emcee_sampler'):
                raise ValueError("No MCMC sampler found. Run run_mcmc() first.")
            sampler = self.emcee_sampler

        chains = sampler.get_chain(flat=True, thin=1, discard=burnin)

        inds = np.random.randint(len(chains), size=1000)
        model_args = chains[inds]
        param_set = self.parse_parameter_set(model_args)

        vrot_ind = list(self.parameters.__dict__.keys()).index('vrot')
        gamma_ind = list(self.parameters.__dict__.keys()).index('gamma')

        subplots_dict = {1:[1, 1], 2:[1, 2], 3:[1,3], 4:[2, 2], 5:[2, 3], 6:[2,3], 7:[2,4], 8:[2,4], 9:[3,3], 10:[3, 4], 11:[3, 4], 12:[3, 4], 13:[3, 5], 14:[3, 5], 15:[3, 5], 16:[4,4]}
        fig, axs = plt.subplots(subplots_dict[len(self.line_list)][0], subplots_dict[len(self.line_list)][1], figsize=(subplots_dict[len(self.line_list)][1]*4, subplots_dict[len(self.line_list)][0]*4))
        axs = axs.ravel()

        for i, line in enumerate(self.line_list.keys()):
            # model_fluxes = self.predict_fluxes_from_nn(line, np.array(param_set, ndmin=2))
            # broadened_wavelength, broadened_fluxes = self.broaden_lines(line, model_fluxes, np.array(param_set[:, vrot_ind], ndmin=1))
            # shifted_wavelengths = self.dopler_shift_lines(broadened_wavelength, np.array(param_set[:,gamma_ind], ndmin=1))
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, np.array(param_set, ndmin=2))

            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)
            # interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, shifted_wavelengths, broadened_fluxes)

            model_mean = np.array(interpolated_fluxes).mean(axis=0)
            model_std = np.array(interpolated_fluxes).std(axis=0)

            axs[i].plot(obs_wavelength, self.observed_flux[obs_inds], 'k-', label='Observed')
            axs[i].plot(obs_wavelength, model_mean, 'r-', label='Best Fit')
            axs[i].fill_between(obs_wavelength, model_mean-model_std, model_mean+model_std, color='lightcoral', alpha=0.8, label='1-sigma')
            axs[i].set_xlabel('Wavelength (Angstrom)')
            axs[i].set_ylabel('Flux')

        plt.show()

    
    def print_MCMC_results(self, sampler=None, burnin=100):
        """
        Print the results of the MCMC simulation.

        Parameters:
        samples (array-like): The samples from the MCMC simulation.
        burnin (int): The number of steps to discard as burn-in.
        """

        if sampler is None:
            if not hasattr(self, 'emcee_sampler'):
                raise ValueError("No MCMC sampler found. Run run_mcmc() first.")
            sampler = self.emcee_sampler

        chains = sampler.get_chain(flat=True, thin=1, discard=burnin)

        print("MCMC Results:")
        for i, param in enumerate(self.mcmc_free_parameters):
            mcmc = np.percentile(chains[:, i], [16, 50, 84])
            errors = np.diff(mcmc)
            print(f"{param} = ".rjust(15) + f" {mcmc[1]}  ( +{errors[1]:.5f}; -{errors[0]:.5f})")
            # print(f"{param}: {mcmc[1]:.4f} ± {std:.4f}")
        print(f"Number of iterations: {sampler.get_chain().shape[0]}")
        print(f"Number of walkers: {sampler.get_chain().shape[1]}")
    # -------------------Nelder-Mead functions--------------------


    def run_Nelder_Mead(self, initial_guess=None, return_result = False):
        """
        Run the Nelder-Mead optimization algorithm to find the best-fit parameters.

        Parameters:
        initial_guess (array-like): Initial guess for the parameters.
        return_result (bool): Whether to explicitly return the optimization result.

        Returns:
        result (OptimizeResult): The optimization result represented as a `OptimizeResult` object.
        """

        if initial_guess is None:
            initial_guess = [self.parameters.__dict__[param].value for param in self.free_parameters]
        
        nll = lambda *args: -self.log_probability(*args)[0]

        result = minimize(nll, initial_guess, method='Nelder-Mead')
        print(result)

        if result.success:
            self.nm_solution = result.x
        
        if return_result:
            return result

    def plot_best_fit(self, model_args=None, save_path=None):
        """
        Plot the best-fit model against the observed data.

        Parameters:
        model_args (array-like): The best-fit parameters.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
        """
        if model_args is None:
            if not hasattr(self, 'nm_solution'):
                raise ValueError("No best-fit parameters found. Run run_Nelder_Mead() first.")
            model_args = self.nm_solution
        best_fit_params = self.parse_parameter_set(model_args)[0]

        vrot_ind = list(self.parameters.__dict__.keys()).index('vrot')
        gamma_ind = list(self.parameters.__dict__.keys()).index('gamma')

        subplots_dict = {1:[1, 1], 2:[1, 2], 3:[1,3], 4:[2, 2], 5:[2, 3], 6:[2,3], 7:[2,4], 8:[2,4], 9:[3,3], 10:[3, 4], 11:[3, 4], 12:[3, 4], 13:[3, 5], 14:[3, 5], 15:[3, 5], 16:[4,4]}
        fig, axs = plt.subplots(subplots_dict[len(self.line_list)][0], subplots_dict[len(self.line_list)][1], figsize=(subplots_dict[len(self.line_list)][1]*4, subplots_dict[len(self.line_list)][0]*4))
        axs = axs.ravel()

        for i, line in enumerate(self.line_list.keys()):
            # model_fluxes = self.predict_fluxes_from_nn(line, np.array(best_fit_params, ndmin=2))
            # broadened_wavelength, broadened_fluxes = self.broaden_lines(line, model_fluxes, np.array(best_fit_params[vrot_ind], ndmin=1))
            # shifted_wavelengths = self.dopler_shift_lines(broadened_wavelength, np.array(best_fit_params[gamma_ind], ndmin=1))
            # Get the model wavelengths and fluxes
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, np.array(best_fit_params, ndmin=2))

            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)
            # interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, shifted_wavelengths, broadened_fluxes)
            axs[i].plot(obs_wavelength, self.observed_flux[obs_inds], 'k-', label='Observed')
            axs[i].plot(obs_wavelength, interpolated_fluxes.T, 'r-', label='Best Fit')
            axs[i].set_xlabel('Wavelength (Angstrom)')
            axs[i].set_ylabel('Flux')
        
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.show()
    

    # -------------------GA functions--------------------


    class ga_result_summary(object):


        def __init__(self, ga_params, population_size, generations, chi2s, fitnesses, probabilities, populations, best_model, best_fitness, free_parameters):
            self.ga_params = ga_params
            self.population_size = population_size
            self.n_generations = generations
            self.reduced_chi_squares = np.array(chi2s)
            self.fitnesses = np.array(fitnesses)
            self.probabilities = np.array(probabilities)
            self.populations = np.array(populations)

            best_mod = []
            error_ranges = []
            probs = self.probabilities.flatten()
            inds = np.where(probs > 0.05)[0]
            self.free_parameters = free_parameters
            for i, param in enumerate(free_parameters):
                best_mod.append(best_model[param])
                param_values = self.populations[:, :, i].flatten()
                param_values = param_values[inds]
                error_ranges.append([np.min(param_values), np.max(param_values)])

            self.best_fit_model = best_mod
            self.best_fit_errors = error_ranges


            self.best_fitness = best_fitness


    def translate_params_to_GA(self):
        """
        Translate the parameters to a format suitable for the genetic algorithm.

        Returns:
        ga_params (array-like): The parameters in the format suitable for the genetic algorithm.
        """
        ga_params = GA.Parameters()
        for param in self.free_parameters:
            name = self.parameters.__dict__[param].name
            bounds = self.parameters.__dict__[param].bounds
            ga_params.add(name, float(bounds[0]), float(bounds[1]), int(6))

        return ga_params
    

    def translate_GA_chromosomes(self, ga_params, chromosome_list):
        """
        Translate the raw GA chromosomes back into the parameter set format used by fwnnfit.

        Parameters:
        ga_params (array-like): The parameters in the format suitable for the genetic algorithm.
        chromosome_list (array-like): The list of raw chromosomes used by the GA.

        Returns:
        ga_params (array-like): The parameters in the format suitable for genetic algorithms.
        """
        keys = list(ga_params.keys())
        keys = self.free_parameters
        model_args = []
        for chromosome in chromosome_list:
            model = []
            for i in keys:
                precision = ga_params[i].precision
                param_min = ga_params[i].min
                param_max = ga_params[i].max
                param_range = param_max - param_min
                value = float('0.' + chromosome[:precision]) * param_range + param_min
                chromosome = chromosome[precision:]
                model.append(value)
            model_args.append(model)
        return np.array(model_args)


    def run_GA(self, n_generations=300, population_size=50, return_result=False):
        """
        Run the genetic algorithm to find the best-fit parameters.

        Parameters:
        n_generations (int): The number of generations to run the genetic algorithm for.
        population_size (int): The number of individuals in the population.
        return_result (bool): Whether to return the result of the genetic algorithm.

        Returns:
        result (GA.Result): The result of the genetic algorithm.
        """

        # set the logf parameter to be fixed at 0.0
        self.parameters.logf.fix(0.0)
        # reinitialize the free parameters array to catch any changed parameters
        self.free_parameters = [param for param in self.parameters.__dict__ if not self.parameters.__dict__[param].fixed]
        
        # translate the parameters to a format suitable for the genetic algorithm
        ga_params = self.translate_params_to_GA()

        # create the initial population of chromosomes
        population_raw = GA.create_chromosome(ga_params, population_size)

        # initialize variables to keep track of the stats per generation
        best_fitness = -999999999
        generation_reduced_chi_squares = []
        generation_fitnesses = []
        generation_parameters = []

        mutation_rate = 0.05

        #Iteration loop to progress through generations of models.
        for generation in trange(n_generations, leave=True, desc='GA generations'):

            #Population is converted from raw chromosomes to input parameters useable by fwnnfit.
            population = GA.batch_translate_chromosomes(ga_params, population_raw, generation)

            model_args = self.translate_GA_chromosomes(ga_params, population_raw)
            generation_parameters.append(model_args)

            # calculate chi2 of each model in the population.
            reduced_chi_squares = self.reduced_chi_square(model_args)
            generation_reduced_chi_squares.append(reduced_chi_squares)

            # calculate fitness of each model in the population.
            fitnesses = len(self.line_list.keys()) / reduced_chi_squares
            generation_fitnesses.append(fitnesses)
            
            # check if best model has changed, if so update best model and best probability.  If not, replace the worst model in the population with the best model.
            if np.max(fitnesses) > best_fitness:
                best_fitness = np.max(fitnesses)
                best_mod = population[np.argmax(fitnesses)]
                best_mod_raw = population_raw[np.argmax(fitnesses)]
            elif best_mod_raw != population_raw[np.argmax(fitnesses)]:
                population_raw = np.delete(population_raw, np.argmin(fitnesses))
                fitnesses = np.delete(fitnesses, np.argmin(fitnesses))
                population_raw = np.append(population_raw, best_mod_raw)
                fitnesses = np.append(fitnesses, best_fitness)

            #With results of probabilities from previous generation the next generation is created.
            population_raw = GA.crossover_and_mutate_raw(population_raw, fitnesses, mutation_rate)
            #Mutuation rate is adjust based on mutation rate of previous generation, to maximise effectiveness of exploration.
            mutation_rate = GA.adjust_mutation_rate(mutation_rate, fitnesses, mut_rate_min = .005)
        
        generation_probabilities = self.calculate_GA_probabilities(np.array(generation_reduced_chi_squares))
        self.GA_results = self.ga_result_summary(ga_params, population_size, n_generations, generation_reduced_chi_squares, generation_fitnesses, generation_probabilities, generation_parameters, best_mod, best_fitness, self.free_parameters)
        # self.GA_results.probabilities = self.calculate_GA_probabilities(self.GA_results.reduced_chi_squares)

        if return_result:
            return self.GA_results


    def calculate_GA_probabilities(self, red_chi2s):
        """
        Calculate the probabilities of each model in the population based on their chi-squared values.

        Parameters:
        red_chi2s (array-like): The chi-squared values for each model in the population.

        Returns:
        probabilities (array-like): The probabilities of each model in the population.
        """

        # calculate degrees of freedom
        degrees_of_freedom = 0
        for line in self.line_list.keys():
            degrees_of_freedom += len(np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0])
        
        degrees_of_freedom -= len(self.free_parameters)

        # normalize chi-squared values
        chi2s = (red_chi2s * degrees_of_freedom) / np.min(red_chi2s)

        probabilities = stats.chi2.sf(chi2s, degrees_of_freedom)

        return probabilities

    def plot_GA_results(self, ga_results=None, diagnostic = 'fitness', save_path=None):
        """
        Plot the results of the genetic algorithm.

        Parameters:
        ga_results (ga_result_summary): The results of the genetic algorithm.
        diagnostic (str): The diagnostic to plot. Options are 'fitness', 'probability', or 'chi_square'.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
        """

        if ga_results is None:
            if not hasattr(self, 'GA_results'):
                raise ValueError("No GA results found. Run run_GA() first.")
            ga_results = self.GA_results


        subplots_dict = {1:[1, 1], 2:[1, 2], 3:[1,3], 4:[2, 2], 5:[2, 3], 6:[2,3], 7:[2,4], 8:[2,4], 9:[3,3], 10:[3, 4], 11:[3, 4], 12:[3, 4], 13:[3, 5], 14:[3, 5], 15:[3, 5], 16:[4,4]}
        fig, axs = plt.subplots(subplots_dict[len(ga_results.free_parameters)][0], subplots_dict[len(ga_results.free_parameters)][1], figsize=(subplots_dict[len(ga_results.free_parameters)][1]*4, subplots_dict[len(ga_results.free_parameters)][0]*3))
        axs = axs.ravel()

        if diagnostic not in ['fitness', 'probability', 'chi_square']:
            raise ValueError("Invalid diagnostic. Choose from 'fitness', 'probability', or 'chi_square'.")
        if diagnostic == 'fitness':
            diagnostic_param = 'fitnesses'
            title = 'Fitness'
        elif diagnostic == 'probability':
            diagnostic_param = 'probabilities'
            title = 'Probability'
        elif diagnostic == 'chi_square':
            diagnostic_param = 'reduced_chi_squares'
            title = 'Reduced Chi-Square'


        diagnostic_values = np.array(ga_results.__dict__[diagnostic_param]).flatten()
        for i, param in enumerate(ga_results.free_parameters):
            # Plot the probabilities for each generation
            param_values = ga_results.populations[:, :, i].flatten()
            generations = np.array([np.arange(ga_results.n_generations)]*ga_results.population_size).T.flatten()
            generations.flatten()
            axs[i].scatter(param_values, diagnostic_values, c= generations, cmap='viridis', alpha=0.5)
            axs[i].set_xlabel(param)
            axs[i].set_ylabel(title)
            axs[i].set_xlim(ga_results.ga_params[param].min, ga_results.ga_params[param].max)
            axs[i].set_ylim(0, np.max(diagnostic_values)*1.1)
            axs[i].set_title(r'%s = $%0.2f \pm \genfrac{}{}{0}{}{%0.2f}{%0.2f}$'%(param, ga_results.best_fit_model[i], ga_results.best_fit_errors[i][1] - ga_results.best_fit_model[i], ga_results.best_fit_model[i] - ga_results.best_fit_errors[i][0]))
            axs[i].fill_betweenx([0, np.max(diagnostic_values)*1.1], ga_results.best_fit_errors[i][0], ga_results.best_fit_errors[i][1], color='lightcoral', alpha=0.3)
        
        if i < len(axs) - 1:
            for j in range(i+1, len(axs)):
                axs[j].axis('off')

        if self.object_name is not None:
            plt.suptitle(f'{self.object_name} GA fit', fontsize=16)

        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()


    def plot_GA_fit(self, ga_results=None, save_path=None):
        """
        Plot the best-fit model from the genetic algorithm against the observed data.

        Parameters:
        ga_results (ga_result_summary): The results of the genetic algorithm.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
        """

        if ga_results is None:
            if not hasattr(self, 'GA_results'):
                raise ValueError("No GA results found. Run run_GA() first.")
            ga_results = self.GA_results

        best_fit_params = ga_results.best_fit_model

        pop = self.GA_results.populations
        population_parameters = pop.reshape(-1, pop.shape[-1])

        probabilities = np.array(ga_results.probabilities).flatten()
        inds = np.where(probabilities > 0.05)[0]

        model_args = population_parameters[inds]
        np.append(model_args, best_fit_params)

        param_set = self.parse_parameter_set(model_args)

        vrot_ind = list(self.parameters.__dict__.keys()).index('vrot')
        gamma_ind = list(self.parameters.__dict__.keys()).index('gamma')


        subplots_dict = {1:[1, 1], 2:[1, 2], 3:[1,3], 4:[2, 2], 5:[2, 3], 6:[2,3], 7:[2,4], 8:[2,4], 9:[3,3], 10:[3, 4], 11:[3, 4], 12:[3, 4], 13:[3, 5], 14:[3, 5], 15:[3, 5], 16:[4,4]}
        fig, axs = plt.subplots(subplots_dict[len(self.line_list)][0], subplots_dict[len(self.line_list)][1], figsize=(subplots_dict[len(self.line_list)][1]*4, subplots_dict[len(self.line_list)][0]*3))
        axs = axs.ravel()

        for i, line in enumerate(self.line_list.keys()):
            # model_fluxes = self.predict_fluxes_from_nn(line, np.array(param_set, ndmin=2))
            # broadened_wavelength, broadened_fluxes = self.broaden_lines(line, model_fluxes, np.array(param_set[:, vrot_ind], ndmin=1))
            # shifted_wavelengths = self.dopler_shift_lines(broadened_wavelength, np.array(param_set[:,gamma_ind], ndmin=1))
            # Get the model wavelengths and fluxes
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, np.array(param_set, ndmin=2))

            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)
            # interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, shifted_wavelengths, broadened_fluxes)

            model_min = np.array(interpolated_fluxes).min(axis=0)
            model_max = np.array(interpolated_fluxes).max(axis=0)

            axs[i].plot(obs_wavelength, self.observed_flux[obs_inds], 'k-', label='Observed')
            axs[i].plot(obs_wavelength, interpolated_fluxes[-1], 'r-', label='Best Fit')
            axs[i].fill_between(obs_wavelength, model_min, model_max, color='lightcoral', alpha=0.5, label='1-sigma', zorder=9)
            axs[i].set_xlabel('Wavelength (Angstrom)')
            axs[i].set_ylabel('Flux')
        
        if self.object_name is not None:
            plt.suptitle(f'{self.object_name} GA fit', fontsize=16)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    
    def print_GA_results(self, ga_results=None):
        """
        Print the results of the genetic algorithm.

        Parameters:
        ga_results (ga_result_summary): The results of the genetic algorithm. If None, uses the results from the last run.
        """

        if ga_results is None:
            if not hasattr(self, 'GA_results'):
                raise ValueError("No GA results found. Run run_GA() first.")
            ga_results = self.GA_results

        print(f"GA Results Summary:")
        for i, param in enumerate(self.free_parameters):
            print(f"{param} = ".rjust(15) + f" {ga_results.best_fit_model[i]}  ( +{ga_results.best_fit_errors[i][1] - ga_results.best_fit_model[i]:.3f}; -{ga_results.best_fit_model[i] - ga_results.best_fit_errors[i][0]:.3f})")
        # print(f"Best fit parameters: {self.GA_results.best_fit_model}")
        # print(f"Best fit errors: {self.GA_results.best_fit_errors}")
        print(f"Best fitness: {self.GA_results.best_fitness}")
        print(f"Number of generations: {self.GA_results.n_generations}")
        print(f"Population size: {self.GA_results.population_size}")