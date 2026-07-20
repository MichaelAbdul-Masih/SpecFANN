import numpy as np
import scipy.interpolate as si
from scipy.signal import fftconvolve
import math
from scipy.special import erf



# -------------------Model Generation functions--------------------


def combined_broadening_vectorized(wavelength,fluxes,vrots,vmacros,inst_res=None,epsilon=0.6):

    c = 299792.458
    # Ensure that vrots and vmacros have minimum values to avoid division by zero or other numerical issues
    vrots[vrots < 1E-6]=1E-6
    vmacros[vmacros < 1E-6]=1E-6

    # convert wavelength array into velocity space, and ensure it is equidistant
    wave_ = np.log(wavelength)
    velo_ = np.linspace(wave_[0],wave_[-1],len(wave_))
    dvelo = velo_[1]-velo_[0]
    vrots_new = vrots/c
    vmacros_new = vmacros/c

    f = si.interp1d(wave_, fluxes, fill_value=np.array([1.0]), bounds_error=False)
    fluxes_ = f(velo_)

    #--------------- Instrumental broadening standard deviation ----------------
    if inst_res is not None:
        inst_fwhm = 1 / inst_res
        inst_std = inst_fwhm / 2.3548
    else:
        inst_std = 0

    # compute the velocity array of the kernels
    max_velocity = np.max(vrots_new) + np.max(vmacros_new) + np.max(inst_std)*5/2
    n = 2*math.ceil(max_velocity/dvelo) + 1
    kernel_velocity = np.linspace(-max_velocity, max_velocity, n)

    #---------------Vrot KERNEL----------------
    y = 1 - (kernel_velocity[None,:]/vrots_new[:,None])**2 # transformation of velocity
    y[y<0]=0

    rot_kernel = (2*(1-epsilon)*np.sqrt(y)+np.pi*epsilon/2.*y)/(np.pi*vrots_new[:,None]*(1-epsilon/3.0))
    rot_kernel /= rot_kernel.sum(axis=1)[:,None]

    #---------------Vmacro KERNEL----------------
    x = (abs(kernel_velocity[None,:])/vmacros_new[:,None])

    lambda0 = np.median(wave_)
    mr = vmacros_new[:,None] * lambda0 / c
    sq_pi = np.sqrt(np.pi)

    macro_kernel = (2/(np.sqrt(np.pi) * mr)) * (np.exp(-x ** 2) + sq_pi * x * (erf(x) - 1.0))
    macro_kernel /= macro_kernel.sum(axis=1)[:,None]

    combined_kernel = fftconvolve(rot_kernel, macro_kernel, mode='same', axes=1)
    #---------------Instrumental KERNEL----------------
    if inst_res is not None and np.ndim(inst_res) == 0:
        inst_kernel = np.exp(-1 * (kernel_velocity)**2 / (2*inst_std**2))
        inst_kernel /= inst_kernel.sum()
        combined_kernel = fftconvolve(combined_kernel, inst_kernel[None,:], mode='same', axes=1)

    elif inst_res is not None and np.ndim(inst_res) == 1:
        inst_kernels = np.exp(-1 * (kernel_velocity[None,:])**2 / (2*inst_std[:,None]**2))
        inst_kernels /= inst_kernels.sum(axis=1)[:,None]
        combined_kernel = fftconvolve(combined_kernel, inst_kernels, mode='same', axes=1)

    #--------------- Convolved KERNEL----------------
    flux_conv = fftconvolve(1-fluxes_,combined_kernel,mode='same', axes=1)


    f = si.interp1d(np.exp(velo_), 1-flux_conv, fill_value=np.array([1.0]), bounds_error=False)
    broadened_fluxes = f(wavelength)
    return wavelength, broadened_fluxes


def inst_broadening_vectorized(wavelength,fluxes,inst_res):

    c = 299792.458
    # convert wavelength array into velocity space, and ensure it is equidistant
    wave_ = np.log(wavelength)
    velo_ = np.linspace(wave_[0],wave_[-1],len(wave_))
    dvelo = velo_[1]-velo_[0]

    f = si.interp1d(wave_, fluxes, fill_value=np.array([1.0]), bounds_error=False)
    fluxes_ = f(velo_)

    #--------------- Instrumental broadening standard deviation ----------------
    if inst_res is not None:
        inst_fwhm = 1 / inst_res
        inst_std = inst_fwhm / 2.3548
    else:
        inst_std = 0

    # compute the velocity array of the kernels
    max_velocity = np.max(inst_std)*5/2
    n = 2*math.ceil(max_velocity/dvelo) + 1
    kernel_velocity = np.linspace(-max_velocity, max_velocity, n)

    #---------------Instrumental KERNEL----------------
    if np.ndim(inst_res) == 0:
        inst_kernel = np.exp(-1 * (kernel_velocity)**2 / (2*inst_std**2))
        inst_kernel /= inst_kernel.sum()
        flux_conv = fftconvolve(1-fluxes_,inst_kernel[None,:],mode='same', axes=1)

    elif np.ndim(inst_res) == 1:
        inst_kernels = np.exp(-1 * (kernel_velocity[None,:])**2 / (2*inst_std[:,None]**2))
        inst_kernels /= inst_kernels.sum(axis=1)[:,None]
        flux_conv = fftconvolve(1-fluxes_,inst_kernels,mode='same', axes=1)


    f = si.interp1d(np.exp(velo_), 1-flux_conv, fill_value=np.array([1.0]), bounds_error=False)
    broadened_fluxes = f(wavelength)
    return wavelength, broadened_fluxes



def generate_model(obj, param_set):
    """
    Generate a model based on the provided parameters.

    Parameters:
    param_set (array-like): The parameters for the model.

    Returns:
    models (dict): A dictionary of models for each line.
    """

    param_set = np.array(param_set, ndmin=2)

    models = {}
    for line in obj.line_list.keys():
        # Generate the model for each line
        wavelengths, fluxes = generate_model_per_line(obj, line, param_set)
        models[line] = {'wavelengths': wavelengths, 'fluxes': fluxes}

    return models


def generate_model_per_line(obj, line, param_set):
    """
    Generate a model based on the provided parameters.

    Parameters:
    param_set (array-like): The parameters for the model.

    Returns:
    models (dict): A dictionary of models for each line.
    """

    vrot_ind = list(obj.parameters.__dict__.keys()).index('vrot')
    vmacro_ind = list(obj.parameters.__dict__.keys()).index('vmacro')
    inst_res_ind = list(obj.parameters.__dict__.keys()).index('inst_res')
    gamma_ind = list(obj.parameters.__dict__.keys()).index('gamma')

    if obj.sbf.use_specfann_broadening:
        # Use the neural network to predict the fluxes for the line
        model_fluxes = obj.sbf.predict_fluxes_from_nn(obj.parameters, obj.line_list, line, param_set)
        # Broaden the lines
        broadened_wavelength, broadened_fluxes = broaden_lines(obj, line, model_fluxes, param_set[:, vrot_ind], param_set[:, vmacro_ind], param_set[:, inst_res_ind])
    else:
        # Use the neural network to predict the fluxes for the line and apply broadening
        broadened_wavelength, broadened_fluxes = obj.sbf.predict_fluxes_from_nn(obj.parameters, obj.line_list, line, param_set)

    # Doppler shift the lines
    shifted_wavelengths = dopler_shift_lines(broadened_wavelength, param_set[:, gamma_ind])

    return shifted_wavelengths, broadened_fluxes


def broaden_lines(obj, line, fluxes, vrot, vmacro, inst_res):
    """
    Broaden the spectral lines using rotational broadening.

    Parameters:
    line (str): The name of the line to be broadened.
    fluxes (array-like): The flux values corresponding to the line.
    vrot (float): The rotational velocity to apply for broadening.
    vmacro (float): The macroturbulent velocity to apply for broadening.
    inst_res (float): The instrumental resolution to apply for broadening.

    Returns:
    broadened_wavelength (array-like): The broadened wavelength array.
    broadened_fluxes (list of array-like): The broadened flux arrays for each input flux.
    """

    wavelength = obj.line_list[line].wavelength
    new_wavelength = np.arange(wavelength[0], wavelength[-1], 0.01)
    unbroadened_fluxes = si.interp1d(wavelength, fluxes, bounds_error=False, fill_value=(1.0,1.0))(new_wavelength)

    broadened_wavelength, broadened_fluxes = combined_broadening_vectorized(new_wavelength, unbroadened_fluxes, vrot, vmacro, inst_res)

    return np.array(broadened_wavelength), np.array(broadened_fluxes)


def dopler_shift_lines(wavelengths, rv):
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


def parse_parameter_set(self, model_args, free_parameters=None):
    """
    Parse the model arguments to return full parameter set including fixed params.

    Parameters:
    model_args (array-like): The sampled values of the free parameters.

    Returns:
    parameter_set (array-like): Full parameter set.
    """

    if free_parameters is None:
        free_parameters = self.free_parameters

    params = list(self.parameters.__dict__.keys())
    parameter_set = []
    for param in params:
        if param in free_parameters:
            parameter_set.append(np.array(model_args.T[free_parameters.index(param)], ndmin=1))
        else:
            parameter_set.append([self.parameters.__dict__[param].value]*len(np.array(model_args, ndmin=2)))

    return np.array(parameter_set).T
