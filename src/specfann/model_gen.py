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
        if type(obj).__name__ == 'single_star':
            wavelengths, fluxes = generate_model_per_line(obj, line, param_set)
        elif type(obj).__name__ == 'composite':
            wavelengths, fluxes = generate_composite_model_per_line(obj, line, param_set)
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
        broadened_wavelength, broadened_fluxes = broaden_lines(obj.line_list[line].wavelength, model_fluxes, param_set[:, vrot_ind], param_set[:, vmacro_ind], param_set[:, inst_res_ind])
    else:
        # Use the neural network to predict the fluxes for the line and apply broadening
        broadened_wavelength, broadened_fluxes = obj.sbf.predict_fluxes_from_nn(obj.parameters, obj.line_list, line, param_set)

    # Doppler shift the lines
    shifted_wavelengths = dopler_shift_lines(broadened_wavelength, param_set[:, gamma_ind])

    return shifted_wavelengths, broadened_fluxes


def generate_composite_model_per_line(obj, line, param_set):
    """
    Generate a model based on the provided parameters.

    Parameters:
    param_set (array-like): The parameters for the model.

    Returns:
    models (dict): A dictionary of models for each line.
    """

    vrot1_ind = list(obj.parameters.__dict__.keys()).index('vrot_1')
    vrot2_ind = list(obj.parameters.__dict__.keys()).index('vrot_2')
    vmacro1_ind = list(obj.parameters.__dict__.keys()).index('vmacro_1')
    vmacro2_ind = list(obj.parameters.__dict__.keys()).index('vmacro_2')
    inst_res_ind = list(obj.parameters.__dict__.keys()).index('inst_res')
    rv1_ind = list(obj.parameters.__dict__.keys()).index('rv_1')
    delta_rv_ind = list(obj.parameters.__dict__.keys()).index('delta_rv')
    lr1_ind = list(obj.parameters.__dict__.keys()).index('lr_1')

    rv2 = param_set[:, rv1_ind] + param_set[:, delta_rv_ind]

    if obj.sbf.use_specfann_broadening:
        # Use the neural network to predict the fluxes for the line
        model_fluxes_1 = obj.sbf.predict_fluxes_from_nn(obj.parameters, obj.line_list, line, param_set, suffix='_1')
        model_fluxes_2 = obj.sbf.predict_fluxes_from_nn(obj.parameters, obj.line_list, line, param_set, suffix='_2')

        # Broaden the lines
        broadened_wavelength_1, broadened_fluxes_1 = broaden_lines(obj.line_list[line].wavelength, model_fluxes_1, param_set[:, vrot1_ind], param_set[:, vmacro1_ind], inst_res=None)
        broadened_wavelength_2, broadened_fluxes_2 = broaden_lines(obj.line_list[line].wavelength, model_fluxes_2, param_set[:, vrot2_ind], param_set[:, vmacro2_ind], inst_res=None)

        # combine the broadened lines
        combined_wavelength, combined_fluxes = combine_binary_lines(broadened_wavelength_1, broadened_fluxes_1, broadened_wavelength_2, broadened_fluxes_2, param_set[:, rv1_ind], rv2, param_set[:, lr1_ind])

        # applly instrumental broadening to the combined line
        broadened_wavelength, broadened_fluxes = inst_broadening_vectorized(combined_wavelength, combined_fluxes, param_set[:, inst_res_ind])

        return broadened_wavelength, broadened_fluxes

    else:
        # Use the neural network to predict the fluxes for the line and apply broadening
        broadened_wavelength_1, broadened_fluxes_1 = obj.sbf.predict_fluxes_from_nn(obj.parameters, obj.line_list, line, param_set, suffix='_1')
        broadened_wavelength_2, broadened_fluxes_2 = obj.sbf.predict_fluxes_from_nn(obj.parameters, obj.line_list, line, param_set, suffix='_2')
        # combine the broadened lines
        combined_wavelength, combined_fluxes = combine_binary_lines(broadened_wavelength_1, broadened_fluxes_1, broadened_wavelength_2, broadened_fluxes_2, param_set[:, rv1_ind], rv2, param_set[:, lr1_ind])

        return combined_wavelength, combined_fluxes


def broaden_lines(wavelength, fluxes, vrot, vmacro, inst_res):
    """
    Broaden the spectral lines using rotational broadening.

    Parameters:
    wavelength (array-like): The wavelength array of the line to be broadened.
    fluxes (array-like): The flux values corresponding to the line.
    vrot (float): The rotational velocity to apply for broadening.
    vmacro (float): The macroturbulent velocity to apply for broadening.
    inst_res (float): The instrumental resolution to apply for broadening.

    Returns:
    broadened_wavelength (array-like): The broadened wavelength array.
    broadened_fluxes (list of array-like): The broadened flux arrays for each input flux.
    """

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


def combine_binary_lines(wavelengths_1, fluxes_1, wavelengths_2, fluxes_2, rv_1, rv_2, lr1=0.5):
    """
    Combine the fluxes of two binary components.

    Parameters:
    wavelengths_1 (array-like): The wavelengths of the first component.
    fluxes_1 (array-like): The fluxes of the first component.
    wavelengths_2 (array-like): The wavelengths of the second component.
    fluxes_2 (array-like): The fluxes of the second component.
    rv_1 (array-like): The radial velocity of the first component in km/s.
    rv_2 (array-like): The radial velocity of the second component in km/s.
    lr1 (array-like): The light ratio of the first component.

    Returns:
    combined_fluxes (array-like): The combined fluxes of the two components.
    """

    shifted_wavelengths_1 = dopler_shift_lines(wavelengths_1, rv_1)
    shifted_wavelengths_2 = dopler_shift_lines(wavelengths_2, rv_2)
    wavelengths = np.arange(min(shifted_wavelengths_1.min(), shifted_wavelengths_2.min()), max(shifted_wavelengths_1.max(), shifted_wavelengths_2.max()), 0.01)

    scaled_fluxes_1 = (fluxes_1 - 1.0) * lr1[:, None] + 1.0
    scaled_fluxes_2 = (fluxes_2 - 1.0) * (1.0 - lr1[:, None]) + 1.0

    # f1 = si.interp1d(shifted_wavelengths_1, scaled_fluxes_1, bounds_error=False, fill_value=(1.0, 1.0))
    # f2 = si.interp1d(shifted_wavelengths_2, scaled_fluxes_2, bounds_error=False, fill_value=(1.0, 1.0))
    interp_flux1 = np.empty((len(shifted_wavelengths_1), len(wavelengths)))
    interp_flux2 = np.empty((len(shifted_wavelengths_2), len(wavelengths)))
    for i in range(len(interp_flux1)):
        interp_flux1[i] = np.interp(wavelengths, shifted_wavelengths_1[i], scaled_fluxes_1[i], left=1.0, right=1.0)
        interp_flux2[i] = np.interp(wavelengths, shifted_wavelengths_2[i], scaled_fluxes_2[i], left=1.0, right=1.0)

    combined_fluxes = interp_flux1 * interp_flux2

    return wavelengths, combined_fluxes


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
