import numpy as np
import keras


# -------------------------------------------------------------------
# -----------------------Parameter functions-------------------------
# -------------------------------------------------------------------



class parameters(object):
    """
    Class to hold the parameters for the model
    """

    def __init__(self, vmacro = 0, inst_res = 10000):
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
        vmacro (float): Macroturbulent velocity in km/s.
        inst_res (float): Instrumental resolving power (R = lambda/delta_lambda).
        gamma (float): Systemic radial velocity in km/s.
        """

        self.vmacro = self.parameter('vmacro', vmacro, bounds=[0, 500], latex_string=r'$v_\mathrm{macro}$', unit=r'km s$^{-1}$')
        self.inst_res = self.parameter('inst_res', inst_res, bounds=None, fixed=True, latex_string=r'$\mathcal{R}$')
        self.logf = self.parameter('logf', 0.0, bounds=[-10, 10], fixed=True, hidden=True, latex_string=r'$\log f$')  # log of the variance scaling factor

    def summary(self, show_hidden=False):
        """
        Print a summary of the parameters.
        """
        print("Parameters:")
        for param in self.__dict__.values():
            if isinstance(param, self.parameter):
                if not param._hidden or show_hidden:
                    print(f"{param.name}: {param.value} (fixed: {param.fixed}, bounds: {param.bounds})")

    class parameter(object):
        """
        Class to hold individual parameters
        """

        def __init__(self, name, value, bounds=None, fixed=False, latex_string=None, unit=None, hidden=False):
            """
            Initialize the parameter object.
            Parameters:
            name (str): The name of the parameter.
            value (float): The value of the parameter.
            bounds (list): A list containing the lower and upper bounds for the parameter.
            latex_string (str): The LaTeX string for the parameter.
            unit (str): The unit of the parameter.
            hidden (bool): Whether the parameter is hidden.
            """
            self.name = name
            self.value = value
            self.fixed = fixed
            self.bounds = bounds if bounds is not None else [None, None]
            self.latex_string = latex_string
            self.unit = unit
            self._hidden = hidden

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



# -------------------------------------------------------------------
# -----------------------Line list functions-------------------------
# -------------------------------------------------------------------

class line_to_fit(object):
    """
    Class to hold information about the lines to be fitted
    """

    def __init__(self, line, nn_path='bundles/', nn_model_string = 'fluxes_$LINE$_model.keras', nn_wavelength_string = 'wnew_$LINE$.npy', fit_range=None, components_dict=None):
        """
        Initialize the line_to_fit object.
        Parameters:
        line (str): The name of the line to be fitted.
        fit_range (list): The range of wavelengths to be fitted.
        components_dict (dict): The individual transitions that make up the line (keys are species, values are lists of wavelengths).
        """

        self.line_name = line
        model_filename = '/'.join([nn_path, nn_model_string.replace('$LINE$', line)])
        wavelength_filename = '/'.join([nn_path, nn_wavelength_string.replace('$LINE$', line)])
        
        self.model = keras.saving.load_model(model_filename)
        self.wavelength = np.load(wavelength_filename)
        if fit_range is None:
            total_range = np.max(self.wavelength) - np.min(self.wavelength)
            self.fit_range = [np.min(self.wavelength) + total_range/4, np.max(self.wavelength) - total_range/4]
        else:
            self.fit_range = fit_range

        self.components_dict = components_dict


def add_line(obj, line, fit_range=None):
    """
    Add a line to the list of lines to be fitted.

    Parameters:
    line (str): The name of the line to be fitted.
    fit_range (list): The range of wavelengths to be fitted for the line.
    """
    if line not in obj.line_list.keys():
        obj.line_list[line] = line_to_fit(line, nn_path=obj.nn_bundle_path, nn_model_string=obj.nn_model_string, nn_wavelength_string=obj.nn_wavelength_string, fit_range=fit_range)


def remove_line(obj, line):
    """
    Remove a line from the list of lines to be fitted.

    Parameters:
    line (str): The name of the line to be removed.
    """
    if line in obj.line_list.keys():
        del obj.line_list[line]
    else:
        print(f"Line {line} not found in the list of lines to be fitted.")



