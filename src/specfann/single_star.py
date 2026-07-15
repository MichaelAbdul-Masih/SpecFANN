import os
import numpy as np
import matplotlib.pyplot as plt

from . import params
from . import io_functions
from . import model_gen
from . import fitting



class single_star(object):
    """
    Class to fit the single star spectra using neural networks
    """

    def __init__(self, bundle_path=None, bundle_name=None):
        default_bundle_name = 'MW_v1.3'

        self.observed_wavelength = None
        self.observed_flux = None
        self.observed_error = None

        self.parameters = params.parameters()

        if bundle_path is None:
            if bundle_name is None:
                self.nn_bundle_name = default_bundle_name
                if os.path.exists(os.path.expanduser(f'~/.specfann/bundles/{default_bundle_name}/')):
                    bundle_path = os.path.expanduser(f'~/.specfann/bundles/{default_bundle_name}/')
                else:
                    bundle_path = os.path.join(os.path.dirname(__file__), f'bundles/{default_bundle_name}/')
            else:
                self.nn_bundle_name = bundle_name
                if os.path.exists(os.path.expanduser('~/.specfann/bundles/{}/'.format(bundle_name))):
                    bundle_path = os.path.expanduser('~/.specfann/bundles/{}/'.format(bundle_name))
                else:
                    bundle_path = os.path.join(os.path.dirname(__file__), 'bundles/%s/' % bundle_name)
        
        self.nn_bundle_path = bundle_path
        self.set_nn_bundle_path(bundle_path)

        self.line_list = {}

        self.free_parameters = [param for param in self.parameters.__dict__ if not self.parameters.__dict__[param].fixed]

        self.n_walkers = 150
        self.n_steps = 1000

        self.object_name = None
        self.n_evaluations = 0
    

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['sbf']
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        temp_parameters = self.parameters

        try:
            self.set_nn_bundle_path(self.nn_bundle_path)
        except:
            bundle_path = os.path.expanduser(f'~/.specfann/bundles/{self.nn_bundle_name}/')
            self.set_nn_bundle_path(bundle_path)
        
        self.parameters = temp_parameters
        
    

    # ----------------------IO----------------------


    def load_observed_data(self, observed_wavelength, observed_flux, observed_error=None, snr_region=[4220, 4240]):
        """
        Load the observed data into the class.

        Parameters:
        observed_wavelength (array-like): The observed wavelengths in Angstroms.
        observed_flux (array-like): The observed flux values corresponding to the wavelengths.
        observed_error (array-like, optional): The errors in the observed flux values. If not provided, errors will be estimated based on the SNR of the data.
        snr_region (list, optional): The wavelength region to use for SNR calculation. Default is [4220, 4240].
        """

        self.observed_wavelength, self.observed_flux, self.observed_error = io_functions.load_observed_data(observed_wavelength, observed_flux, observed_error, snr_region)


    def set_nn_bundle_path(self, nn_bundle_path):
        """
        Set the path to the neural net bundle files.

        Parameters:
        nn_bundle_path (str): The path to the bundle files.
        """

        io_functions.set_nn_bundle_path(self, nn_bundle_path)


    def add_line(self, line, fit_range=None):
        """
        Add a line to the list of lines to be fitted.

        Parameters:
        line (str): The name of the line to be fitted.
        fit_range (list): The range of wavelengths to be fitted for the line.
        """

        params.add_line(self, line, fit_range)


    def remove_line(self, line):
        """
        Remove a line from the list of lines to be fitted.

        Parameters:
        line (str): The name of the line to be removed.
        """

        params.remove_line(self, line)


    def save(self, filename):
        '''
        Save the current state in a pickle file

        Parameters:
        filename (str): The name of the file that the bundle will be saved to
        '''
        io_functions.save(self, filename)


    # -------------------Model Generation functions--------------------


    def generate_model(self, param_set):
        """
        Generate a model based on the provided parameters.

        Parameters:
        param_set (array-like): The parameters for the model.

        Returns:
        models (dict): A dictionary of models for each line.
        """

        return model_gen.generate_model(self, param_set)
    

    def generate_model_per_line(self, line, param_set):
        """
        Generate a model based on the provided parameters.

        Parameters:
        param_set (array-like): The parameters for the model.

        Returns:
        models (dict): A dictionary of models for each line.
        """

        return model_gen.generate_model_per_line(self, line, param_set)

    
    # -------------------Cost functions--------------------


    def log_likelihood(self, param_set, fuzz=False):
        """
        Calculate the log likelihood of the model given the observed data.

        Parameters:
        param_set (array-like): The full parameter set including free and fixed parameters.

        Returns:
        log_likelihoods (array-like): The log likelihoods for each model.
        """

        param_set = np.array(param_set, ndmin=2)
        self.n_evaluations += len(param_set)

        log_likelihoods = np.zeros(len(param_set))
        for line in self.line_list.keys():
            # Get the model wavelengths and fluxes
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, param_set)

            # Interpolate the model lines to the observed wavelengths
            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = fitting.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)

            # Calculate the log likelihood
            if fuzz:
                logf_ind = list(self.parameters.__dict__.keys()).index('logf')
                logf = param_set[:, logf_ind]
                error = np.sqrt(self.observed_error[obs_inds] **2 + np.array(10**logf, ndmin=2).T * interpolated_fluxes**2)
                log_likelihoods += fitting.calc_log_likelihoods_with_fuzz(self.observed_flux[obs_inds], error, interpolated_fluxes)
            else:
                log_likelihoods += fitting.calc_log_likelihoods(self.observed_flux[obs_inds], self.observed_error[obs_inds], interpolated_fluxes)

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

        return prior_array


    def log_probability(self, model_args, fuzz=False):

        param_set = model_gen.parse_parameter_set(self, model_args)

        lp = self.log_prior(param_set)

        return lp + self.log_likelihood(param_set, fuzz=fuzz)


    def reduced_chi_square(self, model_args, fuzz=False):
        """
        Calculate the reduced chi-squared statistic for the model parameters.

        Parameters:
        model_args (array-like): The full parameter set including free and fixed parameters.
        fuzz (bool): Whether to include fuzz in the likelihood calculation.

        Returns:
        chi_squared (float): The chi-squared statistic.
        """

        param_set = model_gen.parse_parameter_set(self, model_args)
        param_set = np.array(param_set, ndmin=2)
        self.n_evaluations += len(param_set)

        chi_squares = np.zeros(len(param_set))
        reduced_chi_squares = np.zeros(len(param_set))
        for line in self.line_list.keys():
            # Get the model wavelengths and fluxes
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, param_set)

            # Interpolate the model lines to the observed wavelengths
            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = fitting.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)

            # Calculate the log likelihood
            if fuzz:
                logf_ind = list(self.parameters.__dict__.keys()).index('logf')
                logf = param_set[:, logf_ind]
                error = np.sqrt(self.observed_error[obs_inds] **2 + np.array(10**logf, ndmin=2).T * interpolated_fluxes**2)
                chi_squares += fitting.calc_chi_square(self.observed_flux[obs_inds], error, interpolated_fluxes)
            else:
                chi_squares += fitting.calc_chi_square(self.observed_flux[obs_inds], self.observed_error[obs_inds], interpolated_fluxes)

            reduced_chi_squares += chi_squares / (len(obs_inds) - len(self.free_parameters))

        return reduced_chi_squares


    # -------------------MCMC functions--------------------


    def run_mcmc(self, initial_positions=None, n_walkers=None, n_steps=None, fuzz=False, return_sampler=False):
        """
        Run the MCMC simulation to sample the parameter space.

        Parameters:
        initial_positions (array-like): Initial positions of the walkers in the parameter space.
        n_walkers (int): The number of walkers to use in the MCMC simulation.
        n_steps (int): The number of steps to run the MCMC simulation for.
        """

        fitting.run_mcmc(self, initial_positions, n_walkers, n_steps, fuzz, return_sampler)


    def continue_mcmc(self, sampler=None, n_steps=None, return_sampler=False):
        """
        Continue the MCMC simulation from the last position of the walkers.

        Parameters:
        sampler (array-like): The MCMC sampler to continue.
        n_steps (int): The number of steps to run the MCMC simulation for.
        """

        fitting.continue_mcmc(self, sampler, n_steps, return_sampler)


    def plot_MCMC_results(self, sampler = None, burnin=100, thin=1, save_path=None):
        """
        Plot the results of the MCMC simulation.

        Parameters:
        samples (array-like): The samples from the MCMC simulation.
        """

        fitting.plot_MCMC_results(self, sampler, burnin, thin, save_path)


    def plot_MCMC_fit(self, sampler = None, burnin=100, save_path=None, line_labels=None, component_labels=None):
        """
        Plot the MCMC fit results against the observed data.

        Parameters:
        samples (array-like): The samples from the MCMC simulation.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
        line_labels (bool): Whether to display labels for each line.
        component_labels (bool): Whether to display labels for each component.
        """

        if sampler is None:
            if not hasattr(self, 'emcee_sampler'):
                raise ValueError("No MCMC sampler found. Run run_mcmc() first.")
            sampler = self.emcee_sampler

        chains = sampler.get_chain(flat=True, thin=1, discard=burnin)
        log_probs = sampler.get_log_prob(flat=True, thin=1, discard=burnin)

        inds = np.random.randint(len(chains), size=1000)
        inds_check = inds[np.isfinite(log_probs[inds])]
        model_args = chains[inds_check]
        param_set = model_gen.parse_parameter_set(self, model_args, free_parameters=self.mcmc_free_parameters)

        subplots_dict = {1:[1, 1], 2:[1, 2], 3:[1,3], 4:[2, 2], 5:[2, 3], 6:[2,3], 7:[2,4], 8:[2,4], 9:[3,3], 10:[3, 4], 11:[3, 4], 12:[3, 4], 13:[3, 5], 14:[3, 5], 15:[3, 5], 16:[4,4], 17:[4,5], 18:[4,5], 19:[4,5], 20:[4,5], 21:[4,6], 22:[4,6], 23:[4,6], 24:[4,6], 25:[5,5], 26:[5,6], 27:[5,6], 28:[5,6], 29:[5,6], 30:[5,6], 31:[5,7], 32:[5,7], 33:[5,7], 34:[5,7], 35:[5,7], 36:[6,6], 37:[6,7], 38:[6,7], 39:[6,7], 40:[6,7], 41:[6,8], 42:[6,8], 43:[6,8], 44:[6,8], 45:[6,8]}
        fig, axs = plt.subplots(subplots_dict[len(self.line_list)][0], subplots_dict[len(self.line_list)][1], figsize=(subplots_dict[len(self.line_list)][1]*4, subplots_dict[len(self.line_list)][0]*3))
        axs = axs.ravel()

        for i, line in enumerate(self.line_list.keys()):
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, np.array(param_set, ndmin=2))

            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = fitting.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)

            model_mean = np.array(interpolated_fluxes).mean(axis=0)
            model_std = np.array(interpolated_fluxes).std(axis=0)

            axs[i].plot(obs_wavelength, self.observed_flux[obs_inds], 'k-', label='Observed')
            axs[i].plot(obs_wavelength, model_mean, 'r-', label='Best Fit')
            axs[i].fill_between(obs_wavelength, model_mean-model_std, model_mean+model_std, color='lightcoral', alpha=0.8, label='1-sigma')
            if line_labels:
                axs[i].text(0.025, 0.025, f'{line}', transform=axs[i].transAxes, fontsize=12, verticalalignment='bottom')
            axs[i].set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
            axs[i].set_ylabel('Flux')

        if i < len(axs) - 1:
            for j in range(i+1, len(axs)):
                axs[j].axis('off')

        if self.object_name is not None:
            plt.suptitle(f'{self.object_name} MCMC fit', fontsize=16)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def print_MCMC_results(self, sampler=None, burnin=100, sigma=1, filename=None):
        """
        Print the results of the MCMC simulation.

        Parameters:
        samples (array-like): The samples from the MCMC simulation.
        burnin (int): The number of steps to discard as burn-in.
        """

        fitting.print_MCMC_results(self, sampler, burnin, sigma, filename)


    # -------------------Nested sampling (ultranest) functions--------------------


    def run_nested_sampling(obj, fuzz=False, return_result=False, step_sampler=None, log_dir=None, **kwargs):
        """
        Run nested sampling with ultranest to sample the parameter space.

        Parameters:
        fuzz (bool): Whether to include fuzz in the likelihood calculation.
        return_result (bool): Whether to return the ultranest result object.
        step_sampler: If None, use default exploration. If True or 'slice', use PopulationSimpleSliceSampler
            (popsize=5, nsteps=20, generate_mixture_random_direction). kwargs step_sampler_popsize and
            step_sampler_nsteps override these. Otherwise must be an ultranest step sampler instance to attach.
        log_dir: Optional path to a folder in which to save results (timestamped subfolder log_dir/ns_YYYYMMDD_HHMMSS/).
            If None (default), results are not saved to disk.
        **kwargs: Passed to ultranest.ReactiveNestedSampler.run() (e.g. min_num_live_points, dlogz,
            max_num_improvement_loops). step_sampler_popsize and step_sampler_nsteps are consumed when
            step_sampler is True/'slice'.

        Returns:
        result: Ultranest result object if return_result=True.
        """

        fitting.run_nested_sampling(obj, fuzz, return_result, step_sampler, log_dir, **kwargs)


    def plot_nested_sampling_results(obj, result=None):
        """
        Plot the results of the nested sampling run: ultranest's run and trace plots (from the
        sampler), then a corner plot of the posterior.

        Parameters:
        result: Ultranest result object. If None, uses self.ultranest_result.
        """

        fitting.plot_nested_sampling_results(obj, result)


    def plot_nested_sampling_fit(self, result=None, n_draw=1000, save_path=None, line_labels=False, component_labels=False):
        """
        Plot the nested sampling fit results against the observed data.

        Parameters:
        result: Ultranest result object. If None, uses self.ultranest_result.
        n_draw (int): Number of posterior samples to draw for model spread.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
        line_labels (bool): Whether to display labels for each line.
        component_labels (bool): Whether to display labels for each component.
        """

        if result is None:
            if not hasattr(self, 'ultranest_result'):
                raise ValueError("No nested sampling result found. Run run_nested_sampling() first.")
            result = self.ultranest_result

        flat_samples = self._get_nested_sampling_posterior_samples(result=result, thin=1)
        n_draw = min(n_draw, len(flat_samples))
        inds = np.random.randint(len(flat_samples), size=n_draw)
        model_args = flat_samples[inds]
        param_set = model_gen.parse_parameter_set(self, model_args)

        subplots_dict = {1:[1, 1], 2:[1, 2], 3:[1,3], 4:[2, 2], 5:[2, 3], 6:[2,3], 7:[2,4], 8:[2,4], 9:[3,3], 10:[3, 4], 11:[3, 4], 12:[3, 4], 13:[3, 5], 14:[3, 5], 15:[3, 5], 16:[4,4], 17:[4,5], 18:[4,5], 19:[4,5], 20:[4,5], 21:[4,6], 22:[4,6], 23:[4,6], 24:[4,6], 25:[5,5], 26:[5,6], 27:[5,6], 28:[5,6], 29:[5,6], 30:[5,6], 31:[5,7], 32:[5,7], 33:[5,7], 34:[5,7], 35:[5,7], 36:[6,6], 37:[6,7], 38:[6,7], 39:[6,7], 40:[6,7], 41:[6,8], 42:[6,8], 43:[6,8], 44:[6,8], 45:[6,8]}
        fig, axs = plt.subplots(subplots_dict[len(self.line_list)][0], subplots_dict[len(self.line_list)][1], figsize=(subplots_dict[len(self.line_list)][1]*4, subplots_dict[len(self.line_list)][0]*3))
        axs = axs.ravel()

        for i, line in enumerate(self.line_list.keys()):
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, np.array(param_set, ndmin=2))
            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = fitting.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)
            
            model_mean = np.array(interpolated_fluxes).mean(axis=0)
            model_std = np.array(interpolated_fluxes).std(axis=0)
            axs[i].plot(obs_wavelength, self.observed_flux[obs_inds], 'k-', label='Observed')
            axs[i].plot(obs_wavelength, model_mean, 'r-', label='Best Fit')
            axs[i].fill_between(obs_wavelength, model_mean - model_std, model_mean + model_std, color='lightcoral', alpha=0.8, label='1-sigma')
            if line_labels:
                axs[i].text(0.025, 0.025, f'{line}', transform=axs[i].transAxes, fontsize=12, verticalalignment='bottom')
            axs[i].set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
            axs[i].set_ylabel('Flux')
        
        if i < len(axs) - 1:
            for j in range(i+1, len(axs)):
                axs[j].axis('off')

        plt.show()

    def print_nested_sampling_results(obj, result=None):
        """
        Print the results of the nested sampling run using ultranest built-in summary.

        Parameters:
        result: Ultranest result object. If None, uses self.ultranest_result.
        """

        fitting.print_nested_sampling_results(obj, result)



    # -------------------GA functions--------------------


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

        fitting.run_GA(self, n_generations, population_size, return_result)


    def plot_GA_results(obj, ga_results=None, diagnostic = 'fitness', sigma=2, save_path=None):
        """
        Plot the results of the genetic algorithm.

        Parameters:
        ga_results (ga_result_summary): The results of the genetic algorithm.
        diagnostic (str): The diagnostic to plot. Options are 'fitness', 'probability', or 'chi_square'.
        sigma (int): The number of sigma to use for error bars.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
        """

        fitting.plot_GA_results(obj, ga_results, diagnostic, sigma, save_path)


    def plot_GA_fit(self, ga_results=None, sigma=2, save_path=None, line_labels=False, component_labels=False):
        """
        Plot the best-fit model from the genetic algorithm against the observed data.

        Parameters:
        ga_results (ga_result_summary): The results of the genetic algorithm.
        sigma (int): The number of sigma to use for error bars.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
        line_labels (bool): Whether to display labels for each line.
        component_labels (bool): Whether to display labels for each component.
        """

        if ga_results is None:
            if not hasattr(self, 'GA_results'):
                raise ValueError("No GA results found. Run run_GA() first.")
            ga_results = self.GA_results

        best_fit_params = ga_results.best_fit_model

        pop = self.GA_results.populations
        population_parameters = pop.reshape(-1, pop.shape[-1])

        probabilities = np.array(ga_results.probabilities).flatten()
        if sigma == 1:
            inds = np.where(probabilities > 0.32)[0]
        elif sigma == 2:
            inds = np.where(probabilities > 0.05)[0]

        model_args = population_parameters[inds]
        np.append(model_args, best_fit_params)

        param_set = model_gen.parse_parameter_set(self, model_args, free_parameters=ga_results.free_parameters)

        subplots_dict = {1:[1, 1], 2:[1, 2], 3:[1,3], 4:[2, 2], 5:[2, 3], 6:[2,3], 7:[2,4], 8:[2,4], 9:[3,3], 10:[3, 4], 11:[3, 4], 12:[3, 4], 13:[3, 5], 14:[3, 5], 15:[3, 5], 16:[4,4], 17:[4,5], 18:[4,5], 19:[4,5], 20:[4,5], 21:[4,6], 22:[4,6], 23:[4,6], 24:[4,6], 25:[5,5], 26:[5,6], 27:[5,6], 28:[5,6], 29:[5,6], 30:[5,6], 31:[5,7], 32:[5,7], 33:[5,7], 34:[5,7], 35:[5,7], 36:[6,6], 37:[6,7], 38:[6,7], 39:[6,7], 40:[6,7], 41:[6,8], 42:[6,8], 43:[6,8], 44:[6,8], 45:[6,8]}
        fig, axs = plt.subplots(subplots_dict[len(self.line_list)][0], subplots_dict[len(self.line_list)][1], figsize=(subplots_dict[len(self.line_list)][1]*4, subplots_dict[len(self.line_list)][0]*3))
        axs = axs.ravel()

        for i, line in enumerate(self.line_list.keys()):
            # Get the model wavelengths and fluxes
            model_wavelengths, model_fluxes = self.generate_model_per_line(line, np.array(param_set, ndmin=2))

            obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
            obs_wavelength = self.observed_wavelength[obs_inds]
            interpolated_fluxes = fitting.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)

            model_min = np.array(interpolated_fluxes).min(axis=0)
            model_max = np.array(interpolated_fluxes).max(axis=0)

            axs[i].plot(obs_wavelength, self.observed_flux[obs_inds], 'k-', label='Observed')
            axs[i].plot(obs_wavelength, interpolated_fluxes[-1], 'r-', label='Best Fit')
            axs[i].fill_between(obs_wavelength, model_min, model_max, color='lightcoral', alpha=0.5, label='1-sigma', zorder=9)
            if line_labels:
                axs[i].text(0.025, 0.025, f'{line}', transform=axs[i].transAxes, fontsize=12, verticalalignment='bottom')
            axs[i].set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
            axs[i].set_ylabel('Flux')

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


    def print_GA_results(self, ga_results=None, sigma=2, filename=None):
        """
        Print the results of the genetic algorithm.

        Parameters:
        ga_results (ga_result_summary): The results of the genetic algorithm. If None, uses the results from the last run.
        sigma (int): The number of sigma to use for error bars.
        filename (str): The name of the file to save the results to. If None, results are not saved.
        """

        if ga_results is None:
            if not hasattr(self, 'GA_results'):
                raise ValueError("No GA results found. Run run_GA() first.")
            ga_results = self.GA_results

        fitting.print_GA_results(ga_results, sigma, filename)
