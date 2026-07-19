import os
import numpy as np
import emcee
import corner
from scipy.optimize import minimize
from . import pyGA as GA
import ultranest
import ultranest.popstepsampler
from scipy import stats
from datetime import datetime
from tqdm import trange

import matplotlib.pyplot as plt

from . import model_gen




def interp_model_lines_to_observed(observed_wavelength, model_wavelengths, model_fluxes):
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



# -------------------Cost functions--------------------


def calc_log_likelihoods_with_fuzz(data, error, model):
    data = np.array(data)
    error = np.array(error)
    model = np.array(model)

    log_liklihoods = np.sum(-0.5 * ((data - model)**2 / error**2 + np.log(2*np.pi * error**2)), axis=-1)
    return log_liklihoods


def calc_log_likelihoods(data, error, model):
    data = np.array(data)
    error = np.array(error)
    model = np.array(model)

    log_liklihoods = np.sum(-0.5 * ((data - model)**2 / error**2), axis=-1)
    return log_liklihoods


def calc_chi_square(data, error, model):
    data = np.array(data)
    error = np.array(error)
    model = np.array(model)

    chi_squares = np.sum(((data - model)**2 / error**2), axis=-1)
    return chi_squares



# -------------------MCMC functions--------------------


def run_mcmc(obj, initial_positions=None, n_walkers=None, n_steps=None, fuzz=False, return_sampler=False):
    """
    Run the MCMC simulation to sample the parameter space.

    Parameters:
    initial_positions (array-like): Initial positions of the walkers in the parameter space.
    n_walkers (int): The number of walkers to use in the MCMC simulation.
    n_steps (int): The number of steps to run the MCMC simulation for.
    """

    if n_walkers is None:
        n_walkers = obj.n_walkers
    if n_steps is None:
        n_steps = obj.n_steps

    if fuzz:
        print("Using fuzz in the likelihood calculation.")
        obj.parameters.logf.free()
    else:
        obj.parameters.logf.fix(0.0)

    # reinitialize the free parameters array to catch any changed parameters
    obj.free_parameters = [param for param in obj.parameters.__dict__ if not obj.parameters.__dict__[param].fixed]
    obj.mcmc_free_parameters = obj.free_parameters.copy()


    # Initialize the walkers if not passed to the function
    if initial_positions is None:
        random_positions = []
        for param in obj.free_parameters:
            bounds = obj.parameters.__dict__[param].bounds
            random_positions.append(np.random.uniform(bounds[0], bounds[1], n_walkers*10))

        random_positions = np.array(random_positions).T
        param_set = model_gen.parse_parameter_set(obj, random_positions, free_parameters=obj.mcmc_free_parameters)
        log_prior = obj.log_prior(param_set)
        good_inds = np.where(np.isfinite(log_prior))[0]
        if len(good_inds) < n_walkers:
            raise ValueError("Not enough good initial positions found. Check the parameter bounds to make sure they are compatible with the priors.")
        initial_positions = random_positions[good_inds][:n_walkers]

    elif isinstance(initial_positions, ga_result_summary):
        ga_summary = initial_positions
        initial_positions = []
        for param in obj.free_parameters:
            if param in ga_summary.free_parameters:
                initial_positions.append(np.array(ga_summary.best_fit_model).T[ga_summary.free_parameters.index(param)])
            else:
                initial_positions.append(obj.parameters.__dict__[param].value)
        initial_positions = initial_positions + 1e-4 * np.random.randn(n_walkers, len(obj.free_parameters))

    else:
        initial_positions = initial_positions + 1e-4 * np.random.randn(n_walkers, len(obj.free_parameters))

    # Create the sampler
    sampler = emcee.EnsembleSampler(n_walkers, len(obj.free_parameters), obj.log_probability, args=(fuzz,), vectorize=True)

    # Run the MCMC simulation
    sampler.run_mcmc(initial_positions, n_steps, progress=True)
    obj.emcee_sampler = sampler

    if return_sampler:
        return sampler


def continue_mcmc(obj, sampler=None, n_steps=None, return_sampler=False):
    """
    Continue the MCMC simulation from the last position of the walkers.

    Parameters:
    sampler (array-like): The MCMC sampler to continue.
    n_steps (int): The number of steps to run the MCMC simulation for.
    """

    if not hasattr(obj, 'emcee_sampler'):
        raise ValueError("No MCMC sampler found. Run run_mcmc() first.")

    if n_steps is None:
        n_steps = obj.n_steps

    if sampler is None:
        sampler = obj.emcee_sampler
    # Continue the MCMC simulation
    sampler.run_mcmc(None, n_steps, progress=True)
    obj.emcee_sampler = sampler

    if return_sampler:
        return obj.emcee_sampler


def plot_MCMC_results(obj, sampler = None, burnin=100, thin=1, save_path=None):
    """
    Plot the results of the MCMC simulation.

    Parameters:
    samples (array-like): The samples from the MCMC simulation.
    """

    if sampler is None:
        if not hasattr(obj, 'emcee_sampler'):
            raise ValueError("No MCMC sampler found. Run run_mcmc() first.")
        sampler = obj.emcee_sampler

    samples = sampler.get_chain(discard=burnin)

    fig, axs = plt.subplots(len(obj.mcmc_free_parameters), figsize=(10, 7), sharex=True)
    labels = []
    for i, param in enumerate(obj.mcmc_free_parameters):
        axs[i].plot(samples[:, :, i], "k", alpha=0.3)
        axs[i].set_xlim(0, len(samples))
        
        if obj.parameters.__dict__[param].latex_string is not None:
            param_label = obj.parameters.__dict__[param].latex_string
        else:
            param_label = param
        param_name = param_label
        labels.append(param_name)
        if obj.parameters.__dict__[param].unit is not None:
            param_label += f' ({obj.parameters.__dict__[param].unit})'
        axs[i].set_ylabel(param_label)

    if save_path is not None:
        plt.savefig(save_path.split('.')[0] + '_trace.png')
    else:
        plt.show()

    flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

    fig = corner.corner(flat_samples, labels=labels, show_titles=True)
    plt.show()


def print_MCMC_results(obj, sampler=None, burnin=100, sigma=1, filename=None):
    """
    Print the results of the MCMC simulation.

    Parameters:
    samples (array-like): The samples from the MCMC simulation.
    burnin (int): The number of steps to discard as burn-in.
    """

    if sampler is None:
        if not hasattr(obj, 'emcee_sampler'):
            raise ValueError("No MCMC sampler found. Run run_mcmc() first.")
        sampler = obj.emcee_sampler

    chains = sampler.get_chain(flat=True, thin=1, discard=burnin)

    print("MCMC Results:")
    for i, param in enumerate(obj.mcmc_free_parameters):
        if sigma == 1:
            mcmc = np.percentile(chains[:, i], [16, 50, 84])
        elif sigma == 2:
            mcmc = np.percentile(chains[:, i], [2.5, 50, 97.5])
        else:
            raise ValueError("Sigma must be either 1 or 2.")
        errors = np.diff(mcmc)
        print(f"{param} = ".rjust(15) + f" {mcmc[1]}  ( +{errors[1]:.5f}; -{errors[0]:.5f})")
        # print(f"{param}: {mcmc[1]:.4f} ± {std:.4f}")
    print(f"Number of iterations: {sampler.get_chain().shape[0]}")
    print(f"Number of walkers: {sampler.get_chain().shape[1]}")

    if filename is not None:
        with open(filename, 'w') as f:
            for i, param in enumerate(obj.mcmc_free_parameters):
                if sigma == 1:
                    mcmc = np.percentile(chains[:, i], [16, 50, 84])
                elif sigma == 2:
                    mcmc = np.percentile(chains[:, i], [2.5, 50, 97.5])
                f.write(f"{param} {mcmc[1]} {mcmc[0]} {mcmc[2]}\n")



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
    if ultranest is None:
        raise ImportError("ultranest is required for nested sampling. Install with: pip install ultranest")

    if fuzz:
        obj.parameters.logf.free()
    else:
        obj.parameters.logf.fix(0.0)

    obj.free_parameters = [param for param in obj.parameters.__dict__ if not obj.parameters.__dict__[param].fixed]
    obj.nested_sampling_free_parameters = obj.free_parameters.copy()

    ndim = len(obj.free_parameters)
    param_names = list(obj.free_parameters)
    bounds = [obj.parameters.__dict__[p].bounds for p in obj.free_parameters]

    def prior_transform(cube):
        """Transform unit cube [0,1]^ndim to physical parameter space. Vectorized: cube (n, ndim) -> (n, ndim)."""
        cube = np.array(cube, ndmin=2)
        return np.array([bounds[i][0] + cube[:, i] * (bounds[i][1] - bounds[i][0]) for i in range(ndim)]).T

    def log_probability(theta):
        """Log probability (log prior + log likelihood). Same as MCMC. Non-finite replaced with -1e100 for ultranest."""
        theta = np.array(theta, ndmin=2)
        logp = obj.log_probability(theta, fuzz=fuzz)
        return np.where(np.isfinite(logp), logp, -1e100)

    ns_log_dir = None
    if log_dir is not None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ns_log_dir = os.path.join(log_dir, f'ns_{ts}')
        os.makedirs(ns_log_dir, exist_ok=True)

    sampler = ultranest.ReactiveNestedSampler(
        param_names, log_probability, prior_transform, vectorized=True,
        log_dir=ns_log_dir, resume='overwrite'
    )

    if step_sampler is not None:
        if step_sampler is True or step_sampler == 'slice':
            popsize = kwargs.pop('step_sampler_popsize', 5)
            nsteps = kwargs.pop('step_sampler_nsteps', 20)
            sampler.stepsampler = ultranest.popstepsampler.PopulationSimpleSliceSampler(
                popsize=popsize,
                nsteps=nsteps,
                generate_direction=ultranest.popstepsampler.generate_mixture_random_direction
            )
        else:
            sampler.stepsampler = step_sampler

    result = sampler.run(**kwargs)
    obj.ultranest_result = result
    obj.ultranest_sampler = sampler
    if ns_log_dir is not None:
        obj.ultranest_log_dir = sampler.logs['run_dir'] if sampler.log_to_disk else ns_log_dir
        print(f"Nested sampling results saved to: {obj.ultranest_log_dir}")

    if return_result:
        return result


def _get_nested_sampling_posterior_samples(obj, result=None, thin=1):
    """Get posterior samples (resampled by weight) for plotting/printing."""
    if result is None:
        if not hasattr(obj, 'ultranest_result'):
            raise ValueError("No nested sampling result found. Run run_nested_sampling() first.")
        result = obj.ultranest_result
    if isinstance(result, dict):
        samples = result.get('samples')
        if samples is not None:
            return samples[::thin]
        ws = result.get('weighted_samples', {})
        samples = ws.get('points') if isinstance(ws, dict) else None
        weights = ws.get('weights') if isinstance(ws, dict) else None
    else:
        samples = getattr(result, 'samples', None)
        if samples is not None:
            return samples[::thin]
        ws = getattr(result, 'weighted_samples', None)
        samples = getattr(ws, 'points', None) if ws else None
        weights = getattr(ws, 'weights', None) if ws else None
    if samples is None or weights is None:
        raise ValueError("Ultranest result must have 'samples' or 'weighted_samples' with 'points' and 'weights'.")
    weights = np.asarray(weights) / np.asarray(weights).sum()
    n_resample = min(10000, len(samples))
    indices = np.random.choice(len(samples), size=n_resample, p=weights, replace=True)
    return samples[indices][::thin]


def plot_nested_sampling_results(obj, result=None):
    """
    Plot the results of the nested sampling run: ultranest's run and trace plots (from the
    sampler), then a corner plot of the posterior.

    Parameters:
    result: Ultranest result object. If None, uses self.ultranest_result.
    """
    if result is None:
        if not hasattr(obj, 'ultranest_result'):
            raise ValueError("No nested sampling result found. Run run_nested_sampling() first.")
        result = obj.ultranest_result

    if hasattr(obj, 'ultranest_sampler') and obj.ultranest_sampler is not None:
        obj.ultranest_sampler.plot_run()
        obj.ultranest_sampler.plot_trace()

    flat_samples = _get_nested_sampling_posterior_samples(obj, result=result, thin=1)
    params = obj.nested_sampling_free_parameters
    labels = [p if obj.parameters.__dict__[p].latex_string is None else obj.parameters.__dict__[p].latex_string for p in params]
    fig = corner.corner(flat_samples, labels=labels, show_titles=True)
    plt.show()


def print_nested_sampling_results(obj, result=None):
    """
    Print the results of the nested sampling run using ultranest built-in summary.

    Parameters:
    result: Ultranest result object. If None, uses self.ultranest_result.
    """
    if hasattr(obj, 'ultranest_sampler') and obj.ultranest_sampler is not None:
        obj.ultranest_sampler.print_results()
        return
    if result is None:
        if not hasattr(obj, 'ultranest_result'):
            raise ValueError("No nested sampling result found. Run run_nested_sampling() first.")
        result = obj.ultranest_result
    flat_samples = _get_nested_sampling_posterior_samples(obj, result=result, thin=1)
    n_iterations = result.get('ncall', 0) if isinstance(result, dict) else getattr(result, 'ncall', 0)
    logz = result.get('logz', result.get('logz_mean', np.nan)) if isinstance(result, dict) else getattr(result, 'logz', getattr(result, 'logz_mean', np.nan))
    print("Nested sampling results (fallback):")
    for i, param in enumerate(obj.nested_sampling_free_parameters):
        perc = np.percentile(flat_samples[:, i], [16, 50, 84])
        errors = np.diff(perc)
        print(f"{param} = ".rjust(15) + f" {perc[1]}  ( +{errors[1]:.5f}; -{errors[0]:.5f})")
    print(f"Number of likelihood evaluations: {n_iterations}")
    print(f"Log-evidence (log Z): {logz}")


# -------------------Nelder-Mead functions--------------------


def run_Nelder_Mead(obj, initial_guess=None, return_result = False):
    """
    Run the Nelder-Mead optimization algorithm to find the best-fit parameters.

    Parameters:
    initial_guess (array-like): Initial guess for the parameters.
    return_result (bool): Whether to explicitly return the optimization result.

    Returns:
    result (OptimizeResult): The optimization result represented as a `OptimizeResult` object.
    """

    if initial_guess is None:
        initial_guess = [obj.parameters.__dict__[param].value for param in obj.free_parameters]

    nll = lambda *args: -obj.log_probability(*args)[0]

    result = minimize(nll, initial_guess, method='Nelder-Mead')
    print(result)

    if result.success:
        obj.nm_solution = result.x

    if return_result:
        return result


def plot_NM_fit(self, model_args=None, save_path=None):
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
    best_fit_params = model_gen.parse_parameter_set(self, model_args)[0]

    subplots_dict = {1:[1, 1], 2:[1, 2], 3:[1,3], 4:[2, 2], 5:[2, 3], 6:[2,3], 7:[2,4], 8:[2,4], 9:[3,3], 10:[3, 4], 11:[3, 4], 12:[3, 4], 13:[3, 5], 14:[3, 5], 15:[3, 5], 16:[4,4], 17:[4,5], 18:[4,5], 19:[4,5], 20:[4,5], 21:[4,6], 22:[4,6], 23:[4,6], 24:[4,6], 25:[5,5], 26:[5,6], 27:[5,6], 28:[5,6], 29:[5,6], 30:[5,6], 31:[5,7], 32:[5,7], 33:[5,7], 34:[5,7], 35:[5,7], 36:[6,6], 37:[6,7], 38:[6,7], 39:[6,7], 40:[6,7], 41:[6,8], 42:[6,8], 43:[6,8], 44:[6,8], 45:[6,8]}
    fig, axs = plt.subplots(subplots_dict[len(self.line_list)][0], subplots_dict[len(self.line_list)][1], figsize=(subplots_dict[len(self.line_list)][1]*4, subplots_dict[len(self.line_list)][0]*4))
    axs = axs.ravel()

    for i, line in enumerate(self.line_list.keys()):
        # Get the model wavelengths and fluxes
        model_wavelengths, model_fluxes = self.generate_model_per_line(line, np.array(best_fit_params, ndmin=2))

        obs_inds = np.where((self.observed_wavelength >= self.line_list[line].fit_range[0]) & (self.observed_wavelength <= self.line_list[line].fit_range[1]))[0]
        obs_wavelength = self.observed_wavelength[obs_inds]
        interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, model_wavelengths, model_fluxes)
        # interpolated_fluxes = self.interp_model_lines_to_observed(obs_wavelength, shifted_wavelengths, broadened_fluxes)
        axs[i].plot(obs_wavelength, self.observed_flux[obs_inds], 'k-', label='Observed')
        axs[i].plot(obs_wavelength, interpolated_fluxes.T, 'r-', label='Best Fit')
        axs[i].set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
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
        error_ranges_1sigma = []
        probs = self.probabilities.flatten()
        inds = np.where(probs > 0.05)[0]
        inds_1sigma = np.where(probs > 0.32)[0]
        self.free_parameters = free_parameters
        for i, param in enumerate(free_parameters):
            best_mod.append(best_model[param])
            param_values = self.populations[:, :, i].flatten()
            param_values_2sigma = param_values[inds]
            error_ranges.append([np.min(param_values_2sigma), np.max(param_values_2sigma)])
            param_values_1sigma = param_values[inds_1sigma]
            error_ranges_1sigma.append([np.min(param_values_1sigma), np.max(param_values_1sigma)])
        self.best_fit_model = best_mod
        self.best_fit_errors_2sigma = error_ranges
        self.best_fit_errors_1sigma = error_ranges_1sigma


        self.best_fitness = best_fitness


def _translate_params_to_GA(obj):
    """
    Translate the parameters to a format suitable for the genetic algorithm.

    Returns:
    ga_params (array-like): The parameters in the format suitable for the genetic algorithm.
    """
    ga_params = GA.Parameters()
    for param in obj.free_parameters:
        name = obj.parameters.__dict__[param].name
        bounds = obj.parameters.__dict__[param].bounds
        ga_params.add(name, float(bounds[0]), float(bounds[1]), int(6))

    return ga_params


def _translate_GA_chromosomes(free_parameters, ga_params, chromosome_list):
    """
    Translate the raw GA chromosomes back into the parameter set format used by SpecFANN.

    Parameters:
    ga_params (array-like): The parameters in the format suitable for the genetic algorithm.
    chromosome_list (array-like): The list of raw chromosomes used by the GA.

    Returns:
    ga_params (array-like): The parameters in the format suitable for genetic algorithms.
    """
    keys = list(ga_params.keys())
    keys = free_parameters
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


def run_GA(obj, n_generations=300, population_size=50, return_result=False):
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
    obj.parameters.logf.fix(0.0)
    # reinitialize the free parameters array to catch any changed parameters
    obj.free_parameters = [param for param in obj.parameters.__dict__ if not obj.parameters.__dict__[param].fixed]

    # translate the parameters to a format suitable for the genetic algorithm
    ga_params = _translate_params_to_GA(obj)

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

        model_args = _translate_GA_chromosomes(obj.free_parameters, ga_params, population_raw)
        generation_parameters.append(model_args)

        # calculate chi2 of each model in the population.
        reduced_chi_squares = obj.reduced_chi_square(model_args)
        generation_reduced_chi_squares.append(reduced_chi_squares)

        # calculate fitness of each model in the population.
        fitnesses = len(obj.line_list.keys()) / reduced_chi_squares
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

    generation_probabilities = _calculate_GA_probabilities(obj, np.array(generation_reduced_chi_squares))
    obj.GA_results = ga_result_summary(ga_params, population_size, n_generations, generation_reduced_chi_squares, generation_fitnesses, generation_probabilities, generation_parameters, best_mod, best_fitness, obj.free_parameters)

    if return_result:
        return obj.GA_results


def _calculate_GA_probabilities(obj, red_chi2s):
    """
    Calculate the probabilities of each model in the population based on their chi-squared values.

    Parameters:
    red_chi2s (array-like): The chi-squared values for each model in the population.

    Returns:
    probabilities (array-like): The probabilities of each model in the population.
    """

    # calculate degrees of freedom
    degrees_of_freedom = 0
    for line in obj.line_list.keys():
        degrees_of_freedom += len(np.where((obj.observed_wavelength >= obj.line_list[line].fit_range[0]) & (obj.observed_wavelength <= obj.line_list[line].fit_range[1]))[0])

    degrees_of_freedom -= len(obj.free_parameters)

    # normalize chi-squared values
    chi2s = (red_chi2s * degrees_of_freedom) / np.min(red_chi2s)

    probabilities = stats.chi2.sf(chi2s, degrees_of_freedom)

    return probabilities


def plot_GA_results(obj, ga_results=None, diagnostic = 'fitness', sigma=2, save_path=None):
    """
    Plot the results of the genetic algorithm.

    Parameters:
    ga_results (ga_result_summary): The results of the genetic algorithm.
    diagnostic (str): The diagnostic to plot. Options are 'fitness', 'probability', or 'chi_square'.
    sigma (int): The number of sigma to use for error bars.
    save_path (str): Path to save the plot. If None, the plot will not be saved.
    """

    if ga_results is None:
        if not hasattr(obj, 'GA_results'):
            raise ValueError("No GA results found. Run run_GA() first.")
        ga_results = obj.GA_results


    subplots_dict = {1:[1, 1], 2:[1, 2], 3:[1,3], 4:[2, 2], 5:[2, 3], 6:[2,3], 7:[2,4], 8:[2,4], 9:[3,3], 10:[3, 4], 11:[3, 4], 12:[3, 4], 13:[3, 5], 14:[3, 5], 15:[3, 5], 16:[4,4], 17:[4,5], 18:[4,5], 19:[4,5], 20:[4,5], 21:[4,6], 22:[4,6], 23:[4,6], 24:[4,6], 25:[5,5], 26:[5,6], 27:[5,6], 28:[5,6], 29:[5,6], 30:[5,6], 31:[5,7], 32:[5,7], 33:[5,7], 34:[5,7], 35:[5,7], 36:[6,6], 37:[6,7], 38:[6,7], 39:[6,7], 40:[6,7], 41:[6,8], 42:[6,8], 43:[6,8], 44:[6,8], 45:[6,8]}
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

    if sigma == 1:
        best_fit_errors = ga_results.best_fit_errors_1sigma
    elif sigma == 2:
        best_fit_errors = ga_results.best_fit_errors_2sigma
    else:
        raise ValueError("Invalid sigma value. Choose 1 or 2.")

    diagnostic_values = np.array(ga_results.__dict__[diagnostic_param]).flatten()
    for i, param in enumerate(ga_results.free_parameters):
        # Plot the probabilities for each generation
        param_values = ga_results.populations[:, :, i].flatten()
        generations = np.array([np.arange(ga_results.n_generations)]*ga_results.population_size).T.flatten()
        generations.flatten()
        axs[i].scatter(param_values, diagnostic_values, c= generations, cmap='viridis', alpha=0.5, rasterized=True)
        if obj.parameters.__dict__[param].latex_string is not None:
            param_label = obj.parameters.__dict__[param].latex_string
        else:
            param_label = param
        param_name = param_label
        if obj.parameters.__dict__[param].unit is not None:
            param_label += f' ({obj.parameters.__dict__[param].unit})'
        axs[i].set_xlabel(param_label)
        axs[i].set_ylabel(title)
        axs[i].set_xlim(ga_results.ga_params[param].min, ga_results.ga_params[param].max)
        axs[i].set_ylim(0, np.max(diagnostic_values)*1.1)
        axs[i].set_title(r'%s = $%0.2f \pm \genfrac{}{}{0}{}{%0.2f}{%0.2f}$'%(param_name, ga_results.best_fit_model[i], best_fit_errors[i][1] - ga_results.best_fit_model[i], ga_results.best_fit_model[i] - best_fit_errors[i][0]))
        axs[i].fill_betweenx([0, np.max(diagnostic_values)*1.1], best_fit_errors[i][0], best_fit_errors[i][1], color='lightcoral', alpha=0.3)

    if i < len(axs) - 1:
        for j in range(i+1, len(axs)):
            axs[j].axis('off')

    if obj.object_name is not None:
        plt.suptitle(f'{obj.object_name} GA fit', fontsize=16)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def print_GA_results(ga_results=None, sigma=2, filename=None):
    """
    Print the results of the genetic algorithm.

    Parameters:
    ga_results (ga_result_summary): The results of the genetic algorithm. If None, uses the results from the last run.
    sigma (int): The number of sigma to use for error bars.
    filename (str): The name of the file to save the results to. If None, results are not saved.
    """

    if sigma == 1:
        best_fit_errors = ga_results.best_fit_errors_1sigma
    elif sigma == 2:
        best_fit_errors = ga_results.best_fit_errors_2sigma
    else:
        raise ValueError("Invalid sigma value. Choose 1 or 2.")

    print(f"GA Results Summary ({sigma}-sigma):")
    for i, param in enumerate(ga_results.free_parameters):
        print(f"{param} = ".rjust(15) + f" {ga_results.best_fit_model[i]}  ( +{best_fit_errors[i][1] - ga_results.best_fit_model[i]:.3f}; -{ga_results.best_fit_model[i] - best_fit_errors[i][0]:.3f})")
    print(f"Best fitness: {ga_results.best_fitness}")
    print(f"Number of generations: {ga_results.n_generations}")
    print(f"Population size: {ga_results.population_size}")

    if filename is not None:
        with open(filename, 'w') as f:
            for i, param in enumerate(ga_results.free_parameters):
                f.write(f"{param} {ga_results.best_fit_model[i]} {best_fit_errors[i][0]} {best_fit_errors[i][1]}\n")
