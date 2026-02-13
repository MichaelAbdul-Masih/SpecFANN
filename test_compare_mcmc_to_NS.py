"""
Test script: load SpecFANN NN, run GA then MCMC inference, then compare to an already finished nested sampling run (which was saved in the nested_sampling_results directory).
Optionally, can load existing MCMC posterior points from a file.
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner
import specfann

# Change to script directory so data path works
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load NN (default bundle)
s = specfann.specfann()

# Read observed spectrum
wavelength, flux = np.loadtxt("data/HD214680.txt").T
s.load_observed_data(wavelength, flux)

# Line list (same as tutorial)
s.add_line("HDELTA")
s.add_line("HGAMMA")
s.add_line("HEI4471", fit_range=[4460, 4480])
s.add_line("HEII4541")
print("Full line list:", s.sbf.full_line_list())
print("Selected lines:", list(s.line_list.keys()))

# Fix parameters (same as tutorial)
s.parameters.c.fix()
s.parameters.o.fix()
s.parameters.si.fix()
s.parameters.r.fix(6)
s.parameters.vmacro.fix(10)
s.parameters.inst_res.fix(85000)
s.parameters.gamma.set_bounds([-200, 200])
s.parameters.summary()

# GA + MCMC, or load existing MCMC posterior
mcmc_points_file = "mcmc_posterior_points.txt"
if os.path.isfile(mcmc_points_file):
    with open(mcmc_points_file, "r") as f:
        mcmc_param_names = f.readline().strip().split()
    mcmc_flat = np.loadtxt(mcmc_points_file, skiprows=1)
    mcmc_flat = np.array(mcmc_flat, ndmin=2)
    print(f"Loaded MCMC posterior points from {mcmc_points_file}")
    ran_mcmc = False
else:
    # GA to get a good starting point for MCMC
    s.n_evaluations = 0
    s.run_GA(n_generations=300, population_size=50)
    ga_evals = s.n_evaluations
    print(f"Parameter evaluations (GA): {ga_evals}")
    # MCMC inference using GA best fit as initial positions (same as notebook)
    s.n_evaluations = 0
    s.run_mcmc(initial_positions=s.GA_results.best_fit_model, n_walkers=50, n_steps=5000)
    mcmc_evals = s.n_evaluations
    print(f"Parameter evaluations (MCMC): {mcmc_evals}")
    print(f"Parameter evaluations (total): {ga_evals + mcmc_evals}")
    burnin = 1000
    thin = 10
    mcmc_flat = s.emcee_sampler.get_chain(discard=burnin, flat=True, thin=thin)
    mcmc_param_names = list(s.mcmc_free_parameters)
    np.savetxt(
        mcmc_points_file,
        mcmc_flat,
        header=" ".join(mcmc_param_names),
        comments="",
    )
    ran_mcmc = True

# Fetch the most recent nested sampling run and make an overlay corner plot
def _latest_nested_sampling_equal_weighted_post(log_dir: Path) -> Path:
    if not log_dir.exists():
        raise FileNotFoundError(f"Nested sampling results directory not found: {log_dir}")
    run_dirs = sorted([p for p in log_dir.glob("ns_*") if p.is_dir()], key=lambda p: p.name)
    if len(run_dirs) == 0:
        raise FileNotFoundError(f"No nested sampling runs found under: {log_dir}")
    latest_run = run_dirs[-1]
    chain_path = latest_run / "chains" / "equal_weighted_post.txt"
    if not chain_path.exists():
        raise FileNotFoundError(f"Nested sampling chain file not found: {chain_path}")
    return chain_path


ns_chain_path = _latest_nested_sampling_equal_weighted_post(Path("nested_sampling_results"))
with open(ns_chain_path, "r") as f:
    ns_param_names = f.readline().strip().split()
ns_samples_all = np.loadtxt(ns_chain_path, skiprows=1)
ns_samples_all = np.array(ns_samples_all, ndmin=2)

common_params = [p for p in mcmc_param_names if p in ns_param_names]
if len(common_params) == 0:
    raise ValueError(
        "No overlapping parameter names between MCMC and nested sampling.\n"
        f"MCMC params: {mcmc_param_names}\n"
        f"Nested sampling params: {ns_param_names}"
    )

mcmc_idx = [mcmc_param_names.index(p) for p in common_params]
ns_idx = [ns_param_names.index(p) for p in common_params]
mcmc_samples = mcmc_flat[:, mcmc_idx]
ns_samples = ns_samples_all[:, ns_idx]

# 1/2/3 sigma credible levels
levels = [0.6827, 0.9545, 0.9973]

fig = corner.corner(
    mcmc_samples,
    labels=common_params,
    show_titles=True,
    title_fmt=".5g",
    plot_datapoints=False,
    plot_density=False,
    levels=levels,
    color="C0",
    contour_kwargs={"linewidths": 1.25},
    no_fill_contours=True,
)
corner.corner(
    ns_samples,
    fig=fig,
    labels=common_params,
    show_titles=False,
    plot_datapoints=False,
    plot_density=False,
    levels=levels,
    color="C1",
    contour_kwargs={"linewidths": 1.25},
    no_fill_contours=True,
)

handles = [
    mlines.Line2D([], [], color="C0", label="MCMC"),
    mlines.Line2D([], [], color="C1", label="Nested sampling"),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))
plt.show()

if ran_mcmc:
    s.plot_MCMC_fit(burnin=1000)
    s.print_MCMC_results(burnin=1000)
