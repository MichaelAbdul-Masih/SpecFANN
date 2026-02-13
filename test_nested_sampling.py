"""
Test script: nested sampling inference (ultranest).
Tries to follow the tutorial notebook in terms of setup and parameters.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from ultranest.mlfriends import RobustEllipsoidRegion
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

# Nested sampling inference (ultranest)
s.n_evaluations = 0
s.run_nested_sampling(jitter=False,
                     step_sampler=None,
                     min_num_live_points=500,
                     frac_remain=0.3,
                     dlogz=0.5,
                     max_num_improvement_loops=1,
                     update_interval_volume_fraction=0.5,
                     region_class=RobustEllipsoidRegion,
                     #max_ncalls=500000 # Use this to limit the number of likelihood evaluations
                     )

result = s.ultranest_result
sampler = s.ultranest_sampler
print(f"Total evaluations: {s.n_evaluations}")
print(f"Log evidence: {result['logz']:.2f} ± {result['logzerr']:.2f}")

sampler.print_results()
sampler.plot()
plt.show()

s.plot_nested_sampling_fit(n_draw=1000)
