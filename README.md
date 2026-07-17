# SpecFANN 1.0.0

Introduction
------------
SpecFANN (Spectroscopic Fitting via Artificial Neural Networks) is a deep learning based FASTWIND emulator and spectroscopic fitting suite written in Python.  Using a series of neural networks trained to emulate individual FASTWIND line profiles, SpecFANN allows the user to very quickly produce forward models for any subset of the ~140 included spectral lines, and allows the user to fit observed spectra using various minimization techniques and samplers. SpecFANN is also built in a flexible way such that if users have thier own trained neural networks, these can be dropped in to replace the provided networks, while still being able to use all of the framework set up here. For details on how the code works, please see the corresponding release paper: (coming soon)
<!-- [(coming soon)](https://ui.adsabs.harvard.edu/abs/2020arXiv200309008A/abstract) -->

Installation
------------
There are several ways to install SpecFANN, depending on your specific needs.

### pip
The easiest way to install SpecFANN is via pip:

        pip install specfann

### From source
If you prefer, you can install SpecFANN from the source with the following steps:

*   Clone the git repository to create a local copy.

        git clone https://github.com/MichaelAbdul-Masih/SpecFANN.git

*   Once downloaded, you can perform a local pip install:

        cd SpecFANN/
        pip install .

### Installing the neural network bundles
In addition to the git repository, you will need to download the bundle of neural networks that SpecFANN uses to generates models. These can be downloaded [here](https://cloud.iac.es/index.php/s/H7FdjCcJcaZJSzN).  Alternatively, we have provided a convenience function that will retrieve the bundles directly from the cloud and store them in the default directory `~/.specfann/bundles/`.  This can be done by calling the following specfann function: 

        import specfann
        specfann.install_bundle(bundle_name='MW_v1.4', bundle_path = None)

Where `bundle_name` is the name of the bundle and bundle_path is an optional parameter if you prefer to store the bundle in a location different than the default.  For now, two bundles are available corresponding to Milky Way and SMC metallicity: the `MW_v1.4.tgz` and `SMC_v1.0` bundles (~2GB each).  While SpecFANN is written in a way where the bundle can be stored in any location, we recommend that to begin, you use the convenience function provided above, however if you wish to set up the bundles manually, you can place the bundle in your preferred directory and untar it there.  By default, SpecFANN will look for the `MW_v1.4` bundle in the `~/.specfann/bundles/` directory but an alternative bundle and/or bundle file path can be specified later.

        mv ~/Downloads/MW_v1.4.tgz ~/.specfann/bundles/.
        cd ~/.specfann/bundles/
        tar -xvzf MW_v1.4.tgz


### Test the Installation
To make sure the installation was successful, ensure you are in the correct python environment, initiate a python instance and run the following commands:

        import specfann
        s = specfann.single_star() 

If there is no error message following the second command, then things should be working properly.  If you placed the bundle in a different location than what was recommended above, then in the call to `specfann.single_star()`, you will need to pass the path to the bundle using the `bundle_path = ` argument:

        import specfann
        s = specfann.specfann(bundle_path = 'path/to/MW_v1.4') 


Getting Started
---------------
We've prepared a jupyter notebook that shows the SpecFANN workflow, and what customization options are available to fit your specific science case.  This can be found in the `SpecFANN_Tutorial.ipynb` notebook in the base SpecFANN directory. You will also need the example spectrum in the `data` folder. We recommend that you start here to learn the basics.


Change Log
---------------

### 1.0.0 - official release of SpecFANN v1.0

* If upgrading from a prerelease version (i.e. v0.X.X), you will need to do a clean install. Please note that specfann saved files from previous versions will not work with v1.0.0.  Moving forward we will do our best to ensure backwards compatibility.  Important changes will be documented in the Change Log for each new release.
