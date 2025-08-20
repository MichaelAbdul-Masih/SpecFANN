# SpecFANN 0.1.0

Introduction
------------
SpecFANN (Spectroscopic Fitting via Artificial Neural Networks) is a deep learning based FASTWIND emulator and spectroscopic fitting suite written in Python.  Using a series of neural networks trained to emulate individual FASTWIND line profiles, SpecFANN allows the user to very quickly produce forward models for any subset of the ~140 included spectral lines, and allows the user to fit observed spectra using various minimization techniques and samplers. SpecFANN is also built in a flexible way such that if users have thier own trained neural networks, these can be dropped in to replace the provided networks, while still being able to use all of the framework set up here. For details on how the code works, please see the corresponding release paper: (coming soon)
<!-- [(coming soon)](https://ui.adsabs.harvard.edu/abs/2020arXiv200309008A/abstract) -->

Installation
------------
*   Clone the git repository to create a local copy.

        git clone https://github.com/MichaelAbdul-Masih/SpecFANN.git

*   SpecFANN is written in Python 3 and has several dependencies that are needed to make it run.  For your convenience, we have provided an environment file (SpecFANN_env.yml), which you can use to create an environment that already has all of these dependencies.  If you are using Anaconda you can create the environment as follows:

        cd SpecFANN/
        conda env create -f SpecFANN_env.yml
        conda activate SpecFANN_env

*   If you prefer to create the environment yourself, the minimum package requirements can be found below:

        astropy 7.0.0
        numpy 1.26.4
        matplotlib 3.10.0
        scipy 1.15.3
        keras 3.6.0
        corner 2.2.3
        tqdm 4.67.1

*   In addition to the git repository, you will need to download the bundle of neural networks that SpecFANN uses to generates models. These can be downloaded [here](https://cloud.iac.es/index.php/s/H7FdjCcJcaZJSzN).  For now, only one bundle is available corresponding to Milky Way metallicity: the `MW_v1.0.tgz` bundle (~2GB).  While SpecFANN is written in a way where the bundle can be stored in any location, we recommend that to begin, you place the bundle in the `bundles` folder within the git repo directory and untar it there.  By default, SpecFANN will look for the `MW_v1.0` bundle in the relative path (to specfann.py): `models/MW_v1.0`, but an alternative bundle and/or bundle file path can be specified later.

        mv ~/Downloads/MW_v1.0.tgz ~/SpecFANN/bundles/.
        cd ~/SpecFANN/bundles/
        tar -xvzf MW_v1.0.tgz


SpecFANN can be run directly in the git directory, or if you prefer, you can add the git directory to your Python path and you can run SpecFANN from any location.  


### Test the Installation
To make sure the installation was successful, cd into the SpecFANN git directory, ensure you are in the correct conda environment, initiate a python instance and run the following commands:

        import specfann
        s = specfann.specfann() 

If there is no error message following the second command, then things should be working properly.  If you placed the bundle in a different location than what was recommended above, then in the call to `specfann.specfann()`, you will need to pass the path to the bundle using the `bundle_path = ` argument:

        import specfann
        s = specfann.specfann(bundle_path = 'path/to/MW_v1.0') 


Getting Started
---------------
We've prepared a jupyter notebook that shows the SpecFANN workflow, and what customization options are available to fit your specific science case.  This can be found in the `SpecFANN_Tutorial.ipynb` notebook in the base SpecFANN directory.  We recommend that you start here to learn the basics.
