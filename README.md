# SMAUG2.0

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About

The [original SMAUG code](https://github.com/mdlreyes/SMAUG) was written by M. A. C. de los Reyes to measure Mn abundances. This version of SMAUG has been edited to measure any element in medium-resolution DEIMOS blue spectra.

Much of this README is echoing the original SMAUG repository here for your convenience.

### Prerequisites

The following software packages are required:
* astropy
* matplotlib
* numpy
* pandas
* scipy
* MOOG (SMAUG uses the [moog17scat version from Alex Ji](https://github.com/alexji/moog17scat))

### Installation
Follow the usual steps to clone the repo:
```sh
git clone https://github.com/laurenh435/SMAUG2.0.git
```

Wrap the FORTRAN code smooth_gauss to Python using the following:
```sh
f2py -c -m smooth_gauss smooth_gauss.f
```

## Usage

An example MOOG parameter file that is created by SMAUG to generate a synthetic spectrum is included in this repository (SMAUGexample.par). 

To run SMAUG "simply," an example usage is included in output.py. SMAUGsplit.sh and SMAUGsplitGC.sh are examples of how SMAUG can be run for multiple stars simultaneously to decrease computation time.

Optional: Depending on the line list, may want to change which isotope ratios are specified for MOOG. For the
   s-process runs, Ba, Nd, Eu, CH, and CN are specified. This can be changed in the createPar function in run_moog.py

SMAUG2.0 proceeds roughly as follows:
1. Getting started.
   * Make linelists for 20 angstrom regions around each line of the element of interest (make_linelists.py)
2. Prepare the spectrum.
   * Open the spectrum files and extract stellar parameters (math_spectrum.py)
   * Continuum-normalize the spectrum (continuum_div.py)
   * Perform wavelength correction (wvl_corr_even_newer.py)
   * Mask the observed spectrum to only contain small (20 A) regions around the lines of interest (continuum_div.py)
3. Fit the abundance of the element of interest by creating synthetic spectra with MOOG and comparing them to the observed spectrum.
   * Create synthetic spectra (interp_atmosphere.py and run_moog.py)
   * Fit the observed spectrum with the synthetic spectrum in each of the line regions simultaneously (chi_sq.py)
4. Make some plots (make_plots.py)

<!-- CONTACT -->
## Contact

Lauren Henderson - lhender6@nd.edu

Project Link: [https://github.com/laurenh435/SMAUG2.0](https://github.com/laurenh435/SMAUG2.0)

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

[SMAUG](https://github.com/mdlreyes/SMAUG)
[README.md template](https://github.com/othneildrew/Best-README-Template)