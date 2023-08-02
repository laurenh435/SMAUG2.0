# match_spectrum.py
# Opens observed spectra files; smooths and interpolates synthetic 
# spectrum to match observed spectrum
#
# - open_obs_file: opens observed spectrum
# - smooth_gauss_wrapper: Python wrapper for Fortran code that 
#	    smooths and interpolates synth spectrum to match observed
# 
# Created 9 Feb 18
# Updated 22 June 18
#
# Edit SMAUG to make general -LEH 5/31/2023
###################################################################

import os
import sys
import numpy as np
import math
from astropy.io import fits
from smooth_gauss import smooth_gauss
import matplotlib.pyplot as plt

def open_obs_file(filename, retrievespec=None, specparams=False, objname=None, coords=False, hires=False, inputcoords=None):
	"""Open .fits.gz files with observed spectra.

	Inputs:
	filename - name of file to open

	Keywords:
	retrievespec - if None (default), output number of stars in file; 
					else, retrieve spectrum of nth star from the file (where n = retrievespec)

	specparams	 - if True, output temp, logg, fe, alpha of nth star in file 
					(only works when retrievespec is not None)

	objname 	 - if not None, finds parameters for spectrum of objname
					(only works when specparams=True)

	coords 		 - if True, output the coordinates of the stars in file

	hires 		 - open hi-res spectra (different format than Moogify)

	inputcoords  - if not None (default), use to do coordinate matching as well as object name matching

	Outputs:
	wvl  - rest wavelength array for nth star
	flux - flux array for nth star
	ivar - inverse variance array for nth star
	"""

	if hires:

		print('Opening ', filename)

		fluxtot = []
		wvl = []
		dlam = []

		for i in [17]:
			hdu = fits.open(filename+'_0'+str(i)+'.fits')

			hdr = hdu[0].header

			flux = hdu[0].data
			fluxtot.append(flux)

			# Get wavelength array
			for j in range(len(flux)):
				wvl.append(j*hdr['CDELT1'] + hdr['CRVAL1'])
				dlam.append(0.128/2.355)
				#dlam.append(0.7086)

		return np.asarray(wvl), np.hstack(fluxtot[:]), np.asarray(dlam)

	print('Opening ', filename)
	hdu1 = fits.open(filename)
	#print(hdu1[1].header)
	data = hdu1[1].data

	if retrievespec is not None:

		# If don't need to return other parameters, just return spectrum
		if not specparams:

			# Open SPEC1D file to get optimal extraction
			#spec1d 	= data['SPEC1DFILE'][retrievespec]
			#print('spec1dfile:', data['SPEC1DFILE'][0])

			#manually put in file path for now because Moogify points to a file on Evan's computer?
			#spec1d = '/mnt/c/Research/Spectra/bscl1/IndividualSpectra/spec1d.bscl1.001.1002447.fits.gz'

			# Fix typos in some of the filenames
			#if 'CVnIa' in spec1d:
				#spec1d = spec1d.replace('CVnIa_1200B_','CVnIa_1200B/')
			#elif 'LeoIb' in spec1d:
				#spec1d = spec1d.replace('LeoIb_1200B/', 'LeoIb_1200B/2018mar19/')


			#hdu2 	= fits.open(spec1d)
			#print('got hdu2')
			# wvl_blue 	= hdu2[3].data['LAMBDA'][0]
			# flux_blue 	= hdu2[3].data['SPEC'][0]
			# ivar_blue 	= hdu2[3].data['IVAR'][0]
			# wvl_red 	= hdu2[4].data['LAMBDA'][0]
			# flux_red 	= hdu2[4].data['SPEC'][0]
			# ivar_red 	= hdu2[4].data['IVAR'][0]

			# Concatenate blue and red parts of optimal extraction spectrum
			# wvl  = np.hstack((wvl_blue, wvl_red))
			# flux = np.hstack((flux_blue, flux_red))
			# ivar = np.hstack((ivar_blue, ivar_red))

			# Get spectrum of a single star
			name = data['OBJNAME'][retrievespec]
			print('objname:', name)
			wvl  = data['LAMBDA'][retrievespec]
			flux = data['SPEC'][retrievespec]
			ivar = data['IVAR'][retrievespec]
			dlam = data['DLAM'][retrievespec]
			dlam = 0.7086*np.ones(len(dlam))

			# Check that measured velocity is good
			checkvel = data['GOOD'][retrievespec]
			if checkvel == 0:
				raise ValueError('Velocity is not trustworthy! Skip this star!')

			# Correct for wavelength
			zrest = data['ZREST'][retrievespec]
			wvl = wvl / (1. + zrest)
			print('Redshift: ', zrest)

			return name, wvl, flux, ivar, dlam, zrest

		# Else, return other parameters
		else:

			# If necessary, match object name from other output file
			if objname is not None:

				# Get index of entry that matches object name of spectrum
				namearray = data['OBJNAME']
				index 	  = np.where(namearray==objname)
				#print('test', objname, index)

				# Check that such an entry exists
				if len(index[0]) > 0:
					#print(index[0])

					# If needed, do additional coordinate matching
					if len(index[0]) > 1:
						idxRA = np.argmin(np.abs(inputcoords[0] - data['RA']))
						idxDec = np.argmin(np.abs(inputcoords[1] - data['Dec']))

						if idxRA != idxDec:
							print(idxRA, idxDec)
							raise ValueError('Having trouble with coordinate matching!')
						else:
							idx = idxRA

					else:
						idx = index[0]

					temp 	= int(data['TEFF'][idx])
					logg 	= data['LOGG'][idx]
					fe 		= data['FEH'][idx]
					alpha 	= data['ALPHAFE'][idx]

					fe_err  = data['FEHERR'][idx]
					#zrest = data['ZREST'][index[0]]

					print('Parameters: ', temp, logg, fe, alpha)
					#print('Redshift: ', zrest)

				# If not, then missing best-fit parameters; just end the program
				else:
					raise ValueError('Spectrum not properly reduced! No best-fit parameters')

			else:
				temp 	= int(data['TEFF'][retrievespec])
				logg 	= data['LOGG'][retrievespec]
				fe 		= data['FEH'][retrievespec]
				alpha 	= data['ALPHAFE'][retrievespec]

				fe_err 	= data['FEHERR'][retrievespec]
				#zrest = data['ZREST'][retrievespec]

				print('Parameters: ', temp, logg, fe, alpha)

			return temp, logg, fe, alpha, fe_err #, zrest

	# Else, return number of stars in file (if coords=False) or coordinates of stars in file (if coords = True)
	else:

		if coords:
			RA = data['RA']
			dec = data['DEC']

			return RA, dec

		else:
			wavearray = data['LAMBDA']
			return len(wavearray)

def smooth_gauss_wrapper(lambda1, spec1, lambda2, dlam_in):
	"""
	Written by I. Escala

	A wrapper around the Fortran routine smooth_gauss.f, which
	interpolates the synthetic spectrum onto the wavelength array of the
	observed spectrum, while smoothing it to the specified resolution of the
	observed spectrum.
	Adapted into Python from IDL (E. Kirby)

	Parameters
	----------
	lambda1: array-like: synthetic spectrum wavelength array
	spec1: array-like: synthetic spectrum normalized flux values
	lambda2: array-like: observed wavelength array
	dlam_in: float, or array-like: full-width half max resolution in Angstroms
			to smooth the synthetic spectrum to, or the FWHM as a function of wavelength
			  
	Returns
	-------
	spec2: array-like: smoothed and interpolated synthetic spectrum, matching observations
	"""

	if not isinstance(lambda1, np.ndarray): lambda1 = np.array(lambda1)
	if not isinstance(lambda2, np.ndarray): lambda2 = np.array(lambda2)
	if not isinstance(spec1, np.ndarray): spec1 = np.array(spec1)
	
	#Make sure the synthetic spectrum is within the range specified by the
	#observed wavelength array!!!

	n2 = lambda2.size; n1 = lambda1.size
	
	# def findex(u, v):
	# 	"""
	# 	Return the index, for each point in the synthetic wavelength array, that corresponds
	#     to the bin it belongs to in the observed spectrum
	# 	e.g., lambda1[i-1] <= lambda2 < lambda1[i] if lambda1 is monotonically increasing
	#     The minus one changes it such that lambda[i] <= lambda2 < lambda[i+1] for i = 0,n2-2
	# 	in accordance with IDL
	# 	"""
	# 	result = np.digitize(u, v)-1 # Gives bin i for each lambda1[i] such that lambda2[i] <= lambda1 < lambda2[i+1]
	# 	w = [int((v[i] - u[result[i]])/(u[result[i]+1] - u[result[i]]) + result[i]) for i in range(n2)]
	# 	return np.array(w)
	
	#f = findex(lambda1, lambda2)
	f = np.digitize(lambda2, lambda1)
	f[-1] -= 1

	#print(f[0:20])
	#print(lambda1[0:20])
	#print(lambda2[0:20])

	#print(f)
	
	#Make it such that smooth_gauss.f takes an array corresponding to the resolution
	#each point of the synthetic spectrum will be smoothed to
	if isinstance(dlam_in, list) or isinstance(dlam_in, np.ndarray): dlam = dlam_in
	else: dlam = np.full(n2, dlam_in)
	dlam = np.array(dlam)
	
	dlambda1 = np.diff(lambda1)
	dlambda1 = dlambda1[dlambda1 > 0.]
	halfwindow = int(np.ceil(1.1*5.*dlam.max()/dlambda1.min()))
	
	#Python wrapped fortran implementation of smooth gauss
	spec2 = smooth_gauss(lambda1, spec1, lambda2, dlam, f, halfwindow)
	#temp = np.zeros(500); gauss = np.zeros(500)

	return spec2