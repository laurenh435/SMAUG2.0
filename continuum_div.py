# continuum_div.py
# - masks parts of observed spectra (mask_obs);
# - obtains synthetic spectrum from Ivanna's grid (get_synth); 
# - divides obs/synth, fits spline, and divides obs/spline (divide_spec)
# - necessary input: FITS file that just has list of lines for measuring the abundance of 'element'
# 
# Created 22 Feb 18 by M. de los Reyes
# 
# edited to make SMAUG general -LEH 5/31/2023
###################################################################

import os
import sys
import numpy as np
import numpy.ma as ma

np.set_printoptions(threshold=np.inf)

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import math
import gzip
from astropy.io import fits
from smooth_gauss import smooth_gauss
from interp_atmosphere import interpolateAtm
from match_spectrum import open_obs_file, smooth_gauss_wrapper
from scipy.interpolate import splrep, splev
from make_plots import make_plots

def get_synth(obswvl, obsflux, ivar, dlam, synth=None, temp=None, logg=None, fe=None, alpha=None, stravinsky=False):
	"""Get synthetic spectrum and smooth it to match observed spectrum.

    Inputs:
    For observed spectrum --
    obswvl  -- wavelength array of observed spectrum
    obsflux -- flux array of observed spectrum
    ivar 	-- inverse variance array of observed spectrum
    dlam 	-- FWHM of observed spectrum

    Keywords:
    synth -- if None, get synthetic spectrum from Ivanna's grid;
    		 else, use synth (should be a list of arrays [synthflux, synthwvl])

		    For synthetic spectrum from Ivanna's grid -- 
		    temp  -- effective temperature (K)
		    logg  -- surface gravity
		    fe 	  -- [Fe/H]
		    alpha -- [alpha/Fe]

    Outputs:
    synthflux -- synthetic flux array
    obsflux   -- observed flux array
    obswvl    -- wavelength array
    ivar 	  -- inverse variance array
    """

	# Get synthetic spectrum from grid
	if synth is None:

		# Use modified version of interpolateAtm to get blue synthetic spectrum from Ivanna's grid
		if stravinsky:
			synthflux_blue = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/raid/gridie/bin/', stravinsky=stravinsky)
		else:
			synthflux_blue = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/mnt/c/SpectraGrids/gridie/bin/', stravinsky=stravinsky)
		wvl_range_blue = np.arange(4100., 6300.+0.14, 0.14) #THIS WILL NEED TO CHANGE WHEN WVL RANGE EXPANDS?
		synthwvl_blue  = 0.5*(wvl_range_blue[1:] + wvl_range_blue[:-1])
		#print('bluest gridie:',synthwvl_blue[0])

		# Also get synthetic spectrum for redder part from Evan's grid
		if stravinsky:
			synthflux_red = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/raid/grid7/bin/', stravinsky=stravinsky)
			synthwvl_red  = np.fromfile('/raid/grid7/bin/lambda.bin')
		else:
			synthflux_red = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/mnt/c/SpectraGrids/grid7/bin/', stravinsky=stravinsky)
			synthwvl_red  = np.fromfile('/mnt/c/SpectraGrids/grid7/bin/lambda.bin')
		synthwvl_red  = np.around(synthwvl_red,2)

		# Splice blue + red parts together
		synthflux = np.hstack((synthflux_blue, synthflux_red))
		synthwvl  = np.hstack((synthwvl_blue, synthwvl_red))
		#print(' synthwvl range without mask:',synthwvl[0],synthwvl[-1])

	# Else, use input synthetic spectrum
	else:
		synthflux = synth[0]
		synthwvl  = synth[1]
	
	# Clip synthetic spectrum so it's within range of obs spectrum
	mask = np.where((synthwvl > obswvl[0]) & (synthwvl < obswvl[-1]))
	synthwvl = synthwvl[mask]
	synthflux = synthflux[mask]
	#print(' synthwvl range WITH mask:',synthwvl[0],synthwvl[-1])

	# Interpolate and smooth the synthetic spectrum onto the observed wavelength array
	synthfluxnew = smooth_gauss_wrapper(synthwvl, synthflux, obswvl, dlam)

	# For testing purposes
	'''
	plt.figure()
	plt.plot(synthwvl, synthflux, 'k-', label='Synthetic')
	plt.plot(obswvl, synthfluxnew, 'r-', label='Synthetic (smoothed)')
	#plt.xlim(5000, 5250)
	#plt.ylim(0.9,1.0)
	plt.legend(loc='best')
	plt.savefig('synth_cont.png')
	#plt.show()
	'''

	return synthfluxnew

def mask_obs_for_division(obswvl, obsflux, ivar, element, linegaps, temp=None, logg=None, fe=None, alpha=None, dlam=None, lines='new', hires=False, stravinsky=False):
	"""Make a mask for synthetic and observed spectra.
	Mask out desired element lines for continuum division.
	Split spectra into red and blue parts.

    Inputs:
    For observed spectrum --
    obswvl  -- wavelength array of observed spectrum
    obsflux -- flux array of observed spectrum
    ivar 	-- inverse variance array of observed spectrum
    element -- element you want the abundances of e.g. 'Sr', 'Mn'

    Keywords:
    For synthetic spectrum -- 
    temp  -- effective temperature (K)
    logg  -- surface gravity
    fe 	  -- [Fe/H]
    alpha -- [alpha/Fe]

    dlam  -- FWHM of observed spectrum

    lines -- if 'new', use new revised linelist; else, use original linelist from Judy's code
    hires -- if 'True', don't mask chipgap

    Outputs:
    synthfluxmask -- (masked!) synthetic flux array
    obsfluxmask   -- (masked!) observed flux array
    obswvlmask    -- (masked!) wavelength array
    ivarmask 	  -- (masked!) inverse variance array
    mask 	      -- mask to avoid bad shit (chip gaps, bad pixels, Na D lines)
    """

	#print('masking obs for division')

	# Get smoothed synthetic spectrum and (NOT continuum-normalized) observed spectrum
	synthflux = get_synth(obswvl, obsflux, ivar, dlam, synth=None, temp=temp, logg=logg, fe=fe, alpha=alpha, stravinsky=stravinsky)
	print('got synth')

	# Make a mask
	mask = np.zeros(len(synthflux), dtype=bool)

	# Mask out first and last five pixels
	mask[:5]  = True
	mask[-5:] = True

	# Mask out more of ends of spectra
	#mask[np.where(obswvl < 4650)] = True 
	mask[np.where(obswvl < 4060)] = True #just trying to test how to change the bounds of normalization # had 4150
	mask[np.where(obswvl > 6550)] = True

	# Mask out Halpha, Hbeta, Hgamma
	mask[np.where((obswvl > 4340 - 5) & (obswvl < 4340 + 5))] = True #Hgamma
	mask[np.where((obswvl > 4862 - 5) & (obswvl < 4862 + 5))] = True #Hbeta
	mask[np.where((obswvl > 6563 - 5) & (obswvl < 6563 + 5))] = True #Halpha

	if hires==False:
		# Mask out pixels near chip gap
		chipgap = int(len(mask)/2 - 1)
		print('wavelength of chip gap: ', obswvl[chipgap])
		#print('Chip gap: ', chipgap)
		#mask[(chipgap - 10): (chipgap + 10)] = True
		mask[np.where((obswvl > (obswvl[chipgap] - 20)) & (obswvl < (obswvl[chipgap] + 20)))] = True

	# Mask out any bad pixels
	mask[np.where(synthflux <= 0.)] = True
	#print('Where synthflux < 0: ', obswvl[np.where(synthflux <=0.)])

	mask[np.where(ivar <= 0.)] = True
	#print('Where ivar < 0: ', obswvl[np.where(ivar <=0.)])

	# Mask out pixels around Na D doublet (5890, 5896 A)
	mask[np.where((obswvl > 5884.) & (obswvl < 5904.))] = True

	# Mask out pixels in regions around desired element lines
	elemmask = np.zeros(len(synthflux), dtype=bool)

	# For med-res spectra, mask out pixels in regions around desired element lines
	if hires==False:
		if lines == 'old':
			lines  = np.array([[4744.,4772.],[4773.,4793.],[4813.,4833.],[5384,5404.],[5527.,5547.],[6003.,6031.]])
		elif lines=='new':
			#with resonance lines: lines = np.array([[4729.,4793.],[4813.,4833.],[5384.,5442.],[5506.,5547.],[6003.,6031.],[6374.,6394.],[6481.,6501.]])
			#+/- 10A regions around Mn lines: lines = np.array([[4729.,4793.],[4813.,4833.],[5400.,5430.],[5506.,5547.],[6003.,6031.],[6374.,6394.],[6481.,6501.]])
			#+/- 5A regions around Mn lines: lines = np.array([[4734.1,4744.1],[4749.0,4770.8],[4778.4,4788.4],[4818.5,4828.5],[5402.3,5412.3],[5415.3,5425.3],[5511.8,5521.8],[5532.7,5542.7],[6008.3,6026.8],[6379.7,6389.7],[6486.7,6496.7]])
			#+/- 1A regions around Mn lines:
			lines = np.array(linegaps)

	# For hi-res spectra, mask out pixels in +/- 1A regions around Mn lines
	else:
		lines = np.array([[4738.1,4740.1],[4753.,4755.],[4760.5,4763.3],[4764.8,4767.8],[4782.4,4784.4],[4822.5,4824.5]])
						#,[5402.,5412.],[5415.,5425.],[5511.,5521.],[5532.,5542.],[6008.,6026.],[6379.,6389.],[6486.,6596.]])

	for line in range(len(lines)):
		elemmask[np.where((obswvl > lines[line][0]) & (obswvl < lines[line][1]))] = True
	mask[elemmask] = True

	# Create masked arrays
	synthfluxmask 	= ma.masked_array(synthflux, mask)
	obsfluxmask   	= ma.masked_array(obsflux, mask)
	obswvlmask	  	= ma.masked_array(obswvl, mask)
	ivarmask	  	= ma.masked_array(ivar, mask)

	# Split spectra into blue (index 0) and red (index 1) parts
	if hires==False:
		synthfluxmask 	= [synthfluxmask[:chipgap], synthfluxmask[chipgap:]]
		obsfluxmask		= [obsfluxmask[:chipgap], obsfluxmask[chipgap:]]
		obswvlmask 		= [obswvlmask[:chipgap], obswvlmask[chipgap:]]
		ivarmask 		= [ivarmask[:chipgap], ivarmask[chipgap:]]
		mask 			= [mask[:chipgap], mask[chipgap:]]
	print('finished masking obs for division')

	return synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask

def divide_spec(synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, element, sigmaclip=False, specname=None, outputname=None, hires=False):
	"""Do the actual continuum fitting:
	- Divide obs/synth.
	- Fit spline to quotient. 
		- Use cubic B-spline representation with breakpoints spaced every 150 Angstroms (Kirby+09)
		- Do iteratively, so that pixels that deviate from fit by more than 5sigma are removed for next iteration
		- Don't worry about telluric absorption corrections?
	- Divide obs/spline.

	Do this for blue and red parts of spectra separately, then splice back together.

    Inputs:
    synthfluxmask 	-- smoothed synth spectrum (desired element lines masked out)
    obsfluxmask		-- obs spectrum (desired element lines masked out)
    obswvlmask		-- wavelength (desired element lines masked out)
    ivarmask		-- inverse variance array (desired element lines masked out)
    mask 			-- mask used to mask stuff (desired element lines, bad pixels) out

    Keywords:
    sigmaclip 		-- if 'True', do sigma clipping while spline-fitting
    specname 		-- if not None, make plots of quotient and spline
    outputname 		-- if not None, gives path to save plots to
    hires 			-- if 'True', don't mask chipgap

    Outputs:
    obsflux_norm_final -- continuum-normalized observed flux (blue and red parts spliced together)
    ivar_norm_final    -- continuum-normalized inverse variance (blue and red parts spliced together)
    """

	# Prep outputs of continuum division
	obswvl 		 = []
	obsflux_norm = []
	ivar_norm 	 = []

	# Also prep some other outputs for testing
	quotient = []
	continuum = []
	obsflux = []

	# Do continuum division for blue and red parts separately
	if hires == True:
		synthfluxmask 	= [synthfluxmask]
		obsfluxmask 	= [obsfluxmask]
		obswvlmask 		= [obswvlmask]
		ivarmask 		= [ivarmask]
		mask 			= [mask]

		numparts = [0]

	else:
		numparts = [0,1]

	for ipart in numparts:

		# Convert inverse variance to inverse standard deviation
		newivarmask = ma.masked_array(np.sqrt(ivarmask[ipart].data), mask[ipart])

		# Divide obs/synth
		quotient.append(obsfluxmask[ipart]/synthfluxmask[ipart])

		# First check if there are enough points to compute continuum
		if len(synthfluxmask[ipart].compressed()) < 300:
			print('Insufficient number of pixels to determine the continuum!')
			#return

		# Compute breakpoints for B-spline (in wavelength space, not pixel space)
		def calc_breakpoints_wvl(array, interval):
			"""
			Helper function for use with a B-spline.
			Computes breakpoints for an array given an interval.
			"""

			breakpoints = []
			counter = 0
			for i in range(len(array)):

				if (array[i] - array[counter]) >= interval:
					counter = i
					breakpoints.append(array[i])

			return breakpoints

		# Compute breakpoints for B-spline (in pixel space, not wavelength space)
		def calc_breakpoints_pixels(array, interval):
			"""
			Helper function for use with a B-spline.
			Computes breakpoints for an array given an interval.
			"""

			breakpoints = []
			counter = 0
			for i in range(len(array)):

				if (i - counter) >= interval:
					counter = i
					breakpoints.append(array[i])

			return breakpoints

		# Determine initial spline fit, before sigma-clipping
		if hires:
			breakpoints_old = calc_breakpoints_wvl(obswvlmask[ipart].compressed(), 15.) # Use 50 A spacing
		else:
			breakpoints_old	= calc_breakpoints_pixels(obswvlmask[ipart].compressed(), 300.) # Use 300 pixel spacing

		#print(obswvlmask[ipart].compressed())
		print('breakpoints: ', breakpoints_old)
		# count how many data points are between each knot, including those at the ends
		n_inknots = []
		counter = 0
		for wvl1 in obswvlmask[ipart].compressed():
			if wvl1 < breakpoints_old[0]:
				counter += 1
		n_inknots.append(counter)
		for i in range(len(breakpoints_old)-1):
			counter = 0
			for wvl in obswvlmask[ipart].compressed():
				if (wvl > breakpoints_old[i]) and (wvl < breakpoints_old[i+1]):
					counter += 1
			n_inknots.append(counter)
		counter = 0
		for wvl2 in obswvlmask[ipart].compressed():
			if wvl2 > breakpoints_old[-1]:
				counter += 1
		n_inknots.append(counter)
		print('number of points between knots:',n_inknots)
		# Noticed for stars 3 and 7 that were not running after increasing the mask bounds that the last knot had
		# no data points between it and the end. When removed, the code could run but there appears to be too much noise/too little Sr
		# to produce a reasonable result and the code gives up after a couple abundance guesses.

		#print(quotient[ipart].compressed())
		#print('first obswvlmask entry',obswvlmask[ipart].compressed()[0])
		#print('last obswvlmask entry',obswvlmask[ipart].compressed()[-1])
		if breakpoints_old[-1]==obswvlmask[ipart].compressed()[-1]: #so that each knot, t, has data points in between IS THIS OK TO DO?
			breakpoints_old.pop(-1)
			print('had to get rid of last knot')
		# def is_sorted(array):
		# 	for i in range(len(array)-1):
		# 		if array[i+1] < array[i]:
		# 			return False
		# 	return True
		# print('is x sorted?',is_sorted(obswvlmask[ipart].compressed())) # x must be sorted for interpolation to work
		splinerep_old 	= splrep(obswvlmask[ipart].compressed(), quotient[ipart].compressed(), w=newivarmask.compressed(), t=breakpoints_old)
		continuum_old	= splev(obswvlmask[ipart].compressed(), splinerep_old)

		# Iterate the fit, sigma-clipping until it converges or max number of iterations is reached
		if sigmaclip:
			iternum  = 0
			maxiter  = 3
			clipmask = np.ones(len(obswvlmask[ipart].compressed()), dtype=bool)

			while iternum < maxiter:

				# Compute residual between quotient and spline
				resid = quotient[ipart].compressed() - continuum_old
				sigma = np.std(resid)

				# Sigma-clipping
				clipmask[np.where((resid < -5*sigma) | (resid > 5*sigma))] = False

				# Recalculate the fit after sigma-clipping
				breakpoints_new = calc_breakpoints_pixels((obswvlmask[ipart].compressed())[clipmask], 300.)
				splinerep_new 	= splrep((obswvlmask[ipart].compressed())[clipmask], (quotient[ipart].compressed())[clipmask], w=(newivarmask.compressed())[clipmask], t=breakpoints_new)
				continuum_new 	= splev(obswvlmask[ipart].compressed(), splinerep_new)

				# For testing purposes
				'''
				print('Iteration ', iternum)
				print((obswvlmask[ipart].compressed()[clipmask]).size)

				plt.figure()

				plt.subplot(211)
				plt.title('Iteration '+str(iternum))
				plt.plot(obswvlmask[ipart], quotient, 'b.')
				print('here1')
				plt.plot(obswvlmask[ipart][~clipmask], quotient[~clipmask], 'ko')
				print('here2')
				plt.plot(obswvlmask[ipart].compressed(), continuum_new, 'r-')
				print('here3')

				plt.subplot(212)
				plt.plot(obswvlmask[ipart].compressed(), resid)
				plt.savefig('/mnt/c/Research/Spectra/Sr-SMAUGoutput/normalizetest.png')
				'''

				# Check for convergence (if all points have been clipped)
				if (obswvlmask[ipart].compressed()[clipmask]).size == 0:
					print('Continuum fit converged at iteration ', iternum)
					break 

				else:
					continuum_old = continuum_new
					iternum += 1

			# Compute final spline
			continuum_final = splev(obswvlmask[ipart].data, splinerep_new)

		# If no sigma clipping, just compute final spline from initial spline fit
		else:
			continuum_final = splev(obswvlmask[ipart].data, splinerep_old)

		continuum.append(continuum_final)

		# Now divide obs/spline
		obswvl.append(obswvlmask[ipart].data)
		obsflux.append(obsfluxmask[ipart].data)
		obsflux_norm.append(obsfluxmask[ipart].data/continuum_final)

		# Compute final inverse variance
		ivar_norm.append(ivarmask[ipart].data * np.power(continuum_final, 2.))

		# Plots for testing
		''' 
		# this part also adds to the final fit plots from make_plot
		plt.figure()
		plt.subplot(211)
		plt.plot(obswvlmask[ipart].data, obsfluxmask[ipart].data, 'r-', label='Masked')
		plt.plot(obswvlmask[ipart].compressed(), obsfluxmask[ipart].compressed(), 'k-', label='Observed')
		#plt.plot(obswvlmask[ipart][mask], obsfluxmask[ipart][mask], 'r-', label='Mask')
		plt.legend(loc='best')
		plt.subplot(212)
		plt.plot(obswvlmask[ipart].data, synthfluxmask[ipart].data, 'r-', label='Masked')
		plt.plot(obswvlmask[ipart].compressed(), synthfluxmask[ipart].compressed(), 'k-', label='Synthetic')
		plt.legend(loc='best')
		plt.savefig(outputname+'/'+specname+'_maskedlines'+str(ipart)+'.png')
		'''

		plt.figure()
		plt.plot(obswvlmask[ipart].compressed(), quotient[ipart].compressed(), 'k-', label='Quotient (masked)')
		plt.plot(obswvlmask[ipart].compressed(), continuum_final[~mask[ipart]], 'r-', label='Final spline (masked)')
		plt.legend(loc='best')
		plt.savefig(outputname+'/'+specname+'_quotient'+str(ipart)+'.png')
		plt.close()
		

	# Now splice blue and red parts together
	obswvl_final	 	= np.hstack((obswvl[:]))
	obsflux_norm_final 	= np.hstack((obsflux_norm[:]))
	ivar_norm_final 	= np.hstack((ivar_norm[:]))

	if hires == False:
		obsflux_final 	= np.hstack((obsflux[0].data, obsflux[1].data))
	else:
		obsflux_final = obsflux

	quotient 			= np.hstack((quotient[:]))
	continuum 			= np.hstack((continuum[:]))

	# Make plot to test
	#make_plots('new', specname+'_continuum', obswvl_final, quotient, continuum, outputname, ivar=None, title=None, synthfluxup=None, synthfluxdown=None, resids=False)
	#make_plots('new', specname+'_unnormalized', obswvl_final, obsflux_final, continuum, outputname, ivar=None, title=None, synthfluxup=None, synthfluxdown=None, resids=False)

	return obsflux_norm_final, ivar_norm_final

def mask_obs_for_abundance(obswvl, obsflux_norm, ivar_norm, dlam, element, linegaps, lines = 'new', hires = False):
	"""Make a mask for synthetic and observed spectra.
	Mask out bad stuff + EVERYTHING BUT Mn lines (for actual abundance measurements)

    Inputs:
    Observed spectrum --
    obswvl  	 -- wavelength array of (continuum-normalized!) observed spectrum
    obsflux_norm -- flux array of (continuum-normalized!) observed spectrum
    ivar_norm	 -- inverse variance array of (continuum-normalized!) observed spectrum
    dlam 		 -- FWHM array of (continuum-normalized!) observed spectrum
    element      -- element you want the abundances of e.g. 'Sr', 'Mn'

	Synthetic spectrum --
    synthwvl  -- wavelength array of synthetic spectrum
    synthflux -- flux array of synthetic spectrum

    Keywords:
    lines -- if 'new', use new revised linelist; else, use original linelist from Judy's code
    hires -- if 'True', don't mask chipgap

    Outputs:
    obsfluxmask   -- (masked!) observed flux array
    obswvlmask    -- (masked!) wavelength array
    ivarmask 	  -- (masked!) inverse variance array
    dlammask	  -- (masked!) FWHM array
    skip 		  -- list of lines to NOT skip
    """
    
	# Read in file of target lines
	#fitslines = fits.open('/mnt/c/Research/Sr-SMAUG/'+element+'linelists/'+element+'lines.fits')
	#elemlines = fitslines[1].data

	# Make a mask
	mask = np.zeros(len(obswvl), dtype=bool)

	# Mask out first and last five pixels
	mask[:5]  = True
	mask[-5:] = True

	# Mask out pixels near chip gap
	if hires == False:
		chipgap = int(len(mask)/2 - 1)
		mask[(chipgap - 5): (chipgap + 5)] = True

	# Mask out any bad pixels
	mask[np.where(ivar_norm <= 0.)] = True

	# Mask out pixels around Na D doublet (5890, 5896 A)
	mask[np.where((obswvl > 5884.) & (obswvl < 5904.))] = True

	# Mask out everything EXCEPT desired element lines
	obsfluxmask = []
	obswvlmask  = []
	ivarmask 	= []
	dlammask	= []

	masklist 	= [obsfluxmask, obswvlmask, ivarmask, dlammask]
	arraylist 	= [obsflux_norm, obswvl, ivar_norm, dlam]

	if lines == 'old':
		lines  = np.array([[4744.,4772.],[4773.,4793.],[4813.,4833.],[5384,5404.],[5527.,5547.],[6003.,6031.]])
	elif lines == 'new':

		# lineslist = list()
		# for i in range(len(elemlines[element+'lines'])):
		# 	gap = [elemlines[element+'lines'][i]-10., elemlines[element+'lines'][i]+10.]
		# 	lineslist.append(gap)
		lines = np.array(linegaps)

	for i in range(len(masklist)):
		for line in range(len(lines)):
			masklist[i].append( arraylist[i][np.where(((obswvl > lines[line][0]) & (obswvl < lines[line][1]) & (~mask)))] )

	skip = np.arange(len(lines))
	for line in range(len(lines)):
		#print('checking line',lines[line])

		# Skip spectral regions where the chip gap falls
		if hires == False:
			if (obswvl[chipgap + 5] > lines[line][0]) and (obswvl[chipgap + 5] < lines[line][1]):
				skip = np.delete(skip, np.where(skip==line))
			elif (obswvl[chipgap - 5] > lines[line][0]) and (obswvl[chipgap - 5] < lines[line][1]):
				skip = np.delete(skip, np.where(skip==line))

		# Skip spectral regions that are outside the observed wavelength range (with bad pixels masked out)
		if (lines[line][0] < obswvl[~mask][0]) or (lines[line][1] > obswvl[~mask][-1]):
			skip = np.delete(skip, np.where(skip==line))
		
		# Skip lines that are at edges of spectrum where things go crazy
		print('length of observed spectrum', len(obsflux_norm))
		#print(np.where(obsflux_norm[0:2000] <= 0))
		blue_cutoff = obswvl[np.where(obsflux_norm[0:2000] <= 0)][-1]
		red_cutoff = obswvl[-3000:][np.where(obsflux_norm[-3000:] <= 0)][0]
		print('cutoffs', blue_cutoff, red_cutoff)
		if lines[line][0] > blue_cutoff and lines[line][1] < red_cutoff:
			continue
		else:
			skip = np.delete(skip, np.where(skip==line))			
					

		# 	badspots_red = np.where(self.obsflux_norm[-2000:-1] < 0)
		# 	red_cutoff = self.obswvl[badspots_red][-1]
		# 	print(self.linelists)
		# 	self.newlinelists = []
		# 	for i in range(len(self.elementlines)):
		# 		if self.elementlines[i] > blue_cutoff and self.elementlines[i] < red_cutoff:
		# 			self.newlinelists.append(self.linelists[i])
		# 		else:
		# 			continue

	return np.asarray(obsfluxmask, dtype=object), np.asarray(obswvlmask, dtype=object), np.asarray(ivarmask, dtype=object), np.asarray(dlammask, dtype=object), np.asarray(skip)