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

def get_synth(obswvl, obsflux, ivar, dlam, synth=None, temp=None, logg=None, fe=None, alpha=None, carbon=None, stravinsky=False, smoothed=True):
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
		    carbon -- [C/Fe]
	stravinsky -- whether running on stravinsky or not
	smoothed -- when True (default), smooth the synthetic spectrum to dlam

    Outputs:
    synthflux -- synthetic flux array
    obsflux   -- observed flux array
    obswvl    -- wavelength array
    ivar 	  -- inverse variance array
    """

	# Get synthetic spectrum from grid
	if synth is None:

		# Get synthetic spectrum for G band (4100-4500), specifying [C/Fe]
		if stravinsky:
			synthflux_ch = 1. - interpolateAtm(temp,logg,fe,alpha,carbon=carbon,griddir='/raid/gridch/bin/',gridch=True,stravinsky=stravinsky)
			synthwvl_ch = np.fromfile('/raid/gridch/bin/lambda.bin')
		else:
			synthflux_ch = 1. - interpolateAtm(temp,logg,fe,alpha,carbon=carbon,griddir='/mnt/c/SpectraGrids/gridch/bin/',gridch=True,stravinsky=stravinsky) #this won't work because I don't have this grid on my computer
		synthwvl_ch  = np.around(synthwvl_ch,2)

		# Use modified version of interpolateAtm to get blue synthetic spectrum from Ivanna's grid
		if stravinsky:
			synthflux_blue = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/raid/gridie/bin/', stravinsky=stravinsky)
		else:
			synthflux_blue = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/mnt/c/SpectraGrids/gridie/bin/', stravinsky=stravinsky)
		wvl_range_blue = np.arange(4100., 6300.+0.14, 0.14) # can change this if wavelength range changes
		synthwvl_blue  = 0.5*(wvl_range_blue[1:] + wvl_range_blue[:-1])
		iblue = np.where(synthwvl_blue >= 4500)[0]
		synthflux_blue = synthflux_blue[iblue] # cut blue spectrum to not overlap with gridch synth
		synthwvl_blue = synthwvl_blue[iblue]

		# Also get synthetic spectrum for redder part from Evan's grid
		if stravinsky:
			synthflux_red = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/raid/grid7/bin/', stravinsky=stravinsky)
			synthwvl_red  = np.fromfile('/raid/grid7/bin/lambda.bin')
		else:
			synthflux_red = 1. - interpolateAtm(temp,logg,fe,alpha,griddir='/mnt/c/SpectraGrids/grid7/bin/', stravinsky=stravinsky)
			synthwvl_red  = np.fromfile('/mnt/c/SpectraGrids/grid7/bin/lambda.bin')
		synthwvl_red  = np.around(synthwvl_red,2)
		ired = np.where(synthwvl_red >= 6300)[0]
		synthflux_red = synthflux_red[ired] # cut red spectrum to not overlap with gridie synth
		synthwvl_red = synthwvl_red[ired]

		# Splice blue + red parts together
		synthflux = np.hstack((synthflux_ch, synthflux_blue, synthflux_red))
		synthwvl  = np.hstack((synthwvl_ch, synthwvl_blue, synthwvl_red))

	# Else, use input synthetic spectrum
	else:
		synthflux = synth[0]
		synthwvl  = synth[1]
	
	# Clip synthetic spectrum so it's within range of obs spectrum
	mask = np.where((synthwvl > obswvl[0]) & (synthwvl < obswvl[-1]))
	synthwvl = synthwvl[mask]
	synthflux = synthflux[mask]
	# print(' synthwvl range with mask:',synthwvl[0],synthwvl[-1])

	# Interpolate and smooth the synthetic spectrum onto the observed wavelength array
	if smoothed:
		synthfluxnew = smooth_gauss_wrapper(synthwvl, synthflux, obswvl, dlam)
		return synthfluxnew
	else:
		return synthflux

def mask_obs_for_division(obswvl, obsflux, ivar, element, linegaps, temp=None, logg=None, fe=None, alpha=None, dlam=None, carbon=None, hires=False, stravinsky=False, spec1200G=False):
	"""Make a mask for synthetic and observed spectra.
	Mask out desired element lines for continuum division.
	Split spectra into red and blue parts.

    Inputs:
    For observed spectrum --
    obswvl  -- wavelength array of observed spectrum
    obsflux -- flux array of observed spectrum
    ivar 	-- inverse variance array of observed spectrum
    element -- element you want the abundances of e.g. 'Sr', 'Mn'
	linegaps -- wavelength regions that will be synthesized to measure the abundance you want

    Keywords:
    For synthetic spectrum -- 
    temp  -- effective temperature (K)
    logg  -- surface gravity
    fe 	  -- [Fe/H]
    alpha -- [alpha/Fe]
    dlam  -- FWHM of observed spectrum
	carbon -- [C/Fe]

    hires -- if 'True', don't mask chipgap
	stravinsky -- whether running on stravinsky or not for file paths
	spec1200G -- whether you're adding a 1200G spectrum as well

    Outputs:
    synthfluxmask -- (masked!) synthetic flux array
    obsfluxmask   -- (masked!) observed flux array
    obswvlmask    -- (masked!) wavelength array
    ivarmask 	  -- (masked!) inverse variance array
    mask 	      -- mask to avoid bad shit (chip gaps, bad pixels, Na D lines)
    """

	# Get smoothed synthetic spectrum and (NOT continuum-normalized) observed spectrum
	synthflux = get_synth(obswvl, obsflux, ivar, dlam, synth=None, temp=temp, logg=logg, fe=fe, alpha=alpha, carbon=carbon, stravinsky=stravinsky)
	print('got synth')

	# Make a mask
	mask = np.zeros(len(synthflux), dtype=bool)

	# Mask out first and last five pixels
	mask[:5]  = True
	mask[-5:] = True

	# Mask out more of ends of spectra
	if spec1200G:
		mask[np.where(obswvl < 6400)] = True
		mask[np.where(obswvl > 9000)] = True
	else: 
		mask[np.where(obswvl < 4100)] = True #synthetic grids cut off at 4100 for now
		mask[np.where(obswvl > 6800)] = True #was 6550

	# Mask out Halpha, Hbeta, Hgamma
	mask[np.where((obswvl > 4340 - 5) & (obswvl < 4340 + 5))] = True #Hgamma
	mask[np.where((obswvl > 4862 - 5) & (obswvl < 4862 + 5))] = True #Hbeta
	mask[np.where((obswvl > 6563 - 5) & (obswvl < 6563 + 5))] = True #Halpha

	if hires==False:
		# Find chipgap
		chipgap = int(len(mask)/2 - 1)
		print('wavelength of chip gap: ', obswvl[chipgap])
		#mask[(chipgap - 10): (chipgap + 10)] = True
		#mask[np.where((obswvl > (obswvl[chipgap] - 20)) & (obswvl < (obswvl[chipgap] + 20)))] = True
		# Mask out pixels near chip gap
		chipstart = 0
		for i in range(chipgap-50,chipgap+50):
			if abs(obsflux[i]-obsflux[i+1]) > 400: #if flux jumps a large amount near the chip gap it's probably where it starts going weird
				chipstart = i
				chipend = (chipgap - chipstart) +chipgap+1 #symmetrical around the chipgap
				break
		if chipstart == 0:
			mask[np.where((obswvl > (obswvl[chipgap] - 20)) & (obswvl < (obswvl[chipgap] + 20)))] = True
		else:
			mask[np.where((obswvl > (obswvl[chipstart])) & (obswvl < (obswvl[chipend])))] = True

	# Mask out any bad pixels
	mask[np.where(synthflux <= 0.)] = True

	mask[np.where(ivar <= 0.)] = True

	# Mask out pixels around Na D doublet (5890, 5896 A)
	mask[np.where((obswvl > 5884.) & (obswvl < 5904.))] = True

	# Mask out pixels in regions around desired element lines
	elemmask = np.zeros(len(synthflux), dtype=bool)

	# For med-res spectra, mask out pixels in regions around desired element lines
	lines = np.array(linegaps)

	# For hi-res spectra, mask out pixels in +/- 1A regions around the element lines. This is not built in yet.

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

	return synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, chipgap

def divide_spec(synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, sigmaclip=False, specname=None, outputname=None, hires=False):
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
	print('normalization started')

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
		# print('masked inverse standard deviation')

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

		if breakpoints_old[-1]==obswvlmask[ipart].compressed()[-1]: #so that each knot, t, has data points in between
			breakpoints_old.pop(-1)
			print('had to get rid of last knot')
		def is_sorted(array):
			for i in range(len(array)-1):
				if array[i+1] < array[i]:
					return False
			return True
		# print('is x sorted?',is_sorted(obswvlmask[ipart].compressed())) # x must be sorted for interpolation to work
		splinerep_old 	= splrep(obswvlmask[ipart].compressed(), quotient[ipart].compressed(), w=newivarmask.compressed(), t=breakpoints_old)
		continuum_old	= splev(obswvlmask[ipart].compressed(), splinerep_old)

		# Iterate the fit, sigma-clipping until it converges or max number of iterations is reached
		if sigmaclip:
			#print('doing sigma clipping')
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
				# print('is x sorted?',is_sorted((obswvlmask[ipart].compressed())[clipmask]))
				breakpoints_new = calc_breakpoints_pixels((obswvlmask[ipart].compressed())[clipmask], 300.)

				# If the last breakpoint is too close to the end, the run will fail. Move it slightly to avoid the issue.
				if np.abs((obswvlmask[ipart].compressed())[clipmask][-1] - breakpoints_new[-1]) < 0.0001:
					breakpoints_new[-1] = breakpoints_new[-1]-0.0001
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
				plt.savefig('/mnt/c/Research/Spectra/SMAUGoutput/normalizetest.png')
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
			#print('no sigma clipping in normalization')
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
		plt.close()
		

		plt.figure()
		plt.plot(obswvlmask[ipart].compressed(), quotient[ipart].compressed(), 'k-', label='Quotient (masked)')
		plt.plot(obswvlmask[ipart].compressed(), continuum_final[~mask[ipart]], 'r-', label='Final spline (masked)')
		plt.legend(loc='best')
		plt.savefig(outputname+'/'+specname+'_quotient'+str(ipart)+'.png')
		plt.close()

		fig, axs = plt.subplots(2,1)
		axs[0].plot(obswvlmask[ipart].compressed(), obsfluxmask[ipart].compressed(), 'k-', label='Observed')
		axs[0].plot(obswvlmask[ipart].compressed(), continuum_final[~mask[ipart]], 'r-', label='Final Continuum')
		axs[0].legend(loc='best')
		axs[1].plot(obswvlmask[ipart].compressed(), obsfluxmask[ipart].compressed(), 'k-')
		axs[1].plot(obswvlmask[ipart].compressed(), continuum_final[~mask[ipart]], 'r-')
		axs[1].set_xlim((obswvlmask[ipart].compressed()[0],obswvlmask[ipart].compressed()[0]+500))
		fig.savefig(outputname+'/'+specname+'_continuum'+str(ipart)+'.png')
		plt.close()
		'''

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

def mask_obs_for_abundance(obswvl, obsflux_norm, ivar_norm, dlam, synthflux, element, linegaps, hires = False):
	"""Make a mask for synthetic and observed spectra.
	Mask out bad stuff + EVERYTHING BUT desired element lines (for actual abundance measurements)

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
    hires -- if 'True', don't mask chipgap

    Outputs:
    obsfluxmask   -- (masked!) observed flux array
    obswvlmask    -- (masked!) wavelength array
    ivarmask 	  -- (masked!) inverse variance array
    dlammask	  -- (masked!) FWHM array
    skip 		  -- list of lines to NOT skip
    """

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

	lines = np.array(linegaps)

	for i in range(len(masklist)):
		for line in range(len(lines)):
			masklist[i].append( arraylist[i][np.where(((obswvl > lines[line][0]) & (obswvl < lines[line][1]) & (~mask)))] )
		# 	print('linegap:', lines[line])
		# 	print(masklist[i][-1])
		# if lines[-1] == lines[-2]: #we have the reddest line for Ba or Eu in both blue and 1200G spectra -- need to split those up


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
		#print('length of observed spectrum', len(obsflux_norm))
		#print(np.where(obsflux_norm[0:2000] <= 0))
		if len(np.where(obsflux_norm[0:2000] <= 0)) != 0:
			blue_cutoff = obswvl[np.where(obsflux_norm[0:2000] <= 0)][-1]
			#print(blue_cutoff)
			if lines[line][0] > blue_cutoff:
				continue
			else: 
				skip = np.delete(skip, np.where(skip==line))
		if len(np.where(obsflux_norm[-3000:] <= 0)) != 0:
			red_cutoff = obswvl[-3000:][np.where(obsflux_norm[-3000:] <= 0)][0]
			#print(red_cutoff)
			if lines[line][1] < red_cutoff:
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