# wvl_corr.py
# Functions to do empirical wavelength corrections 
# 
# Created 21 June 18
# Updated 22 June 18
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import math
from interp_atmosphere import find_nearest, interpolateAtm
import pandas
import scipy.optimize
from scipy.signal import find_peaks_cwt

def return_hydrogen_synth(temp,logg,fe,alpha,dlam,wvl_radius=10):
	""" Code from G. Duggan.
		Find and return closest hydrogen synth spec:
		Truncate the returned spectra to h_lines +/- wvl_radius in angstroms
	"""

	# Variables
	step = 0.02 # for full-resolution (0.14 else)
	gauss_sigma = np.around(dlam[0]/step) 	# for smoothing

	# Get subtitle
	hteff  = find_nearest(temp, array=np.array([4500, 5000, 5500, 6000]))
	hlogg  = find_nearest(logg, array=np.array([0.5, 1.0, 1.5, 2.0, 2.5]))
	hfeh   = find_nearest(logg, array=np.array([-2.0, -1.0]))
	halpha = find_nearest(alpha, array=np.array([0.0]))
	subtitle = r'Synth: T$_{eff}$=%i, log(g)=%.2f, [Fe/H]=%.2f, [$\alpha$/Fe]=%.2f'%(hteff, hlogg, hfeh, halpha)

	# Hgamma
	########
	# Wavelength
	wvly  = np.fromfile('/raid/gridch/synths/lambda.bin')
	wvly  = np.around(wvly,2)
	# Relative flux
	fluxy = interpolateAtm(hteff, hlogg, hfeh, halpha, hgrid=True, griddir='/raid/gridch/synths/')
	print(len(fluxy))
	relfluxy = scipy.ndimage.filters.gaussian_filter(1.0-fluxy,gauss_sigma)
	n = int(len(relfluxy)/24)
	relfluxy = relfluxy[n*12:n*13]

	# Hbeta
	#######
	# Wavelength
	wvlb  = np.arange(4851, 4871+step*.1, step)
	# Relative flux
	fluxb = interpolateAtm(hteff, hlogg, hfeh, halpha, hgrid=True, griddir='/raid/gduggan/s/gridhbeta/synths/')
	relfluxb = scipy.ndimage.filters.gaussian_filter(1.0-fluxb,gauss_sigma)
	
	# Halpha
	########
	wvla = np.fromfile('/raid/grid7/synths/lambda.bin')
	wvla = np.around(wvla,2)
	fluxa = interpolateAtm(hteff, hlogg, hfeh, halpha, hgrid=True, griddir='/raid/grid7/synths/')
	relfluxa = scipy.ndimage.filters.gaussian_filter(1.0-fluxa,gauss_sigma)

	# Mask out regions outside the line regions
	h_lines = [4341, 4861] #, 6563]
	masky = (wvly > h_lines[0]-wvl_radius) & (wvly < h_lines[0] + wvl_radius)
	maskb = (wvlb > h_lines[1]-wvl_radius) & (wvlb < h_lines[1] + wvl_radius)
	#maska = (wvla > h_lines[2]-wvl_radius) & (wvla < h_lines[2] + wvl_radius)

	# Put the spectra into arrays
	wvlh = np.array([wvly[masky],wvlb[maskb]]) #,wvla[maska]])
	relfluxh = np.array([relfluxy[masky],relfluxb[maskb]]) #,relfluxa[maska]])
	############# DANGER DANGER DANGER - hard coded reducing flux from hbeta synthesis

	return wvlh, relfluxh, subtitle

def h_continuum(obs_wvl,obs_flux_norm,obs_flux_std,spec_wvl,spec_flux,f_data,wvl_radius=10):

	""" Code from G. Duggan.
		Compute continuum in H line regions.

		Inputs:
		obs_wvl 	  -- observed wavelengths for single slit
		obs_flux_norm -- observed flux (continuum-corrected)
		obs_flux_std  -- observed stddev
		spec_wvl 	  -- synthetic H spectra wavelengths
		spec_flux 	  -- synthetic H spectra flux
		f_data 		  -- function of interpolated observed spectrum

		Outputs:
		f_synth_adj   -- array of (flux * continuum) in H regions
		h_continuum_array -- array of continuums in H regions
		line_data_std -- array of median standard deviations in H regions
	"""

	line_data_std = []
	chipgap = int(len(obs_wvl)/2 - 1)
		
	wvl_begin_gap = obs_wvl[chipgap - 5]
	wvl_end_gap = obs_wvl[chipgap + 5]

	# Loop over H line regions
	h_lines = [4341, 4861] #, 6563]
	for i in range(len(h_lines)):

		# Compute line regions
		data_mask = (obs_wvl>(h_lines[i]-2.*wvl_radius)) & (obs_wvl<(h_lines[i]+2.*wvl_radius))

		# Compute median standard deviation
		line_data_std.append(np.median(obs_flux_std[data_mask]))

		# Check that spectrum isn't awful and covers hydrogen lines
		if ((len(obs_flux_norm[data_mask]) != sum(np.isfinite(obs_flux_norm[data_mask]))) or 
			(h_lines[i]-2.*wvl_radius<obs_wvl[0]) or (h_lines[i]+2.*wvl_radius>obs_wvl[-1]) or 
			((wvl_end_gap>h_lines[i]+2.*wvl_radius>wvl_begin_gap) or (wvl_end_gap>h_lines[i]-2.*wvl_radius>wvl_begin_gap))):
			return [],[],[]
	
	# Compute observed/synthetic
	divided = np.array([f_data(spec_wvl[i])/spec_flux[i] for i in range(len(h_lines))])

	# Fit a line to obs/synth
	fit_param = np.array([np.polyfit(spec_wvl[i],divided[i],1) for i in range(len(h_lines))])

	f_synth_adj = []
	h_continuum_array = []
	for i in range(len(h_lines)):
		continuum = np.poly1d(fit_param[i])
		# Compute continuum in H regions
		h_continuum_array.append(continuum(spec_wvl[i]))
		# Compute continuum*flux in H regions
		f_synth_adj.append(spec_flux[i]*continuum(spec_wvl[i]))

	'''
	fig, axs = plt.subplots(1,3, figsize=(11,5))
	fig.subplots_adjust(bottom=0.17,wspace=0.29)
	#plt.suptitle(subtitle, y = 0.955)
	#xminorLocator = plt. MultipleLocator (1)
	#xmajorLocator = plt. MultipleLocator (10)
	#ymajorLocator = plt. MultipleLocator (0.25)

	axs = axs.ravel()
	for i in range(len(h_lines)):
		if (i==0):
			axs[i].set_ylabel('Normalized Flux')#,fontsize=14, labelpad=10)
		if i==1:
			axs[i].set_xlabel('Wavelength ($\AA$)',labelpad=10)#,fontsize=14, labelpad=10)
		xmajorLocator = plt. MultipleLocator (10)
		axs[i].xaxis.set_major_locator(xmajorLocator)    
		axs[i].plot(spec_wvl[i],divided[i], label='Quotient')
		axs[i].plot(spec_wvl[i],f_synth_adj[i],'r', label='Continuum-corrected synthetic')
		p = np.poly1d(fit_param[i])
		axs[i].plot(spec_wvl[i],p(spec_wvl[i]), label='Fit to quotient')
		axs[i].legend(loc='best')
	plt.show()    
	plt.close()
	'''
	
	return np.asarray(f_synth_adj), np.asarray(h_continuum_array), np.asarray(line_data_std)

def find_wvl_offset(wvlh,f_data,f_synth_adj,wvl_radius=10):
	""" Code from G. Duggan.
		Convolve data with synthetic spectra to find wavelength offset

		Inputs:
		wvlh 	    -- H region wavelengths
		f_data 		-- function of interpolated observed spectrum
		f_synth_adj -- array of (flux * continuum) in H regions
		wvl_radius  -- size of H regions

		Outputs:
		synth_wvls 		 -- median wavelength of each H region
		data_interp_wvls -- wavelength of the correction in each H region
		wvl_data_conv 	 -- wavelength array for each (convolved) H region
		conv_array 		 -- convolution array for each H region
	"""

	synth_wvls = []
	data_interp_wvls = []
	wvl_data_conv = []
	conv_array = []

	# Loop over all H lines
	h_lines = [4341, 4861] #, 6563]
	for i in range(len(h_lines)):

		# Create wavelength grid for the large data wavelength interval that matches synthetic spectra -> wvl_data
		wvl_step = np.around(wvlh[i][1]-wvlh[i][0],2)
		wvl_data = np.arange(h_lines[i]-1.5*wvl_radius, h_lines[i]+1.5*wvl_radius, wvl_step)

		# Compute interpolated observed spectrum in these H regions
		f_data(wvl_data)

		# Convolve observed with synthetic spectra. 
		# Need to move spectra so the absorption is positive and is centered at 0
		maxpt = np.ceil(max(f_data(wvl_data)))
		print('CHECK: ', maxpt)
		conv = np.correlate((1.0-f_data(wvl_data)),(1.0-f_synth_adj[i]),'same')
		true_synth_wvl = np.median(wvlh[i])

		# Normalize convolution
		conv_normalized = conv/np.max(conv)

		# Find peaks of convolution. These are the wavelength corrections!
		peakind = np.asarray(find_peaks_cwt(conv,np.array([20]))) # 20 = expected width of peaks

		# If many corrections are within 10% of the best match, pick the closest one (data_interp_wvl)
		peak_mask = np.where(conv[peakind]/np.max(conv[peakind])>0.9)
		data_interp_wvl = wvl_data[peakind[peak_mask]][np.argmin(np.abs(wvl_data[peakind[peak_mask]]-true_synth_wvl))]

		# Output for each line region
		synth_wvls.append(true_synth_wvl) 			# Median wavelength
		data_interp_wvls.append(data_interp_wvl) 	# Wavelength of the correction 
		wvl_data_conv.append(wvl_data)				# Wavelength array
		conv_array.append(conv) 					# Convolution
	
	return np.asarray(synth_wvls), np.asarray(data_interp_wvls), np.asarray(wvl_data_conv), np.asarray(conv_array)

def fit_wvl(obs_wvl, obs_flux_norm, obs_flux_std, dlam, 
			teff, logg, feh, alphafe, 
			name, directory, wvl_max_shift = 20):
	""" Code from G. Duggan.
		Fine-tune wavelength by matching to Halpha, Hbeta, Hgamma

		Inputs:
		obs_wvl 	  -- observed wavelengths for single slit
		obs_flux_norm -- observed flux (continuum-corrected)
		obs_flux_std -- observed stddev
		dlam 		  -- FWHM from observed FITS file
		teff, logg, feh, alphafe -- params from star
		directory 	  -- where to save output files
		name 		  -- name of star (specname from FITS file)

		Keywords:
		wvl_max_shift -- max allowed wavelength shift (default = 20 Angstroms)

		Outputs:
		wvl_slit_new -- corrected observed wavelengths for single slit
	"""

	chipgap = int(len(obs_wvl)/2 - 1)
		
	wvl_begin_gap = obs_wvl[chipgap - 5]
	wvl_end_gap = obs_wvl[chipgap + 5]
	mask_red_data = obs_wvl > wvl_end_gap

	# Check that all three hydrogen lines are included in the data
	h_lines = [4341, 4861] #, 6563]
	h_lines_blue_chip = (h_lines > obs_wvl[0]) & (h_lines < wvl_begin_gap)
	h_lines_red_chip = (h_lines < obs_wvl[-1]) & (h_lines > wvl_end_gap)
	if np.sum(h_lines_blue_chip)+np.sum(h_lines_red_chip) != len(h_lines):
		print(h_lines_blue_chip,h_lines_red_chip, "Halpha, beta, or gamma is missing. Wavelength is not fine-tuned.")
		return obs_wvl

	f_data = scipy.interpolate.interp1d(obs_wvl, obs_flux_norm) 
				
	title = r'T$_{eff}$=%i, log(g)=%.2f, [Fe/H]=%.2f, [$\alpha$/Fe]=%.2f'%(teff, logg, feh, alphafe)
	outfilename = name+'_wvlfit'    

	# Find corresponding hydrogen line synthetic spectra
	wvlh, relfluxh, subtitle = return_hydrogen_synth(teff,logg,feh,alphafe,dlam)

	# Find continuum 
	f_synth_adj, h_continuum_array, h_data_std = h_continuum(obs_wvl,obs_flux_norm,obs_flux_std,wvlh,relfluxh,f_data)

	# Check that spectra are good in H line regions
	if len(f_synth_adj) == 0:
		print("Skipping %s - non-finite flux or incomplete coverage near H lines"%(name))
		return []
		
	# Convolve data with synthetic spectra
	wvl_radius = 10
	synth_wvls, data_interp_wvls, wvl_data_conv, conv_array = find_wvl_offset(wvlh,f_data,f_synth_adj)
	print('Nothing wrong with checks')

	# Compute wavelength offset for each H region
	diff_wvls = synth_wvls - data_interp_wvls
	
	if np.max(np.abs(diff_wvls))>wvl_max_shift:
		print("Wavelength shift larger than %d ang requested - fail"%wvl_max_shift)
		return []

	# Fit all three Balmer lines
	param = np.polyfit(data_interp_wvls,synth_wvls,1,w=np.reciprocal(h_data_std))
	wvl_old_to_new = np.poly1d(param) 
				
	# Based on the three corrected h-lines, create a new wavelength solution 
	interp_mask = (obs_wvl>min(data_interp_wvls)) & (obs_wvl<max(data_interp_wvls)) 
	wvl_slit_new = wvl_old_to_new(obs_wvl)
	
	# Plot convolution results
	hgamma_data_interp = data_interp_wvls[0]

	fig, axs = plt.subplots(2,3, figsize=(11,8.5))
	fig.subplots_adjust(bottom=0.10,top=0.92,hspace=.18, right=0.95, left=0.11)
	plt.suptitle(subtitle, y = 0.955)
	ymajorLocator = plt. MultipleLocator (0.25)
	axs = axs.ravel()   
	axs[1].set_title(title, y=1.1)
	axs[4].set_xlabel('Wavelength ($\AA$)',labelpad=10)#,fontsize=14, labelpad=10) 
	axs[0].set_ylabel('Normalized Flux')#,fontsize=14, labelpad=10)
	axs[3].set_ylabel('Normalized Flux')#,fontsize=14, labelpad=10)

	#print('CHECK: ', obs_flux_norm[np.where((obs_wvl > 6562) & (obs_wvl < 6564))])

	for i in range(len(h_lines)):
		axs[i].plot(obs_wvl,obs_flux_norm,'k.', label="data")
		axs[i].plot(wvlh[i],relfluxh[i],'g-',label="un-adjusted synth")
		axs[i].plot(wvlh[i],f_synth_adj[i],'b',label="continuum corrected")       
		axs[i].plot(wvlh[i]-diff_wvls[i],f_synth_adj[i],'r',label="wavelength corrected")
	
		axs[i+3].plot(wvlh[i],f_synth_adj[i],'b-',label="continuum corrected")
		axs[i+3].plot(wvl_data_conv[i],f_data(wvl_data_conv[i]),'k-',label="data")
		axs[i+3].plot(wvl_data_conv[i],conv_array[i]/max(conv_array[i]),'m',label="convolution results")
		axs[i+3].plot([synth_wvls[i]]*2,[0,1],'g:',label="synthetic h-line wvl")
		if i == 0:
			axs[i+3].plot([hgamma_data_interp]*2,[0,1],'m:',label="uncorrected h-line wvl")
		else:    
			axs[i+3].plot([data_interp_wvls[i]]*2,[0,1],'m:',label="uncorrected h-line wvl")
   
		axs[i].set_ylim([0,1.2])
		axs[i+3].set_ylim([0,1.2])
		axs[i+3].set_xlim([h_lines[i]-2*wvl_radius,h_lines[i]+2*wvl_radius])
		axs[i].set_xlim([h_lines[i]-wvl_radius,h_lines[i]+wvl_radius])
		xmajorLocator = plt. MultipleLocator (10)
		axs[i+3].xaxis.set_major_locator(xmajorLocator)

	#plt.legend()
	#print('Directory: ', directory)
	#print('Outfilename: ', outfilename)
	plt.savefig(directory+outfilename+'.png')
	plt.close(fig)

	# Plot new wavelength solution results    
	fig2, axs2 = plt.subplots(1,1, figsize=(11,6.0))
	plt.title(title,y=1.07)
	plt.suptitle(subtitle, y = 0.93)
	
	axs2.errorbar(data_interp_wvls,data_interp_wvls-synth_wvls,yerr=h_data_std,fmt='x',label='Synth')
	axs2.plot(obs_wvl,obs_wvl-wvl_slit_new,'-', label='final')
	axs2.plot([hgamma_data_interp], [hgamma_data_interp]-synth_wvls[0],'rx')
	axs2.plot(np.array([4130.6, 4554.0, 4934.1, 5853.7, 6141.7, 6496.9]),[0,0,0,0,0,0], 'ro',label='Ba lines')  
	axs2.plot([4000,7000],[0,0],'--')
	xmajorLocator = plt. MultipleLocator (1000)
	axs2.xaxis.set_major_locator(xmajorLocator)
	axs2.set_xlabel('Current Data Wavelength')
	axs2.set_ylabel('Difference (Current Data - X)')
	plt.savefig(directory+outfilename+'2.png') 
	plt.close(fig2)
	
	# Output wavelength solution
	return wvl_slit_new