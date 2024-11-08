# chi_sq.py
# Computes synthetic spectrum, then compares to observed spectrum
# and finds parameters that minimze chisq measure
# 
# Created 5 Feb 18 by M. de los Reyes
# Updated 2 Nov 18
#
# edited SMAUG to make general -LEH 6/5/2023
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import numpy.ma as ma
import math
from run_moog import runMoog
from smooth_gauss import smooth_gauss
from match_spectrum import open_obs_file, smooth_gauss_wrapper
from continuum_div import get_synth, mask_obs_for_division, divide_spec, mask_obs_for_abundance
from interp_atmosphere import interpolateAtm
import subprocess
from astropy.io import fits
import pandas as pd
import scipy.optimize
#from wvl_corr_new import do_wvl_corr
from wvl_corr_even_newer import do_wvl_corr
import csv
from make_plots import make_plots
#from poly_wvl_corr import fit_wvl2

# Observed spectrum
class obsSpectrum:

	def __init__(self, obsfilename, paramfilename, starnum, wvlcorr, galaxyname, slitmaskname, globular, lines, RA, Dec, element, grating,\
	       atom_num, linelists, linegaps, elementlines, obsspecial=None, plot=False, hires=None, smooth=None, specialparams=None, \
			stravinsky=True, chisq_weight=True, dlamfit=True, file1200G=None):

		# Observed star
		self.obsfilename 	= obsfilename 	# File with observed spectra
		self.paramfilename  = paramfilename # File with parameters of observed spectra
		self.starnum 		= starnum		# Star number (index)
		self.galaxyname 	= galaxyname 	# Name of galaxy
		self.slitmaskname 	= slitmaskname 	# Name of slitmask
		self.globular 		= globular		# Parameter marking if globular cluster
		self.lines 			= lines 		# Parameter marking whether or not to use revised or original linelist
		self.element        = element       # element you want the abundances of e.g. 'Mn', 'Sr'
		self.grating        = grating       # grating used for the slitmask (i.e. '1200B'), used to set dlam
		self.atom_num       = atom_num      # atomic number of the element you want the abundances of
		self.linelists      = linelists     # list of line list file names for MOOG
		self.linegaps       = linegaps      # list of spectrum sections that will be synthesized by MOOG
		self.elementlines   = elementlines  # list of lines from the element of interest that the linegaps are centered on
		self.stravinsky     = stravinsky    # whether SMAUG is running on Stravinsky (currently must run on Stravinsky as the files are)
		self.chisq_weight   = chisq_weight     # maximum chisq value allowed for a line section; if None, will not do chisq cut    

		# If observed spectrum comes from moogify file (default), open observed file and continuum normalize as usual
		if obsspecial is None:

			# Output filename
			if self.globular:
				self.outputname = '/home/lhender6/Research/SMAUGoutput/glob/'+galaxyname
			else:
				if self.stravinsky:
					self.outputname = '/home/lhender6/Research/SMAUGoutput/'+slitmaskname
				else:
					self.outputname = '/mnt/c/Research/SMAUGoutput/'+slitmaskname #'/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname

			# Open observed spectrum
			self.specname, self.obswvl, self.obsflux, self.ivar, self.dlam, self.zrest = \
				open_obs_file(self.obsfilename, grating=self.grating, slitmask=self.slitmaskname, retrievespec=self.starnum)


			if file1200G is not None:
				self.specname1200G, self.obswvl1200G, self.obsflux1200G, self.ivar1200G, self.dlam1200G, self.zrest1200G = \
					open_obs_file(file1200G, grating='1200G', slitmask=self.slitmaskname, retrievespec=self.starnum, objname=self.specname, inputcoords=[RA, Dec])
				print('1200G wavelength range: ',self.obswvl1200G[0][0], self.obswvl1200G[0][-1])
				print('1200G dlam: ', self.dlam1200G[0][0])
				self.obswvl1200G = self.obswvl1200G[0]
				self.obsflux1200G = self.obsflux1200G[0]
				self.ivar1200G = self.ivar1200G[0]
				self.dlam1200G = self.dlam1200G[0]
			
				if self.element == 'Ba' and self.obswvl[-1] > 6497:
					file1200G = None # we don't need 1200G if the blue spectrum has the Ba6497 line
					print('not using 1200G for this star')

			#self.elementlines = [lines for lines in self.elementlines if lines > self.obswvl[0]] 

			# Get measured parameters from observed spectrum
			self.temp, self.logg, self.fe, self.alpha, self.fe_err = open_obs_file(self.paramfilename, retrievespec=self.starnum, \
									  specparams=True, objname=self.specname, inputcoords=[RA, Dec])
			
			# Calculate log(L/Lsun) to get [C/Fe]
			if self.globular:
				chfile = '/raid/caltech/moogify/'+self.galaxyname+self.slitmaskname+'/moogifych.fits.gz'
			else:
				if stravinsky:
					chfile = '/home/lhender6/Research/Spectra/'+self.slitmaskname+'/moogifych.fits.gz'
				else:
					chfile = '/mnt/c/Research/Spectra/'+self.slitmaskname+'/moogifych.fits.gz'
			if os.path.exists(chfile):
				hdu1 = fits.open(chfile)
				chdata = hdu1[1].data
				chnames = chdata['OBJNAME']
				chnames = list(chnames)
				if str(self.specname) in chnames:
					chindex = chnames.index(str(self.specname))
					print('carbon in moogifych!')
					self.carbon = chdata['CFE'][chindex] #[C/Fe]
				else:
					lum = (4*np.pi*6.6743*(10**(-8))*0.75*1.99*(10**(33))*5.6704*(10**(-5))*self.temp**4)/(10**self.logg)
					logl = np.log10(lum/(3.846*(10**(33))))
					#print('luminosity test:')
					#print('L/Lsun',lum/(3.846*(10**(33))) )
					#print('logl',logl)
					if logl <= 1.6:
						self.carbon = 0.12 #[C/Fe]
					else:
						self.carbon = 1.42 - 0.82*logl[0]
			else:
				lum = (4*np.pi*6.6743*(10**(-8))*0.75*1.99*(10**(33))*5.6704*(10**(-5))*self.temp**4)/(10**self.logg)
				logl = np.log10(lum/(3.846*(10**(33))))
				#print('luminosity test:')
				#print('L/Lsun',lum/(3.846*(10**(33))) )
				#print('logl',logl)
				if logl <= 1.6:
					self.carbon = 0.12 #[C/Fe]
				else:
					try:
						self.carbon = 1.42 - 0.82*logl[0]
					except IndexError:
						self.carbon = 1.42 - 0.82*logl
					
			print('carbon:', self.carbon)

			#print('got measured parameters')
			if specialparams is not None:
				self.temp = specialparams[0]
				self.logg = specialparams[1]
				self.fe = specialparams[2]
				self.alpha = specialparams[3]

			if plot:
				# Plot observed spectrum
				plt.figure(figsize=(12,7))
				plt.plot(self.obswvl, self.obsflux, 'k-', alpha=0.7)
				if file1200G is not None:
					plt.plot(self.obswvl1200G, self.obsflux1200G, 'g-', alpha=0.7)
				for line in self.elementlines:
					plt.axvspan(line-10, line+10, alpha=0.5, color='pink')
				plt.savefig(self.outputname+'/'+self.specname+self.element+str(self.starnum)+'_obs.png')
				plt.close()

			# Get synthetic spectrum, split both obs and synth spectra into red and blue parts
			self.synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, self.chipgap = mask_obs_for_division(self.obswvl, self.obsflux, self.ivar,\
										   self.element, self.linegaps,temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha,\
											dlam=self.dlam, carbon=self.carbon, lines=self.lines, stravinsky=self.stravinsky)
			if file1200G is not None:
				print('masking 1200G for division')
				self.synthfluxmask1200G, obsfluxmask1200G, obswvlmask1200G, ivarmask1200G, mask1200G, self.chipgap1200G = mask_obs_for_division(self.obswvl1200G, self.obsflux1200G, self.ivar1200G,\
										   self.element, self.linegaps,temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha,\
											dlam=self.dlam1200G, carbon=self.carbon, lines=self.lines, stravinsky=self.stravinsky, spec1200G=True)


			if plot:
				# print('obswvl',obswvlmask)
				# print('synth',self.synthfluxmask)
				# Plot spliced synthetic spectrum
				plt.figure()
				plt.plot(obswvlmask[0], self.synthfluxmask[0], 'b-')
				plt.plot(obswvlmask[1], self.synthfluxmask[1], 'r-')
				plt.savefig(self.outputname+'/'+self.specname+self.element+str(self.starnum)+'_synth.png')
				plt.close()
				if file1200G is not None:
					plt.figure()
					plt.plot(obswvlmask1200G[0], self.synthfluxmask1200G[0], 'b-')
					plt.plot(obswvlmask1200G[1], self.synthfluxmask1200G[1], 'r-')
					plt.savefig(self.outputname+'/'+self.specname+self.element+'_synth1200G.png')
					plt.close()

			# Compute continuum-normalized observed spectrum
			self.obsflux_norm, self.ivar_norm = divide_spec(self.synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, self.element, \
						   sigmaclip=True, specname=self.specname, outputname=self.outputname)
			print('normalization done!')
			if file1200G is not None:
				print('normalizing 1200G')
				self.obsflux_norm1200G, self.ivar_norm1200G = divide_spec(self.synthfluxmask1200G, obsfluxmask1200G, obswvlmask1200G, ivarmask1200G, mask1200G, self.element, \
						   sigmaclip=True, specname=self.specname, outputname=self.outputname)

			# Try to fit dlam if you want
			if dlamfit:
				self.dlam = self.Fitdlam()

			# Fit C in more detail by interpolating in gridch and adjusting normalization
			##### NOT DOING THIS EXTRA STEP################
			# fit_params = self.CurveFitCarbon()
			# self.carbon = fit_params[0]
			# renormalize with the new carbon value
			# self.synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, self.chipgap = mask_obs_for_division(self.obswvl, self.obsflux, self.ivar,\
			# 							   self.element, self.linegaps,temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha,\
			# 								dlam=self.dlam, carbon=self.carbon, lines=self.lines, stravinsky=self.stravinsky)
			# self.obsflux_norm, self.ivar_norm = divide_spec(self.synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, self.element, \
			# 			   sigmaclip=True, specname=self.specname, outputname=self.outputname)
			###############################################


			if plot:
				# Plot continuum-normalized observed spectrum
				plt.figure(figsize=(12,7))
				plt.plot(self.obswvl, self.obsflux_norm, 'k-', alpha=0.7)
				if file1200G is not None:
					plt.plot(self.obswvl1200G, self.obsflux_norm1200G, 'g-', alpha=0.7)
				for line in self.elementlines:
					plt.axvspan(line-10, line+10, alpha=0.5, color='pink')
				#what's red and what's blue?
				#plt.axvspan(4749, 4759, alpha=0.5, color='blue')
				#plt.axvspan(4335, 4345, alpha=0.5, color='red')
				plt.ylim((0,5))
				plt.savefig(self.outputname+'/'+self.specname+self.element+'_obsnormalized'+str(self.starnum)+'.png')
				plt.close()
				np.savetxt(self.outputname+'/'+self.specname+self.element+'_obsnormalized'+str(self.starnum)+'.txt',np.asarray((self.obswvl,self.obsflux_norm)).T)
				if file1200G is not None:
					np.savetxt(self.outputname+'/'+self.specname+self.element+'_obsnormalized'+str(self.starnum)+'1200G.txt',np.asarray((self.obswvl1200G,self.obsflux_norm1200G)).T)

				obsfluxnorm_mask = np.concatenate((ma.masked_array(np.array(self.obsflux_norm)[:self.chipgap],mask[0]), ma.masked_array(np.array(self.obsflux_norm)[self.chipgap:], mask[1])))
				full_obswvlmask = np.concatenate((obswvlmask[0],obswvlmask[1]))
				full_synthfluxmask = np.concatenate((self.synthfluxmask[0],self.synthfluxmask[1]))
				full_ivarmask = np.concatenate((ma.masked_array(np.array(self.ivar_norm)[:self.chipgap],mask[0]), ma.masked_array(np.array(self.ivar_norm)[self.chipgap:], mask[1])))


				plt.figure()
				plt.plot(obswvlmask[0], self.synthfluxmask[0], 'b-')
				plt.plot(obswvlmask[1], self.synthfluxmask[1], 'r-')
				plt.plot(self.obswvl, self.obsflux_norm, 'k-', alpha=0.7)
				plt.xlim((5200,5300))
				plt.ylim((0.6,1.1))
				plt.savefig(self.outputname+'/'+self.specname+self.element+str(self.starnum)+'_NaDsynth.png')
				plt.close()

				modelfilenameblue = self.outputname+'/'+self.specname+self.element+'_synthblue'+str(self.starnum)+'.csv'
				# Define columns
				columnstr = ['wavelength_blue','synth_blue']
				columns = np.asarray([obswvlmask[0], self.synthfluxmask[0]])
				with open(modelfilenameblue, 'w') as csvfile:
					datawriter = csv.writer(csvfile, delimiter=',')
					datawriter.writerow(columnstr)
					# Write data
					for j in range(len(obswvlmask[0])):
						datawriter.writerow(columns[:,j])

				modelfilenamered = self.outputname+'/'+self.specname+self.element+'_synthred'+str(self.starnum)+'.csv'
				# Define columns
				columnstr = ['wavelength_red','synth_red']
				columns = np.asarray([obswvlmask[1], self.synthfluxmask[1]])
				with open(modelfilenamered, 'w') as csvfile:
					datawriter = csv.writer(csvfile, delimiter=',')
					datawriter.writerow(columnstr)
					# Write data
					for j in range(len(obswvlmask[1])):
						datawriter.writerow(columns[:,j])

				# fig, axs = plt.subplots(2,2, figsize=(25,25))
				# axs[0,0].plot(full_obswvlmask, obsfluxnorm_mask, 'deepskyblue', alpha=0.5, label='obs')
				# axs[0,0].plot(full_obswvlmask, full_synthfluxmask, 'tomato', alpha=0.5, label='synth')
				# axs[0,0].set_xlim((4100,6500)) #range where spectrum is normalized
				# axs[0,0].set_ylim((0.0,1.2))
				# axs[0,0].legend(loc='best')
				# axs[0,1].plot(full_obswvlmask, obsfluxnorm_mask, 'deepskyblue', alpha=0.5)
				# axs[0,1].plot(full_obswvlmask, full_synthfluxmask, 'tomato', alpha=0.5)
				# axs[0,1].set_xlim((4100,4600))
				# axs[0,1].set_ylim((0.0,1.2))
				# axs[1,0].errorbar(full_obswvlmask, obsfluxnorm_mask, yerr=np.power(full_ivarmask,-0.5), color='deepskyblue', alpha=0.7, fmt='o', markersize=6, zorder=3)
				# axs[1,0].plot(full_obswvlmask, full_synthfluxmask, 'tomato', alpha=0.9)
				# axs[1,0].set_xlim((4200,4230))
				# axs[1,0].set_ylim((0.0,1.2))
				# axs[1,1].errorbar(full_obswvlmask, obsfluxnorm_mask, yerr=np.power(full_ivarmask,-0.5), color='deepskyblue', alpha=0.7, fmt='o', markersize=6, zorder=3)
				# axs[1,1].plot(full_obswvlmask, full_synthfluxmask, 'tomato', alpha=0.5)
				# axs[1,1].set_xlim((4850,4870))
				# axs[1,1].set_ylim((0.0,1.2))
				# fig.savefig(self.outputname+'/'+self.specname+self.element+'_normalization.png')
				# plt.close()

				# for testing only
				# for gap in self.linegaps:
				# 	plt.figure()
				# 	plt.plot(full_obswvlmask, obsfluxnorm_mask, 'deepskyblue', alpha=0.7, label='obs')
				# 	plt.plot(full_obswvlmask, full_synthfluxmask, 'tomato', alpha=0.7, label='synth')
				# 	plt.xlim(gap)
				# 	plt.savefig(self.outputname+'/'+self.specname+self.element+'_'+str(gap[0])+'.png')
				# 	plt.close()


			if wvlcorr:
				print('Doing wavelength correction...')
				# print(self.obsflux_norm)
				# print(self.obswvl)
				# print(chipgap)

				try:
					# Compute standard deviation
					# contdivstd = np.zeros(len(self.ivar_norm))+np.inf
					# contdivstd[self.ivar_norm > 0] = np.sqrt(np.reciprocal(self.ivar_norm[self.ivar_norm > 0]))

					# Wavelength correction
					#self.obswvl = fit_wvl(self.obswvl, self.obsflux_norm, contdivstd, self.dlam, 
						#self.temp, self.logg, self.fe, self.alpha, self.specname, self.outputname+'/')
					#self.obswvl = self.fit_wvl2() #my version that totally doesn't work lol
					self.obswvl = do_wvl_corr(self.obswvl, self.obsflux_norm, self.ivar_norm, self.outputname, self.specname, self.dlam)


					print('Done with wavelength correction!')

					plt.figure()
					plt.plot(obswvlmask[0], self.synthfluxmask[0], 'g-')
					plt.plot(obswvlmask[1], self.synthfluxmask[1], 'r-')
					plt.plot(self.obswvl, self.obsflux_norm, 'k-', alpha=0.7)
					plt.xlim((5200,5300))
					plt.ylim((0.6,1.1))
					plt.savefig(self.outputname+'/'+self.specname+self.element+str(self.starnum)+'_NaDsynth_postwvlcorr.png')
					plt.close()

				except Exception as e:
					print(repr(e))
					print('Couldn\'t complete wavelength correction for some reason.')

			# Crop observed spectrum into regions around lines
			self.obsflux_fit, self.obswvl_fit, self.ivar_fit, self.dlam_fit, self.skip = mask_obs_for_abundance(self.obswvl, self.obsflux_norm,\
															self.ivar_norm, self.dlam, self.synthfluxmask, self.element, self.linegaps, lines=self.lines)
			# Stack 1200G spectrum onto the end if using
			if file1200G is not None:
				self.obsflux_fit1200G, self.obswvl_fit1200G, self.ivar_fit1200G, self.dlam_fit1200G, self.skip1200G = mask_obs_for_abundance(self.obswvl1200G, self.obsflux_norm1200G,\
												        self.ivar_norm1200G, self.dlam1200G, self.synthfluxmask1200G, self.element, self.linegaps, lines=self.lines)
				if self.element == 'Ba':
					if self.obswvl[-1] > 6497 and self.obswvl1200G[0] < 6497:
						self.elementlines.append(6496.91)
						self.linegaps.append([6486.91, 6506.91])
						self.linelists.append('full_linelists/Ba6496.txt')
						index = [self.skip[-1]+1]
						self.skip = np.hstack((self.skip, index))
				if self.element == 'Eu':
					if self.obswvl[-1] > 6645 and self.obswvl1200G[0] < 6645:
						self.linegaps.append([6486.91, 6506.91])
						index = [self.skip[-1]+1]
						self.skip = np.hstack((self.skip, index))

				print(self.obsflux_fit.dtype)
				self.obsflux_fit = np.array(self.obsflux_fit.tolist() + [np.asarray(self.obsflux_fit1200G[-1].tolist())], dtype = object)
				self.obswvl_fit = np.array(self.obswvl_fit.tolist() + [np.asarray(self.obswvl_fit1200G[-1].tolist())], dtype = object)
				self.ivar_fit = np.array(self.ivar_fit.tolist() + [np.asarray(self.ivar_fit1200G[-1].tolist())], dtype = object)
				self.dlam_fit = np.array(self.dlam_fit.tolist() + [np.asarray(self.dlam_fit1200G[-1].tolist())], dtype = object)			

			if ((self.element == 'Sr') & (self.skip == [1])).all():
				raise ValueError('Only Sr4607 available! Skipping #'+str(self.starnum))

		# Else, check if we need to open a hi-res file to get the spectrum
		elif hires is not None:

			# Output filename
			if self.globular:
				self.outputname = '/raid/madlr/glob/'+galaxyname+'/'+'hires'
			else:
				self.outputname = '/mnt/c/Research/SMAUG/dsph/'+galaxyname+'/'+'hires'

			# Open observed spectrum
			self.specname = hires
			self.obswvl, self.obsflux, self.dlam = open_obs_file(self.obsfilename, hires=True)

			# Get measured parameters from obsspecial keyword
			self.temp		= obsspecial[0]
			self.logg		= obsspecial[1]
			self.fe 		= obsspecial[2]
			self.alpha 		= obsspecial[3]
			self.fe_err 	= obsspecial[4]
			self.zrest 		= obsspecial[5]

			self.ivar = np.ones(len(self.obsflux))

			# Correct for wavelength
			self.obswvl = self.obswvl / (1. + self.zrest)
			print('Redshift: ', self.zrest)

			# Get synthetic spectrum, split both obs and synth spectra into red and blue parts
			self.synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, self.chipgap = mask_obs_for_division(self.obswvl, self.obsflux, self.ivar,\
										   self.linegaps, temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, dlam=self.dlam,\
											lines=self.lines, hires=True, stravinsky=self.stravinsky)

			# Compute continuum-normalized observed spectrum
			self.obsflux_norm, self.ivar_norm = divide_spec(self.synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, self.element, specname=self.specname, outputname=self.outputname, hires=True)

			if smooth is not None:

				# Crop med-res wavelength range to match hi-res spectrum
				smooth = smooth[np.where((smooth > self.obswvl[0]) & (smooth < self.obswvl[-1]))]

				# Interpolate and smooth the synthetic spectrum onto the observed wavelength array
				self.dlam = np.ones(len(smooth))*0.7086
				self.obsflux_norm = smooth_gauss_wrapper(self.obswvl, self.obsflux_norm, smooth, self.dlam)
				self.obswvl = smooth
				self.ivar_norm = np.ones(len(self.obswvl))*1.e4

			if plot:
				# Plot continuum-normalized observed spectrum
				plt.figure()
				plt.plot(self.obswvl, self.obsflux_norm, 'k-')
				plt.savefig(self.outputname+'/'+self.specname+'_obsnormalized'+str(self.starnum)+'.png')
				plt.close()

			# Crop observed spectrum into regions around element lines
			self.obsflux_fit, self.obswvl_fit, self.ivar_fit, self.dlam_fit, self.skip = mask_obs_for_abundance(self.obswvl, self.obsflux_norm,\
												        self.ivar_norm, self.dlam, self.synthfluxmask, self.element, self.linegaps, lines=self.lines, hires=True)

		# Else, take both spectrum and observed parameters from obsspecial keyword
		else:

			# Output filename
			self.outputname = '/raid/madlr/test/'+slitmaskname

			self.obsflux_fit = obsspecial[0]
			self.obswvl_fit = obsspecial[1]
			self.ivar_fit 	= obsspecial[2]
			self.dlam_fit 	= obsspecial[3]
			self.skip 		= obsspecial[4]
			self.temp		= obsspecial[5]
			self.logg		= obsspecial[6]
			self.fe 		= obsspecial[7]
			self.alpha 		= obsspecial[8]
			self.fe_err 	= obsspecial[9]

			self.specname 	= self.slitmaskname

		# Splice together element line regions of observed spectra
		print('Skip: ', self.skip)

		self.obsflux_final = np.hstack((self.obsflux_fit[self.skip]))
		self.obswvl_final = np.hstack((self.obswvl_fit[self.skip]))
		self.ivar_final = np.hstack((self.ivar_fit[self.skip]))
		#self.dlam_final = np.hstack((self.dlam_fit[self.skip]))
		#print(len(self.obswvl_final))

	def chisqdlam(self, obswvlmask, dlam):
		# Mask out wacky ends of the spectrum
		mask = np.zeros(len(self.obswvl), dtype=bool)
		mask[np.where(self.obswvl < 4500)] = True
		mask[np.where(self.obswvl > (self.obswvl[self.chipgap]-20))] = True
		obswvlmask = ma.masked_array(self.obswvl, mask).compressed()
		# print(obswvlmask[0], obswvlmask[-1])
		obsfluxmask = ma.masked_array(self.obsflux_norm, mask).compressed()
		ivarmask = ma.masked_array(self.ivar_norm, mask).compressed()
		dlam_shaped = np.empty(len(obswvlmask))
		dlam_shaped.fill(dlam)
		# Get synth with dlam guess, calculate reduced chisq
		synth = get_synth(obswvlmask, obsfluxmask, ivarmask, dlam_shaped, synth=None, temp=self.temp, logg=self.logg, \
							fe=self.fe, alpha=self.alpha, carbon=self.carbon, stravinsky=self.stravinsky)
		# redchisq = np.sum(np.power(np.array(obsfluxmask) - np.array(synth), 2.) * np.array(ivarmask)) / (len(obsfluxmask) - 1.)
		# print(redchisq)
		return synth #redchisq
	
	def Fitdlam(self):
		'''Use curve_fit to fit dlambda to the normalized spectrum.
		Return dlambda as an array with a value for each observed point.
		'''
		print('Fitting dlam')
		mask = np.zeros(len(self.obswvl), dtype=bool)
		# Fit between 4500 and the chip gap
		mask[np.where(self.obswvl < 4500)] = True
		mask[np.where(self.obswvl > (self.obswvl[self.chipgap]-20))] = True
		obswvlmask = ma.masked_array(self.obswvl, mask).compressed()
		# print(obswvlmask[0], obswvlmask[-1])
		obsfluxmask = ma.masked_array(self.obsflux_norm, mask).compressed()
		ivarmask = ma.masked_array(self.ivar_norm, mask).compressed()
		# Perform the curve fit
		best_params, covar = scipy.optimize.curve_fit(self.chisqdlam, obswvlmask, obsfluxmask, p0=[self.dlam[0]], sigma=np.sqrt(np.float64(np.reciprocal(ivarmask))))
		new_dlam = best_params[0]
		print('Best dlam:', new_dlam)
		# If dlam is crazy, revert back to the original value
		if (new_dlam > (self.dlam[0]+0.2)) or (new_dlam < (self.dlam[0]-0.3)):
			print('dlam fit probably not trustworthy')
			new_dlam = self.dlam[0]
		# Make it the right size
		dlam_shaped = np.empty(len(self.obsflux_norm))
		dlam_shaped.fill(new_dlam)
		
		# Plot to test
		# synth = get_synth(self.obswvl, self.obsflux_norm, self.ivar_norm, dlam_shaped, synth=None, temp=self.temp, logg=self.logg, \
		# 					fe=self.fe, alpha=self.alpha, carbon=self.carbon, stravinsky=self.stravinsky)
		# plt.figure()
		# plt.plot(self.obswvl, self.obsflux_norm, 'k-', alpha=0.5)
		# plt.plot(self.obswvl, synth, 'r-', alpha=0.5)
		# # plt.xlim([4845,4945])
		# plt.ylim([0,3])
		# plt.savefig(self.outputname+'/'+self.specname+self.element+'_dlamtest.png')
		# plt.close()

		return dlam_shaped
	
	def CarbonSynth(self, obswvl, carbon, a, b, c):
		''' Function to curvefit. Interpolates grdich to carbon values
		and multiplies by a polynomial to correct the flux.
		fe, alpha, temp, logg are already fixed for the star.
		Inputs:
		carbon -- [C/Fe] guess
		obsflux -- observed flux normalized intially (obsflux_norm)
		'''
		#get interpolated synthetic spectrum
		synthflux_ch = 1. - interpolateAtm(self.temp,self.logg,self.fe,self.alpha,carbon=carbon,griddir='/raid/gridch/bin/',gridch=True,stravinsky=True)
		synthwvl_ch = np.fromfile('/raid/gridch/bin/lambda.bin')
		synthwvl_ch  = np.around(synthwvl_ch,2)
		obsmask = np.where((obswvl > 4100) & (obswvl < 4500))
		obswvl_cut = obswvl[obsmask]
		obsflux_cut = self.obsflux_norm[obsmask]
		ivar_cut = self.ivar_norm[obsmask]
		dlam_cut = self.dlam[obsmask]
		
		#smooth to the proper resolution
		synthmask = np.where(synthwvl_ch > obswvl_cut[0])
		synthwvl_ch = synthwvl_ch[synthmask]
		synthflux_ch = synthflux_ch[synthmask]
		synthfluxnew = smooth_gauss_wrapper(synthwvl_ch, synthflux_ch, obswvl_cut, dlam_cut)
		#apply the polynomial - note that now the synthflux follows the same wavelengths as obswvl
		synthfluxnew = synthfluxnew * (a + b*obswvl + c*(obswvl**2)) #adjust flux normalization of the gridch spectrum
		# synthfluxnew = synthfluxnew * (a + b*obswvl) #adjust flux normalization of the gridch spectrum

		return synthfluxnew

	def CurveFitCarbon(self):
		''' Actually perform the curvefitting.
		Inputs:
		startcarbon -- [C/Fe] used for inital normalization
		obsflux -- observed flux normalized (obsflux_norm)
		'''
		#cut obsflux, obswvl, dlam, and ivar_final to be 4100-4500A
		obsmask = np.where((self.obswvl > 4100) & (self.obswvl < 4500))
		obswvl_cut = self.obswvl[obsmask]
		obsflux_cut = self.obsflux_norm[obsmask]
		ivar_cut = self.ivar_norm[obsmask]
		print('initial [C/Fe] value:',self.carbon)

		#do the curve fitting
		best_params, covar = scipy.optimize.curve_fit(self.CarbonSynth, obswvl_cut, obsflux_cut, p0=[self.carbon,1,0,0], sigma=np.sqrt(np.float64(np.reciprocal(ivar_cut))))
		print('carbon curvefit results:',best_params)
		best_carbonsynth = self.CarbonSynth(obswvl_cut, best_params[0],best_params[1],best_params[2],best_params[3])

		#plot for testing
		plt.figure()
		plt.plot(obswvl_cut, obsflux_cut, 'deepskyblue', alpha=0.5, label='observed')
		plt.plot(obswvl_cut, best_carbonsynth, 'forestgreen', alpha=0.5, label='adjusted synth')
		plt.xlim([4100,4500])
		plt.ylim([0.2,1.2])
		plt.title('gridch interpolation test')
		plt.legend(loc='best')
		plt.savefig(self.outputname+'/'+self.specname+str(self.starnum)+'_gridchtest.png')
		plt.close()

		return best_params

	# Define function to minimize
	def synthetic(self, obswvl, elem, full=True, doquotientfit=True, coeffs=None, alternate_Fe=None):
		"""Get synthetic spectrum for fitting.

		Inputs:
		obswvl  -- independent variable (wavelength)
		parameters to fit:
			elem    -- element abundance e.g. Mn, Sr
			dlam    -- FWHM to be used for smoothing. Note that fitting dlam only works for constant dlam for the whole spectrum

		Keywords:
		full 	-- if True, splice together all desired element line regions; else, keep as array

		Outputs:
		synthflux -- array-like, output synthetic spectrum
		"""

		if elem > 10 or elem < -10:
			print('Unphysical guess for [X/H]:', elem)
			raise ValueError('Guesses went crazy. Skip this star')
		# Compute synthetic spectrum
		print('Computing synthetic spectrum with parameters: ', elem)#, dlam)
		if alternate_Fe is not None:
			fe=alternate_Fe
		else:
			fe = self.fe
		synth = runMoog(temp=self.temp, logg=self.logg, fe=fe, alpha=self.alpha, carbon=self.carbon, specname=str(self.starnum), slitmask=self.slitmaskname, linelists=self.linelists, skip=self.skip,\
		   atom_nums=[self.atom_num], elements=[self.element], abunds=[elem], lines=self.lines, stravinsky=self.stravinsky)
		# print('got synth!')
		# Loop over each line
		synthflux = []

		if doquotientfit:
			self.bestquotientfits = []
		for i in range(len(self.skip)):

			synthregion = synth[i]

			# Smooth each region of synthetic spectrum to match each region of continuum-normalized observed spectrum

			# uncomment this if dlam is not a fitting parameter
			newsynth = get_synth(self.obswvl_fit[self.skip[i]], self.obsflux_fit[self.skip[i]], self.ivar_fit[self.skip[i]],\
			 self.dlam_fit[self.skip[i]], synth=synthregion, stravinsky=self.stravinsky)

			# uncomment this if dlam is a fitting parameter
			# dlam_fit = np.empty(len(self.obswvl_fit[self.skip[i]]))
			# dlam_fit.fill(dlam)
			# newsynth = get_synth(self.obswvl_fit[self.skip[i]], self.obsflux_fit[self.skip[i]], self.ivar_fit[self.skip[i]],\
			# 			dlam_fit, synth=synthregion, stravinsky=self.stravinsky)
			
			# renormalize each spectrum section
			if doquotientfit: 
				# print('doing quotient fit')
				newsynth, popt = self.quotientfit(newsynth, i)
				self.bestquotientfits.append(popt)
			else:
				# if you want the synth with no change, coeffs = [[0,1],[0,1],...]
				newsynth,popt = self.quotientfit(newsynth, i, coeffs=coeffs[i])

			#correct for offset normalization
			#newsynth = newsynth*(a + b*self.obswvl_fit[self.skip[i]] + c*self.obswvl_fit[self.skip[i]]**2)

			synthflux.append(newsynth)

		# print('Finished smoothing synthetic spectrum!')

		# If necessary, splice everything together
		if full:
			synthflux = np.hstack(synthflux[:])

		#for debugging, plot each guess that SMAUG makes
		# mask = np.where((self.obswvl > 4597) & (self.obswvl < 4618))
		# figsize = (32,15)
		# plt.figure(figsize=figsize)
		# plt.plot(self.obswvl[mask][0:48], synthflux, 'r-')
		# plt.scatter(self.obswvl[mask],self.obsflux_norm[mask])
		# plt.title(self.specname + str(elem))
		# plt.ylim(0.75,1.1)
		# plt.savefig(self.outputname+'/'+self.specname+'_'+str(elem)+'.png')
		# plt.close()

		return synthflux
	
	def line(self,x,A,B):
		''' for fitting a line to the quotient when getting the synthetic spectrum
		'''
		return A*x + B
	
	def quotientfit(self,synth, i, coeffs=None):
		''' fit a line to obs_flux/synth and apply
		that to the synthetic spectrum
		i -- index of self.skip that you're fitting
		'''
		if coeffs is None:
			quotient = self.obsflux_fit[self.skip[i]]/synth
			# mask out hbeta
			hbetamask = np.zeros(len(self.obswvl_fit[self.skip[i]]), dtype=bool)
			hbetamask[np.where((self.obswvl_fit[self.skip[i]] > 4860) & (self.obswvl_fit[self.skip[i]] < 4864))] = True
			obswvl_hbetamask = np.ma.masked_array(self.obswvl_fit[self.skip[i]], hbetamask).compressed()
			quotient_hbetamask = np.ma.masked_array(quotient, hbetamask).compressed()
			ivar_hbetamask = np.ma.masked_array(self.ivar_fit[self.skip[i]], hbetamask).compressed()
			popt, pcov = scipy.optimize.curve_fit(self.line, obswvl_hbetamask, quotient_hbetamask, p0=[0,1], sigma=np.sqrt(np.float64(np.reciprocal(ivar_hbetamask))))
			#popt, pcov = scipy.optimize.curve_fit(self.line, np.array(self.obswvl_fit[self.skip[i]],dtype='float64'), np.array(quotient,dtype='float64'), p0=[0,1], sigma=np.sqrt(np.float64(np.reciprocal(self.ivar_fit[self.skip[i]]))))
			#print('fit to quotient:',popt)
		else:
			popt = coeffs
		newsynth = synth * (popt[0]*np.array(self.obswvl_fit[self.skip[i]],dtype='float64') + popt[1])
		return newsynth, popt

	def minimize_scipy(self, params0, output=False, plots=False, hires=False):
		"""Minimize residual using scipy.optimize Levenberg-Marquardt.

		Inputs:
		params0  -- initial guesses for parameters:
			smoothivar0 -- if applicable, inverse variance to use for smoothing

		Keywords:
		plots  -- if 'True', also plot final fit & residual
		output -- if 'True', also output a file (default='False')
		hires  -- if 'True', zoom in a bit on plots to better display hi-res spectra

		Outputs:
		fitparams -- best fit parameters
		rchisq	  -- reduced chi-squared
		"""

		# Do minimization
		print('Starting minimization! Initial guesses: ', params0)
		self.guesshigh = False
		#best_fit, covar = scipy.optimize.curve_fit(self.synthetic, self.obswvl_final, self.obsflux_final, p0=[params0,1,0,0], sigma=np.sqrt(np.float64(np.reciprocal(self.ivar_final))), epsfcn=0.01)
		best_fit, covar = scipy.optimize.curve_fit(self.synthetic, self.obswvl_final, self.obsflux_final, p0=params0, sigma=np.sqrt(np.float64(np.reciprocal(self.ivar_final))), epsfcn=0.01)
        # PARAMS0 HAD BRACKETS AROUND IT
		#for some reason sometimes the np.reciprocal function produces floats instead of float64 so needed to add that in...
		error = np.sqrt(np.diag(covar))[0]

		print('Answer: ', best_fit)
		print('Error: ', error)
		best_elem = best_fit[0]
		# best_dlam = best_fit[1]
		fe_ratio = best_elem - self.fe
		print('[X/Fe]:', fe_ratio)

		#PLOTTING TWICE JUST FOR TESTING CHI SQUARE WEIGHT
		# finalsynth = self.synthetic(self.obswvl_final, best_elem, full=True)
		# finalsynthup 	= self.synthetic(self.obswvl_final, best_elem + error, full=True, doquotientfit=False, coeffs=self.bestquotientfits)
		# finalsynthdown 	= self.synthetic(self.obswvl_final, best_elem - error, full=True, doquotientfit=False, coeffs=self.bestquotientfits)
		# make_plots(self.lines, self.elementlines, self.linegaps, self.specname+'_UNWEIGHTED_', self.obswvl_final, self.obsflux_final, finalsynth,\
	    #     self.outputname, self.element, self.skip, ivar=self.ivar_final, synthfluxup=finalsynthup, synthfluxdown=finalsynthdown, hires=hires)
		self.chisq_cut = 5.0
		if self.chisq_weight:
			#calculate reduced chisquare of each line section, then weight sigma based on that chisq and run curve_fit again
			finalsynth_split = self.synthetic(self.obswvl_final, best_elem, full=False)
			chisq_shaped = [] # reduced chisq for each line, repeated so that it's the same shape as self.ivar_fit
			weight_ivar = [] # should also be the same shape as self.ivar_fit. This is 1/chisquared * inverse variance
			self.weights_output = np.zeros(len(self.linegaps)) #overall weight for each line to add to output file: (1/redchisq)*k (I think?)
			self.redchisq_output = np.zeros(len(self.linegaps))
			self.lines_used = np.zeros(len(self.linegaps))
			run_again = False
			for i in range(len(self.skip)):
				if self.element == 'Y': #mask out Hbeta line
					hbetamask = np.zeros(len(self.obswvl_fit[self.skip[i]]), dtype=bool)
					hbetamask[np.where((self.obswvl_fit[self.skip[i]] > 4860) & (self.obswvl_fit[self.skip[i]] < 4864))] = True
					obsflux_hbetamask = np.ma.masked_array(self.obsflux_fit[self.skip[i]], hbetamask).compressed()
					ivar_hbetamask = np.ma.masked_array(self.ivar_fit[self.skip[i]], hbetamask).compressed()
					finalsynth_hbetamask = np.ma.masked_array(finalsynth_split[i], hbetamask).compressed()
					chisq = np.sum(np.power(obsflux_hbetamask - finalsynth_hbetamask, 2.) * ivar_hbetamask) / (len(obsflux_hbetamask) - 1.)
				else:
					chisq = np.sum(np.power(self.obsflux_fit[self.skip[i]] - finalsynth_split[i], 2.) * self.ivar_fit[self.skip[i]]) / (len(self.obsflux_fit[self.skip[i]]) - 1.)
				print('line gap:', self.linegaps[self.skip[i]])
				print('chisq = ',chisq)
				self.redchisq_output[self.skip[i]] = chisq
				# self.weights_output[self.skip[i]] = (1/chisq)
			
			# take average of the chsiq
			redchisq_actual = [redchisq for redchisq in self.redchisq_output if redchisq != 0]
			avg_redchisq = np.mean(redchisq_actual)
			med_redchisq = np.median(redchisq_actual)
			std_redchisq = np.std(redchisq_actual)
			print('average reduced chisq:', avg_redchisq)
			print('median reduced chisq:', med_redchisq)
			print('standard deviation:', std_redchisq)
			for chisqval in self.redchisq_output:
				if (chisqval > (med_redchisq+2*std_redchisq)) and (chisqval > self.chisq_cut):
					run_again = True
				# if chisq > self.chisq_cut: # have just an upper limit on redchisq instead of weighting each line
				# 	run_again = True
			# 	chisq_shaped.append(np.full(len(self.ivar_fit[self.skip[i]]), chisq))
			# 	weight_ivar.append((1/chisq)*self.ivar_fit[self.skip[i]])
			# weight_ivar_final = np.hstack(weight_ivar)
			# #kl = [(np.sum(self.ivar_final))/(np.sum(weight_ivar[j])) for j in range(len(weight_ivar))] #should the lower sum just be over the one line or over all points?
			# k = (np.sum(self.ivar_final))/(np.sum(weight_ivar_final)) #actually I think there is just 1 k value, not line-dependent
			# self.weights_output = self.weights_output*k
			# k_shaped = np.full(len(self.ivar_final),k)
			# new_sigma = np.sqrt(np.float64(np.reciprocal(np.array(weight_ivar_final))))/np.sqrt(k_shaped)
			# new_ivar = np.array(weight_ivar_final)*k
			# print('Running curve_fit with weighted sigma')
			# best_fit, covar = scipy.optimize.curve_fit(self.synthetic, self.obswvl_final, self.obsflux_final, p0=[best_elem], sigma=new_sigma, epsfcn=0.01)
			# error = np.sqrt(np.diag(covar))[0]
			# print('New answer: ', best_fit)
			# print('New error: ', error)
			# best_elem = best_fit[0]
			# # best_dlam = best_fit[1]
			# fe_ratio = best_elem - self.fe
			# print('New [X/Fe]:', fe_ratio)
			############################################### 
			### FOR TESTING
			###############
			if run_again:
				if len(self.skip) == 0:
					raise ValueError('All chisq too high. Skipping #'+str(self.starnum))
				# self.skip = [self.skip[i] for i in range(len(self.skip)) if self.redchisq_output[i] < self.chisq_cut]
				self.skip = [self.skip[i] for i in range(len(self.skip)) if ((self.redchisq_output[i] < (med_redchisq+2*std_redchisq)) or (self.redchisq_output[i] < 5))]
				print('new self.skip based on chi square',self.skip, ' Will run again.')
				#re-mask the observed spectrum to exclude the right lines
				# self.obsflux_fit, self.obswvl_fit, self.ivar_fit, self.dlam_fit, self.skip = mask_obs_for_abundance(self.obswvl, self.obsflux_norm,\
				# 											self.ivar_norm, self.dlam, self.synthfluxmask, self.element, self.linegaps, lines=self.lines, hires=True
				self.obsflux_final = np.hstack((self.obsflux_fit[self.skip]))
				self.obswvl_final = np.hstack((self.obswvl_fit[self.skip]))
				self.ivar_final = np.hstack((self.ivar_fit[self.skip]))
				best_fit, covar = scipy.optimize.curve_fit(self.synthetic, self.obswvl_final, self.obsflux_final, p0=[best_elem], sigma=np.sqrt(np.float64(np.reciprocal(self.ivar_final))), epsfcn=0.01)
				error = np.sqrt(np.diag(covar))[0]
				print('New answer: ', best_fit)
				print('New error: ', error)
				best_elem = best_fit[0]
				fe_ratio = best_elem - self.fe
				print('New [X/Fe]:', fe_ratio)
			for index in self.skip:
				self.lines_used[index] = 1
			new_ivar = self.ivar_final #for the rest of the code to write out correctly
			new_sigma = np.sqrt(np.float64(np.reciprocal(np.array(np.hstack(self.ivar_fit[self.skip])))))
		else: #these still need to be defined so that everything writes out correctly - but it's the same as no weights being used
			new_sigma = np.sqrt(np.float64(np.reciprocal(np.array(np.hstack(self.ivar_fit)))))
			self.weights_output = np.zeros(len(self.linegaps))
			self.redchisq_output = np.zeros(len(self.linegaps))
			new_ivar = self.ivar_final
			for index in self.skip:
				self.lines_used[index] = 1
				
		# Do some checks
		if len(np.atleast_1d(best_elem)) == 1:
			#finalsynth = self.synthetic(self.obswvl_final, best_elem, best_fit[1], best_fit[2], best_fit[3], full=True)
			#print('fitting final synth: check if the fits are the same')
			finalsynth = self.synthetic(self.obswvl_final, best_elem, full=True)
			#print('should be all the best fits:',self.bestquotientfits)
		else:
			finalsynth = self.synthetic(self.obswvl_final, best_elem[0], best_elem[1], full=True)

		

		# Output the final data
		if output:

			if error < 10: # higher error is not worth plotting
				if len(np.atleast_1d(best_elem)) == 1:
					#best_coeffs = self.bestquotientfits
					finalsynthup 	= self.synthetic(self.obswvl_final, best_elem + error, full=True, doquotientfit=False, coeffs=self.bestquotientfits)
					finalsynthdown 	= self.synthetic(self.obswvl_final, best_elem - error, full=True, doquotientfit=False, coeffs=self.bestquotientfits)
					finalsynthup02 	= self.synthetic(self.obswvl_final, best_elem + 0.2, full=True, doquotientfit=False, coeffs=self.bestquotientfits)
					finalsynthdown02 = self.synthetic(self.obswvl_final, best_elem - 0.2, full=True, doquotientfit=False, coeffs=self.bestquotientfits)

					#print('got synthup and synthdown')
					# finalsynthup 	= self.synthetic(self.obswvl_final, best_elem + error, best_fit[1], best_fit[2], best_fit[3], full=True)
					# finalsynthdown 	= self.synthetic(self.obswvl_final, best_elem - error, best_fit[1], best_fit[2], best_fit[3], full=True)
					# finalsynthup 	= self.synthetic(self.obswvl_final, best_elem + 0.15, full=True) #this is what I've been doing so far
					# finalsynthdown 	= self.synthetic(self.obswvl_final, best_elem - 0.15, full=True)
					# finalsynthup 	= self.synthetic(self.obswvl_final, best_elem, full=True)
					# finalsynthdown 	= self.synthetic(self.obswvl_final, -10, full=True) #basically none

					synthflux_no_elem = self.synthetic(self.obswvl_final, -8.0, full=True, doquotientfit=False, coeffs=self.bestquotientfits)
				else:
					#finalsynthup = self.synthetic(self.obswvl_final, best_elem[0] + error[0], best_elem[1], full=True)
					#finalsynthdown = self.synthetic(self.obswvl_final, best_elem[0] - error[0], best_elem[1], full=True)
					finalsynthup = self.synthetic(self.obswvl_final, best_elem[0] + 0.15, best_elem[1], full=True)
					finalsynthdown = self.synthetic(self.obswvl_final, best_elem[0] - 0.15, best_elem[1], full=True)
					
					synthflux_no_elem = self.synthetic(self.obswvl_final, best_elem - 10, full=True, doquotientfit=False, coeffs=self.bestquotientfits)
					

				# Create file
				print('filename:', self.outputname,'/',self.specname,self.element,str(self.starnum),'_data.csv')
				filename = self.outputname+'/'+self.specname+self.element+str(self.starnum)+'_data.csv'
				#print('best elem', best_elem)

				# Define columns
				columnstr = ['wvl','obsflux','synthflux','synthflux_up','synthflux_down','ivar','ivar_weighted']
				columns = np.asarray([self.obswvl_final, self.obsflux_final, finalsynth, finalsynthup, finalsynthdown, self.ivar_final, new_ivar])
				print('trying to open file')
				with open(filename, 'w') as csvfile:
					print('opened file')
					datawriter = csv.writer(csvfile, delimiter=',')

					# Write header
					datawriter.writerow(['['+self.element+'/H]', best_elem])
					#print('wrote header')
					if len(np.atleast_1d(best_elem)) > 1:
						datawriter.writerow(['dlam', best_elem])
						#print('did this')
					datawriter.writerow(columnstr)

					# Write data
					for i in range(len(finalsynth)):
						datawriter.writerow(columns[:,i])

				# Make plots
				if plots:
					if error > 0.2:
						synthflux02_above = True
					else:
						synthflux02_above = False
					make_plots(self.lines, self.elementlines, self.linegaps, self.specname+'_', self.starnum, self.obswvl_final, self.obsflux_final, finalsynth,\
					self.outputname, self.element, self.skip, ivar=self.ivar_final, synthfluxup=finalsynthup, synthfluxdown=finalsynthdown, \
					synthfluxup02=finalsynthup02, synthfluxdown02=finalsynthdown02, synthflux02_above=synthflux02_above, synthflux_no_elem=synthflux_no_elem, hires=hires) #plot with no elem line
					# make_plots(self.lines, self.elementlines, self.linegaps, self.specname+'_', self.starnum, self.obswvl_final, self.obsflux_final, finalsynth,\
					# self.outputname, self.element, self.skip, ivar=self.ivar_final, synthfluxup=finalsynthup, synthfluxdown=finalsynthdown, \
					# hires=hires)
			else:
				make_plots(self.lines, self.elementlines, self.linegaps, self.specname+'_', self.starnum, self.obswvl_final, self.obsflux_final, finalsynth,\
					self.outputname, self.element, self.skip, ivar=self.ivar_final)

		elif plots:
			make_plots(self.lines, self.elementlines, self.linegaps, self.specname+'_', self.obswvl_final, self.obsflux_final, finalsynth,\
	       self.outputname, self.element, self.skip, ivar=self.ivar_final, hires=hires)
		
		#print('finalsynth:', finalsynth)

		return best_fit, error, new_sigma

	def plot_chisq(self, params0, minimize=True, output=False, plots=False, save=False):
		"""Plot chi-sq as a function of [X/H].

		Inputs:
		params0 -- initial guesses for parameters:
			smoothivar0 -- if applicable, inverse variance to use for smoothing

		Keywords:
		minimize -- if 'True' (default), params0 is an initial guess, and code will minimize;
					else, params0 must be a list containing the best-fit element abundance and the error [elem_result, elem_error]
		plots    -- if 'True', also plot final fit & residual (note: only works if minimize=True)

		Outputs:
		fitparams -- best fit parameters
		rchisq	  -- reduced chi-squared
		"""

		if minimize:
			elem_result, elem_error, new_sigma = self.minimize_scipy(params0, plots=plots, output=output)
			#print('elem_result:',elem_result)
		else:
			elem_result = [params0[0]]
			elem_error  = params0[1]
		#quotient_coeffs = self.bestquotientfits

		#return (remove comment if creating MOOG output files for testing purposes)

		elem_list = np.array([-3,-2,-1.5,-1,-0.5,-0.1,0,0.1,0.5,1,1.5,2,3])*elem_error + elem_result[0]
		# print('elem_list',elem_list)
		chisq_list = np.zeros(len(elem_list)) #this will be reduced chi sq
		notredchisq_list = np.zeros(len(elem_list))

		#If [X/H] error is small enough, make reduced chi-sq plots
		if elem_error < 1.0:
			for i in range(len(elem_list)):
				#print('redchisq test',self.obswvl_final)
				# finalsynth = self.synthetic(self.obswvl_final, elem_list[i], elem_result[1], elem_result[2], elem_result[3]) #, dlam)
				#finalsynth = self.synthetic(self.obswvl_final, elem_list[i], doquotientfit=False, coeffs=self.bestquotientfits) #, dlam)
				finalsynth = self.synthetic(self.obswvl_final, elem_list[i])
				chisq = np.sum(np.power(self.obsflux_final - finalsynth, 2.) * (np.power(np.reciprocal(new_sigma), 2.))) / (len(self.obsflux_final) - 1.)

				# chisq = np.sum(np.power(self.obsflux_final - finalsynth, 2.) * self.ivar_final) / (len(self.obsflux_final) - 1.)
				chisq_list[i] = chisq
				notredchisq = np.sum(np.power(self.obsflux_final - finalsynth, 2.) * (np.power(np.reciprocal(new_sigma), 2.)))
				notredchisq_list[i] = notredchisq

			# Create file to output chisq calculation

			chisqfilename = self.outputname+'/'+self.specname+self.element+'_chisq'+str(self.starnum)+'.csv'
			# Define columns
			columnstr = ['['+self.element+'/H] guess','red chisq','chisq']
			columns = np.asarray([elem_list, chisq_list, notredchisq_list])
			with open(chisqfilename, 'w') as csvfile:
				datawriter = csv.writer(csvfile, delimiter=',')

				# Write header
				datawriter.writerow(['['+self.element+'/H]', elem_result])
				# if len(np.atleast_1d(elem_result)) > 1:
				# 	datawriter.writerow(['dlam', elem_result])
				datawriter.writerow(columnstr)
				# Write data
				for j in range(len(elem_list)):
					datawriter.writerow(columns[:,j])

			plt.figure()
			plt.title('Star '+self.specname+' '+self.element, fontsize=18)
			plt.plot(elem_list, chisq_list, '-o')
			plt.ylabel(r'$\chi^{2}_{red}$', fontsize=16)
			plt.xlabel('['+self.element+'/H]', fontsize=16)
			plt.savefig(self.outputname+'/'+self.specname+self.element+'_redchisq'+str(self.starnum)+'.png')
			plt.close()

			if save:
				np.savetxt(self.outputname+'/'+self.specname+'_redchisq'+str(self.starnum)+'.txt',np.asarray((elem_list - self.fe, chisq_list)).T,header="["+self.element+"/Fe], redchisq")
		else:
			# finalsynth = self.synthetic(self.obswvl_final, elem_list[6], elem_result[1], elem_result[2], elem_result[3]) #, dlam)
			#finalsynth = self.synthetic(self.obswvl_final, elem_list[6], doquotientfit=False, coeffs=self.bestquotientfits) #, dlam)
			finalsynth = self.synthetic(self.obswvl_final, elem_list[6])
			chisq_list[6] = np.sum(np.power(self.obsflux_final - finalsynth, 2.) * self.ivar_final) / (len(self.obsflux_final) - 1.)

		return elem_result, elem_error, chisq_list[6], self.dlam[0]

def test_hires(starname, starnum, galaxyname, slitmaskname, temp, logg, feh, alpha, zrest):

	filename = '/raid/keck/hires/'+galaxyname+'/'+starname+'/'+starname #+'_017.fits'

	# Try fitting directly to hi-res spectrum
	#test = obsSpectrum(filename, filename, 0, True, galaxyname, slitmaskname, True, 'new', obsspecial=[temp, logg, feh, alpha, 0.0, zrest], plot=False, hires=starname).minimize_scipy(feh, output=False, plots=True, hires=True)

	# Smooth hi-res spectrum to med-res before fitting
	obswvl = obsSpectrum('/raid/caltech/moogify/n5024b_1200B/moogify.fits.gz', '/raid/caltech/moogify/n5024b_1200B/moogify.fits.gz', starnum, True, galaxyname, slitmaskname, True, 'new', plot=False).obswvl
	test = obsSpectrum(filename, filename, 0, True, galaxyname, slitmaskname, True, 'new', obsspecial=[temp, logg, feh, alpha, 0.0, zrest], plot=True, hires=starname, smooth=obswvl).minimize_scipy(feh, output=False, plots=True, hires=True)

	return

def main():
	filename = '/raid/caltech/moogify/n5024b_1200B/moogify.fits.gz'
	#paramfilename = '/raid/m31/dsph/scl/scl1/moogify7_flexteff.fits.gz'
	paramfilename = '/raid/caltech/moogify/n5024b_1200B/moogify.fits.gz'
	galaxyname = 'n5024'
	slitmaskname = 'n5024b_1200B'

	# Code for Evan for Keck 2019A proposal
	#test1 = obsSpectrum(filename, paramfilename, 16, True, galaxyname, slitmaskname, False, 'new', plot=True).minimize_scipy(-2.68, output=True)
	#test2 = obsSpectrum(filename, paramfilename, 30, True, galaxyname, slitmaskname, False, 'new', plot=True).minimize_scipy(-1.29, output=True)
	#test2 = obsSpectrum(filename, paramfilename, 26, True, galaxyname, slitmaskname, False, 'new', plot=True).plot_chisq(-1.50, output=True, plots=False)

	# Code to check hi-res spectra
	#test_hires('B9354', 9, 'n5024','hires', 4733, 1.6694455544153846, -1.8671022414349092, 0.2060026649715580, -0.00022376)
	#test_hires('S16', 3, 'n5024','hires', 4465, 1.1176236470540364, -2.0168930661196254, 0.2276681163556594, -0.0002259)
	#test_hires('S230', 8, 'n5024','hires', 4849, 1.6879225969314575, -1.9910418985188603, 0.23366356933861662, -0.0002172)
	#test_hires('S29', 4, 'n5024','hires', 4542, 1.1664302349090574, -2.0045057512527262, 0.18337140203171015, -0.00023115)
	#test_hires('S32', 5, 'n5024','hires', 4694, 1.3708726167678833, -2.2178865839654534, 0.23014964700722065, -0.00022388)

	# Code to test linelist
	#test = obsSpectrum(filename, paramfilename, 4, True, galaxyname, slitmaskname, True, 'new', plot=True).minimize_scipy(-2.0045057512527262, output=True, plots=True)

	#print('we done')
	#test = obsSpectrum(filename, 57).plot_chisq(-2.1661300692266998)

	# Get data for single star in Scl
	obsSpectrum('/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', '/raid/caltech/moogify/bscl5_1200B/moogify.fits.gz', 66, False, 'scl', 'bscl5_1200B', False, 'new', plot=True).minimize_scipy(-1.8616617309640884, output=True)

if __name__ == "__main__":
	main()