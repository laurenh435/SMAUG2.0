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
import math
from run_moog import runMoog
from smooth_gauss import smooth_gauss
from match_spectrum import open_obs_file, smooth_gauss_wrapper
from continuum_div import get_synth, mask_obs_for_division, divide_spec, mask_obs_for_abundance
import subprocess
from astropy.io import fits
import pandas
import scipy.optimize
from wvl_corr import fit_wvl
import csv
from make_plots import make_plots

# Observed spectrum
class obsSpectrum:

	def __init__(self, obsfilename, paramfilename, starnum, wvlcorr, galaxyname, slitmaskname, globular, lines, RA, Dec, element,\
	       atom_num, linelists, linegaps, elementlines, obsspecial=None, plot=False, hires=None, smooth=None, specialparams=None):

		# Observed star
		self.obsfilename 	= obsfilename 	# File with observed spectra
		self.paramfilename  = paramfilename # File with parameters of observed spectra
		self.starnum 		= starnum		# Star number
		self.galaxyname 	= galaxyname 	# Name of galaxy
		self.slitmaskname 	= slitmaskname 	# Name of slitmask
		self.globular 		= globular		# Parameter marking if globular cluster
		self.lines 			= lines 		# Parameter marking whether or not to use revised or original linelist
		self.element        = element       # element you want the abundances of e.g. 'Mn', 'Sr'
		self.atom_num       = atom_num      # atomic number of the element you want the abundances of
		self.linelists      = linelists     # list of line list file names for MOOG
		self.linegaps       = linegaps
		self.elementlines   = elementlines    

		# If observed spectrum comes from moogify file (default), open observed file and continuum normalize as usual
		if obsspecial is None:

			# Output filename
			if self.globular:
				self.outputname = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname
			else:
				self.outputname = '/mnt/c/Research/Spectra/Sr-SMAUGoutput' #'/raid/madlr/dsph/'+galaxyname+'/'+slitmaskname

			# Open observed spectrum
			self.specname, self.obswvl, self.obsflux, self.ivar, self.dlam, self.zrest = open_obs_file(self.obsfilename, retrievespec=self.starnum)

			#self.elementlines = [lines for lines in self.elementlines if lines > self.obswvl[0]] 

			# Get measured parameters from observed spectrum
			self.temp, self.logg, self.fe, self.alpha, self.fe_err = open_obs_file(self.paramfilename, self.starnum, \
									  specparams=True, objname=self.specname, inputcoords=[RA, Dec])
			print('got measured parameters')
			if specialparams is not None:
				self.temp = specialparams[0]
				self.logg = specialparams[1]
				self.fe = specialparams[2]
				self.alpha = specialparams[3]

			if plot:
				# Plot observed spectrum
				plt.figure()
				plt.plot(self.obswvl, self.obsflux, 'k-')
				for line in self.elementlines:
					plt.axvspan(line-10, line+10, alpha=0.5, color='pink')
				plt.savefig(self.outputname+'/'+self.specname+self.element+'_obs.png')
				plt.close()

			# Get synthetic spectrum, split both obs and synth spectra into red and blue parts
			synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs_for_division(self.obswvl, self.obsflux, self.ivar,\
										   self.element, self.linegaps,temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, dlam=self.dlam, lines=self.lines)

			if plot:
				# Plot spliced synthetic spectrum
				plt.figure()
				plt.plot(obswvlmask[0], synthfluxmask[0], 'b-')
				plt.plot(obswvlmask[1], synthfluxmask[1], 'r-')
				plt.savefig(self.outputname+'/'+self.specname+self.element+'_synth.png')
				plt.close()

			# Compute continuum-normalized observed spectrum
			self.obsflux_norm, self.ivar_norm = divide_spec(synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, self.element, \
						   sigmaclip=True, specname=self.specname, outputname=self.outputname)
			
			# Get rid of lines outside of spectrum range
			# badspots_blue = np.where(self.obsflux_norm[0:2000] < 0)
			# blue_cutoff = self.obswvl[badspots_blue][-1]
			# badspots_red = np.where(self.obsflux_norm[-2000:-1] < 0)
			# red_cutoff = self.obswvl[badspots_red][-1]
			# print(self.linelists)
			# self.newlinelists = []
			# for i in range(len(self.elementlines)):
			# 	if self.elementlines[i] > blue_cutoff and self.elementlines[i] < red_cutoff:
			# 		self.newlinelists.append(self.linelists[i])
			# 	else:
			# 		continue
			# print('revised linelist:', self.newlinelists)

			#self.elementlines = [lines for lines in self.elementlines if lines > blue_cutoff and lines < red_cutoff]
			#print('lines in spectrum range:', self.elementlines)


			if plot:
				# Plot continuum-normalized observed spectrum
				plt.figure()
				plt.plot(self.obswvl, self.obsflux_norm, 'k-')
				for line in self.elementlines:
					plt.axvspan(line-10, line+10, alpha=0.5, color='pink')
				#what's red and what's blue?
				#plt.axvspan(4749, 4759, alpha=0.5, color='blue')
				#plt.axvspan(4335, 4345, alpha=0.5, color='red')
				plt.ylim((0,5))
				plt.savefig(self.outputname+'/'+self.specname+self.element+'_obsnormalized.png')
				plt.close()
				np.savetxt(self.outputname+'/'+self.specname+self.element+'_obsnormalized.txt',np.asarray((self.obswvl,self.obsflux_norm)).T)

			if wvlcorr:
				print('Doing wavelength correction...')

				try:
					# Compute standard deviation
					contdivstd = np.zeros(len(self.ivar_norm))+np.inf
					contdivstd[self.ivar_norm > 0] = np.sqrt(np.reciprocal(self.ivar_norm[self.ivar_norm > 0]))

					# Wavelength correction
					self.obswvl = fit_wvl(self.obswvl, self.obsflux_norm, contdivstd, self.dlam, 
						self.temp, self.logg, self.fe, self.alpha, self.specname, self.outputname+'/')

					print('Done with wavelength correction!')

				except Exception as e:
					print(repr(e))
					print('Couldn\'t complete wavelength correction for some reason.')

			# Crop observed spectrum into regions around Mn lines
			self.obsflux_fit, self.obswvl_fit, self.ivar_fit, self.dlam_fit, self.skip = mask_obs_for_abundance(self.obswvl, self.obsflux_norm,\
												        self.ivar_norm, self.dlam, self.element, self.linegaps, lines=self.lines)

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
			synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask = mask_obs_for_division(self.obswvl, self.obsflux, self.ivar,\
										   self.linegaps, temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, dlam=self.dlam, lines=self.lines, hires=True)

			# Compute continuum-normalized observed spectrum
			self.obsflux_norm, self.ivar_norm = divide_spec(synthfluxmask, obsfluxmask, obswvlmask, ivarmask, mask, self.element, specname=self.specname, outputname=self.outputname, hires=True)

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
				plt.savefig(self.outputname+'/'+self.specname+'_obsnormalized.png')
				plt.close()

			# Crop observed spectrum into regions around element lines
			self.obsflux_fit, self.obswvl_fit, self.ivar_fit, self.dlam_fit, self.skip = mask_obs_for_abundance(self.obswvl, self.obsflux_norm,\
												        self.ivar_norm, self.dlam, self.element, self.linegaps, lines=self.lines, hires=True)

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

	# Define function to minimize
	def synthetic(self, obswvl, elem, full=True):
		"""Get synthetic spectrum for fitting.

		Inputs:
		obswvl  -- independent variable (wavelength)
		parameters to fit:
			elem    -- element abundance e.g. Mn, Sr
			dlam    -- FWHM to be used for smoothing

		Keywords:
		full 	-- if True, splice together all desired element line regions; else, keep as array

		Outputs:
		synthflux -- array-like, output synthetic spectrum
		"""

		# Compute synthetic spectrum
		print('Computing synthetic spectrum with parameters: ', elem) #, dlam)
		synth = runMoog(temp=self.temp, logg=self.logg, fe=self.fe, alpha=self.alpha, linelists=self.linelists,\
		   atom_nums=[self.atom_num], elements=[self.element], abunds=[elem], solar=[5.43], lines=self.lines)
		# Loop over each line
		synthflux = []
		for i in self.skip:

			synthregion = synth[i]
			#print(len(synthregion), len(self.obswvl_fit[i]),len(self.obsflux_fit[i]),len(self.ivar_fit[i]), len(self.dlam_fit[i]))

			# Smooth each region of synthetic spectrum to match each region of continuum-normalized observed spectrum

			# uncomment this if dlam is not a fitting parameter
			newsynth = get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit[i], self.dlam_fit[i], synth=synthregion)

			# uncomment this if dlam is a fitting parameter
			#newsynth = get_synth(self.obswvl_fit[i], self.obsflux_fit[i], self.ivar_fit[i], dlam, synth=synthregion)

			synthflux.append(newsynth)

		#print('Finished smoothing synthetic spectrum!')

		# If necessary, splice everything together
		if full:
			synthflux = np.hstack(synthflux[:])

		return synthflux

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
		best_elem, covar = scipy.optimize.curve_fit(self.synthetic, self.obswvl_final, self.obsflux_final, p0=[params0], sigma=np.sqrt(np.reciprocal(self.ivar_final)), epsfcn=0.01)
		error = np.sqrt(np.diag(covar))

		print('Answer: ', best_elem)
		print('Error: ', error)

		# Do some checks
		if len(np.atleast_1d(best_elem)) == 1:
			finalsynth = self.synthetic(self.obswvl_final, best_elem, full=True)
		else:
			finalsynth = self.synthetic(self.obswvl_final, best_elem[0], best_elem[1], full=True)

		# Output the final data
		if output:

			if len(np.atleast_1d(best_elem)) == 1:
				#finalsynthup 	= self.synthetic(self.obswvl_final, best_mn + error, full=True)
				#finalsynthdown 	= self.synthetic(self.obswvl_final, best_mn - error, full=True)
				finalsynthup 	= self.synthetic(self.obswvl_final, best_elem + 0.15, full=True)
				finalsynthdown 	= self.synthetic(self.obswvl_final, best_elem - 0.15, full=True)
			else:
				#finalsynthup = self.synthetic(self.obswvl_final, best_mn[0] + error[0], best_mn[1], full=True)
				#finalsynthdown = self.synthetic(self.obswvl_final, best_mn[0] - error[0], best_mn[1], full=True)
				finalsynthup = self.synthetic(self.obswvl_final, best_elem[0] + 0.15, best_elem[1], full=True)
				finalsynthdown = self.synthetic(self.obswvl_final, best_elem[0] - 0.15, best_elem[1], full=True)

			# Create file
			filename = self.outputname+'/'+self.specname+self.element+'_data.csv'

			# Define columns
			columnstr = ['wvl','obsflux','synthflux','synthflux_up','synthflux_down','ivar']
			columns = np.asarray([self.obswvl_final, self.obsflux_final, finalsynth, finalsynthup, finalsynthdown, self.ivar_final])

			with open(filename, 'w') as csvfile:
				datawriter = csv.writer(csvfile, delimiter=',')

				# Write header
				datawriter.writerow(['['+self.element+'/H]', best_elem[0]])
				if len(np.atleast_1d(best_elem)) > 1:
					datawriter.writerow(['dlam', best_elem[1]])
				datawriter.writerow(columnstr)

				# Write data
				for i in range(len(finalsynth)):
					datawriter.writerow(columns[:,i])

			# Make plots
			if plots:
				make_plots(self.lines, self.elementlines, self.specname+'_', self.obswvl_final, self.obsflux_final, finalsynth,\
	        self.outputname, self.element, self.skip, ivar=self.ivar_final, synthfluxup=finalsynthup, synthfluxdown=finalsynthdown, hires=hires)

		elif plots:
			make_plots(self.lines, self.elementlines, self.specname+'_', self.obswvl_final, self.obsflux_final, finalsynth,\
	       self.outputname, self.element, self.skip, ivar=self.ivar_final, hires=hires)
		
		#print('finalsynth:', finalsynth)

		return best_elem, error

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
			elem_result, elem_error = self.minimize_scipy(params0, plots=plots, output=output)
			print('elem_result:',elem_result)
		else:
			elem_result = [params0[0]]
			elem_error  = [params0[1]]

		#return (remove comment if creating MOOG output files for testing purposes)

		elem_list = np.array([-3,-2,-1.5,-1,-0.5,-0.1,0,0.1,0.5,1,1.5,2,3])*elem_error[0] + elem_result[0]
		chisq_list = np.zeros(len(elem_list))

		#If [X/H] error is small enough, make reduced chi-sq plots
		if elem_error[0] < 1.0:
			for i in range(len(elem_list)):
				finalsynth = self.synthetic(self.obswvl_final, elem_list[i]) #, dlam)
				chisq = np.sum(np.power(self.obsflux_final - finalsynth, 2.) * self.ivar_final) / (len(self.obsflux_final) - 1.)
				chisq_list[i] = chisq

			plt.figure()
			plt.title('Star '+self.specname+' '+self.element, fontsize=18)
			plt.plot(elem_list, chisq_list, '-o')
			plt.ylabel(r'$\chi^{2}_{red}$', fontsize=16)
			plt.xlabel('['+self.element+'/H]', fontsize=16)
			plt.savefig(self.outputname+'/'+self.specname+self.element+'_redchisq.png')
			plt.close()

			if save:
				np.savetxt(self.outputname+'/'+self.specname+'_redchisq.txt',np.asarray((elem_list - self.fe, chisq_list)).T,header="["+self.element+"/Fe], redchisq")
		else:
			finalsynth = self.synthetic(self.obswvl_final, elem_list[6]) #, dlam)
			chisq_list[6] = np.sum(np.power(self.obsflux_final - finalsynth, 2.) * self.ivar_final) / (len(self.obsflux_final) - 1.)

		return elem_result, elem_error, chisq_list[6]

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