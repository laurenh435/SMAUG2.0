# output.py
# Produces nice outputs
#
# Created 5 June 18 by M. de los Reyes
#
# Attempting to edit this to run for just one star on my computer -LEH 3/31/23
# New copy of this to make general for any element -LEH 5/31/2023
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
from match_spectrum import open_obs_file
from continuum_div import get_synth, mask_obs_for_division, divide_spec, mask_obs_for_abundance
from make_linelists import split_list
import subprocess
from astropy.io import fits, ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas
import scipy.optimize
import chi_sq
from make_plots import make_plots

def run_chisq(filename, paramfilename, galaxyname, slitmaskname, element, atom_num, startstar=0, globular=False, lines='new', plots=False, wvlcorr=True, stravinsky=False, membercheck=None, memberlist=None, velmemberlist=None):
	""" Measure abundances from a FITS file.

	Inputs:
	filename 		-- file with observed spectra
	paramfilename 	-- file with parameters of observed spectra
	galaxyname		-- galaxy name, options: 'scl'
	slitmaskname 	-- slitmask name, options: 'scl1'
	element         -- element you want the abundances of e.g. 'Sr', 'Mn'
	atom_num        -- atomic number of the element you want the abundance of

	Keywords:
	startstar		-- if 0 (default), start at beginning of file and write new datafile;
						else, start at #startstar and just append to datafile
	globular 		-- if 'False' (default), put into output path of galaxy;
						else, put into globular cluster path
	lines 			-- if 'new' (default), use new revised linelist;
						else, use original linelist from Judy's code
	plots 			-- if 'False' (default), don't plot final fits/resids while doing the fits;
						else, plot them
	wvlcorr 		-- if 'True' (default), do linear wavelength corrections following G. Duggan's code for 900ZD data;
						else (for 1200B data), don't do corrections
	membercheck 	-- do membership check for this object
	memberlist		-- member list (from Evan's Li-rich giants paper)
	velmemberlist 	-- member list (from radial velocity check)

	"""

	# Output filename
	if globular:
		outputname = '/mnt/c/Research/glob/'+galaxyname+'/'+slitmaskname+'.csv'
	else:
		if stravinsky:
			outputname = '/home/lhender6/Research/Spectra/Sr-SMAUGoutput/'+slitmaskname+element+'.csv'
		else:
			outputname = '/mnt/c/Research/Spectra/Sr-SMAUGoutput/'+slitmaskname+element+'.csv' #using this for now - I suppose there will be a separate one for each element?

	# Open new file
	if startstar<1:
		with open(outputname, 'w+') as f:
			f.write('Name\tRA\tDec\tTemp\tlog(g)\t[Fe/H]\terror([Fe/H])\t[alpha/Fe]\t['+element+\
	   '/H]\terror(['+element+'/H])\tchisq(reduced)\n')

	# Prep for member check
	if membercheck is not None:

		# Check if stars are in member list from Evan's Li-rich giants paper
		table = ascii.read(memberlist)
		memberindex = np.where(table.columns[0] == membercheck)
		membernames = table.columns[1][memberindex]

		# Also check if stars are in member list from velocity cut
		if velmemberlist is not None:
			oldmembernames = membernames
			membernames = []
			velmembernames = np.genfromtxt(velmemberlist,dtype='str')
			for i in range(len(oldmembernames)):
				if oldmembernames[i] in velmembernames:
					membernames.append(oldmembernames)

			membernames = np.asarray(membernames)

	# Get number of stars in file
	Nstars = open_obs_file(filename)

	# Get coordinates of stars in file
	RA, Dec = open_obs_file(filename, coords=True)

	# Make line lists for the element of interest
	if stravinsky:
		linelists, linegaps, elementlines = split_list('/home/lhender6/Research/Sr-SMAUG/full_linelists/full_lines_sprocess.txt', atom_num, element, stravinsky=stravinsky)
	else:
		linelists, linegaps, elementlines = split_list('/mnt/c/Research/Sr-SMAUG/full_linelists/full_lines_sprocess.txt', atom_num, element, stravinsky=stravinsky)
	if element == 'Sr': #synth grids don't go below 4100 A, so get rid of Sr 4077 line for now
		elementlines.pop(0)
		linegaps.pop(0)
		linelists.pop(0)
	print('element lines:', elementlines)
	print('line gaps:',linegaps)
	#split_list('/mnt/c/Research/Sr-SMAUG/full_linelists/full_lines_sprocess.txt', atom_num, element) -- for s-process elements

	# Run chi-sq fitting for all stars in file
	for i in range(7, 8): #range(startstar, Nstars)

		try:
			# Get metallicity of star to use for initial guess
			print('Getting initial metallicity')
			temp, logg, fe, alpha, fe_err = open_obs_file(filename, retrievespec=i, specparams=True)

			# Get dlam (FWHM) of star to use for initial guess
			specname, obswvl, obsflux, ivar, dlam, zrest = open_obs_file(filename, retrievespec=i)

			# Check for bad parameter measurement
			if np.isclose(temp, 4750.) and np.isclose(fe,-1.5) and np.isclose(alpha,0.2):
				print('Bad parameter measurement! Skipped #'+str(i+1)+'/'+str(Nstars)+' stars'+'\n')
				continue

			# Do membership check
			if membercheck is not None:
				if specname not in membernames:
					print('Not in member list! Skipped '+specname+'\n')  #'+str(i+1)+'/'+str(Nstars)+' stars')
					continue

			# Run optimization code
			star = chi_sq.obsSpectrum(filename, paramfilename, i, wvlcorr, galaxyname, slitmaskname, globular, lines, RA[i], \
			     Dec[i], element, atom_num, linelists, linegaps, elementlines, plot=True, stravinsky=stravinsky)
			best_elem, error, finalchisq = star.plot_chisq(fe, output=True, plots=plots)

		except Exception as e:
			print(repr(e))
			print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars'+'\n')
			continue

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars'+'\n')
		#print('test', star.specname, RA[i], Dec[i], star.temp, star.logg, star.fe, star.fe_err, star.alpha, best_elem, error, finalchisq)

		with open(outputname, 'a') as f:
			f.write(star.specname+'\t'+str(RA[i])+'\t'+str(Dec[i])+'\t'+str(star.temp)+'\t'+str(star.logg)+'\t'+str(star.fe)+'\t'+str(star.fe_err)+'\t'+str(star.alpha)+'\t'+str(best_elem[0])+'\t'+str(error[0])+'\t'+str(finalchisq)+'\n')

	return

def make_chisq_plots(filename, paramfilename, galaxyname, slitmaskname, element, startstar=0, globular=False, stravinsky=False):
	""" Plot chisq contours for stars whose [X/H] abundances have already been measured.

	Inputs:
	filename 		-- file with observed spectra
	paramfilename 	-- file with parameters of observed spectra
	galaxyname		-- galaxy name, options: 'scl'
	slitmaskname 	-- slitmask name, options: 'scl1'

	Keywords:
	globular 		-- if 'False', put into output path of galaxy; else, put into globular cluster path

	"""

	# Input filename
	if globular:
		file = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'.csv'
	else:
		if stravinsky:
			file = '/home/lhender6/Research/Spectra/Sr-SMAUGoutput/'+slitmaskname+element+'.csv' #using this for now
		else:
			file = '/mnt/c/Research/Spectra/Sr-SMAUGoutput/'+slitmaskname+element+'.csv' #using this for now

	name  = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')
	elem    = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=8)
	elemerr = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=9)
	#dlam = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=8)
	#dlamerr = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=9)

	# Get number of stars in file with observed spectra
	Nstars = open_obs_file(filename)

	# Plot chi-sq contours for each star
	for i in range(startstar, 1): #ended at Nstars

		try:

			# Check if parameters are measured
			temp, logg, fe, alpha, fe_err = open_obs_file(filename, retrievespec=i, specparams=True)
			if np.isclose(1.5,logg) and np.isclose(fe,-1.5) and np.isclose(fe_err, 0.0):
				print('Bad parameter measurement! Skipped #'+str(i+1)+'/'+str(Nstars)+' stars')
				continue

			# Open star
			star = chi_sq.obsSpectrum(filename, paramfilename, i, False, galaxyname, slitmaskname, globular, 'new')

			# Check if star has already had [Mn/H] measured
			if star.specname in name:

				# If so, plot chi-sq contours if error is < 1 dex
				idx = np.where(name == star.specname)
				if elemerr[idx][0] < 1:
					params0 = [elem[idx][0], elemerr[idx][0]]
					best_elem, error = star.plot_chisq(params0, minimize=False, plots=True, save=True)

		except Exception as e:
			print(repr(e))
			print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars')
			continue

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars')

	return

def plot_fits_postfacto(filename, paramfilename, galaxyname, slitmaskname, element, startstar=0, globular=False, lines='new', mn_cluster=None):
	""" Plot fits, residuals, and ivar for stars whose [X/H] abundances have already been measured.

	Inputs:
	filename 		-- file with observed spectra
	paramfilename 	-- file with parameters of observed spectra
	galaxyname		-- galaxy name, options: 'scl'
	slitmaskname 	-- slitmask name, options: 'scl1'

	Keywords:
	startstar		-- if 0 (default), start at beginning of file and write new datafile;
						else, start at #startstar and just append to datafile
	globular 		-- if 'False' (default), put into output path of galaxy;
						else, put into globular cluster path
	lines 			-- if 'new' (default), use new revised linelist;
						else, use original linelist from Judy's code
	mn_cluster 		-- if not None (default), also plot spectrum with [Mn/H] = mean [Mn/H] of cluster

	"""

	# Input filename
	if globular:
		file = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'.csv'
	else:
		file = '/mnt/c/Research/Spectra/Sr-SMAUGoutput/'+slitmaskname+element+'.csv' #using this for now

	# Output filepath
	if globular:
		outputname = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname
	else:
		outputname = '/mnt/c/Research/Spectra/Sr-SMAUGoutput/'+slitmaskname+element #using this for now

	name  = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, dtype='str')
	mn    = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=8)
	mnerr = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=9)

	# Get number of stars in file with observed spectra
	Nstars = open_obs_file(filename)

	# Open file to store reduced chi-sq values
	chisqfile = outputname+'_chisq.txt'
	with open(chisqfile, 'w+') as f:
		#print('made it here')
		f.write('Star'+'\t'+'Line'+'\t'+'redChiSq (best['+element+'/H])'+'\t'+'redChiSq (best['+element+'/H]+0.15)'+'\t'+'redChiSq (best['+element+'/H]-0.15)'+'\n')

	# Plot spectra for each star
	for i in range(startstar, 3): #ended at Nstars

		try:

			# Check if parameters are measured
			temp, logg, fe, alpha, fe_err = open_obs_file(filename, retrievespec=i, specparams=True)
			if np.isclose(1.5,logg) and np.isclose(fe,-1.5) and np.isclose(fe_err, 0.0):
				print('Bad parameter measurement! Skipped #'+str(i+1)+'/'+str(Nstars)+' stars')
				continue

			# Open star
			star = chi_sq.obsSpectrum(filename, paramfilename, i, False, galaxyname, slitmaskname, globular, lines, plot=True) #missing a bunch of arguments?
			#self, obsfilename, paramfilename, starnum, wvlcorr, galaxyname, slitmaskname, globular, lines, RA, Dec, element, atom_num, linelists, linegaps, obsspecial=None, plot=False, hires=None, smooth=None, specialparams=None

			# Check if star has already had [Mn/H] measured
			if star.specname in name:

				# If so, open data file for star
				if globular:
					datafile = '/raid/madlr/glob/'+galaxyname+'/'+slitmaskname+'/'+str(star.specname)+element+'_data.csv'
				else:
					datafile = '/mnt/c/Research/Spectra/Sr-SMAUGoutput/'+str(star.specname)+element+'_data.csv'

				# Get observed and synthetic spectra and inverse variance array
				obswvl 		= np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=0)
				obsflux 	= np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=1)
				synthflux 	= np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=2)
				#synthfluxup = np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=3)
				#synthfluxdown = np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=4)
				ivar 		= np.genfromtxt(datafile, delimiter=',', skip_header=2, usecols=5)

				idx = np.where(name == star.specname)

				synthfluxup 	= star.synthetic(obswvl, mn[idx] + 0.15, full=True)
				synthfluxdown 	= star.synthetic(obswvl, mn[idx] - 0.15, full=True)
				synthflux_nomn 	= star.synthetic(obswvl, -10.0, full=True)

				if mn_cluster is not None:
					synthflux_cluster = [mn_cluster, star.synthetic(obswvl, mn_cluster, full=True)]
				else:
					synthflux_cluster=None

				if mnerr[idx][0] < 1:
					# Run code to make plots
					make_plots(lines, star.specname+'_', obswvl, obsflux, synthflux, outputname, ivar=ivar, resids=True, synthfluxup=synthfluxup, synthfluxdown=synthfluxdown, synthflux_nomn=synthflux_nomn, synthflux_cluster=synthflux_cluster, title=None, savechisq=chisqfile)

					# Write all plotting data to a file
					hdr = 'Star '+str(star.specname)+'\n'+'obswvl\tobsflux\tsynthflux\tsynthfluxup\tsynthfluxdown\tsynthflux_nomn\n'
					np.savetxt(outputname+'/'+str(star.specname)+element+'_finaldata.csv', np.asarray((obswvl,obsflux,synthflux,synthfluxup,synthfluxdown,synthflux_nomn)).T, header=hdr)

		except Exception as e:
			print(repr(e))
			print('Skipped star #'+str(i+1)+'/'+str(Nstars)+' stars')
			continue

		print('Finished star '+star.specname, '#'+str(i+1)+'/'+str(Nstars)+' stars')

	return

def main():
	# run_chisq('/mnt/c/Research/Spectra/bscl1/moogify.fits.gz', '/mnt/c/Research/Spectra/bscl1/moogify.fits.gz', 'scl', 'bscl1', 'Sr',\
	#    atom_num=38, startstar=0, globular=False, lines='new', plots=True, wvlcorr=True, stravinsky=False)
	run_chisq('/home/lhender6/Research/Spectra/bscl1/moogify.fits.gz', '/home/lhender6/Research/Spectra/bscl1/moogify.fits.gz', 'scl', 'bscl1', 'Sr',\
	   atom_num=38, startstar=0, globular=False, lines='new', plots=True, wvlcorr=False, stravinsky=True)

    
if __name__ == "__main__":
	main()