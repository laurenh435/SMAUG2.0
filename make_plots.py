# make_plots.py
# Make plots for other codes
# 
# Created 9 Nov 18
# Updated 9 Nov 18
###################################################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import math

# Code to make plots
def make_plots(lines, specname, obswvl, obsflux, synthflux, outputname, resids=True, ivar=None, title=None, synthfluxup=None, synthfluxdown=None, synthflux_nomn=None, synthflux_cluster=None, savechisq=None, hires=False):
	"""Make plots.

	Inputs:
	lines -- which linelist to use? Options: 'new', 'old'
	specname -- name of star
	obswvl 	-- observed wavelength array
	obsflux -- observed flux
	synthflux 	-- synthetic flux
	outputname 	-- where to output file

	Keywords:
	resids  -- plot residuals if 'True' (default); else, don't plot residuals
	ivar 	-- inverse variance; if 'None' (default), don't plot errorbars
	title 	-- plot title; if 'None' (default), then plot title = "Star + ID"
	synthfluxup & synthfluxdown -- if not 'None' (default), then plot synthetic spectrum as region between [Mn/H]_best +/- 0.3dex
	synthflux_nomn 		-- if not 'None' (default), then plot synthetic spectrum with [Mn/H] = -10.0
	synthflux_cluster 	-- if not 'None' (default), then plot synthetic spectrum with mean [Mn/H] of cluster; in format synthflux_cluster = [mean [Mn/H], spectrum]
	savechisq 	-- if not 'None' (default), compute & save reduced chi-sq for each line in output file with path savechisq
	hires 	-- if 'False', plot as normal; else, zoom in a bit to show hi-res spectra

	Outputs:
	"""

	# Define lines to plot
	if lines == 'new':
		#linelist = np.array([4739.,4754.,4761.5,4765.5,4783.,4823.,5394.,5399.,
		#					 5407.,5420.,5432.,5516.,5537.,6013.,6016.,6021.,6384.,6491.])
		#linelist = np.array([4739.1, 4754.0, 4761.9, 4765.8, 4783.4, 4823.5, 
		#					 5394.6, 5399.5, 5407.3, 5420.3, 5432.3, 5516.8,
		#					 5537.7, 6013.3, 6016.6, 6021.8, 6384.7, 6491.7])
		#linewidth = np.array([1.,1.,1.5,1.,1.,1.,
		#					  1.,1.,1.,1.,1.,1.,
		#					  1.,1.,1.,1.,1.,1.])

		linelist = np.array([4739.1, 4754.0, 4761.9, 4765.8, 4783.4, 
							 4823.5, 5407.3, 5420.3, 5516.8, 5537.7, 
							 6013.3, 6016.6, 6021.8, 6384.7, 6491.7])
		linewidth = np.array([1.,1.,1.5,1.,1.,
							  1.,1.,1.,1.,1.,
							  1.,1.,1.,1.,1.])

		nrows = 3
		ncols = 5
		figsize = (32,15)
		#figsize = (40,15)
		#figsize = (20,12)

	elif lines == 'old':
		linelist = np.array([4739.,4783.,4823.,5394.,5432.,5516.,5537.,6013.,6021.,6384.,6491.])
		linewidth = np.ones(len(linelist))

		nrows = 3
		ncols = 4
		figsize = (20,15)

	# Define title
	if title is None:
		title = 'Star'+specname

	# Plot showing fits
	#f, axes = plt.subplots(nrows, ncols, sharey='row', num=1, figsize=figsize)
	plt.figure(num=1, figsize=figsize)
	plt.title(title)

	# Plot showing residuals
	if resids:
		plt.figure(num=2, figsize=figsize)
		plt.title(title)

	# Plot showing ivar
	if ivar is not None:
		plt.figure(num=3, figsize=figsize)
		plt.title(title)

	# Prep for computing reduced chi-sq:
	if savechisq is not None:
		chisq = np.zeros(len(linelist))
		chisq_up = np.zeros(len(linelist))
		chisq_down = np.zeros(len(linelist))

	for i in range(len(linelist)):
		#f = plt.figure(1)
		#for i, ax in enumerate(f.axes):

		# Range over which to plot
		#if hires == False:
		lolim = linelist[i] - 10
		uplim = linelist[i] + 10
		#else:
		#	lolim = linelist[i] - 5
		#	uplim = linelist[i] + 5

		# Make mask for wavelength
		try:
			mask = np.where((obswvl > lolim) & (obswvl < uplim))

			if len(mask[0]) > 0:

				if ivar is not None:
					yerr=np.power(ivar[mask],-0.5)
				else:
					yerr=None

				# Plot fits
				with plt.rc_context({'axes.linewidth':4, 'axes.edgecolor':'#594F4F', 'xtick.color':'#594F4F', 'ytick.color':'#594F4F'}):
					plt.figure(1)

					if i==0:
						ax = plt.subplot(nrows,ncols,i+1)
					else:
						plt.subplot(nrows,ncols,i+1) #,sharey=ax)

					plt.axvspan(linelist[i] - linewidth[i], linelist[i] + linewidth[i], color='green', zorder=1, alpha=0.25)

					# Plot synthetic spectrum
					if (synthfluxup is not None) and (synthfluxdown is not None):
						plt.fill_between(obswvl[mask], synthfluxup[mask], synthfluxdown[mask], facecolor='red', edgecolor='red', alpha=0.75, linewidth=0.5, label='Synthetic', zorder=2)
					else:
						plt.plot(obswvl[mask], synthflux[mask], color='r', alpha=0.5, linestyle='-', linewidth=2, label='Synthetic', zorder=100)

					# Plot synthetic spectrum with basically no [Mn/Fe]
					if synthflux_nomn is not None:
						plt.plot(obswvl[mask], synthflux_nomn[mask], 'b-', label='[Mn/H] = -10.0', zorder=2)

					# Plot synthetic spectrum with mean [Mn/Fe] of cluster
					if synthflux_cluster is not None:
						plt.plot(obswvl[mask], synthflux_cluster[1][mask], color='purple', linestyle='--', linewidth=2, label='<[Mn/H]>='+str(synthflux_cluster[0]), zorder=2)

					# Plot observed spectrum
					#if hires == False:
					plt.errorbar(obswvl[mask], obsflux[mask], yerr=yerr, color='k', fmt='o', markersize=6, label='Observed', zorder=3)
					#else:
					#plt.plot(obswvl[mask], obsflux[mask], 'k-', label='Observed')

					#plt.xticks([linelist[i]], fontsize=18)
					plt.yticks(fontsize=10)

					plt.xlim((lolim, uplim))
					plt.ylim((0.75, 1.10))

					if i==0:
						leg = plt.legend(fancybox=True, framealpha=0.5, loc='best')
						for text in leg.get_texts():
							plt.setp(text, color='#594F4F', fontsize=18)

				# Compute reduced chi-sq
				if savechisq is not None:
					current_chisq = np.sum(np.power(obsflux[mask] - synthflux[mask], 2.) * ivar[mask]) / (len(obsflux[mask]) - 1.)
					current_chisq_up = np.sum(np.power(obsflux[mask] - synthfluxup[mask], 2.) * ivar[mask]) / (len(obsflux[mask]) - 1.)
					current_chisq_down = np.sum(np.power(obsflux[mask] - synthfluxdown[mask], 2.) * ivar[mask]) / (len(obsflux[mask]) - 1.)

					chisq[i] = current_chisq
					chisq_up[i] = current_chisq_up
					chisq_down[i] = current_chisq_down

				if resids:
					# Only plot residuals if synth spectrum has been smoothed to match obswvl
					plt.figure(2)
					plt.subplot(nrows,ncols,i+1)
					plt.axvspan(linelist[i] - linewidth[i], linelist[i] + linewidth[i], color='green', alpha=0.25)
					plt.errorbar(obswvl[mask], obsflux[mask] - synthflux[mask], yerr=yerr, color='k', fmt='o', label='Residuals')
					plt.axhline(0, color='r', linestyle='solid', label='Zero')

				if ivar is not None:
					# Plot ivar
					plt.figure(3)
					plt.subplot(nrows,ncols,i+1)
					plt.axvspan(linelist[i] - linewidth[i], linelist[i] + linewidth[i], color='green', alpha=0.25)
					plt.errorbar(obswvl[mask], ivar[mask], color='k', linestyle='-')
					#plt.axhline(0, color='r', linestyle='solid', label='Zero')

		except:
			#ax.set_visible(False)
			continue

	# Legend for plot showing fits
	fig = plt.figure(1)
	#fig.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center', color='#594F4F')
	#fig.text(0.06, 0.5, 'Relative flux', fontsize=18, ha='center', va='center', rotation='vertical', color='#594F4F')
	#plt.ylabel('Relative flux')
	#plt.xlabel('Wavelength (A)')

	plt.savefig(outputname+'/'+specname+'finalfits.png',bbox_inches='tight') #,transparent=True)
	plt.close(1)

	if resids:
		fig2 = plt.figure(2)
		fig2.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center')
		fig2.text(0.06, 0.5, 'Residuals', fontsize=18, ha='center', va='center', rotation='vertical')
		plt.legend(loc='best')
		plt.savefig(outputname+'/'+specname+'resids.png',bbox_inches='tight')
		plt.close(2)

	if ivar is not None:
		fig3 = plt.figure(3)
		fig3.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center')
		fig3.text(0.06, 0.5, 'Inverse variance', fontsize=18, ha='center', va='center', rotation='vertical')
		#plt.legend(loc='best')
		plt.savefig(outputname+'/'+specname+'ivar.png',bbox_inches='tight')
		plt.close(3)

	# Save the reduced chi-sq values!
	if savechisq is not None:
		with open(savechisq, 'a') as f:
			for i in range(len(linelist)):
				f.write(specname[:-1]+'\t'+str(i)+'\t'+str(chisq[i])+'\t'+str(chisq_up[i])+'\t'+str(chisq_down[i])+'\n')

	return