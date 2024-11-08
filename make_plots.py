# make_plots.py
# Make plots for other codes
# 
# Created 9 Nov 18
# Updated 9 Nov 18
#
# Edited to make general 6/9/2023 -LEH
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
def make_plots(lines, linelist, linegaps, specname, starnum, obswvl, obsflux, synthflux, outputname, element, skip, resids=True, \
			   ivar=None, title=None, synthfluxup=None, synthfluxdown=None, synthfluxup02=None, synthfluxdown02=None, synthflux02_above=False, \
				synthflux_no_elem=None, synthflux_cluster=None, additional_synth=None, add_synth_val=None, savechisq=None, hires=False):
	"""Make plots.

	Inputs:
	lines -- which linelist to use? Options: 'new', 'old'
	linelist -- list of the synthesized lines of the element of interest
	specname -- name of star
	obswvl 	-- observed wavelength array
	obsflux -- observed flux
	synthflux 	-- synthetic flux
	outputname 	-- where to output file
	element -- string of element of interest e.g. 'Mn', 'Sr'

	Keywords:
	resids  -- plot residuals if 'True' (default); else, don't plot residuals
	ivar 	-- inverse variance; if 'None' (default), don't plot errorbars
	title 	-- plot title; if 'None' (default), then plot title = "Star + ID"
	synthfluxup & synthfluxdown -- if not 'None' (default), then plot synthetic spectrum as region between [X/H]_best +/- 0.3dex
	synthflux_no_elem	-- if not 'None' (default), then plot synthetic spectrum with [X/H] = -10.0
	synthflux_cluster 	-- if not 'None' (default), then plot synthetic spectrum with mean [X/H] of cluster; in format synthflux_cluster = [mean [X/H], spectrum]
	savechisq 	-- if not 'None' (default), compute & save reduced chi-sq for each line in output file with path savechisq
	hires 	-- if 'False', plot as normal; else, zoom in a bit to show hi-res spectra

	Outputs:
	"""
	print('making plots')

	# Define lines to plot
	newlinelist = []
	for j in linelist:
		for index in skip:
			if j > linegaps[index][0] and j < linegaps[index][1]:
				newlinelist.append(j)
		#newlinelist.append(linelist[j])
	if lines == 'new':
		linewidth = np.ones(len(newlinelist)) #sets width of green overlay
		if element == 'Nd':
			nrows = 4
			ncols = int(len(newlinelist)/4)+1
			figsize = (60,60)
		else:
			nrows = 2
			ncols = int(len(newlinelist)/2) + 1
			figsize = (32,15)
			# USE ABOVE USUALLY
			# nrows = 2
			# ncols = 1
			#figsize = (40,15)
			# figsize = (20,15)

	elif lines == 'old':
		linelist = np.array([4739.,4783.,4823.,5394.,5432.,5516.,5537.,6013.,6021.,6384.,6491.])
		linewidth = np.ones(len(linelist))

		nrows = 3
		ncols = 4
		figsize = (20,15)

	# Define title
	if title is None:
		title = 'Star'+specname+element

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
		chisq = np.zeros(len(newlinelist))
		chisq_up = np.zeros(len(newlinelist))
		chisq_down = np.zeros(len(newlinelist))

	#print('length synth, obswvl',len(synthfluxup),len(obswvl))
	for i in range(len(newlinelist)):
		#f = plt.figure(1)
		#for i, ax in enumerate(f.axes):

		# Range over which to plot
		#if hires == False:
		lolim = newlinelist[i] - 10
		uplim = newlinelist[i] + 10
		#print(lolim, uplim)
		#else:
		#	lolim = linelist[i] - 5
		#	uplim = linelist[i] + 5

		# Make mask for wavelength
		try:
			mask = np.where((obswvl > lolim) & (obswvl < uplim))
			# for testing
			# plt.figure()
			# plt.plot(obswvl[mask], synthfluxup[mask], 'k-', label='im trying')
			# plt.plot(obswvl[mask], synthfluxdown[mask], 'r-', label='down')
			# plt.legend(loc='best')
			# plt.savefig(outputname+'/'+specname+'_TEST'+str(i)+'.png')
			# plt.close()

			if len(mask[0]) > 0:

				if ivar is not None:
					yerr=np.power(ivar[mask],-0.5)
					yerr = np.float64(yerr)
					#print(type(yerr[0]))
				else:
					yerr=None

				# Plot fits
				with plt.rc_context({'axes.linewidth':4, 'axes.edgecolor':'#594F4F', 'xtick.color':'#594F4F', 'ytick.color':'#594F4F','xtick.major.width':2,\
						 'ytick.major.width':2,'xtick.minor.width':1,'ytick.minor.width':1,'xtick.major.size':6,'ytick.major.size':6,'xtick.direction':'in','ytick.direction':'in'}):
					plt.figure(1)

					if i==0:
						ax = plt.subplot(nrows,ncols,i+1)
					else:
						plt.subplot(nrows,ncols,i+1) #,sharey=ax)

					plt.axvspan(newlinelist[i] - linewidth[i], newlinelist[i] + linewidth[i], color='green', zorder=1, alpha=0.25)

					# Plot synthetic spectrum with basically no [X/Fe]
					if synthflux_no_elem is not None:
						plt.plot(obswvl[mask], synthflux_no_elem[mask], 'b-', label='['+element+'/H] = -8.0', zorder=2)

					# Plot synthetic spectrum best fit
					if (synthfluxup is not None) and (synthfluxdown is not None):
						# print('type of x and y')
						# print(type(obswvl[mask][0]))
						# print(type(synthfluxup[mask][0]))
						# print(type(synthfluxdown[mask][0]))
						obswvl = np.float64(obswvl) #so that wvl is the same type as the synth flux

						plt.fill_between(obswvl[mask], synthfluxup[mask], synthfluxdown[mask], facecolor='red', edgecolor='red', alpha=0.75, linewidth=0.5,\
		                                   label='Synthetic', zorder=3)
					else:
						plt.plot(obswvl[mask], synthflux[mask], color='r', alpha=0.5, linestyle='-', linewidth=2, label='Synthetic', zorder=100)

					# Plot +/-0.2 dex region
					if (synthfluxup02 is not None) and (synthfluxdown02 is not None):
						if synthflux02_above:
							zorder = 100
						else:
							zorder = 2
						obswvl = np.float64(obswvl)
						plt.fill_between(obswvl[mask], synthfluxup02[mask], synthfluxdown02[mask], facecolor='pink', edgecolor='purple', alpha=0.50, linewidth=0.5,\
		                                   label='Synthetic +/- 0.2', zorder=zorder)
					if additional_synth is not None:
						plt.plot(obswvl[mask], additional_synth[mask], 'g-', label='['+element+'/H] = '+str(add_synth_val), zorder=100)

					# Plot synthetic spectrum with mean [X/Fe] of cluster
					if synthflux_cluster is not None:
						plt.plot(obswvl[mask], synthflux_cluster[1][mask], color='purple', linestyle='--', linewidth=2, label='<['+element+'/H]>='+str(synthflux_cluster[0]), zorder=2)

					# Plot observed spectrum
					#if hires == False:
					#print('obsflux',obsflux[mask])
					plt.errorbar(obswvl[mask], obsflux[mask], yerr=yerr, color='k', fmt='o', markersize=6, label='Observed', zorder=3)
					#else:
					#plt.plot(obswvl[mask], obsflux[mask], 'k-', label='Observed')

					# plt.xticks(fontsize=12)
					# plt.yticks(fontsize=15)
					plt.xticks(fontsize=20)
					plt.yticks(fontsize=20)

					plt.xlim((lolim, uplim))
					# plt.ylim((0.50, 1.15))
					plt.ylim((0, 1.15))

					if i==0:
						leg = plt.legend(fancybox=True, framealpha=0.5, fontsize=18, loc='best') 
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
					plt.axvspan(newlinelist[i] - linewidth[i], newlinelist[i] + linewidth[i], color='green', alpha=0.25)
					plt.errorbar(obswvl[mask], obsflux[mask] - synthflux[mask], yerr=yerr, color='k', fmt='o', label='Residuals')
					plt.axhline(0, color='r', linestyle='solid', label='Zero')

				if ivar is not None:
					# Plot ivar
					plt.figure(3)
					plt.subplot(nrows,ncols,i+1)
					plt.axvspan(newlinelist[i] - linewidth[i], newlinelist[i] + linewidth[i], color='green', alpha=0.25)
					plt.errorbar(obswvl[mask], ivar[mask], color='k', linestyle='-')
					#plt.axhline(0, color='r', linestyle='solid', label='Zero')

		except:
			#ax.set_visible(False)
			continue

	# Legend for plot showing fits
	fig = plt.figure(1)
	fig.text(0.5, 0.04, 'Wavelength ('+r'$\AA$'+')', fontsize=28, ha='center', va='center', color='#594F4F') #fontsize=18
	fig.text(0.06, 0.5, 'Normalized flux', fontsize=28, ha='center', va='center', rotation='vertical', color='#594F4F') #fontsize=18
	#plt.ylabel('Relative flux')
	#plt.xlabel('Wavelength (A)')

	plt.savefig(outputname+'/'+specname+element+'finalfits'+str(starnum)+'.png',bbox_inches='tight') #,transparent=True)
	plt.close(1)

	if resids:
		fig2 = plt.figure(2)
		fig2.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center')
		fig2.text(0.06, 0.5, 'Residuals', fontsize=18, ha='center', va='center', rotation='vertical')
		plt.legend(loc='best')
		plt.savefig(outputname+'/'+specname+element+'resids'+str(starnum)+'.png',bbox_inches='tight')
		plt.close(2)

	if ivar is not None:
		fig3 = plt.figure(3)
		fig3.text(0.5, 0.04, 'Wavelength (A)', fontsize=18, ha='center', va='center')
		fig3.text(0.06, 0.5, 'Inverse variance', fontsize=18, ha='center', va='center', rotation='vertical')
		#plt.legend(loc='best')
		plt.savefig(outputname+'/'+specname+element+'ivar'+str(starnum)+'.png',bbox_inches='tight')
		plt.close(3)

	# Save the reduced chi-sq values!
	if savechisq is not None:
		with open(savechisq, 'a') as f:
			for i in range(len(linelist)):
				f.write(specname[:-1]+'\t'+str(i)+'\t'+str(chisq[i])+'\t'+str(chisq_up[i])+'\t'+str(chisq_down[i])+'\n')

	

	return