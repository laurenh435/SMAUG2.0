# run_moog.py
# Runs MOOG
# Outputs spliced synthetic spectrum
#
# - createPar: for a given *.atm file and linelist, output *.par file
# - runMoog: runs MOOG for each Mn linelist (calls createPar), splices output spectrum
#
# Created 4 Jan 18 by M. de los Reyes
# making SMAUG general -LEH 5/31/2023
###################################################################

import os
import sys
import glob
import numpy as np
import math
from interp_atmosphere import checkFile, getAtm, writeAtm
from isotopes import isotope_ratio
import subprocess
import pandas
import tempfile
import shutil
	
def createPar(name, atom_nums, logg, specname, atmfile='', linelist='', directory='', stravinsky=False):
	"""Create *.par file using *.atm file and linelist."""

	# Open linelist and get wavelength range to synthesize spectrum
	wavelengths = np.genfromtxt(linelist, skip_header=1, usecols=0)
	wavelengthrange = [ math.floor(wavelengths[0]),math.ceil(wavelengths[-1]) ]
	#print('wavelength range:',wavelengthrange)

	# Define filename
	filestr = directory + name + '.par'

	# Check if file already exists
	exists, readytowrite = checkFile(filestr)
	if readytowrite:

		# Outfile names:
		# if stravinsky:
		# 	out1 = '\'/raid/lhender6/temp/'+name+'.out1\''
		# 	out2 = '\'/raid/lhender6/temp/'+name+'.out2\''
		# else:
		# out1 = '\'temp'+specname+'/'+name+'.out1\'' THESE WORK WITH CWD ENDING IN SMAUG/
		# out2 = '\'temp'+specname+'/'+name+'.out2\''
		out1 = '\''+name+'.out1\''
		out2 = '\''+name+'.out2\''

		# If file exists, open file
		with open(filestr, 'w+') as file:

			# Print lines of .par file
			file.write('synth'+'\n')
			file.write('terminal       '+'\'x11\''+'\n')
			file.write('standard_out   '+out1+'\n')
			file.write('summary_out    '+out2+'\n')
			file.write('model_in       '+'\''+atmfile+'\''+'\n')
			file.write('lines_in       '+'\''+'../'+linelist+'\''+'\n')
			#file.write('lines_in       '+'\'/raid/lhender6/lines/Sr4215.txt\''+'\n')
			file.write('strong        1'+'\n') #changed this to 1 so that MOOG will take a strong line list
			file.write('stronglines_in '+'\'../full_linelists/blue.strong\''+'\n')
			file.write('atmosphere    1'+'\n')
			file.write('molecules     1'+'\n')
			if atom_nums[0] == 12:
				file.write('damping       0'+'\n') #for some reason Mg triplet needs damping=0 for accurate abundances
			else:
				file.write('damping       1'+'\n')
			file.write('trudamp       0'+'\n')
			file.write('lines         1'+'\n')
			file.write('flux/int      0'+'\n')
			file.write('plot          0'+'\n')
			file.write('synlimits'+'\n')
			file.write('  '+'{0:.3f}'.format(wavelengthrange[0])+' '+'{0:.3f}'.format(wavelengthrange[1])+'  0.02  1.00'+'\n')
			# Calculate C12 C13 ratio to get isotope abundances for CH and CN - calculation from Kirby+ 2015
			if logg <= 2.0:
				c12c13 = 6.0
			elif logg > 2.7:
				c12c13 = 50.0
			else:
				try:
					c12c13 = 63.0 * logg[0] - 120.0
				except IndexError:
					c12c13 = 63.0 * logg - 120.0 
			c12reciprocal = np.round((c12c13 + 1)/c12c13, decimals=2)
			c13reciprocal = np.round(1/(1-(1/c12reciprocal)), decimals=2)
			# Mg isotope ratios are taken to be solar from Asplund+ 2009 for MgH
			mg24reciprocal = 1.27
			mg25reciprocal = 10.00
			mg26reciprocal = 9.08
			# Get ratios of isotopes of element of interest
			# 2nd entry to isotope_ratio is fraction of the element created by s-process (for running with line lists based on full_lines_sprocess):
			Ba_isotope_reciprocals, Ba_isotopes = isotope_ratio(56,stravinsky=stravinsky)
			Nd_isotope_reciprocals, Nd_isotopes = isotope_ratio(60,stravinsky=stravinsky)
			Eu_isotope_reciprocals, Eu_isotopes = isotope_ratio(63,stravinsky=stravinsky)
			all_reciprocals = Ba_isotope_reciprocals + Nd_isotope_reciprocals + Eu_isotope_reciprocals
			all_reciprocals.append(c12reciprocal)
			all_reciprocals.append(c13reciprocal)
			all_reciprocals.append(c12reciprocal)
			all_reciprocals.append(c13reciprocal)
			all_reciprocals.append(mg24reciprocal)
			all_reciprocals.append(mg25reciprocal)
			all_reciprocals.append(mg26reciprocal)
			all_isotopes = Ba_isotopes + Nd_isotopes + Eu_isotopes + ['106.00112','106.00113','607.01214','607.01314','112.00124','112.00125','112.00126']
			file.write('isotopes     '+str(len(all_isotopes))+'         1'+'\n')
			for i in range(len(all_isotopes)):
				file.write(' '+str(all_isotopes[i])+'      '+str(all_reciprocals[i])+'\n') 
			file.write('obspectrum    0')
	
	return filestr, wavelengthrange, len(all_isotopes)

def runMoog(temp, logg, fe, alpha, carbon, specname, slitmask, linelists, skip, directory='/mnt/c/Research/SMAUG/output/', atom_nums=None, elements=None, abunds=None, stravinsky=False):
	"""Run MOOG for each desired element linelist and splice spectra.

	Inputs:
	temp 	 -- effective temperature (K)
	logg 	 -- surface gravity
	fe 		 -- [Fe/H]
	alpha 	 -- [alpha/Fe]
	carbon   -- [C/Fe]
	specname -- star index
	slitmask 
	linelists
	skip     -- indices of lines in linegaps that we are running

	Keywords:
	directory -- directory to write MOOG output to [default = '/mnt/c/Research/SMAUG/output/']
	atom_nums -- list of atomic numbers of elements to add to the list of atoms
    elements  -- list of element symbols you want added to the list of atoms e.g. 'Mn', 'Sr'
	abunds 	  -- list of elemental abundances corresponding to list of elements
	stravinsky -- whether running on stravinsky or not

	Outputs:
	spectrum -- spliced synthetic spectrum
	"""

	# Define temporary directory to store tempfiles
	#tempdir = tempfile.mkdtemp() + '/'
	#tempdir = '/home/lhender6/Research/SMAUG/temp/' # to watch what's going on 
	if stravinsky:
		tempdir = '/home/lhender6/Research/SMAUG/temp'+specname+slitmask+'/'
	else:
		tempdir = '/mnt/c/Research/SMAUG/temp'+specname+slitmask+'/'
	if not os.path.exists(tempdir):
		os.makedirs(tempdir)

	spectrum  = []

	# Create identifying filename (including all parameters + linelist used)
	name, shortfile = getAtm(temp, logg, fe, alpha, directory='') # Add all parameters to name

	# Add the new elements to filename, if any
	if atom_nums is not None:

		for i in range(len(atom_nums)):

			abund = int(abunds[i]*10)
			elementname = elements[i]

			# Note different sign conventions for abundances
			if abund < 0:
				elementstr 	= elementname + '{:03}'.format(abund)
			else:
				elementstr	= elementname + '_' + '{:02}'.format(abund)

			name = name + elementstr

	# Create *.atm file (for use with each linelist)
	# Solar abundances from Asplund et al. 2009
	all_solar = [12.00,10.93, 1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93,\
	  6.24, 7.60, 6.45, 7.51, 5.41, 7.12, 5.50, 6.40, 5.03, 6.34, 3.15, 4.95, \
		3.93, 5.64, 5.43, 7.50, 4.99, 6.22, 4.19, 4.56, 3.04, 3.65, 2.30, \
			3.34, 2.54, 3.25, 2.52, 2.87, 2.21, 2.58, 1.46, 1.88,-5.00, 1.75,\
				0.91, 1.57, 0.94, 1.71, 0.80, 2.04, 1.01, 2.18, 1.55, 2.24, 1.08,\
					2.18, 1.10, 1.58, 0.72, 1.42, -5.00, 0.96, 0.52, 1.07, 0.30,\
						1.10, 0.48, 0.92, 0.10, 0.84, 0.10, 0.85,-0.12, 0.85, 0.26,\
							1.40, 1.38, 1.62, 0.92, 1.17, 0.90, 1.75, 0.65,-5.00,\
								-5.00,-5.00,-5.00,-5.00,-5.00, 0.02, -5.00,-0.54,-5.00,-5.00,-5.00] 
	if atom_nums is not None:
		solar = []
		for atom in atom_nums:
			solar.append(all_solar[atom-1])
	else:
		solar = None
	fullatmfile, atmfile = writeAtm(temp, logg, fe, alpha, carbon, atom_nums=atom_nums, elements=elements, abunds=abunds, solar=solar, dir=tempdir, stravinsky=stravinsky)
	# if stravinsky: WORKS FOR CWD ENDING IN SMAUG/
	# 	atmfile = 'temp'+specname+'/'+ atmfile #same length as just writing /raid/... or /home/...
	# 	# atmfile = 'temp/'+ atmfile 
	# else:
	# 	atmfile = 'temp'+specname+'/'+ atmfile
		# atmfile = 'temp/'+ atmfile
	
	#atmfile = '/raid/lhender6/temp/'+ atmfile
	#print('atmfile in run_moog:', atmfile)
	# Loop over all linelists
	for i in skip:

		# Create *.par file
		parname = name + '_' + linelists[i][-8:-4]
		parfile, wavelengthrange, nisotopes = createPar(parname, atom_nums, logg, specname, atmfile, linelists[i], directory=tempdir, stravinsky=stravinsky)
		#print('parfile:', parfile)
		#parfile = '/mnt/c/Research/SMAUG/temp/myparfile.par'
		
		# Run MOOG
		if stravinsky:
			p = subprocess.Popen(['/raid/moog/moog17scat/MOOG', parfile], cwd='/home/lhender6/Research/SMAUG/temp'+specname+slitmask+'/', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		else:
			p = subprocess.Popen(['/mnt/c/Research/moog17scat/MOOG', parfile], cwd='/mnt/c/Research/SMAUG/temp'+specname+slitmask+'/', stdout=subprocess.PIPE, stderr=subprocess.PIPE)


		# Wait for MOOG to finish running
		p.communicate()

		# Create arrays of wavelengths and fluxes
		outfile = tempdir+parname+'.out2'
		wavelength = np.linspace(wavelengthrange[0],wavelengthrange[1],math.ceil((wavelengthrange[1]-wavelengthrange[0])/0.02), endpoint=True)
		skiprows = np.arange(nisotopes+2)
		skiprows = np.append(skiprows,-1)
		data = pandas.read_csv(outfile,skiprows=skiprows, delimiter=' ').to_numpy() #skiprows=[0,1,-1], delimiter=' ').to_numpy()
		flux = data[~np.isnan(data)][:-1]

		spectrum.append([1.-flux, wavelength])

	# Output synthetic spectrum in a format that continuum_div functions will understand (list of arrays)

	# Clean out the temporary directory
	shutil.rmtree(tempdir)

	return spectrum