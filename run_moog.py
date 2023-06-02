# run_moog.py
# Runs MOOG 8 times, one for each Mn linelist
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
import subprocess
import pandas
import tempfile
import shutil
	
def createPar(name, atmfile='', linelist='', directory=''):
	"""Create *.par file using *.atm file and linelist."""

	# Open linelist and get wavelength range to synthesize spectrum
	wavelengths = np.genfromtxt(linelist, skip_header=1, usecols=0)
	wavelengthrange = [ math.floor(wavelengths[0]),math.ceil(wavelengths[-1]) ]

	# Define filename
	filestr = directory + name + '.par' #took out slash between directory and name
	#print('filestr:', filestr)

	# Check if file already exists
	exists, readytowrite = checkFile(filestr)
	if readytowrite:

		# Outfile names:
		out1 = '\''+directory+name+'.out1\'' #took out slash between directory and name
		out2 = '\''+directory+name+'.out2\'' #took out slash between directory and name
		#print('out2:',out2)

		# If file exists, open file
		with open(filestr, 'w+') as file:

			# Print lines of .par file
			file.write('synth'+'\n')
			file.write('terminal       '+'\'x11\''+'\n')
			file.write('standard_out   '+out1+'\n')
			file.write('summary_out    '+out2+'\n')
			file.write('model_in       '+'\''+atmfile+'\''+'\n')
			file.write('lines_in       '+'\''+linelist+'\''+'\n')
			file.write('strong        0'+'\n')
			file.write('atmosphere    1'+'\n')
			file.write('molecules     1'+'\n')
			file.write('damping       1'+'\n')
			file.write('trudamp       0'+'\n')
			file.write('lines         1'+'\n')
			file.write('flux/int      0'+'\n')
			file.write('plot          0'+'\n')
			file.write('synlimits'+'\n')
			file.write('  '+'{0:.3f}'.format(wavelengthrange[0])+' '+'{0:.3f}'.format(wavelengthrange[1])+'  0.02  1.00'+'\n')
			file.write('obspectrum    0')

	return filestr, wavelengthrange

def runMoog(temp, logg, fe, alpha, directory='/mnt/c/Research/SMAUG/moogspectra/', elements=None, abunds=None, solar=None, lines='new'):
	"""Run MOOG for each desired element linelist and splice spectra.

	Inputs:
	temp 	 -- effective temperature (K)
	logg 	 -- surface gravity
	fe 		 -- [Fe/H]
	alpha 	 -- [alpha/Fe]

	Keywords:
	dir 	 -- directory to write MOOG output to [default = '/mnt/c/Research/SMAUG/moogspectra/']
	elements -- list of atomic numbers of elements to add to the *.atm file
	abunds 	 -- list of elemental abundances corresponding to list of elements
	lines    -- if 'new', use new revised linelist; else, use original linelist from Judy's code

	Outputs:
	spectrum -- spliced synthetic spectrum
	"""

	# Define temporary directory to store tempfiles
	tempdir = tempfile.mkdtemp() + '/'
	#print(tempdir)
	#tempdir = '/mnt/c/Research/SMAUG/temp/' #making a temp folder so I can maybe see what's going on

	# Define list of linelists
	if lines == 'new':
		linelists = np.array(['Mn47394783_new','Mn4823','Mn54075420_new','Mn55165537','Mn60136021','Mn6384','Mn6491'])
	elif lines == 'old':
		linelists = np.array(['linelist_Mn47544762','linelist_Mn4783','linelist_Mn4823','linelist_Mn5394','linelist_Mn5537','linelist_Mn60136021']) 
	
	spectrum  = []

	# Create identifying filename (including all parameters + linelist used)
	name = getAtm(temp, logg, fe, alpha, directory='') # Add all parameters to name
	#print('name:',name)

	# Add the new elements to filename, if any
	if elements is not None:

		for i in range(len(elements)):

			abund = int(abunds[i]*10)
			if elements[i] == 25:
				elementname = 'mn'

			# Note different sign conventions for abundances
			if abund < 0:
				elementstr 	= elementname + '{:03}'.format(abund)
			else:
				elementstr	= elementname + '_' + '{:02}'.format(abund)

			name = name + elementstr

	# Create *.atm file (for use with each linelist)
	atmfile = writeAtm(temp, logg, fe, alpha, elements=elements, abunds=abunds, solar=solar, dir=tempdir)
	#print('atmfile in run_moog:', atmfile)
	# Loop over all linelists
	for i in range(len(linelists)):

		# Create *.par file
		parname = name + '_' + linelists[i][9:]
		#print('parname:', parname)
		parfile, wavelengthrange = createPar(parname, atmfile, '/mnt/c/Research/SMAUG/Mnlinelists/finallinelists/'+linelists[i], directory=tempdir)
		#print('parfile:', parfile)


		# Run MOOG
		p = subprocess.Popen(['./MOOG', parfile], cwd='/mnt/c/Research/moog17scat/', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		#print('MOOG is running')


		# Wait for MOOG to finish running
		p.communicate()

		# Create arrays of wavelengths and fluxes
		outfile = tempdir+'/'+parname+'.out2'

		wavelength = np.linspace(wavelengthrange[0],wavelengthrange[1],math.ceil((wavelengthrange[1]-wavelengthrange[0])/0.02), endpoint=True)
		data = pandas.read_csv(outfile, skiprows=[0,1,-1], delimiter=' ').to_numpy()
		flux = data[~np.isnan(data)][:-1]

		spectrum.append([1.-flux, wavelength])

	# Output synthetic spectrum in a format that continuum_div functions will understand (list of arrays)

	# Clean out the temporary directory
	shutil.rmtree(tempdir)

	return spectrum