# interp_atmosphere.py
# Given atmospheric params and [Mn/H], creates *.atm file
# 
# Contains the following functions: 
# - checkFile: check if input file exists and is non-empty
# - round_to: round either up/down to a given precision
# - find_nearest: find nearest temperatures above/below an 
#						input temp, given uneven temp array
# - getAtm: get path to a *.atm file
# - readAtm: read in a *.atm file and put contents into a list
# - interpolateAtm: for given parameters (T, logg, Fe/H, alpha/Fe), 
#				interpolate from ATLAS9 grid7 and output atmosphere
# - writeAtm: given atmosphere, put contents into a *.atm file
# 
# Created 2 Nov 17 by M. de los Reyes
# edited to make general 6/5/2023 -LEH
###################################################################

import os
import numpy as np
import math
import gzip
import subprocess

def checkFile(filestr, overridecheck=True):
	"""Check if file exists and is non-empty.

	Inputs:
	filestr -- filename

	Keyword arguments:
    overridecheck -- 'True' if file is writeable and I want to make sure I don't accidentally overwrite it

    Outputs:
    exists -- 'True' if file exists already, 'False' if not
    readytowrite -- 'True' if writeable file is safe to overwrite, 'False' if not
    """

	# Check if file exists and has contents
	if os.path.isfile(filestr) and os.path.getsize(filestr) > 0:

		# If this is for a writeable file, make sure I don't accidentally overwrite something
		if overridecheck:
			#override = input('Warning: File '+filestr+' already exists! Do you want to overwrite? (y/n)\n')
			#if override == 'y' or override == 'Y' or override == 'yes':
			#	exists = True
			#	readytowrite = True
			#else:
			#	exists = True
			#	readytowrite = False
			#print('Overwriting file '+filestr+'!')
			exists = True
			readytowrite = True

		# Otherwise, don't need to worry about overwriting
		else:
			exists = True
			readytowrite = True

	# If file doesn't exist or has no contents, don't have to worry about overwriting
	else:
		exists = False
		readytowrite = True
		#print('File '+filestr+' doesn\'t exist... yet?')

	return exists, readytowrite

def round_to(n, precision, updown):
	"""Round number n (either up or down) to given precision."""
	precision = float(precision)

	if updown == 'up':
		roundn = (math.ceil(n / precision) * precision)
	else:
		roundn = (math.floor(n / precision) * precision)
    
	return roundn

def find_nearest(temp, array=None):
	""" Given a value ("temp"), find nearest value in grid.
		If array == None, use normal temperature array and find nearest temp (both up and down) in grid; 
		else, use given array and just return nearest value.
	"""

	# Array of temperatures in grid
	if array is not None:
		tarray = array

		# Find what grid value is closest to input value
		idx = (np.abs(tarray-temp)).argmin()
		neartemp 	= tarray[idx]

	else:
		tarray = np.array([3500, 3600, 3700, 3800, 3900,
					4000, 4100, 4200, 4300, 4400,
					4500, 4600, 4700, 4800, 4900,
					5000, 5100, 5200, 5300, 5400,
					5500, 5600, 5800, 6000, 6200,
					6400, 6600, 6800, 7000, 7200,
					7400, 7600, 7800, 8000])

		# Find what grid temp is closest to the input temp
		idx = (np.abs(tarray-temp)).argmin()
		neartemp 	= tarray[idx]

	# Define output variables
	uptemp 		= 0
	downtemp	= 0
	error 		= False

	# If the closest grid temp is **lower** than input temp
	if neartemp < temp:

		# Check to make sure the input temp isn't outside the grid
		if idx == len(tarray) - 1:
			error = True

		# If ok, then can find the closest grid temp that's **higher** than input temp
		else:
			downtemp = neartemp
			uptemp 	 = tarray[idx+1]

	# If the closest grid temp is **higher** than input temp
	elif neartemp > temp:

		# Check to make sure the input temp isn't outside the grid
		if idx == 0:
			error = True 

		# If ok, then can find the closest grid temp that's **lower** than input temp
		else:
			downtemp = tarray[idx-1]
			uptemp	 = neartemp

	# Check if input temp is equal to one of the grid temps
	else:
		uptemp	 = neartemp
		downtemp = neartemp 

	# Return temperatures and error message if input temp is outside grid range
	return uptemp, downtemp, error

def getAtm(temp, logg, fe, alpha, directory):
	"""Get path of grid file."""

	temp  = temp
	logg  = int(logg*10)
	fe    = int(fe*10)
	alpha = int(alpha*10)

	filebase	= 't' + str(temp) + 'g_' + '{:02}'.format(logg)

	# Note different sign conventions for [Fe/H], [alpha/Fe]
	if fe < 0:
		fepart 	= 'f' + '{:03}'.format(fe)
	else:
		fepart	= 'f_' + '{:02}'.format(fe)

	if alpha < 0:
		alphapart 	= 'a' + '{:03}'.format(alpha)
	else:
		alphapart 	= 'a_' + '{:02}'.format(alpha)

	#if directory[-1] != '/':
		#directory = directory+'/'

	filestr	= directory + filebase + fepart + alphapart
	shortfile = filebase + fepart + alphapart

	return filestr, shortfile

def readAtm(temp, logg, fe, alpha, carbon=None, gridch=False, inputdir='/mnt/c/Research/SMAUG/grid7/atmospheres/', stravinsky=False):
	"""Read file from grid.

	Inputs:
    temp -- effective temperature (K)
    logg -- surface gravity
    fe -- [Fe/H]
    alpha -- [alpha/Fe]

    Keywords:
    carbon -- [C/Fe]
    gridch -- whether gridch is being used - so that C abundance must be specified
    inputdir -- input directory
    	if '/raid/grid7/atmospheres/' (default): return '.atm' file
    	if 'raid/gridie/bin/': return '.bin.gz' file

	"""

	tempnew  = temp
	loggnew  = int(logg*10)
	fenew    = int(fe*10)
	alphanew = int(alpha*10)

	# Directory to read atmospheres from
	directory	= inputdir + 't' + str(tempnew) + '/g_' + '{:02}'.format(loggnew) + '/' 
	#print('readAtm directory', directory)

	# Atmosphere to read
	if stravinsky:
		if inputdir == '/raid/grid7/atmospheres/':
			filestr, shortfile = getAtm(temp, logg, fe, alpha, directory)
			filestr = filestr + '.atm'
			contents = np.genfromtxt(filestr, skip_header=3, max_rows=72, usecols=None, autostrip=True)
		elif inputdir == '/raid/gridch/bin/':
			# each gridch file has 24 spectra (see [C/Fe] range below) all in a list - so need to find the right one for the carbon abundance you want
			filestr, shortfile = getAtm(temp, logg, fe, alpha, directory)
			filestr = filestr + '.bin.gz'
			n = 2858 # for bin. for hired, n=20001
			acfe = np.array([-2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2, 0.4,  0.6,  0.8,  1.0,  1.4, 1.8,  2.2,  2.6,  3.0,  3.5])
			icfe = np.where(acfe == carbon)[0]
			if icfe.size != 1:
				raise ValueError('[C/Fe] out of range.')
			moogspec_ch = np.zeros(n, dtype=np.float32)
			with gzip.open(filestr, 'rb') as f:
				bstring = f.read()
				full_contents = np.frombuffer(bstring, dtype=np.float32) #this has all 24 spectra [C/Fe] = -2.4-3.5
				istart = n * icfe[0]
				iend = istart + n
				contents = full_contents[istart:iend]
				f.close()
		else:
			filestr, shortfile = getAtm(temp, logg, fe, alpha, directory)
			filestr = filestr + '.bin.gz' 
			with gzip.open(filestr, 'rb') as f:
				bstring = f.read()
				contents = np.fromstring(bstring, dtype=np.float32)
				f.close()

	else:
		if inputdir == '/mnt/c/SpectraGrids/grid7/atmospheres/':
			filestr, shortfile = getAtm(temp, logg, fe, alpha, directory)
			filestr = filestr + '.atm'
			contents = np.genfromtxt(filestr, skip_header=3, max_rows=72, usecols=None, autostrip=True)
		else: #if inputdir == '/mnt/c/Research/SMAUG/gridie/bin/':
			filestr, shortfile = getAtm(temp, logg, fe, alpha, directory)
			filestr = filestr + '.bin.gz' 
			with gzip.open(filestr, 'rb') as f:
				bstring = f.read()
				contents = np.fromstring(bstring, dtype=np.float32)
				f.close()

	return contents

def interpolateAtm(temp, logg, fe, alpha, carbon=None, hgrid=False, griddir='/mnt/c/SpectraGrids/grid7/atmospheres/', gridch=False, stravinsky=False):
	"""Interpolate atmosphere from grid of atmospheres.

    Inputs:
    temp 	-- effective temperature (K)
    logg 	-- surface gravity
    fe 	  	-- [Fe/H]
    alpha 	-- [alpha/Fe]

    Keywords:
    carbon  -- [C/Fe] for gridch
    hgrid 	-- if 'True', use hydrogen grids; else use normal grids for temp
    griddir -- where to get atmospheres from
    gridch  -- whether to include [C/Fe] in the interpolation
    """

	# Change input parameters to correct format for filenames
	tempnew  = temp
	loggnew  = logg*10
	fenew 	 = fe*10
	alphanew = alpha*10

	if hgrid == False:
		# Get nearest gridpoints for each parameter
		tempUp, tempDown, tempError = find_nearest(tempnew)

		loggUp = round_to(loggnew, 5, 'up')/10.
		loggDown = round_to(loggnew, 5, 'down')/10.

		feUp = round_to(fenew, 1, 'up')/10.
		feDown = round_to(fenew, 1, 'down')/10.


		alphaUp = round_to(alphanew, 1, 'up')/10.
		alphaDown = round_to(alphanew, 1, 'down')/10.

		if gridch:
			carbonnew = carbon*10
			carray = 10*np.array([-2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2, 0.4,  0.6,  0.8,  1.0,  1.4, 1.8,  2.2,  2.6,  3.0,  3.5])
			if carbon < -2.4:
				carbonUp = carbonDown = -2.4
			elif carbon > 3.5:
				carbonUp = carbonDown = 3.5
			else:
				carbonUp, carbonDown, carbonError = find_nearest(carbonnew,array=carray) #Use this because array is not regular the whole way through
				carbonUp = carbonUp/10
				carbonDown = carbonDown/10
			#print('carbon:', carbon, carbonUp, carbonDown)
			if carbonUp == 3.5 or carbonDown == -2.4:
				print('Synth carbon may be off since measured carbon is ', carbon, 'which is out of range')
				#raise ValueError('[C/Fe] = '+ str(carbon) + ' is out of range!')

		# print('Up and Down values in interpolateAtm')
		# print('temp:',temp, tempUp,tempDown)
		# print('logg:',logg, loggUp,loggDown)
		# print('alpha:', alpha, alphaUp,alphaDown)


		# Check that points are within range of grid
		if tempError:
			raise ValueError('T = ' + str(temp) + ' is out of range!')

		if loggUp > 5.0 or loggDown < 0:
			raise ValueError('log(g) = ' + str(logg) + ' is out of range!')

		elif feUp > 0 or feDown < -5.0:
			raise ValueError('[Fe/H] = ' + str(fe) + ' is out of range!')

		elif alphaUp > 1.2 or alphaDown < -0.8:
			raise ValueError('[alpha/Fe] = ' + str(alpha) + ' is out of range!')
	

		# Grid isn't uniform, so do additional checks to make sure points are within range of grid
		elif ((loggUp < 0.5) or (loggDown < 0.5)) and ((tempUp >= 7000) or (tempDown >= 7000)):
			raise ValueError('T = ' + str(temp) + ' and log(g) = ' + str(logg) + ' out of range!')

	else:
		# Get nearest gridpoints for each parameter
		tempUp = tempDown = find_nearest(tempnew, array=np.array([4500, 5000, 5500, 6000]))
		loggUp = loggDown = find_nearest(loggnew, array=np.array([0.5, 1.0, 1.5, 2.0, 2.5]))
		feUp = feDown = find_nearest(loggnew, array=np.array([-2.0, -1.0]))
		alphaUp = alphaDown = find_nearest(alphanew, array=np.array([0.0]))

	# If within grid, interpolate!

	# Calculate intervals for each variable
	# (quantities needed for interpolation)
	#######################################

	# Temperature interval
	## Check if input temp exactly matches one of the grid points
	if tempUp == tempDown:

		# If so, interpolation interval is just one point,
		# so interval is just one point, and delta(T) and n(T) are both 1
		tempInterval = np.array([temp])
		tempDelta 	 = np.array([1])
		nTemp 		 = 1

	## If not, then input temp is between two grid points
	else:
		tempInterval = np.array([tempDown, tempUp])
		tempDelta	 = np.absolute(np.array([tempUp, tempDown]) - temp)
		nTemp 		 = 2

	# Repeat for other variables:
	if loggUp == loggDown:
		loggInterval = np.array([logg])
		loggDelta 	 = np.array([1])
		nLogg		 = 1
	else:
		loggInterval = np.array([loggDown, loggUp])
		loggDelta	 = np.absolute(np.array([loggUp, loggDown]) - logg)
		nLogg 		 = 2

	if feUp == feDown:
		feInterval	= np.array([fe])
		feDelta 	= np.array([1])
		nFe		 	= 1
	else:
		feInterval	= np.array([feDown, feUp])
		feDelta		= np.absolute(np.array([feUp, feDown]) - fe)
		nFe 		= 2

	if alphaUp == alphaDown:
		alphaInterval	= np.array([alpha])
		alphaDelta 		= np.array([1])
		nAlpha	 		= 1
	else:
		alphaInterval	= np.array([alphaDown, alphaUp])
		alphaDelta		= np.absolute(np.array([alphaUp, alphaDown]) - alpha)
		nAlpha 			= 2
	
	if gridch:
		if carbonUp == carbonDown:
			carbonInterval	= np.array([carbonUp])
			carbonDelta 		= np.array([1])
			ncarbon	 		= 1
		else:
			carbonInterval	= np.array([carbonDown, carbonUp])
			carbonDelta		= np.absolute(np.array([carbonUp, carbonDown]) - carbon)
			ncarbon         = 2

	# Do interpolation!
	###################
	if gridch: #include [C/Fe] in interpolation
		for i in range(nTemp):
			for j in range(nLogg):
				for m in range(nFe):
					for n in range(nAlpha):
						for p in range(ncarbon):

							# Read in grid point (atmosphere file)
							iflux = readAtm(tempInterval[i],loggInterval[j],feInterval[m],alphaInterval[n], carbon=carbonInterval[p], gridch=True, inputdir=griddir, stravinsky=stravinsky) #[:,0]

							# Compute weighted sum of grid points
							## If first iteration, initialize flux as weighted value of lowest grid point 
							if (i==0) & (j==0) & (m==0) & (n==0) & (p==0):
								flux = tempDelta[i]*loggDelta[j]*feDelta[m]*alphaDelta[n]*carbonDelta[p] * iflux

							## Else, start adding weighted values of other grid points
							else:
								flux = flux + tempDelta[i]*loggDelta[j]*feDelta[m]*alphaDelta[n]*carbonDelta[p] * iflux
	else:
		for i in range(nTemp):
			for j in range(nLogg):
				for m in range(nFe):
					for n in range(nAlpha):

						# Read in grid point (atmosphere file)
						iflux = readAtm(tempInterval[i],loggInterval[j],feInterval[m],alphaInterval[n], inputdir=griddir, stravinsky=stravinsky) #[:,0]

						# Compute weighted sum of grid points
						## If first iteration, initialize flux as weighted value of lowest grid point 
						if (i==0) & (j==0) & (m==0) & (n==0):
							flux = tempDelta[i]*loggDelta[j]*feDelta[m]*alphaDelta[n] * iflux

						## Else, start adding weighted values of other grid points
						else:
							flux = flux + tempDelta[i]*loggDelta[j]*feDelta[m]*alphaDelta[n] * iflux

	# Normalize by dividing by correct intervals
	quotient = 1.0
	if nTemp == 2:
		quotient = quotient * (tempUp - tempDown)
	if nLogg == 2:
		quotient = quotient * (loggUp - loggDown)
	if nFe == 2:
		quotient = quotient * (feUp - feDown)
	if nAlpha == 2:
		quotient = quotient * (alphaUp - alphaDown)
	if gridch:
		if ncarbon == 2:
			quotient = quotient * (carbonUp - carbonDown)

	flux = flux/(quotient*1.0)

	return flux

def writeAtm(temp, logg, fe, alpha, carbon, dir='/mnt/c/Research/SMAUG/atm/', atom_nums=None, elements=None, abunds=None, solar=None, stravinsky=False):
	"""Create *.atm file

    Inputs:
    temp 	 -- effective temperature (K)
    logg 	 -- surface gravity
    fe 		 -- [Fe/H]
    alpha 	 -- [alpha/Fe]
	carbon   -- [C/Fe]

    Keywords:
    dir 	  -- directory to write atmospheres to [default = '/mnt/c/Research/SMAUG/atm/']
    atom_nums -- list of atomic numbers of elements to add to the list of atoms
    elements  -- list of element symbols you want added to the list of atoms e.g. 'mn', 'sr'
    abunds 	  -- list of elemental abundances corresponding to list of elements
    solar 	  -- solar abundances corresponding to list of elements
	stravinsky -- whether running on stravinsky for file names
    """

	# Atmosphere to write
	filestr, shortfile = getAtm(temp, logg, fe, alpha, dir)
	filestr = filestr[:-4] # Remove '.atm' (in case additional elements need to be added to name)
	shortfile = shortfile[:-4]
	printstr = str(temp) + './' + ('%.2f' % float(logg)) + '/' + ('%5.2f' % float(fe)) + '/' + ('%5.2f' % float(alpha))

	# Check if file already exists
	exists, readytowrite = checkFile(filestr+'.atm', overridecheck=False)
	if exists:
		print('atm file already exists')
		return filestr+'.atm', shortfile+'.atm'

	else:
		# Get atmosphere data
		#####################
		if stravinsky:
			atmosphere = interpolateAtm(temp,logg,fe,alpha,griddir='/raid/grid7/atmospheres/',stravinsky=stravinsky)
		else:
			atmosphere = interpolateAtm(temp,logg,fe,alpha,stravinsky=stravinsky)
		#print('parameters for atm file',temp,logg,fe,alpha)

		# Header text
		#############
		headertxt = str('KURUCZ\n' +
					printstr +
					'\nntau=      72')

		# Footer text
		#############
		# microturbvel 	= atmosphere[0,6]
		if np.isscalar(logg):
			microturbvel = (2.13 - 0.23*logg) * 1e5
		else:
			microturbvel = (2.13 - 0.23*logg[0]) * 1e5
		# print('TEST: ', microturbvel, test)
		# XXX CHECK TO MAKE SURE THIS IS THE SAME AS XI COMPUTED FROM LOGG XXX

		alphatxt 	= str('\n      6       ' + ('%5.2f' % float(8.43 + carbon + fe)) + #not an alpha element but also add carbon abundance to the atmosphere file (this should give [C/Fe])
					'\n      12      ' + ('%5.2f' % float(7.60 + fe + alpha)) +
					'\n      14      ' + ('%5.2f' % float(7.51 + fe + alpha)) +
					'\n      16      ' + ('%5.2f' % float(7.12 + fe + alpha)) +
					'\n      18      ' + ('%5.2f' % float(6.40 + fe + alpha)) +
					'\n      20      ' + ('%5.2f' % float(6.34 + fe + alpha)) +
					'\n      22      ' + ('%5.2f' % float(4.95 + fe + alpha)) ) #solar values plus fe and alpha for the star

		# If not adding any elements, use default NATOMS footer
		if atom_nums is None:
			natoms = 7
			atomstxt = str( ('%.3E' % microturbvel) +
					'\nNATOMS    ' + str(natoms) + '   ' + ('%5.2f' % float(fe)) + alphatxt)

		# If adding elements, first make sure that number of elements matches number of abundances
		elif len(atom_nums) != len(abunds):
			print('Error: length of element array doesn\'t match length of abundances array')
			raise

		else:
			natoms = 7 + len(atom_nums)
			atomstxt = str( ('%.3E' % microturbvel) +
					'\nNATOMS    ' + str(natoms) + '   ' + ('%5.2f' % float(fe)) + alphatxt)

			# Add the new elements
			for i in range(len(atom_nums)):
				atomstxt = str( atomstxt + 
					'\n      '+str(atom_nums[i])+'      ' + ('%5.2f' % float(abunds[i] + solar[i])) )

				# Also add new elements to filename
				abund = int(abunds[i]*10)
				elementname = elements[i]

				# Note different sign conventions for abundances
				if abund < 0:
					elementstr 	= elementname + '{:03}'.format(abund)
				else:
					elementstr	= elementname + '_' + '{:02}'.format(abund)

				filestr = filestr + elementstr
				shortfile = shortfile + elementstr

		# Create final footer by adding NMOL footer to NATOMS footer
		footertxt = str( atomstxt +
					'\nNMOL       15' +
					'\n   101.0   106.0   107.0   108.0   112.0   606.0   607.0   608.0   707.0' +
					'\n   708.0   808.0 10108.0 60808.0     6.1     7.1     8.1   12.1' )

		# Save file
		###########
		np.savetxt(filestr+'.atm', atmosphere, header=headertxt, delimiter=' ', 
			fmt=['%10.9E','%9.1f','%10.4E','%10.4E','%10.4E','%10.4E','%10.4E'],
			footer=footertxt, comments='')

		return filestr+'.atm', shortfile+'.atm'