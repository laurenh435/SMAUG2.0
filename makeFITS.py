# makeFITS.py
# make simple FITS file with list of the desired element lines for masking the spectrum
# used in continuum_div.py
#
# created 5/31/2023 -LEH
##############################################################################

from astropy.io import fits
import numpy as np

#if changing element, keep file format 'c/mnt/Research/Sr-SMAUG/[element]linelists/[element]lines.fits
outpath = '/mnt/c/Research/Sr-SMAUG/Mnlinelists/Mnlines.fits'
linearray = np.array([4739.1, 4754.0, 4761.5, 4762.3, 4765.8, 4766.4, 4783.5, 4823.5, 5399.5, 5407.3, 5420.3, 5516.8, 5537.7, 6013.5, 6016.6, 6021.8, 6384.7, 6491.7])
c1 = fits.Column(name='Mnlines',array=linearray,format='D') #'D' saves the values as double precision floats (64-bit)
t = fits.BinTableHDU.from_columns([c1])
t.writeto(outpath, overwrite=True)

#tests

fitslines = fits.open('/mnt/c/Research/Sr-SMAUG/Mnlinelists/Mnlines.fits')
elemlines = fitslines[1].data
print(elemlines['Mnlines'])
gap = [elemlines['Mnlines'][0]-1., elemlines['Mnlines'][0]+1.]
print(gap)
			
#lines = np.zeros(len(elemlines['Mnlines']))
lines = list()
for i in range(len(elemlines['Mnlines'])):
	gap = [elemlines['Mnlines'][i]-1., elemlines['Mnlines'][i]+1.]
	lines.append(gap)
linesarray = np.array(lines)
print(lines)