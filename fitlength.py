from astropy.io import fits
import sys

def getlength(file):
    hdu1 = fits.open(file)
    # print(hdu1[1].header)
    data = hdu1[1].data
    # wavearray = data['LAMBDA']
    namearray = data['OBJNAME']
    Nstars = len(namearray)
    print(Nstars)
    return Nstars

def main():
    getlength(sys.argv[1])

if __name__ == "__main__":
    main()