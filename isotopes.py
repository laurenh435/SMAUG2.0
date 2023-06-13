# isotopes.py
# Use table 1 from Sneden+ 2008 to get isotope ratio of a given element based on the
# ratio of s to r process
#
# written 6/7/2023 -LEH
#######################################################################################

import numpy as np

class element:
    def __init__(self, Z):
        self.Z = Z
        self.isotopes = []
        self.Ns = []
        self.Nr = []

def isotope_ratio(atom_num, star_sfrac):
    '''given an element and the fraction of that element created
    by the s-process, return a list of what fraction of that element is 
    each isotope
    star_sfrac = between 0 and 1
    '''
    objs, Z_list = readTable()
    index = np.where(np.array((Z_list)) == atom_num)[0][0]
    stot = np.sum(objs[index].Ns)
    rtot = np.sum(objs[index].Nr)
    #Th and U are r-process only: so they must have star_sfrac = 0
    if stot > 0:
        sfrac = objs[index].Ns/stot
    else:
        sfrac = objs[index].Ns
    if rtot > 0:
        rfrac = objs[index].Nr/rtot
    else:
        rfrac = objs[index].Nr
    #print('sfrac:',sfrac)
    isotope_fracs = [] #fraction of each isotope
    moog_isotopes = [] #string of isotope identifiers for MOOG e.g. 38.1086 for 86SrII
    for i in range(len(objs[index].isotopes)):
        val = star_sfrac*sfrac[i] + (1-star_sfrac)*rfrac[i]
        isotope_fracs.append(val)
        print('isotope fraction:', objs[index].isotopes[i], val)
        moog_isotopes.append(str(atom_num)+'.1'+str(objs[index].isotopes[i]).zfill(3))
    #MOOG takes reciprocal of isotope fraction as input
    iso_reciprocal = []
    for fraction in isotope_fracs:
        if fraction == 0: #just in case one of the isotope fractions is 0
            iso_reciprocal.append(999999999)
        else:
            iso_reciprocal.append(1/fraction)
    #iso_reciprocal = 1/np.array(isotope_fracs)
    print('reciprocals:', iso_reciprocal)
    #print('check:', np.sum(np.array(isotope_fracs)))

    
    return iso_reciprocal, moog_isotopes

def readTable():
    ''' read in table 1 from Sneden+ 2008
    '''
    tablepath = '/mnt/c/Research/Sr-SMAUG/isotope_table.txt'
    tablelines = read_file(tablepath, header=True)
    element_symbols = []
    Z_list = []
    for line in tablelines:
        symbol, Z, A, Ns, Nr = parse_line(line)
        #print(symbol, Z, A, Ns, Nr)
        if Z > 0:
             element_symbols.append(symbol)
             Z_list.append(Z)
    objs = [element(Z_list[i]) for i in range(len(element_symbols))]
    index = -1
    for line in tablelines:
        symbol, Z, A, Ns, Nr = parse_line(line)
        if Z == 0:
            objs[index].isotopes.append(A)
            objs[index].Ns.append(Ns)
            objs[index].Nr.append(Nr)
        else:
            index += 1
            objs[index].isotopes.append(A)
            objs[index].Ns.append(Ns)
            objs[index].Nr.append(Nr)

    return objs, Z_list

def read_file(filepath, header=False):
    '''Read data from file line by line.
    
    header: if header is true delete the first line of data
    '''
    ifile = open(filepath)
    lines = ifile.readlines()
    #remove header
    if header:
         lines.pop(0)
    ifile.close()
    return lines

def parse_line(line):
    elements = line.split()
    if len(elements) < 5:
        elements = [0,0]+elements
    element = elements[0]
    Z = int(float(elements[1]))
    A = int(float(elements[2]))
    Ns = float(elements[3])
    Nr = float(elements[4])
    return element, Z, A, Ns, Nr

if __name__ == "__main__":
    isotope_reciprocals = isotope_ratio(38, 0.5)