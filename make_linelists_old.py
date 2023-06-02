# make_linelists_old.py
# Given an element of interest, combines line lists from Ivanna and from linemake to make
# a full line list with background lines and hyperfine splitting for the element of interest
# doesn't really work... see make_linelists.py
#
# created 6/1/2023 -LEH
##############################################################################################

from astropy.io import ascii
import numpy as np
import string

def combine_lines(atom_num,element):
    '''combine strong lines, background lines, and hyperfine lines for a 
    specific element into one line list

    atom_num = atomic number of the element of interest
    element = symbol of element on interest e.g. 'mn', 'sr'
    '''
    #open the line lists
    filepath = '/mnt/c/Research/Sr-SMAUG/full_linelists/'
    strongfile = filepath+'bluestrong.txt'
    stronglines = read_file(strongfile,' ', header=False)
    valdnist_file = filepath+'valdnist4163.txt'
    valdnistlines = read_file(valdnist_file,' ', header=True)
    linemake_file = filepath+'full_linemake.txt'
    linemakelines = read_file(linemake_file, ' ',header=True)

    #keep only desired element lines in linemake lines (to get hyperfine splitting)
    linemakelines = [rows for rows in linemakelines if int(rows[1])==atom_num or int(rows[1])==atom_num]
    #print(linemakelines)

    #remove that element's lines from Ivanna's list so we can insert the hyperfine splitting
    valdnistlines = [rows for rows in valdnistlines if int(rows[1])!=atom_num or int(rows[1])!=atom_num]
    #print(valdnistlines)

    #combine the two lists into one, still ordered by wavelength
    full_lines = []
    linemakelinestemp = linemakelines[0:10]
    valdnistlinestemp = valdnistlines[2000:2100]
    num_linemake = len(linemakelinestemp)
    num_valdnist = len(valdnistlinestemp)
    print(num_linemake, num_valdnist)
    i, j = 0, 0
    while i < num_linemake and j < num_valdnist:
         if linemakelinestemp[i][0] < valdnistlinestemp[j][0]:
              full_lines.append(linemakelinestemp[i])
              i += 1
    else:
         full_lines.append(valdnistlinestemp[j])
         j += 1
    full_lines = full_lines + linemakelinestemp[i:] + valdnistlinestemp[j:]

    #output into a file
    result = "\n".join("\t".join(map(str,l)) for l in full_lines)
    newfile = filepath+'full_lines_'+element+'.txt'
    ofile=open(newfile, 'w')
    ofile.write(result)
    ofile.close()

    return

def read_file(filepath,delimiter,header=False):
    '''Open file, read data, close file, return data as arrays
    data is returned as a list of lists of floats.
    Gets rid of entries that include letters.

    header: if header is true delete the first line of data
    '''
    alphabet = string.ascii_uppercase+string.ascii_lowercase
    data = list()
    ifile = open(filepath)
    line = ifile.readline()
    #read file
    while line:
        if(line != ''):
            if delimiter == None:
                data.append(line.strip('\n').split())
            else:
                data.append(line.strip('\n').split(delimiter))
            line = ifile.readline()
    #remove header
    if header:
         data.pop(0)
    #get rid of blank spaces, remove any entries that contain letters, and convert to floats
    for i in range(len(data)): 
         data[i] = [x for x in data[i] if x != '' and check_pres(x,alphabet)]
    for i in range(len(data)):
         for j in range(len(data[i])):
              data[i][j] = float(data[i][j])             
    ifile.close()

    return(data)

def check_pres(sub, test_str):
    '''check to see if any elements in a string match 
    the test string
    '''
    for ele in sub:
        if ele in test_str:
            return 0
    return 1

if __name__ == "__main__":
    #Sr:38, Mn:25
	combine_lines(38, 'sr')
