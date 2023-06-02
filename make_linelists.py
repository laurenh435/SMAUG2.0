# make_linelists.py
# Given an element of interest, combines line lists from Ivanna and from linemake to make
# a full line list with background lines and hyperfine splitting for the element of interest
#
# created 6/2/2023 -LEH
##############################################################################################

def combine_lines(atom_num,element):
    '''combine strong lines, background lines, and hyperfine lines for a 
    specific element into one line list

    atom_num = atomic number of the element of interest
    element = symbol of element on interest e.g. 'mn', 'sr'
    '''
    #open the line lists
    filepath = '/mnt/c/Research/Sr-SMAUG/full_linelists/'
#     strongfile = filepath+'bluestrong.txt'
#     stronglines = read_file(strongfile, header=False)
    valdnist_file = filepath+'valdnist4163.txt'
    valdnistlines = read_file(valdnist_file, header=True)
    linemake_file = filepath+'full_linemake.txt'
    linemakelines = read_file(linemake_file, header=True)

    #keep only desired element lines in linemake lines (to get hyperfine splitting)
    linemakelines = [rows for rows in linemakelines if int(float(rows[10:20].strip()))==atom_num]

    #remove that element's lines from Ivanna's list so we can insert the hyperfine splitting
    valdnistlines = [rows for rows in valdnistlines if int(float(rows[10:20].strip()))!=atom_num]

    #combine the two lists into one, still ordered by wavelength and write into a new file
    newfile = filepath+'full_lines_'+element+'.txt'
    ofile=open(newfile, 'w')
    #linemakelinest = linemakelines[0:10] for testing
    #valdnistlinest = valdnistlines[300:400]
    num_linemake = len(linemakelines)
    num_valdnist = len(valdnistlines)
    print(num_linemake, num_valdnist)
    i, j = 0, 0
    while i < num_linemake and j < num_valdnist:
        if float(linemakelines[i][0:10].strip()) < float(valdnistlines[j][0:10].strip()):
            ofile.write(linemakelines[i])
            i += 1
        else:
            ofile.write(valdnistlines[j])
            j += 1
    for k in linemakelines[i:]:
         ofile.write(k)
    for m in valdnistlines[j:]:
         ofile.write(m)
    ofile.close()

    return


def read_file(filepath, header=False):
    '''Read data from file line by line.
    
    header: if header is true delete the first line of data
    '''
    ifile = open(filepath)
    lines = ifile.readlines()
    #remove header
    if header:
         lines.pop(0)
    #print(lines)
    #print(lines[4][0:10])
    #print(lines[4][10:20])
    ifile.close()
    return lines

if __name__ == "__main__":
    #Sr:38, Mn:25
	combine_lines(38, 'sr')

