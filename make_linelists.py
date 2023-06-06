# make_linelists.py
# Given an element of interest, combines line lists from Ivanna and from linemake to make
# a full line list with background lines and hyperfine splitting for the element of interest
#
# created 6/2/2023 -LEH
##############################################################################################

def combine_lines(atom_nums, filestr):
    '''combine strong lines, background lines, and hyperfine lines for a 
    specific element into one line list

    atom_num -- list of atomic numbers of the elements of interest
    filestr  -- filename will be '/mnt/c/Research/Sr-SMAUG/full_linelists/fulllines_[filestr].txt'
    '''
    #open the line lists
    filepath = '/mnt/c/Research/Sr-SMAUG/full_linelists/'
#     strongfile = filepath+'bluestrong.txt'
#     stronglines = read_file(strongfile, header=False)
    valdnist_file = filepath+'valdnist4163.txt'
    valdnistlines = read_file(valdnist_file, header=True)
    linemake_file = filepath+'full_linemake.txt'
    linemakelines = read_file(linemake_file, header=True)

    atomlines = list()
    for atom in atom_nums:
        #keep only desired element lines in linemake lines to get hyperfine splitting
        newlinemakelines = [rows for rows in linemakelines if int(float(rows[10:20].strip()))==atom]
        atomlines += newlinemakelines
        #remove that element's lines from Ivanna's list
        valdnistlines = [rows for rows in valdnistlines if int(float(rows[10:20].strip()))!=atom]

    #combine the two lists into one, still ordered by wavelength, and write into a new file
    newfile = filepath+'full_lines_'+filestr+'.txt'
    ofile=open(newfile, 'w')
    num_atomlines = len(atomlines)
    num_valdnist = len(valdnistlines)
    print('number of lines from linemakeL',num_atomlines, 'number of lines from Ivannas list:',num_valdnist)
    i, j = 0, 0
    while i < num_atomlines and j < num_valdnist:
        if float(atomlines[i][0:10].strip()) < float(valdnistlines[j][0:10].strip()):
            ofile.write(atomlines[i])
            i += 1
        else:
            ofile.write(valdnistlines[j])
            j += 1
    for k in atomlines[i:]:
         ofile.write(k)
    for m in valdnistlines[j:]:
         ofile.write(m)
    ofile.close()

    return

def split_list(full_list, atom_num, element):
    '''splits full line list as created in combine_lines into +/- 10 A
    bands around the element's lines in the reference line list

    inputs:
    full_list -- file path to full line list
    atom_num  -- atomic number of the element of interest
    element   -- symbol of element on interest e.g. 'mn', 'sr'
    '''
    filepath = '/mnt/c/Research/Sr-SMAUG/full_linelists/'
    full_lines = read_file(full_list)
    ref_lines = read_file(filepath+'Ji20_linelist.moog', header=True)
    #get reference lines only for element of interest
    keep_lines = [rows for rows in ref_lines if int(float(rows[15:20].strip()))==atom_num]
    print('number of element lines:', len(keep_lines))

    # need to make this into a file that can be used to make the masks in continuum_div.py!!!!

    print('lines:', keep_lines)

    #split full line list into +/-10A gaps to run with MOOG
    linelists = list() #line list names to give to MOOG
    for i in keep_lines:
        gap = [float(i[0:10].strip())-10, float(i[0:10].strip())+10]
        gap_lines = [lines for lines in full_lines if float(lines[0:10].strip())>gap[0] and float(lines[0:10].strip())<gap[1]]
        newfile = filepath+element+str(int(float(i[0:10].strip())))+'.txt'
        linelists.append(newfile)
        ofile=open(newfile, 'w')
        for j in gap_lines:
            ofile.write(j)
        ofile.close()

    return linelists
     


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
    combine_lines([38,39,40,56,57,58,60,63], 'sprocess')
    #split_list('/mnt/c/Research/Sr-SMAUG/full_linelists/full_lines_sprocess.txt', 38, 'sr')

