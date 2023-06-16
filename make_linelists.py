# make_linelists.py
# Given an element of interest, combines line lists from Ivanna and from linemake to make
# a full line list with background lines and hyperfine splitting for the element of interest
#
# Also split up full line list into +/- 10 Angstrom regions around lines of interest
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

    atomlines = []
    for atom in atom_nums:
        #keep only desired element lines in linemake lines to get hyperfine splitting
        newlinemakelines = [rows for rows in linemakelines if int(float(rows[10:20].strip()))==atom]
        atomlines.append(newlinemakelines)
        #remove that element's lines from Ivanna's list
        valdnistlines = [rows for rows in valdnistlines if int(float(rows[10:20].strip()))!=atom]

    #combine the two lists into one, still ordered by wavelength, and write into a new file
    newfile = filepath+'full_lines_'+filestr+'.txt'
    ofile=open(newfile, 'w')
    # num_atomlines = len(atomlines)
    # num_valdnist = len(valdnistlines)
    #print('number of lines from linemake:',num_atomlines, 'number of lines from Ivannas list:',num_valdnist)
    total = valdnistlines
    for element in atomlines:
        i, j = 0, 0
        natomlines = len(element)
        ntotal = len(total)
        #print(natomlines,ntotal)
        while i < natomlines and j < ntotal:
            if float(element[i][0:10].strip()) < float(total[j][0:10].strip()):
                total.insert(j,element[i])
                ntotal = len(total)
                i += 1
            else:
                j += 1
        for k in element[i:]:
            total.append(k)
    for p in total:
        ofile.write(p)
    ofile.close()

    return

def split_list(full_list, atom_num, element):
    '''splits full line list as created in combine_lines into +/- 10 A
    bands around the element's lines in the reference line list

    inputs:
    full_list -- file path to full line list
    atom_num  -- atomic number of the element of interest
    element   -- symbol of element on interest e.g. 'mn', 'sr'

    outputs:
    linelists    -- list of line list file names as strings
    newgaps      -- +/- 10 angstrom gaps around lines of interest with overlapping gaps combined
    elementlines -- lines from element of interest from 'Ji20_linelist.moog'
    '''
    filepath = '/mnt/c/Research/Sr-SMAUG/full_linelists/'
    full_lines = read_file(full_list)
    ref_lines = read_file(filepath+'Ji20_linelist.moog', header=True)
    #get reference lines only for element of interest
    keep_lines = [rows for rows in ref_lines if int(float(rows[15:20].strip()))==atom_num]
    #print('number of element lines:', len(keep_lines))

    #split full line list into +/-10A gaps to run with MOOG
    linelists = list() #line list names to give to MOOG
    gaps = []
    elementlines = []
    for i in keep_lines:
        elementlines.append(int(float(i[0:10].strip())))
        gap = [float(i[0:10].strip())-10, float(i[0:10].strip())+10]
        gaps.append(gap)

    #put lines and gaps in order
    elementlines.sort()
    def sortfirst(val):
        return val[0]
    gaps.sort(key=sortfirst)
    #print('oldgaps:',gaps)

    #combine gaps that are overlapping
    newgaps = []
    oldnumber = len(gaps)
    gapkey = []
    newkey = ''
    a = 0
    b = 1
    counter = 0
    extend = False
    while a < oldnumber:
        if b < oldnumber:
            if gaps[a][1] > gaps[b][0]:
                extend = True
                newgap = [gaps[a][0],gaps[b][1]]
                newkey += str(int(elementlines[a]))
                b += 1
                a+=1
                counter += 1
                #print('need to extend. b=',b)
            elif extend == False and gaps[a][1] < gaps[b][0]:
                newgap = gaps[a]
                newgaps.append(newgap)
                gapkey.append(str(int(elementlines[a])))
                a+=1
                b+=1
                extend = False
                counter = 0
                #print('did not need to extend. a,b=',a,b)
            else:
                counter += 1
                newgaps.append([gaps[b-counter][0],gaps[b-1][1]])
                counter = 0
                a = b
                b = a+1
                extend = False
                #print('extended. a,b=',a,b)
                gapkey.append(newkey)
                
        elif extend == False and b >= oldnumber:
            newgaps.append(gaps[-1])
            gapkey.append(str(int(elementlines[-1])))
            break
        else:
            newgaps.append([gaps[a][0],gaps[-1][1]])
            gapkey.append(newkey)
            break
    
    #write line list files
    for k in range(len(newgaps)):
        gap_lines = [lines for lines in full_lines if float(lines[0:10].strip())>newgaps[k][0] and float(lines[0:10].strip())<newgaps[k][1]]
        newfile = filepath+element+gapkey[k]+'.txt'
        linelists.append(newfile)
        ofile=open(newfile, 'w')
        ofile.write(element+' '+gapkey[k]+' +/-10 A'+'\n') #MOOG wants first line to be a header of some sort
        for j in gap_lines:
            ofile.write(j)
        ofile.close()

    return linelists, newgaps, elementlines


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
    #split_list('/mnt/c/Research/Sr-SMAUG/full_linelists/full_lines_mn.txt', 25, 'Mn')

