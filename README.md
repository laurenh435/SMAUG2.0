# Sr-SMAUG
My copy of SMAUG edited to make it general for any element.
Original was written for Mn by M. de los Reyes (see de los Reyes 2020).

Inputs:



To change element:
1. In output.py, line 98:
   linelists, linegaps, elementlines = split_list('/mnt/c/Research/Sr-SMAUG/full_linelists/full_lines_sprocess.txt', 
   atom_num, element)
   May have to change the input full line list. If running for an s-process element (Sr, Y, Zr, Ba, La, Ce, Nd, Eu), 
   full_lines_sprocess is fine. Otherwise, you may have to make one with the combine_lines function in 
   make_linelists.py
2. In the main input, change atomic number and element name.
3. Optional: Depending on the line list, may want to change which isotope ratios are specified for MOOG. For the
   s-process runs, Ba, Nd, Eu, CH, and CN are specified. This can be changed in the createPar function in run_moog.py

If making this public, include example par file?
Eventually change name because the current one is dumb

