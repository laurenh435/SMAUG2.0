# wvl_corr_new.py
#
# Perform wavelength correction.
#
# written by V. Manwadkar, adapted from the wvl_corr.py code from Mia
# edited 7/14/2023 to work with my SMAUG version -LEH
#########################################################################

#Backend for python3 on mahler
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    import matplotlib.pyplot as plt
    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc('text', usetex=False)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1])
    return


def gen_mock_spec(flux,ivari,N = 100):
    '''
    Given the observed flux values and the corresponding uncertainties, it resamples it has a normal distribution
    to get different realizations of the spectra

    this is done to account for the observed uncertainty while computing the lines
    '''
    
    mus = flux
    sigs = np.sqrt(1/ivari)
    
    mock_specs = np.random.normal(mus,sigs,size = (N,len(mus)) )
    return mock_specs

def get_lines(obswvl, obsflux, ivar, mock=True):
    '''
    This function computes the location of the line in the observed spectra
    which will be used to compute the wavelength correction ...

    As of now, I am implementing three smoothing schemes:
    1. Savitsky-Golay filter (there is a smoothing window and a corresponding polynomail order for fit)
    2. Gaussian filter with 2 sigma
    3. Gaussian filter with 3 sigma

    Using them in combination appears to do well, but might seem arbitrary for this choice+combination... 

    ''' 
    
    wvtr = np.linspace(np.min(obswvl),np.max(obswvl),1000)
    
    ##using this let us generate the mock spectra
    if mock == True:
        mock_specs = gen_mock_spec(obsflux,ivar)
        
        all_sgfs = []
        all_g2s = []
        all_g4s = []
        
        for mi in mock_specs:
            ##these combos appear to do well..
            f_g2 = gaussian_filter1d(mi, 2)
            f_g4 = gaussian_filter1d(mi, 3)
            f_sgf = savgol_filter(mi, 9, 2)

            ### then the goal is to fit the above data using a spline??? and then see how it does in identifying the minimum?
            g2_spl = UnivariateSpline(obswvl,f_g2,s=0)
            g4_spl = UnivariateSpline(obswvl,f_g4,s=0)
            sgf_spl = UnivariateSpline(obswvl,f_sgf,s=0)


            ##identify the minima of each of these points ... 
            sgf_wvlmin_i = wvtr[np.argmin(sgf_spl(wvtr))]
            g2_wvlmin_i = wvtr[np.argmin(g2_spl(wvtr))]
            g4_wvlmin_i = wvtr[np.argmin(g4_spl(wvtr))]
            
            all_sgfs.append(sgf_wvlmin_i)
            all_g2s.append(g2_wvlmin_i)
            all_g4s.append(g4_wvlmin_i)
            
        return all_g2s, all_g4s, all_sgfs


    else:
        #we just look at the original spectra for this 

        ##these combos appear to do well..
        f_g2 = gaussian_filter1d(obsflux, 2)
        f_g4 = gaussian_filter1d(obsflux, 3)
        f_sgf = savgol_filter(obsflux, 9, 2)

        ### then the goal is to fit the above data using a spline??? and then see how it does in identifying the minimum?
        g2_spl = UnivariateSpline(obswvl,f_g2)
        g4_spl = UnivariateSpline(obswvl,f_g4)
        sgf_spl = UnivariateSpline(obswvl,f_sgf)

        ##identify the minima of each of these points ... 
        sgf_wvlmin = wvtr[np.argmin(sgf_spl(wvtr))]
        g2_wvlmin = wvtr[np.argmin(g2_spl(wvtr))]
        g4_wvlmin = wvtr[np.argmin(g4_spl(wvtr))]

        return g2_wvlmin, g4_wvlmin, sgf_wvlmin
    

def plotting_wvlmin(ax,obswvl_1,obsflux_1,ivar_1, f_g2_1, f_g3_1, f_sgf_1,hgamma, wvl_radius, dlam, legend = False):
    '''
    This is the plotting function and also returns the median value of line and standard deviation in it. Used to compute uncertainty on the wavelength offsets
    '''
    
    std_1 = np.sqrt(1/ivar_1)

    ax.errorbar(obswvl_1, obsflux_1,color = "k",yerr = std_1,ls ="",capsize =2,markersize = 3,marker = "o")

#     for i in range(len(mock_specs)):
#         ax.plot(obswvl_1,mock_specs_hg[i],color = "k",lw = 0.25,alpha = 0.35,ls = "dotted")

    ax.vlines(x = hgamma, ymin = np.min(obsflux_1-std_1)*0.95,ymax = np.max(obsflux_1+std_1)*1.05,color = "r",lw = 2,alpha = 0.4)

    ax.plot(obswvl_1,f_g2_1,color = "seagreen",lw = 1,label = r"Gaussian Filter 2sigma")
    ax.plot(obswvl_1,f_g3_1,color = "darkorange",lw = 1,label = r"Gaussian Filter 4sigma")
    ax.plot(obswvl_1,f_sgf_1,color = "mediumblue",lw = 1,label = r"Savitsky-Golay Filter (w9,d2)")

    ### then the goal is to fit the above data using a spline??? and then see how it does in identifying the minimum?
    g2_spl = UnivariateSpline(obswvl_1,f_g2_1,s=0.)
    g3_spl = UnivariateSpline(obswvl_1,f_g3_1,s=0)
    sgf_spl = UnivariateSpline(obswvl_1,f_sgf_1,s=0)

    wvtr = np.linspace(hgamma - wvl_radius,hgamma + wvl_radius,1000)

    ax.plot(wvtr, g2_spl(wvtr),lw = 1, ls = "--",color = "seagreen")
    ax.plot(wvtr, g3_spl(wvtr),lw = 1, ls = "--",color = "darkorange")
    ax.plot(wvtr, sgf_spl(wvtr),lw = 1, ls = "--",color = "mediumblue")

    ##identify the minima of each of these points ... 

    sgf_wvlmin = wvtr[np.argmin(sgf_spl(wvtr))]
    g2_wvlmin = wvtr[np.argmin(g2_spl(wvtr))]
    g3_wvlmin = wvtr[np.argmin(g3_spl(wvtr))]
    
    ax.vlines(x = sgf_wvlmin,ymin = np.min(obsflux_1-std_1)*0.95,ymax = np.max(obsflux_1+std_1)*1.05,color = "mediumblue",lw = 1,alpha = 0.7 )
    ax.vlines(x = g2_wvlmin,ymin = np.min(obsflux_1-std_1)*0.95,ymax = np.max(obsflux_1+std_1)*1.05,color = "seagreen",lw = 1,alpha = 0.7 )
    ax.vlines(x = g3_wvlmin,ymin = np.min(obsflux_1-std_1)*0.95,ymax = np.max(obsflux_1+std_1)*1.05,color = "darkorange",lw = 1,alpha = 0.7 )

    all_g2s, all_g3s, all_sgfs = get_lines(obswvl_1, obsflux_1, ivar_1,mock=True)
    all_mins = all_sgfs + all_g2s #+ all_g3s

    min_median = np.median(all_mins)
    min_std = np.std(all_mins)

    ##we need to set the minimum uncertainty on this as the FWHM of the spectra
    if min_std < np.mean(dlam)/2.354: #converting FWHM to sigma:
        min_std = np.mean(dlam)/2.354

    ax.fill_betweenx(y = [np.min(obsflux_1-std_1)*0.95, np.max(obsflux_1+std_1)*1.05],
                      x1 = min_median - min_std, x2 = min_median + min_std, color = "lightseagreen",
                      alpha = 0.4,edgecolor = "None",label = r"Identified Line \pm 1sigma")

    if legend == True:
        ax.legend(frameon=False,fontsize = 8,loc = "lower left")

    try:
        ax.set_ylim([np.min(obsflux_1-std_1)*0.95,np.max(obsflux_1+std_1)*1.05])
    except:
        pass
    
    ax.set_xlim([hgamma - wvl_radius, hgamma + wvl_radius] )

    
    return min_median, min_std
    


def do_wvl_corr(obswvl, obsflux_norm, ivar, outputname, specname, dlam):
    '''
    The main function that does the full wavelength corrections

    I need to do something in the case where we do not have any data around some wavelenghts
    '''

    #read the lines which we are going for wavelength correction from iniconf file
    #true_lines = np.array(iniconf["wvl corr"]["true_lines"].split(",")).astype(float)
    true_lines = [4340.47,4861.35,5183.60,5889.95,6562.79]

    #wvl_radius = float(iniconf["wvl corr"]["wvl_radius"])
    wvl_radius = 4.0

    chipgap = int(len(obswvl)/2 - 1)
    wvl_begin_gap = obswvl[chipgap - 5]
    wvl_end_gap = obswvl[chipgap + 5]

    #generate the masks 
    all_masks = []
    true_lines_f = []
    for ti in true_lines:
        # Check that all three hydrogen lines are included in the data
        if ti > obswvl[0] and ti < wvl_begin_gap or ti < obswvl[-1] and ti > wvl_end_gap:
            #this means that this line is included in the data
            true_lines_f.append(ti)
            mask_i = (obswvl > ti - wvl_radius ) & (obswvl < ti + wvl_radius)
            all_masks.append(mask_i)

    print("The following lines are found in the data : ", true_lines_f )

    if len(true_lines_f) == 0:
        raise ValueError("No of the inputted lines found! Check that the spectra data is appropriate.")

    #now using these final lines we do the wvl offset calculation
    
    all_f_g2 = []
    all_f_g3 = []
    all_f_sgf = []

    for mask_i in all_masks:
        ##these combos appear to do well..
        f_g2_i = gaussian_filter1d(obsflux_norm[mask_i], 2) #gaussian 2 sigma filter
        f_g3_i = gaussian_filter1d(obsflux_norm[mask_i], 3) #gaussian 3 sigma filter
        f_sgf_i = savgol_filter(obsflux_norm[mask_i], 9, 2) #savitsky-golay filter

        all_f_g2.append(f_g2_i)
        all_f_g3.append(f_g3_i)
        all_f_sgf.append(f_sgf_i)

    #now let us begin the plotting!


    meas_lines = []
    meas_errs = []

    #####################
    #####################

    plot_pretty()

    fig,ax = plt.subplots(3,2, figsize = (10,9))
    plt.subplots_adjust(wspace = 0.1, hspace = 0.1)

    all_axes = [ax[0,0],ax[0,1],ax[1,0],ax[1,1], ax[2,1] ]

    for i,mask_i in enumerate(all_masks):

        obswvl_i= obswvl[mask_i]
        obsflux_norm_i = obsflux_norm[mask_i]
        ivar_i = ivar[mask_i]

        f_g2_i = all_f_g2[i]
        f_g3_i = all_f_g3[i]
        f_sgf_i =all_f_sgf[i] 

        if i == 2:
            legend = True
        else:
            legend = False

        line_i_min, line_i_std = plotting_wvlmin(all_axes[i],obswvl_i,obsflux_norm_i,ivar_i, f_g2_i, f_g3_i, f_sgf_i,true_lines_f[i],wvl_radius,dlam, legend )
        meas_lines.append(line_i_min)
        meas_errs.append(line_i_std)

    #the folder where all the fitting plots and outputs will be stored
    #filename = iniconf["chisq fitting"]["output_path"] + "/" + specname + "_wvlfit_new.png"
    filename = outputname + "/" + specname + "_wvlfit_new.png"
    plt.savefig(filename) 
    plt.close()

    #####################
    #####################

    meas_lines = np.array(meas_lines)
    meas_errs = np.array(meas_errs)

    wvl_offsets = meas_lines - true_lines_f

    #now we fit the offsets to correct the spectrum 

    wvtr = np.linspace(4100,6800,100)

    plt.figure(figsize = (4,3))
    plt.errorbar(meas_lines, wvl_offsets,yerr = meas_errs,ls = "",color="k",capsize = 3,marker = "o")

    #poly_order = int(iniconf["wvl corr"]["npoly_order"])
    poly_order = 2

    if len(meas_lines) == 5:
        poly_order = 3
    if len(meas_lines) == 4:
        poly_order = 2
    if len(meas_lines) == 3:
        poly_order = 1
    if len(meas_lines) == 2:
        poly_order = 0

    param_use = np.polyfit(meas_lines,wvl_offsets,poly_order,w=np.reciprocal(meas_errs))

    wvl_offset_func = np.poly1d(param_use) 
    plt.plot(wvtr, wvl_offset_func(wvtr))
    plt.xlabel("Measured - Truth",fontsize = 12)
    plt.ylabel("Measured Wavelength",fontsize = 12)

    #filename = iniconf["chisq fitting"]["output_path"] + "/" + specname + "_wvloffset_new.png"
    filename = outputname + "/" + specname + "_wvloffset_new.png"
    plt.savefig(filename)
    plt.close()

    ### now using this offset function let us compute the corrected wavelenghts 

    obswvl_new = obswvl - wvl_offset_func(obswvl) 

    return obswvl_new
        