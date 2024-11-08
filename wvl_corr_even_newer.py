# wvl_corr_even_newer.py
#
# Perform wavelength correction.
# Attempting to make some improvements to Viraj's wavelength correction code
# (including using a lower-order polynomial and doing separate corrections for red and blue sides of chip gap)
#
# written by LEH, adapted from code by V. Manwadkar 9/1/2023
##################################################################################################################

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

def plotting_wvlmin(ax, obswvl_1, obsflux_1, ivar_1, f_g2_1, f_g3_1, f_sgf_1, line, wvl_radius, dlam, legend = False):
    '''
    This is the plotting function and also returns the median value of line and standard deviation in it. Used to compute uncertainty on the wavelength offsets
    '''
    
    std_1 = np.sqrt(1/ivar_1)

    ax.errorbar(obswvl_1, obsflux_1,color = "k",yerr = std_1,ls ="",capsize =2,markersize = 3,marker = "o")

#     for i in range(len(mock_specs)):
#         ax.plot(obswvl_1,mock_specs_hg[i],color = "k",lw = 0.25,alpha = 0.35,ls = "dotted")

    ax.vlines(x = line, ymin = np.min(obsflux_1-std_1)*0.95,ymax = np.max(obsflux_1+std_1)*1.05,color = "r",lw = 2,alpha = 0.4)

    ax.plot(obswvl_1,f_g2_1,color = "seagreen",lw = 1,label = r"Gaussian Filter 2sigma")
    ax.plot(obswvl_1,f_g3_1,color = "darkorange",lw = 1,label = r"Gaussian Filter 4sigma")
    ax.plot(obswvl_1,f_sgf_1,color = "mediumblue",lw = 1,label = r"Savitsky-Golay Filter (w9,d2)")

    ### then the goal is to fit the above data using a spline??? and then see how it does in identifying the minimum?
    g2_spl = UnivariateSpline(obswvl_1,f_g2_1,s=0.)
    g3_spl = UnivariateSpline(obswvl_1,f_g3_1,s=0)
    sgf_spl = UnivariateSpline(obswvl_1,f_sgf_1,s=0)

    wvtr = np.linspace(line - wvl_radius,line + wvl_radius,1000)

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
    
    ax.set_xlim([line - wvl_radius, line + wvl_radius] )

    
    return min_median, min_std

def do_wvl_corr(obswvl, obsflux, ivar, outputname, specname, dlam):
    '''
    Correct wavelength by finding strong lines in the spectrum and fitting a polynomial
    to the offsets from where the lines should be. Do this for both sides of the chipgap.
    '''
    # define the lines to use: using Halpha, Hbeta, Hgamma, Mg line, one Na doublet line
    # should we just fit the whole Na doublet? So that we don't accidentally line up to the wrong one?
    true_lines = [4340.47,4861.35,5183.60,5889.95,6562.79]

    # define radius to search for the line in the spectrum
    wvl_radius = 4.0

    # split spectrum into red and blue components
    ichipgap = int(len(obswvl)/2 - 1) #index of chipgap
    wvl_begin_gap = obswvl[ichipgap - 5]
    wvl_end_gap = obswvl[ichipgap + 5]
    chipgap_region = obswvl[ichipgap-5:ichipgap+6]
    # print('chipgap_region length', len(chipgap_region))
    ired = np.where(np.array(obswvl) > wvl_end_gap)
    iblue = np.where(np.array(obswvl) < wvl_begin_gap)
    obswvl_split = [obswvl[iblue[0]],obswvl[ired[0]]]
    obswvl_red = obswvl[ired[0]]
    obswvl_blue = obswvl[iblue[0]]
    obsflux_split = [obsflux[iblue[0]],obsflux[ired[0]]]
    obsflux_red = obsflux[ired[0]]
    obsflux_blue = obsflux[iblue[0]]
    ivar_split = [ivar[iblue[0]],ivar[ired[0]]]
    dlam_split = [dlam[iblue[0]],dlam[ired[0]]]
    blue_lines = [line for line in true_lines if line < wvl_begin_gap]
    red_lines = [line for line in true_lines if line > wvl_end_gap]
    lines_split = [blue_lines, red_lines]

    # generate the masks 
    all_masks = [[],[]]
    true_lines_f = [[],[]]
    for i in range(len(obswvl_split)):
        for ti in lines_split[i]:
            if ti > obswvl_split[i][0] and ti < obswvl_split[i][-1]:
                # this means that this line is included in the data
                true_lines_f[i].append(ti)
                mask_i = (obswvl_split[i] > ti - wvl_radius ) & (obswvl_split[i] < ti + wvl_radius)
                all_masks[i].append(mask_i)

    print("The following lines are found in the data : ", true_lines_f )
    if (len(true_lines_f[0]) == 0) or (len(true_lines_f[1]) == 0):
        raise ValueError("No inputted lines found! Check that the spectra data is appropriate.")
    
    # now using these final lines we do the wvl offset calculation
    all_f_g2 = [[],[]]
    all_f_g3 = [[],[]]
    all_f_sgf = [[],[]]

    for j in range(len(all_masks)):
        for mask_i in all_masks[j]:
            # these combos appear to do well..
            f_g2_i = gaussian_filter1d(obsflux_split[j][mask_i], 2) # gaussian 2 sigma filter
            f_g3_i = gaussian_filter1d(obsflux_split[j][mask_i], 3) # gaussian 3 sigma filter
            f_sgf_i = savgol_filter(obsflux_split[j][mask_i], 9, 2) # savitsky-golay filter
            all_f_g2[j].append(f_g2_i)
            all_f_g3[j].append(f_g3_i)
            all_f_sgf[j].append(f_sgf_i)
    
    # begin plotting
    meas_lines = [[],[]]
    meas_errs = [[],[]]

    plot_pretty() # sets plotting parameters
    num_lines = len(true_lines_f[0]) + len(true_lines_f[1])
    nrows = 2
    ncols = len(max(true_lines_f, key=len))
    # print(ncols)
    #ncols = int(num_lines/2)+1
    fig,ax = plt.subplots(nrows,ncols, figsize = (12,9)) # was 3, 2, figsize
    # print('intialized the figure')
    plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
    # print(len(true_lines_f[0]))
    # print(range(len(true_lines_f[0])))
    ax_blue = [ax[0,i] for i in range(len(true_lines_f[0]))]
    # print('made ax_blue')
    ax_red = [ax[1,i] for i in range(len(true_lines_f[1]))]
    all_axes = [ax_blue, ax_red]
    #all_axes = [[ax[0,0],ax[0,1],ax[1,0]],[ax[1,1], ax[2,1]] ] #fix this so that it's automated or something like in Mia's make plots
    
    # print(len(all_masks))
    for j in [0,1]:
        # print(j)
        for i,mask_i in enumerate(all_masks[j]):
            obswvl_i= obswvl_split[j][mask_i]
            obsflux_norm_i = obsflux_split[j][mask_i]
            ivar_i = ivar_split[j][mask_i]

            f_g2_i = all_f_g2[j][i]
            f_g3_i = all_f_g3[j][i]
            f_sgf_i = all_f_sgf[j][i] 

            if i == 2:
                legend = True
            else:
                legend = False
            line_i_min, line_i_std = plotting_wvlmin(all_axes[j][i],obswvl_i,obsflux_norm_i,ivar_i, f_g2_i, f_g3_i, f_sgf_i,true_lines_f[j][i],wvl_radius,dlam_split[j], legend )
            meas_lines[j].append(line_i_min)
            meas_errs[j].append(line_i_std)

    # the folder where all the fitting plots and outputs will be stored
    #filename = iniconf["chisq fitting"]["output_path"] + "/" + specname + "_wvlfit_new.png"
    filename = outputname + "/" + specname + "_wvlfit_new.png"
    plt.savefig(filename) 
    plt.close()

    ###################
    ###################
    # print(meas_lines)
    # print(meas_errs)
    meas_lines_blue = np.array(meas_lines[0])
    meas_lines_red = np.array(meas_lines[1])

    wvl_offsets = [meas_lines_blue - np.array(true_lines_f[0]), meas_lines_red - np.array(true_lines_f[1])]
    # now we fit the offsets to correct the spectrum 
    print('wvl offsets not cut',wvl_offsets)
    meas_lines = [[meas_lines[0][i] for i in range(len(wvl_offsets[0])) if (wvl_offsets[0][i] < 1.5) and (wvl_offsets[0][i] > -1.5)], [meas_lines[1][i] for i in range(len(wvl_offsets[1])) if (wvl_offsets[1][i] < 1.5) and (wvl_offsets[1][i] > -1.5)]]
    # meas_lines = [[meas_lines[0][i] for i in range(len(wvl_offsets[0])) if wvl_offsets[0][i] > -1.5], [meas_lines[1][i] for i in range(len(wvl_offsets[1])) if wvl_offsets[1][i] > -1.5]]
    meas_errs = [[meas_errs[0][i] for i in range(len(wvl_offsets[0])) if (wvl_offsets[0][i] < 1.5) and (wvl_offsets[0][i] > -1.5)], [meas_errs[1][i] for i in range(len(wvl_offsets[1])) if (wvl_offsets[1][i] < 1.5) and (wvl_offsets[1][i] > -1.5)]]
    # meas_errs = [[meas_errs[0][i] for i in range(len(wvl_offsets[0])) if wvl_offsets[0][i] > -1.5], [meas_errs[1][i] for i in range(len(wvl_offsets[1])) if wvl_offsets[1][i] > -1.5]]
    meas_lines_blue = np.array(meas_lines[0])
    meas_lines_red = np.array(meas_lines[1])
    wvl_offsets = [[offset_blue for offset_blue in wvl_offsets[0] if offset_blue < 1.5], [offset_red for offset_red in wvl_offsets[1] if offset_red < 1.5]]
    wvl_offsets = [[offset_blue for offset_blue in wvl_offsets[0] if offset_blue > -1.5], [offset_red for offset_red in wvl_offsets[1] if offset_red > -1.5]]
    
    print('now with cut', wvl_offsets)
    plt.figure(figsize = (5,4))
    colors = ['b','r']

    #poly_order = int(iniconf["wvl corr"]["npoly_order"])
    poly_order = 1
    #wvtr = np.linspace(4100,6800,100)
    wvtr_blue = np.linspace(obswvl[0],obswvl[ichipgap],100)
    wvtr_red = np.linspace(obswvl[ichipgap],obswvl[-1],100)
    wvtr = [wvtr_blue, wvtr_red]
    obswvl_new = [[],[]]

    for j in range(len(meas_lines)):
        plt.errorbar(meas_lines[j], wvl_offsets[j],yerr = meas_errs[j],ls = "",color=colors[j],capsize = 3,marker = "o")
        if len(meas_lines[j]) ==  1:
            wvl_offset_func = np.poly1d(wvl_offsets[j])
        elif len(meas_lines[j]) == 0:
            wvl_offset_func = np.poly1d([0])
        else:
            param_use = np.polyfit(meas_lines[j],wvl_offsets[j],poly_order,w=np.reciprocal(meas_errs[j]))
            wvl_offset_func = np.poly1d(param_use)
        obswvl_new[j] = obswvl_split[j] - wvl_offset_func(obswvl_split[j]) 
        plt.plot(wvtr[j], wvl_offset_func(wvtr[j])) #is this going to stretch out the polynomial or something?
        
    plt.ylabel("Difference",fontsize = 12)
    plt.xlabel("Wavelength",fontsize = 12)
    #filename = iniconf["chisq fitting"]["output_path"] + "/" + specname + "_wvloffset_new.png"
    filename = outputname + "/" + specname + "_wvloffset_new.png"
    plt.savefig(filename)
    plt.close()

    # join blue and red sides of chipgap
    obswvl_new_full = np.hstack([obswvl_new[0], chipgap_region, obswvl_new[1]])
    # note: will need to make sure that it matches up with obsflux too
    # print(obswvl_new[0][-5:])
    # print(wvl_begin_gap, wvl_end_gap)
    # print(chipgap_region)
    # print(obswvl_new[1][:5])

    return obswvl_new_full

if __name__ == "__main__":
    obswvl = np.arange(4000,6000,10)
    obsflux = np.ones(len(obswvl))
    
    do_wvl_corr(np.array(obswvl),np.array(obsflux))

