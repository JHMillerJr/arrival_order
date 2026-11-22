#> name: subhalo_radius_test.py
#> author: John Miller Jr
#> descrp: tests to see how far away from the einstein radius subhalos need to be generated

""" #> IMPORTS =======================
================================== """

#> imports
import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy

#> adding opus path
sys.path.insert(0, '../opus')

#> modules
from modules.units import u; u=u()
import modules.error as error
import generate
import params
import order
import plot


""" #> FOLDER ========================
================================== """

#> checks to see if data already exists
def check_folder(outFolder):
    
    #> declaration
    path = generate.dataDir + outFolder
    
    #> if folder/data does not exist
    if not (os.path.isdir(path) and os.listdir(path)): return True
    
    #> if it does & do not want to override
    if input(f'> data already exists in {path}, override it? [y/n] ').strip().lower() != 'y': return False
    
    #> if it does & want to override
    shutil.rmtree(path)
    return True


""" #> INCREMENT =====================
================================== """

#> keeps subhalos inside the diameter [ER] & redoes the mass sheet for that cone
def parse_subhalos(bprofiles, diameter, max_ER):
    
    #> iterating thru galaxies
    for b in bprofiles:
        
        #> declarations
        subh = b['subh'][0]
        tnfw = b['tnfw'][0]
        ER   = subh['cone_opening_angle_arcsec'] / max_ER   # arcsec
        R    = 0.5 * diameter * ER                          # arcsec
        
        #> distance from the host center
        r = np.hypot(tnfw['x0'] - subh['x0'],
                     tnfw['y0'] - subh['y0'])               # arcsec
        
        #> keeping subhalos inside R
        mask = r <= R
        for key in tnfw.keys():
            tnfw[key] = tnfw[key][mask]
    
    return bprofiles


#> main increment function
def func(numGals, numSource, **kwargs):
    
    #> declarations
    min_ER = 2
    max_ER = 12
    paramRanges = params.paramRanges()
    
    #> run folder inside ./data/
    outFolder = kwargs.pop('outFolder', '+unsorted')
    if not check_folder(outFolder): return
    
    #> realizations >= 10: random sources at every window (not the max ER sources)
    rand_srcs = int(outFolder.rstrip('/').split('_')[-1]) >= 10
    
    #>>># GETTING ER #<<<#
    
    #> einstein radius of the galaxy (no subhalos)
    er_out = generate.gen(numGals=numGals, numSource=1,
                          paramRanges=paramRanges,
                          priors=False, mult=[], ex=False, 
                          subh=False, ER=True, **kwargs)
    ER     = er_out['ER'][0,1] # einstein radius in arcsec
        
    
    #>>># LARGEST WINDOW FIRST #<<<#
    paramRanges['subh'][0]['cone_opening_angle_arcsec'] = max_ER * ER        # arcsec
    
    #> max ER diameter first
    max_ER_out = generate.gen(numGals=numGals, numSource=numSource,
                              paramRanges=paramRanges,
                              priors=False, mult=[], ex=False, 
                              subh=True, verbose=True, **kwargs)
    
    generate.save(max_ER_out, folder=outFolder, suffix=f'subhalo_diameter_{max_ER}ER', paramRanges=paramRanges)
    
    #> iterating through each point
    for i in range(min_ER, max_ER):
        
        #> copying max ER bprofiles
        bprofiles = deepcopy(max_ER_out['bprofiles'])
        
        bprofiles = parse_subhalos(bprofiles, i, max_ER)
        bprofiles[0].pop('subh')
        
        #> sources: max ER ones, or random (new seed per window, so they differ)
        srcs = max_ER_out['srcs']
        seed = kwargs['seed']
        if rand_srcs: srcs = None
        if rand_srcs: seed = kwargs['seed'] + 1000 * i
        
        #> lensing!
        out = generate.gen(numGals=numGals, numSource=numSource,
                           sources=srcs,
                           bprofiles=bprofiles,
                           paramRanges=paramRanges,
                           priors=False, mult=[], ex=False, 
                           subh=True, **(kwargs | {'seed': seed}))
        
        #> quads in both runs
        quad    = (out['nims'] == 5) & (max_ER_out['nims'] == 5)
        ims_i   = out['images'][quad][:,:5,:]                      # (Nq, 5, 2) arcsec
        ims_max = max_ER_out['images'][quad][:,:5,:]               # (Nq, 5, 2) arcsec
        
        #> distance from each image to every image of the max ER run
        dx = ims_i[:,:,None,0] - ims_max[:,None,:,0]
        dy = ims_i[:,:,None,1] - ims_max[:,None,:,1]
        
        #> matched image in the max ER run (index of the closest)
        jm   = np.hypot(dx, dy).argmin(axis=2)                     # (Nq, 5)
        rows = np.arange(len(jm))[:,None]                          # (Nq, 1)
        
        #> vector shift of the 4 bright images [arcsec]
        dvec = (ims_i - ims_max[rows, jm])[:,:4,:]                 # (Nq, 4, 2)
        
        #> removing the common shift
        drel = dvec - dvec.mean(axis=1, keepdims=True)
        
        #> calculating image difference
        image_diff = np.hypot(drel[:,:,0], drel[:,:,1])
        
        #> saving: ./data/<outFolder>/<time>_<bin>/
        parent_dir = generate.save(out, folder=outFolder, suffix=f'subhalo_diameter_{i}ER', paramRanges=paramRanges)
                
        np.save(parent_dir+f'image_diffs_{i}ER', image_diff)
        
    return


""" #> ANALYZE =======================
================================== """

#> loads every image_diffs.npy in outFolder, keyed by the cone diameter [ER]
def load_image_diffs(outFolder):
    
    #> imports
    import re
    import glob
    
    #> declarations
    path  = generate.dataDir + outFolder           # ./data/<outFolder>/
    files = glob.glob(path + '*/image_diffs*.npy')  # one per run folder
    
    #> loading
    diffs = {}
    for f in files:
        i        = int(re.search(r'image_diffs_(\d+)ER', f).group(1)) # diameter [ER]
        diffs[i] = np.load(f)                                         # (Nq, 4) arcsec
    
    #> sorted by diameter
    return dict(sorted(diffs.items()))


#> loads nims & images of every run folder in outFolder, keyed by the cone diameter [ER]
def load_runs(outFolder):
    
    #> imports
    import re
    import glob
    
    #> declarations
    path = generate.dataDir + outFolder                  # ./data/<outFolder>/
    dirs = glob.glob(path + '*subhalo_diameter_*ER/')    # one per run folder
    
    #> loading
    runs = {}
    for d in dirs:
        i    = int(re.search(r'subhalo_diameter_(\d+)ER', d).group(1)) # diameter [ER]
        name = os.path.basename(os.path.normpath(d))                    # run folder name
        runs[i] = {'nims':   np.load(os.path.join(d, name + '_nims.npy')),     # (N,)
                   'images': np.load(os.path.join(d, name + '_images.npy')),   # (N, w, 2)
                   'srcs':   np.load(os.path.join(d, name + '_srcs.npy')),
                   'bprofiles': np.load(os.path.join(d, name + '_bprofiles.npy'), allow_pickle=True)}
    
    #> sorted by diameter
    return dict(sorted(runs.items()))


#> checks how the number of quads, image positions & arrival order change with subhalo window size
def analyze(outFolder, numSource, start=0, end=9):
    
    #>>># LOADING #<<<#
    
    #> declarations
    n_max = 10
    num_real = end - start + 1 # number of realizations (start to end, inclusive)
    
    #> iterating thru realizations
    non_quads, med_diffs, mismatches, nims = [], [], [], []
    for i in range(start, end+1):
        
        #> declarations
        folder = outFolder + f'realization_{i}/'
        
        #> loading runs & image_diff files from outFolder
        runs  = load_runs(folder)
        diffs = load_image_diffs(folder)
        
        #> iterating thru ER windows: non-quads & morph vs fermat mismatches
        non_quad, mismatch, nim = [], [], []
        for key in runs.keys():
            nim.append([np.sum(runs[key]['nims'] == n) for n in range(n_max+1)]) # sources w/ n images
            quad = runs[key]['nims'] == 5                    # (N,)
            ims  = runs[key]['images'][quad][:,:5,:]         # (Nq, 5, 2) arcsec (fermat order)
            non_quad.append(np.sum(~quad))
            if quad.any(): mismatch.append(np.sum(order.morph_vs_fermat(ims)[1] == 0))
            else:          mismatch.append(0)
        
        #> iterating thru ER windows
        med_diff = []
        for key in diffs.keys():
            med_diff.append(np.median(diffs[key]))
        
        #> saving realization
        non_quads.append(non_quad)
        mismatches.append(mismatch)
        med_diffs.append(med_diff)
        nims.append(nim)
    
    #> arrays (num_real, number of windows)
    ER_runs    = np.array(list(runs.keys()))      # diameters w/ a run [ER]
    ER_diffs   = np.array(list(diffs.keys()))     # diameters w/ an image difference [ER]
    non_quads  = np.array(non_quads)              # sources that are not quads
    mismatches = np.array(mismatches)             # quads w/ morph order != fermat order
    med_diffs  = np.array(med_diffs) * 1e3        # arcsec --> mas
    nims       = np.array(nims)                   # (num_real, number of windows, n_max+1) sources w/ n images
    
    #>>># PLOTTING #<<<#
    
    #> initializing plot
    plot.setStyle()
    fig, axes = plt.subplots(1,3,figsize=(18,6))
    
    #> labels
    ylabels = ['% Non-Quads', 'Median Image Difference [mas]', '% Order Mismatch']
    
    #> one color per realization
    colors = plt.cm.tab10(np.arange(num_real) % 10)
    
    #> plotting!
    for i, ax in enumerate(axes.flat):
        
        #> initializing
        plot.styleAx(ax, xlabel='Subhalo Window [ER]', ylabel=ylabels[i])
        
        #> non-quads
        if i == 0:
            nq_perc = non_quads / numSource * 100 # % of sources
            for k in range(num_real):
                ax.plot(ER_runs, nq_perc[k], c=colors[k], alpha=0.8, lw=1.5)
            mean0, = ax.plot(ER_runs, nq_perc.mean(axis=0), marker='o', c='k', lw=2, label='all (mean)')
            
            #> non-quads by number of images
            lines0 = [mean0]                                           # legend handles
            styles = iter(['--', ':', '-.', (0, (5, 1)), (0, (1, 3))]) # line style per number of images
            for n in range(n_max+1):
                if n == 5 or not nims[:,:,n].any() or n in [4,6]: continue
                line, = ax.plot(ER_runs, nims[:,:,n].mean(axis=0) / numSource * 100, marker='s', ms=4, c='k', lw=1, ls=next(styles), label=f'{n} images (mean)')
                lines0.append(line)
            ax.legend(handles=lines0, fontsize=12)
        
        #> median image difference
        if i == 1:
            for k in range(num_real):
                ax.plot(ER_diffs, med_diffs[k], c=colors[k], alpha=0.8, lw=1.5, label=f'r{start+k}')
            med1, = ax.plot(ER_diffs, np.median(med_diffs, axis=0), marker='o', c='k', lw=2, label='median')
            ax.legend(handles=[med1], fontsize=12)
        
        #> mismatches
        if i == 2:
            mm_perc = mismatches / (numSource - non_quads) * 100 # % of quads
            for k in range(num_real):
                ax.plot(ER_runs, mm_perc[k], c=colors[k], alpha=0.8, lw=1.5)
            mean2, = ax.plot(ER_runs, mm_perc.mean(axis=0), marker='o', c='k', lw=2, label='mean')
            ax.legend(handles=[mean2], fontsize=12)
    
    #> realization legend (shared, right of the plots)
    handles, legend_labels = axes.flat[1].get_legend_handles_labels()
    fig.legend(handles[:num_real], legend_labels[:num_real], loc='center left', bbox_to_anchor=(0.91, 0.5), fontsize=12)
    
    #> plotting, saving, etc.
    plot.finish(fig, outFile=f'subhalo_radius_test_r{start}-{end}', loc='../arrival_order/figures/')
        
    return


#> plotting sourcs
def plot_quad_sources(outFolder, start=0, end=9):
    
    #> imports
    import order
    
    #> iterating thru realizations (start to end, inclusive)
    for i in range(start, end+1):
        
        #> declarations
        folder = outFolder + f'realization_{i}/'
        
        #> loading max run
        runs  = load_runs(folder)
        b     = runs[12]
        
        #> initializing plot
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        plot.styleAx(ax, title=f'Realization {i}', sky=True)
        
        #> quads only (first 5 entries are the images, rest is nan padding)
        quad = b['nims'] == 5                                   # (N,)
        ims  = b['images'][quad][:,:5,:]                        # (Nq, 5, 2) arcsec (fermat order)
        quad_srcs = b['srcs'][quad]                                  # (Nq, 2) arcsec

        #> getting mismatch (1 = morph order matches fermat order)
        order_match = order.morph_vs_fermat(ims)[1]             # (Nq,)
        
        #> getting mismatches
        mask = (order_match == 0.0)
        match_sources    = quad_srcs[~mask]
        mismatch_sources = quad_srcs[mask]
        
        #> plotting!
        if len(match_sources) != 0: 
            ax.scatter(match_sources[:,0], match_sources[:,1], c='0.8', alpha=0.1, s=10)
        if len(mismatch_sources) != 0:
            ax.scatter(mismatch_sources[:,0], mismatch_sources[:,1], c='r', alpha=0.3, s=10)

        ax.scatter([], [], c='0.8', label='m')
        ax.scatter([], [], c='r', label='mm')
        ax.legend(fontsize=12)
        
        #> % mismatch of the quads (bottom right)
        perc = mask.sum() / len(mask) * 100 # %
        ax.annotate(f'%mm = {perc:.2f}', xy=(0.97, 0.03), xycoords='axes fraction', ha='right', va='bottom', fontsize=12)
        plot.finish(fig, outFile=f'realization_{i}', 
                         loc=f'../arrival_order/figures/subhalo_radius_test/')
    
    return

#> plotting sources + caustics (left) & subhalos (right) at every window for a given realization
def plot_inc(outFolder, realization=0):
    
    #> imports
    from matplotlib.lines import Line2D
    
    #> declarations
    folder = outFolder + f'realization_{realization}/'
    max_ER = 12 
    nph    = 50
    
    #> loading runs
    runs = load_runs(folder)
    
    #> r = 12 run
    b_12  = runs[max_ER]['bprofiles'][0]
    mbmin = np.min(b_12['tnfw'][0]['m_bound'])
    mbmax = np.max(b_12['tnfw'][0]['m_bound'])
    
    #> getting ER
    ER = b_12['subh'][0]['cone_opening_angle_arcsec'] / max_ER 
    
    #> half widths of every plot
    lim_src  = 1.2 * np.nanmax(np.abs(runs[max_ER]['srcs']))
    lim_subh = 1.05 * 0.5 * max_ER * ER
    
    #> circles
    theta = np.linspace(0, 2*np.pi, 1000) # rad
    
    #> titles
    titles = ['Sources + Caustics', 'Subhalos']
    
    #> figure folder (made if needed)
    fig_dir = f'../arrival_order/figures/subhalo_radius_test/realization_{realization}/'
    os.makedirs(fig_dir, exist_ok=True)
    
    #> iterating thru windows
    for key in runs.keys():
        
        #> declarations
        run   = runs[key]
        bprof = run['bprofiles'][0]
        tnfw  = bprof['tnfw'][0]
        
        #> initializing plot
        fig, axes = plt.subplots(1,2,figsize=(12,6))
        fig.suptitle(f'Realization {realization}, R={key}*ER', fontsize=15, fontweight='bold')
        
        #> plotting!
        for i, ax in enumerate(axes.flat):
            
            #> initializing
            plot.styleAx(ax, title=titles[i], sky=True)
            
            #> sources + caustics
            if i == 0:
                
                #> quads only
                quad   = run['nims'] == 5
                double = run['nims'] == 3
                seven  = run['nims'] == 7
                ims    = run['images'][quad][:,:5,:]
                quad_srcs   = run['srcs'][quad]
                double_srcs = run['srcs'][double]
                seven_srcs  = run['srcs'][seven]
                
                #> getting mismatch
                order_match = order.morph_vs_fermat(ims)[1]
                
                #> getting mismatches
                mask = (order_match == 0.0)
                match_sources    = quad_srcs[~mask]
                mismatch_sources = quad_srcs[mask]
                
                #> plotting sources
                if len(match_sources) != 0: 
                    ax.scatter(match_sources[:,0], match_sources[:,1], c='0.8', alpha=0.1, s=10)
                if len(mismatch_sources) != 0:
                    ax.scatter(mismatch_sources[:,0], mismatch_sources[:,1], c='r', alpha=0.3, s=10)
                if len(double_srcs) != 0:
                    ax.scatter(double_srcs[:,0], double_srcs[:,1], c='b', alpha=0.3, s=10)
                if len(seven_srcs) != 0:
                    ax.scatter(seven_srcs[:,0], seven_srcs[:,1], c='tab:green', alpha=1.0, s=10)
                
                #> caustics of this window (tangential, radial)
                maps = plot.lensMaps(bprof, nph)
                _, (tcaus, rcaus) = plot.critCaustics(maps, bprof)
                for line in tcaus: ax.plot(line[:,0], line[:,1], c='k', lw=1)
                for line in rcaus: ax.plot(line[:,0], line[:,1], c='k', lw=1, ls='--')
                
                #> legend
                ax.scatter([], [], c='0.8', label='m')
                ax.scatter([], [], c='r', label='mm')
                if len(double_srcs) != 0: ax.scatter([], [], c='b', label='double')
                if len(seven_srcs) != 0: ax.scatter([], [], c='tab:green', label='seven')
                ax.legend(fontsize=12, loc='upper right')
                
                #> % mismatch of the quads
                perc = mask.sum() / len(mask) * 100 # %
                
                #> % mismatch (bottom right)
                ax.annotate(f'%mm = {perc:.1f}', xy=(0.97, 0.03), xycoords='axes fraction', ha='right', va='bottom', fontsize=12)
                
                #> same window for every plot (RA increases to the left)
                ax.set_xlim(lim_src, -lim_src)
                ax.set_ylim(-lim_src, lim_src)
                ax.set_aspect('equal')
            
            #> subhalos
            if i == 1:
                
                #> getting sizes
                s = (tnfw['m_bound'] - mbmin) / (10**10 - mbmin) # []
                
                #> plotting subhalos
                ax.scatter(tnfw['x0'], tnfw['y0'], s=s*10000, c='k', alpha=0.5)
                ax.scatter([], [], label=f"#={len(tnfw['x0'])}", c='w', s=0)
                
                #> einstein radius & window
                R = 0.5 * key * ER # arcsec
                ax.plot(ER*np.cos(theta), ER*np.sin(theta), c='r', lw=3)
                ax.plot(R*np.cos(theta), R*np.sin(theta), c='b', ls='--', lw=2)
                ax.legend(loc='upper right')
                
                #> same window for every plot
                ax.set_xlim(lim_subh, -lim_subh)
                ax.set_ylim(-lim_subh, lim_subh)
                ax.set_aspect('equal')
        
        #>>># SIZE LEGEND #<<<#
        
        #> marker for log10 m = 6, 7, 8, 9 (same scaling as the subhalos)
        handles = []
        for lm in [6, 7, 8, 9]: # log10(solMass)
            s_lm = (10**lm - mbmin) / (10**10 - mbmin) * 10000 # pt^2 (scatter area)
            ms   = np.sqrt(s_lm)                               # pt (marker width)
            handles.append(Line2D([], [], ls='', marker='o', ms=ms, c='k', alpha=0.5, label=rf'$10^{{{lm:.0f}}}$'))
        
        #> log mass range (legend title)
        title = (r'$m_{bound}$ [M$_\odot$]' + '\n' +
                 r'$\log m_{min}$ = ' + f'{np.log10(mbmin):.2f}' + '\n' +
                 r'$\log m_{max}$ = ' + f'{np.log10(mbmax):.2f}')
        
        #> legend (white space right of the figure)
        fig.legend(handles=handles, title=title, loc='center left', bbox_to_anchor=(0.91, 0.5), 
                   fontsize=12, title_fontsize=12, labelspacing=1.5, frameon=False)
        
        #> plotting, saving, etc. (./figures/subhalo_radius_test/realization_<i>/)
        plot.finish(fig, outFile=f'realization_{realization}_ER{key}', loc=fig_dir)
    
    return


#> plotting sourcs
def plot_all_subhalos(outFolder, start=0, end=9):
    
    #> iterating thru realizations (start to end, inclusive)
    for i in range(start, end+1):
        
        #> declarations
        folder = outFolder + f'realization_{i}/'
        
        #> loading max run
        runs  = load_runs(folder)
        b     = runs[12]['bprofiles'][0]['tnfw'][0]
        
        #> getting sizes
        mbmin = np.min(b['m_bound'])
        mbmax = np.max(b['m_bound'])
        s = np.array([ (mb - mbmin) / (mbmax-mbmin) for mb in b['m_bound'] ])
        
        #> getting ER
        subh_12 = runs[12]['bprofiles'][0]['subh'][0]
        ER = subh_12['cone_opening_angle_arcsec'] / 12   # arcsec
        
        #> initializing plot
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        plot.styleAx(ax, title=f'Realization {i}', sky=True)
        
        ax.scatter(b['x0'], b['y0'], s=s*1000, c='k', alpha=0.5)
        ax.scatter([],[],label=f'#={len(b['x0'])}', c='w',s=0)
        
        theta = np.linspace(0, 2*np.pi, 1000)
        ax.plot(ER*np.cos(theta), ER*np.sin(theta), c='r', lw=3)
        
        ax.legend()
        
        #> % mismatch of the quads (bottom right)
        plot.finish(fig, outFile=f'realization_{i}_subhalos', 
                         loc=f'../arrival_order/figures/subhalo_radius_test/')
    
    return


""" #> CONSISTENCY CHECKS ============
================================== """

#> checking sources
def sources(outFolder, start=0, end=9):
    
    #> declarations
    num_radii = 11
    num_real = end - start + 1 # number of realizations (start to end, inclusive)
    
    #> iterating thru each realization
    srcs_diff = np.zeros(shape=(num_real, num_radii, 1))
    for i in range(start, end+1):
    
        #> declarations
        folder = outFolder + f'realization_{i}/'
        
        #> loading runs from realization
        runs  = load_runs(folder)
        
        #> comparing sources
        dummy = np.zeros(shape=(len(runs.keys()), 1))
        for j, key in enumerate(runs.keys()):
            # runs[key]['srcs'][0] = (0.0, 0.0) # to test if it is working
            dummy[j] = ((runs[12]['srcs'] - runs[key]['srcs']) != 0.0).any()
        
        srcs_diff[i-start] = dummy
        
    #> checking if any != 0
    for i, real in enumerate(srcs_diff):
        if real.any():
            nonzero = np.nonzero(real)
            error.highlight(f'For realization {start+i} the radii {nonzero} had different sources somewhere!')
    
    print('> If there is no error message, then everything is good! Yippee!')
    
    return

#> checks to make sure subhalos are the same from run-to-run (minus larger radii ones)
def subhalos(outFolder, start=0, end=9):
    
    #> declarations
    num_radii = 11
    num_real = end - start + 1 # number of realizations (start to end, inclusive)
    tnfw_keys = ['x0', 'y0', 'm_bound', 'rs']
    
    #> iterating thru each realization
    subh_diff = np.zeros(shape=(num_real, num_radii, len(tnfw_keys)), dtype=bool)
    for i in range(start, end+1):
    
        #> declarations
        folder = outFolder + f'realization_{i}/'
        
        #> loading runs from realization
        runs  = load_runs(folder)
        radii = list(runs.keys()) # diameters [ER]
        
        #> to test if it is working
        # if i == 3:
        #     t = runs[6]['bprofiles'][0]['tnfw'][0]
        #     for key_ in t.keys(): t[key_] = t[key_][1:]
        # if i == 7: runs[9]['bprofiles'][0]['tnfw'][0]['rs'][0] *= 1.001
        
        #> getting max r=12 data & ER
        tnfw_12 = runs[12]['bprofiles'][0]['tnfw'][0]
        subh_12 = runs[12]['bprofiles'][0]['subh'][0]
        ER = subh_12['cone_opening_angle_arcsec'] / 12   # arcsec
        
        #> comparing subhalos
        for j, key in enumerate(radii):
            
            c12 = deepcopy(tnfw_12)
            
            #> current data
            d = runs[key]['bprofiles'][0]['tnfw'][0]
            
            #>>># CUTTING R=12 SUBHALOS #<<<#
            
            #> parsing 
            R    = 0.5 * int(key) * ER
            
            #> distance from the host center
            r = np.hypot(c12['x0'] - subh_12['x0'],  
                         c12['y0'] - subh_12['y0'])               # arcsec
            
            #> keeping subhalos inside R
            mask = r <= R
            for key_ in c12.keys():
                c12[key_] = c12[key_][mask]
                
            
            #>>># COMPARING #<<<#
            
            #> true where a param differs (or the number of subhalos does)
            for k, tnfw_key in enumerate(tnfw_keys):
                subh_diff[i-start,j,k] = not np.array_equal(c12[tnfw_key], d[tnfw_key])
        
    #> checking if any differ
    for i, real in enumerate(subh_diff):
        if real.any():
            for k, tnfw_key in enumerate(tnfw_keys):
                bad = [radii[j] for j in np.nonzero(real[:,k])[0]] # diameters [ER]
                if len(bad) > 0: error.highlight(f'For realization {start+i} the radii {bad} ER had different {tnfw_key} somewhere!')
        
    print('> If there is no error message, then everything is good! Yippee!')
    
    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> outfolder (./data/<outFolder>/)
    
    
    #> declarations
    numGals   = 1
    numSource = 10000
    redshifts = (0.5, 1.0)
    start, end = 0, 9 # realizations
    
    outFolder = '../../arrival_order/data/subhalo_radius_test/'
    
    #> iterating through subhalo realizations
    for i in range(start, end+1):
        
        folder = outFolder + f'realization_{i}'
        
        print(folder)
    
        #> kwargs
        kwargs = {'method': 'lpot',
                  'seed': 42 + i,
                  'nfw_priors': False,
                  'hern_priors': False,
                  'outFolder': folder,
                  'redshifts': redshifts,
                  'order_comp': True}
        
        plot_inc(outFolder, realization=i)
        
        #> incrementing
        # func(numGals, numSource, **kwargs)
        
    analyze(outFolder, numSource, start, end)
    #plot_quad_sources(outFolder, start, end)
    
    #plot_all_subhalos(outFolder, start, end)
    
    #> consistency checks
    # sources(outFolder, start, end)
    # subhalos(outFolder, start, end)

    # end
# thank