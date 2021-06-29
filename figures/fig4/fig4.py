import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from matplotlib import rc
from matplotlib import rcParams
from matplotlib.patches import Patch
from os.path import abspath, dirname, join

# directories
WD = dirname(abspath(__file__))
RES_DIR = join(WD, *['..', '..', 'results'])
OUT_DIR = dirname(abspath(__file__))

# results
SVM_PATH = join(RES_DIR, 'sims_svm.csv')
KOM_PATH = join(RES_DIR, 'sims_kom.csv')
KCB_PATH = join(RES_DIR, 'sims_kcb.csv')
LEE_PATH = join(RES_DIR, 'sims_lee.csv')
CARD_PATH = join(RES_DIR, 'sims_card.csv')

# kernels and scenarios
kernels = ['linear', 'poly', 'rbf']
scenarios = ['G', 'cw']

# plot params
# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amsmath}')
rcParams["mathtext.fontset"] = 'cm'
constrained_layout = True # use tight layout if False

if __name__ == '__main__':
    # SVM data
    svm = pd.read_csv(SVM_PATH, index_col=[0,1,2,3])
    
    # hatches
    hatches = [None, '///', '...']
    
    ## Boxplots ##

    # data
    res_kom = pd.read_csv(KOM_PATH, index_col=[0,1,2])
    res_lee = pd.read_csv(LEE_PATH, index_col=[0,1,2])
    res_kcb = pd.read_csv(KCB_PATH, index_col=[0,1])
    res_card = pd.read_csv(CARD_PATH, index_col=[0,1,2])
    
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 4.333), sharey=False,
                            constrained_layout=constrained_layout)
    if not constrained_layout: fig.subplots_adjust(wspace=0.1, hspace=0.2)
    
    for i in range(2):
        # read in data and compute statistics
        scenario = scenarios[i]
        
        # true PATE
        tau = -0.4 if scenario == 'G' else 10
        
        # SVM data
        sub = svm.xs([scenario], level=['scenario'])
        
        # SVM solutions
        l_ind = -35 if scenario == 'G' else -25
        p_ind = -50 if scenario == 'G' else -35
        r_ind = -65 if scenario == 'G' else -50
        
        # SVM parameters
        C_l = np.unique(sub.xs(['linear'], level=['kernel']).index.droplevel(0))
        C_p = np.unique(sub.xs(['poly'], level=['kernel']).index.droplevel(0))
        C_r = np.unique(sub.xs(['rbf'], level=['kernel']).index.droplevel(0))
        
        # SVM estimates
        ate_l = sub.xs(['linear', C_l[l_ind]], level=['kernel', 'C']).ate.dropna()
        ate_p = sub.xs(['poly', C_p[p_ind]], level=['kernel', 'C']).ate.dropna()
        ate_r = sub.xs(['rbf', C_r[r_ind]], level=['kernel', 'C']).ate.dropna()
        
        # kom data
        ate_kom = res_kom.xs([scenario], level=['scenario']).ate.dropna()
        
        # kcb data
        ate_kcb = res_kcb.xs([scenario], level=['scenario']).ate.dropna()
        
        # cardmatch data
        ate_card = res_card.xs([scenario], level=["scenario"]).ate.dropna()
        
        # propensity score data
        ate_lgr_d = res_lee.xs([scenario, 'LGR'], 
                               level=['scenario', 'method']).ate
        ate_rfr_d = res_lee.xs([scenario, 'RFRST'], 
                               level=['scenario', 'method']).ate
        
        # initialize figure
        axs[i].axhline(y=tau, linestyle='--', color='#d62728', label='True', alpha=1, 
                       lw=1, zorder=9)
        
        # svm boxplots
        bp_svm = axs[i].boxplot(np.array([ate_l, ate_p, ate_r], dtype=object).T,
                            positions=[0.5, 1, 1.5],
                            medianprops={'color' : 'black', 'lw' : 2},
                            widths=0.35,
                            patch_artist=True, showfliers=False, zorder=10)
        
        for patch, hatch in zip(bp_svm['boxes'], hatches):
            patch.set_facecolor("white")
            patch.set(hatch=hatch)
            
        # kom boxplots
        bp_kom = axs[i].boxplot(np.array([
                                      ate_kom.xs('linear', level='kernel'), 
                                      ate_kom.xs('poly', level='kernel'),
                                      ate_kom.xs('rbf', level='kernel')], 
                                     dtype=object).T,
                            positions=[2.5, 3, 3.5],
                            medianprops={'color' : 'black', 'lw' : 2},
                            widths=0.35,
                            patch_artist=True, showfliers=False, zorder=10)
        
        for patch, hatch in zip(bp_kom['boxes'], hatches):
            patch.set_facecolor("white")
            patch.set(hatch=hatch)
            
        # kcb boxplots
        bp_kcb = axs[i].boxplot(ate_kcb,
                            positions=[5],
                            medianprops={'color' : 'black', 'lw' : 2},
                            widths=0.35,
                            patch_artist=True, showfliers=False, zorder=10)
        
        for patch in bp_kcb['boxes']:
            patch.set_facecolor("white")
            patch.set(hatch=hatches[-1])
            
        # cardmatch boxplots
        bp_card = axs[i].boxplot(np.array([
                                       ate_card.xs('linear', level='type'), 
                                       ate_card.xs('poly', level='type')], 
                                      dtype=object).T,
                             positions=[6.75, 7.25],
                             medianprops={'color' : 'black', 'lw' : 2},
                             widths=0.35,
                             patch_artist=True, showfliers=False, zorder=10)
        
        for patch, hatch in zip(bp_card['boxes'], hatches[:2]):
            patch.set_facecolor("white")
            patch.set(hatch=hatch)
        
        # propensity score boxplots
        bp_prop = axs[i].boxplot(np.array([
                                       ate_lgr_d, 
                                       ate_rfr_d], 
                                      dtype=object).T,
                             positions=[9, 11],
                             medianprops={'color' : 'black', 'lw' : 2},
                             widths=0.35,
                             patch_artist=True, showfliers=False, zorder=10)
        
        for patch in bp_prop['boxes']:
            patch.set_facecolor("white")
            patch.set(hatch=hatches[0])
        
        # labels
        axs[i].set_xticks([1, 3, 5, 7, 9, 11])
        if scenario != 'cw':
            axs[i].set_ylim(-0.57, -0.17)
        axs[i].set_xticklabels(['SVM', 'KOM', 'KCB', 'CARD', 'GLM', 'RFRST'])
        axs[i].tick_params(labelsize=10)
        if i == 0:
            axs[i].set_ylabel(r'ATE', fontsize=12)
        axs[i].set_xlabel(r'Method', fontsize=12)
        axs[i].grid(axis='y')
        
        # legend
        if i == 0:
            legend_elements = [
                               Patch(hatch=hatches[0], facecolor='white', 
                                     edgecolor='k', label='Linear'),
                               Patch(hatch=hatches[1], facecolor='white', 
                                     edgecolor='k', label='Polynomial'),
                               Patch(hatch=hatches[2], facecolor='white', 
                                     edgecolor='k', label='RBF')]
            axs[i].legend(handles=legend_elements, loc='upper left',
                          framealpha=1, prop={"size": 8})
        
        # layout
        axs[i].set_aspect(1 / axs[i].get_data_ratio())    
    
    # layout and save
    fig.align_ylabels()
    if not constrained_layout: fig.tight_layout()
    plt.savefig(join(OUT_DIR,'fig4.pdf'), dpi=200, bbox_inches='tight')
    