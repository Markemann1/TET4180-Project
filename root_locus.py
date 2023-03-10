import dynamic as dps
import modal_analysis as dps_mdl
import plotting as dps_plt
import k2a_hygov_tgov as model_data
import importlib
import numpy as np
import matplotlib.pyplot as plt
import utility_functions as dps_uf


model = model_data.load()
if not model['gov_on']: model.pop('gov', None)
if not model['avr_on']: model.pop('avr', None)
if not model['pss_on']: model.pop('pss', None)
fig, ax = plt.subplots(1)

ps = dps.PowerSystemModel(model)
ps.power_flow()
ps.init_dyn_sim()


# Index to access relevant models, e.g. generators, avrs, govs etc.
# Without index, the change is applied to all instances of specified model, e.g. all 'GEN' generators, or all 'TGR' avrs

#index = dps_uf.lookup_strings('G1', ps.gen_mdls['GEN'].par['name'])  # Index for G1 and G3, GOV

#index = dps_uf.lookup_strings('G1', ps.gen_mdls['GEN'].par['name'])  # Index for G1, AVR
#index = dps_uf.lookup_strings('G3', ps.gen_mdls['GEN'].par['name'])  # Index for G3, AVR

#index = dps_uf.lookup_strings('G1', ps.gen_mdls['GEN'].par['name'])  # Index for G1, H
index = dps_uf.lookup_strings('G3', ps.gen_mdls['GEN'].par['name'])  # Index for G3, H

for i in range(30):
    ps.init_dyn_sim()
    print(index)
    ps.gov_mdls['TGOV1'].par['R'][index] = 0.1 + i*0.01   #change in TGOV1 in G3 or G4
    ps.gov_mdls['HYGOV'].par['R'][index] = 0.1 + i*0.01   #change in HYGOV in G1 or G2
    ps.avr_mdls['SEXS'].par['K'][index] = 10 + i*10       #change in SEXS in any G
    ps.gen_mdls['GEN'].par['H'][index] = 1 + i*0.3        #change in H in any G
    
    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    
    if i == 0:
        dps_plt.plot_eigs_2(ps_lin.eigs, fig, ax, col = [1, 0, 0], label = 'k={:.1f}'.format(0+i*5))
    else:
        dps_plt.plot_eigs_2(ps_lin.eigs, fig, ax, col = [i/40, i/50, i/30], label = 'k={:.1f}'.format(0+i*5))

    #pf, rev_Abs, rev_ang = ps_lin.pf_table()
    # Plot eigenvalues
    #dps_plt.plot_eigs(ps_lin.eigs)
    #sc = ax.scatter(ps_lin.eigs.real, ps_lin.eigs.imag)
    #ax.axvline(0, color='k', linewidth=0.5)
    #ax.axhline(0, color='k', linewidth=0.5)
    #ax.grid(True)
    #print(ps_lin.eigs)

plt.show()