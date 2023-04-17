import matplotlib.pyplot as plt
import dynpssimpy.dynamic as dps
import dynpssimpy.modal_analysis as dps_mdl
import dynpssimpy.plotting as dps_plt
import importlib
importlib.reload(dps)

# Load model
import dynpssimpy.ps_models.k2a_base_case_with_AVRs_and_GOVs as model_data
model = model_data.load()

# Power system model
ps = dps.PowerSystemModel(model=model)
# Power flow calculation
ps.power_flow()
# Initialization
ps.init_dyn_sim()

# Plot eigenvalues in C)
# Sensitivity analysis)
ps.init_dyn_sim()
fig, ax = plt.subplots(1)
for i in range(40):
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # First - increase 'H' by 0.25 each iteration
    # Change H in G3:
    # model['generators']['GEN'][1][6] = 1 + i * 0.25

    # Second - increase gain (K) at AVR in G3:
    model['avr']['SEXS'][3][2] = 1 + i * 20
    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    # Plot eigenvalues
    dps_plt.plot_eigs_sensitivity(ps_lin.eigs, fig, ax, col=[0.1 , i / 40 , 0.8], label='k={:.1f}'.format(0 + i * 5))
    # plt.axis([-1.5, 0.5, -0.2, 8.5])
plt.show()