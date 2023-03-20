
import dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import utility_functions as dps_uf

np.set_printoptions(precision=3)


if __name__ == '__main__':

    # Load model
    #
    from dynpssimpy.ps_models import k2a as model_data

    #
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # ps.use_numba = True
    ps.power_flow()
    ps.init_dyn_sim()
    #
    #ps.build_y_bus_red(ps.buses['name'])
    ps.build_y_bus_red(['B2'])
    # ps.reduced_bus_idx
    np.max(ps.ode_fun(0.0, ps.x0))
    t_end = 5

    # Add small perturbation to initial angle of first generator
    x0 = ps.x0.copy()
    #
    #x0[ps.gen_mdls['GEN'].state_idx['angle'][0]] += 1
    #
    # Solver
    sol = dps_uf.ModifiedEuler(ps.ode_fun, 0, x0, t_end, max_step=5e-3)
    #
    # Initialize simulation
    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()
    #
    #
    print('Compute active power: ')
    print('index B1, B2 and B3 =', ps.get_bus_idx(['B1', 'B2', 'B3']))
    print('index B1, B2 and B3 (red =)', ps.get_bus_idx_red(['B1', 'B2', 'B3']))
    from_bus=input('from_bus:')
    from_i=int(from_bus)-1
    to_bus=input('to bus:')
    to_j=int(to_bus)-1
    print('Power will be computed from node:',from_i+1, 'to node:',to_j+1)
    P_e_stored=[]
    P_m_stored=[]
    Pflow_stored=[]
    Qflow_stored=[]
    # print('Y11=',ps.y_bus_red_mod(1,1),'Y12=',ps.y_bus_red_mod(1,2))
    print('Admittance between',from_i+1,'and',to_j+1,'is', -ps.y_bus[from_i, to_j])
    print('state description: ',ps.state_desc)
    print('x0 = ', ps.x0)
    print('----------------------------------')
    
    v_bus_full = ps.red_to_full.dot(ps.v_red)
    v_bus_mag = np.abs(v_bus_full)
    v_bus_ang = np.angle(v_bus_full)
    
    print('V1=', "{:.2f}".format(v_bus_mag[0]*20), ' angle = ', "{:.2f}".format(v_bus_ang[0]*(180/3.14)))
    print('V2=', "{:.2f}".format(v_bus_mag[1]*20), ' angle = ', "{:.2f}".format(v_bus_ang[1]*(180/3.14)))
    print('V3=', "{:.2f}".format(v_bus_mag[2]*20), ' angle = ', "{:.2f}".format(v_bus_ang[2]*(180/3.14)))
    print('V4=', "{:.2f}".format(v_bus_mag[3]*20), ' angle = ', "{:.2f}".format(v_bus_ang[3]*(180/3.14)))
    print('----------------------------------')
    
    v_bus_full_conj = np.conj(v_bus_full)
    
    Sflow15 = ps.y_bus[0, 4] * (v_bus_full[4] - v_bus_full[0]) * v_bus_full_conj[0]
    Pflow15 = Sflow15.real
    Qflow15 = -Sflow15.imag
    
    Sflow26 = ps.y_bus[1, 5] * (v_bus_full[5] - v_bus_full[1]) * v_bus_full_conj[1]
    Pflow26 = Sflow26.real
    Qflow26 = -Sflow26.imag
    
    Sflow67 = ps.y_bus[5, 6] * (v_bus_full[6] - v_bus_full[5]) * v_bus_full_conj[5]
    Pflow67 = Sflow67.real
    Qflow67 = -Sflow67.imag
    
    Sflow56 = ps.y_bus[4, 5] * (v_bus_full[5] - v_bus_full[4]) * v_bus_full_conj[4]
    Pflow56 = Sflow56.real
    Qflow56 = -Sflow56.imag
    
    Sflow78 = ps.y_bus[6, 7] * (v_bus_full[7] - v_bus_full[6]) * v_bus_full_conj[6]
    Pflow78 = Sflow78.real
    Qflow78 = -Sflow78.imag
    
    Sflow311 = ps.y_bus[2, 10] * (v_bus_full[10] - v_bus_full[2]) * v_bus_full_conj[2]
    Pflow311 = Sflow311.real
    Qflow311 = -Sflow311.imag
    
    Sflow410 = ps.y_bus[3, 9] * (v_bus_full[9] - v_bus_full[3]) * v_bus_full_conj[3]
    Pflow410 = Sflow410.real
    Qflow410 = -Sflow410.imag

    Sflow109 = ps.y_bus[9, 8] * (v_bus_full[8] - v_bus_full[9]) * v_bus_full_conj[9]
    Pflow109 = Sflow109.real
    Qflow109 = -Sflow109.imag

    Sflow1110 = ps.y_bus[10, 9] * (v_bus_full[9] - v_bus_full[10]) * v_bus_full_conj[10]
    Pflow1110 = Sflow1110.real
    Qflow1110 = -Sflow1110.imag     
    
    Sflow98 = ps.y_bus[8, 7] * (v_bus_full[7] - v_bus_full[8]) * v_bus_full_conj[8]
    Pflow98 = Sflow98.real
    Qflow98 = -Sflow98.imag
    

    print('Ybus =', ps.y_bus)
    print('----------------------------------')
    
    #print('Ybus15 =',ps.y_bus[0,4])
    print('P from 1 to 5 = ', "{:.2f}".format(Pflow15*1000), ' MW')
    print('Q from 1 to 5 = ', "{:.2f}".format(Qflow15*1000), ' MVAr')
    print('----------------------------------')
    
    #print('Ybus26 =',ps.y_bus[1,5])
    print('P from 2 to 6 =', "{:.2f}".format(Pflow26*1000), ' MW')
    print('Q from 2 to 6 = ',"{:.2f}".format(Qflow26*1000), ' MW')
    print('----------------------------------')
    
    #print('Ybus67 =',ps.y_bus[5,6])
    print('P from 6 to 7 =', "{:.2f}".format(Pflow67*1000), ' MW')
    print('Q from 6 to 7 = ',"{:.2f}".format(Qflow67*1000), ' MW')
    print('----------------------------------')
    
    #print('Ybus65 =',ps.y_bus[5,4])
    print('P from 5 to 6 =', "{:.2f}".format(Pflow56*1000), ' MW')
    print('Q from 5 to 6= ',"{:.2f}".format(Qflow56*1000), ' MW')
    print('----------------------------------')
    
    #print('Ybus78 =',ps.y_bus[6,7])
    print('P from 7 to 8 =', "{:.2f}".format(Pflow78*1000), ' MW')
    print('Q from 7 to 8 = ',"{:.2f}".format(Qflow78*1000), ' MW')
    print('----------------------------------')
    
    #print('Ybus311 =',ps.y_bus[2,10])
    print('P from 3 to 11 =', "{:.2f}".format(Pflow311*1000), ' MW')
    print('Q from 3 to 11 = ',"{:.2f}".format(Qflow311*1000), ' MW')
    print('----------------------------------')
    
    #print('Ybus410 =',ps.y_bus[3,9])
    print('P from 4 to 10 =', "{:.2f}".format(Pflow410*1000), ' MW')
    print('Q from 4 to 10 = ', "{:.2f}".format(Qflow410*1000), ' MW')
    print('----------------------------------')
    
    #print('Ybus109 =',ps.y_bus[9,8])
    print('P from 10 to 9 =', "{:.2f}".format(Pflow109*1000), ' MW')
    print('Q from 10 to 9 = ', "{:.2f}".format(Qflow109*1000), ' MW')
    print('----------------------------------')
    
    #print('Ybus1011 =',ps.y_bus[9,10])
    print('P from 11 to 10 =', "{:.2f}".format(Pflow1110*1000), ' MW')
    print('Q from 11 to 10 = ', "{:.2f}".format(Qflow1110*1000), ' MW')
    print('----------------------------------')
    
    #print('Ybus119 =',ps.y_bus[10,8])
    print('P from 8 to 9 =', "{:.2f}".format(Pflow98*1000), ' MW')
    print('Q from 8 to 9 = ', "{:.2f}".format(Qflow98*1000), ' MW')
    print('----------------------------------')
    
    #
    # Run simulation
    while t < t_end:
        #
        v_bus_full = ps.red_to_full.dot(ps.v_red)
        v_bus_mag = np.abs(v_bus_full)
        #
        v_bus_full_conj = np.conj(v_bus_full)
        #
        # Computing power flow on selected branch
        #
        Sflow=ps.y_bus[from_i, to_j]*(v_bus_full[to_j]-v_bus_full[from_i])*v_bus_full_conj[from_i]
        Pflow=Sflow.real
        Qflow=-Sflow.imag
        # print('t =',t,', Power from',from_i+1,'to',to_j+1,'=',Pflow)
        if 1 < t:
            ps.y_bus_red_mod[2, 2] = 1e6
        else:
            ps.y_bus_red_mod[2, 2] = 0

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

        # Legger til nye outputs
        P_e_stored.append(ps.gen_mdls['GEN'].output['P_e'].copy())
        P_m_stored.append(ps.gen_mdls['GEN'].input['P_m'].copy())
        Pflow_stored.append(Pflow.copy())
        Qflow_stored.append(Qflow.copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    Pelec=np.array(P_e_stored)
    Pmech=np.array(P_m_stored)
    Pe1= Pelec[:,0]
    Pm1= Pmech[:,0]
    # Convert dict to pandas dataframe
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Plot angle and speed
    fig, ax = plt.subplots(3)
    ax[2].set_xlabel('time (s)')
    ax[0].set_ylabel('Speed (p.u.)')
    ax[1].set_ylabel('Power angle (p.u.)')
    ax[2].set_ylabel('Eq_dash')
    fig.suptitle('Generator speed, angle and Eq_dash')
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
    ax[2].plot(result[('Global', 't')], result.xs(key='e_q_t', axis='columns', level=1))
    # ax[3].plot(result[('Global', 't')], np.array(P_e_stored))
    plt.show()
    #
    # Plotting generator powers
    plt.figure()
    
    #plt.plot(result[('Global', 't')], np.array(P_e_stored))
    
    #plt.plot(result[('Global', 't')], np.array(P_e_stored), result[('Global', 't')], np.array(P_m_stored))
    #plt.plot(result[('Global', 't')], Pe1)
    #plt.plot(result[('Global', 't')], Pm1)
    #plt.legend(['G1 Pe', 'G1 Pm'], loc='lower right', shadow=True)
    #plt.xlabel('time (s)')
    #plt.ylabel('power (p.u.)')
    #plt.title('Generator power')
    #plt.show()
    #
    
    angle = result.xs(key='angle', axis='columns', level=1)
    G1_angle = angle.iloc[:,0]
    #print('angle = ', G1_angle)
    
    plt.figure()
    plt.plot(result[('Global', 't')],Pe1, result[('Global', 't')],Pm1)
    
    plt.xlabel('power angle (p.u.)')
    plt.ylabel('power (p.u.)')
    plt.title('Generator power')
    plt.show()
    
    # Plotting line powers
    plt.figure()
    plt.plot(result[('Global', 't')], np.array(Pflow_stored), result[('Global', 't')], np.array(Qflow_stored))
    plt.legend(['Pflow','Qflow'], loc='lower center', shadow=True)
    plt.xlabel('time (s)')
    plt.ylabel('power flow(p.u.)')
    plt.title('Line active and reactive power flow')
    plt.show()