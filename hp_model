import os
import warnings
import numpy as np
import openpnm as op
import scipy as sp
from scipy import special
from scipy.optimize import curve_fit
import random
import matplotlib.pyplot as plt
import openpnm.models as mods
import sys
import pandas as pd
np.set_printoptions(precision=5)
np.random.seed(11)
ws = op.Workspace()
ws.settings['loglevel'] = 40
warnings.filterwarnings('ignore')
proj = ws.new_project()
#######################################################################
#1 setting up network
pn = op.network.Cubic(shape=[50, 20, 20], spacing=[0.00006, 0.00006, 0.00006], connectivity=26, project=proj, name='pn') #add_boundary_pores=['left', 'right']
op.io.VTK.save(pn, filename="hp")
op.io.CSV.save(pn, filename="hp")
# print(pn.am)
# print(pn.conns)
# print(pn.coords)
# print(pn.im)
print("No of throats before elem =",pn.num_throats())
print("No of pores before elem =",pn.num_pores())
print(f"Does 'hp.vtk' exist? {os.path.isfile('hp.vtp')}")
print("A Cubic Lattice is created!")
#Elimination of network
op.topotools.reduce_coordination(network=pn, z=3.9)
# print(pn['throat.conns'])
#checking the pore connectivity
h = pn.check_network_health()
op.topotools.trim(network=pn, pores=h['trim_pores'])
h = pn.check_network_health()
print(h)
op.io.VTK.save(pn, filename="hp_r")
op.io.CSV.save(pn, filename="hp_r")
print(f"Does 'hp_r.vtk' exist? {os.path.isfile('hp_r.vtp')}")
fig = plt.figure(figsize=(16, 8))
plt.hist(pn.num_neighbors(pn.Ps), edgecolor='k')
fig.patch.set_facecolor('white')
plt.xlabel('Pore avg coordination no', fontsize = 20)
plt.ylabel('No of Pores', fontsize = 20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title('Pore coordination number', fontweight ="bold", fontsize = 22)
plt.savefig("hp_r.jpg")
print("Elimation is done!")

######################################################################
#2 set up geometries
np.random.seed(11)
Ps = pn.pores('all')
Ts = pn.throats('all')
# Ps = pn['pore.coords'][:, 0] < pn['pore.coords'][:, 0].mean()
# Ts = pn.find_neighbor_throats(pores=Ps, mode='xnor')
# Ps = pn.pores('*boundary', mode='not') #internal pores
# Ts = pn.find_neighbor_throats(pores=Ps, mode='xnor', flatten=True)

geo = op.geometry.SpheresAndCylinders(network=pn,pores=Ps,throats=Ts)
print("Stick&Ball Geometry is created!")
mean, sigma = 3.90704784,0.09975134512         #1.000050001,0.000005000250213
geo['pore.diameter'] = np.random.lognormal(mean,sigma,pn.Np)*0.000001 # Units of meters
# print(geo['pore.diameter'])
print("Min pore dia before truc =",np.amin(geo['pore.diameter']))
print("Max pore dia before truc =",np.amax(geo['pore.diameter']))
op.io.VTK.save(geo, filename="hp_r_g")

#finding the meand and std for the generated data
print("Mean of pore dia(m) = ",geo['pore.diameter'].mean())
print("Variance of pore dia(m) = ",geo['pore.diameter'].var())
print("Std of pore dia(m) = ",geo['pore.diameter'].std())

#Eliminating the pore bodies with cooridnation number = 1
Ps = pn.num_neighbors(pores=pn.Ps)
op.topotools.trim(pn, pores=Ps==1)
op.io.VTK.save(pn, filename="hp_p_trim")

print("No of pores after elem =",pn.num_pores())
while (Ps == 1).any():
    Ps = pn.num_neighbors(pores=pn.Ps)
    op.topotools.trim(pn, pores=Ps==1)
print("No of pores after final iteration elem =",pn.num_pores())
h = pn.check_network_health()
print(h)

fig = plt.figure(figsize=(16, 8))
plt.hist(geo['pore.diameter'], edgecolor='k', bins = 50)
plt.xlabel('Pore Diameter', fontsize = 20)
plt.ylabel('No of Pores', fontsize = 20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title('Pore coordination number before trunc', fontweight ="bold", fontsize=22)
plt.savefig("hp_r_g.jpg")
print("Random Pore dia data is generated for lognormal dist with n_mean = "+str(mean)+",n_sigma = "+str(sigma))
# np.set_printoptions(threshold=sys.maxsize)   #Shows the full data in the numpy array without truncation in console
# geo['pore.diameter']
geo['pore.diameter'].tofile('hp_r_g.csv',sep=',')
print('pore diametercsv file is generated as hp_r_g.csv')
# op.io.CSV.save(geo['pore.diameter'], filename="hp_r_g")
########################################################################
#Random numbers generation using lognormal dist and trucating it
dummy = np.array([])
while True:
    dummy = np.random.lognormal(mean, sigma, size=pn.Np+1000)*0.000001 # Units of meters
    dummy = dummy[ (dummy >= 0.00001) & (dummy <= 0.00008) ]#*0.000001#*0.000001
    if len(dummy) >= pn.Np:
        break

geo['pore.diameter'] = dummy[:pn.Np]

#finding the meand and std for the truncated data
print("Mean of pore dia(m) after trunc = ",geo['pore.diameter'].mean())
print("Variance of pore dia(m) after trunc = ",geo['pore.diameter'].var())
print("Std of pore dia(m) after trunc = ",geo['pore.diameter'].std())

##########################################################################

fig = plt.figure(figsize=(16, 8))
plt.hist(geo['pore.diameter'], edgecolor='k', bins = 50)
plt.xlabel('Pore Diameter', fontsize = 20)
plt.ylabel('No of Pores', fontsize = 20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title('Pore coordination number after trunc', fontweight ="bold", fontsize=22)
plt.savefig("hp_r_g_t.jpg")
print("No of Pores= ",len(geo['pore.diameter'])) #check for length of generated pore.dia
op.io.VTK.save(geo, filename="hp_r_g_pd")
print("Real soil Pores are created in Network")
#########################################################################
#throat radius generation
P12 = pn['throat.conns']   #indices of the pore-pair(2 pore bodies) connected by each throat
D12 = geo['pore.diameter'][P12] #combining arrays to provide the diameters of the pores on each end of a throat
Dt = np.amin(D12, axis=1)  #performing column operation to find the smaller pore dia in each pair 
geo['throat.diameter'] = Dt*0.8 #Assigning dia for throat as (min dia of pore body*0.8) it is connected
op.io.VTK.save(geo, filename="hp_r_g_td")
print("Real soil Pore throat radius are assigned to Network")

#calculation of pore volume
Rp = geo['pore.diameter']/2  
geo['pore.volume'] = (4/3)*3.14159*(Rp)**3
print("Min pore vol = ",np.amin(geo['pore.volume']))
print("Max pore vol = ",np.amax(geo['pore.volume']))
print("Real soil Pore body volumes are calculated in the Network")

#calculation of length of throat
C2C = 0.00006 # The center-to-center distance between pores
Rp12 = Rp[pn['throat.conns']] #combining arrays to provide radius of throat at each connection(indices)
geo['throat.length'] = C2C - np.sum(Rp12, axis=1) #Assigning lenth for each throat based on pore body radius
op.io.VTK.save(geo, filename="hp_r_g_tl")
print("Real soil Pore throat lengths are assigned to Network")

#calculation of throat volume
Rt = geo['throat.diameter']/2
Lt = geo['throat.length']
geo['throat.volume'] = 3.14159*(Rt)**2*Lt
print("Min throat vol = ",np.amin(geo['throat.volume']))
print("Max throat vol = ",np.amax(geo['throat.volume']))
op.io.VTK.save(geo, filename="hp_r_g_tv")
print("Real soil Pore throat volumes are calculated in the Network")

# calculating cross-section area of pore and throat
geo["pore.area"] = np.pi * np.square(geo['pore.diameter']) / 4
geo["throat.area"] = geo['throat.cross_sectional_area']


# Let's pull out only the pore properties from the geometry
pore_data_sheet = pd.DataFrame({i: geo[i] for i in geo.props(element='pore')})
pore_data_sheet.to_csv('pore_data_sheet.csv')


# Let's pull out only the pore properties from the geometry
throat_data_sheet = pd.DataFrame({i: geo[i] for i in geo.props(element='throat')})
throat_data_sheet.to_csv('throat_data_sheet.csv')

#Calculaton of porosity of the network

vol_total = (pn._shape * pn._spacing).prod()
print(vol_total)
vol_pores = geo['pore.volume'].sum()
vol_throats = geo['throat.volume'].sum()
print(vol_pores)
print(vol_throats)
porosity = (vol_pores + vol_throats) / vol_total
print('Porosity:', porosity)

print(max(geo['pore.diameter']))

#############################################################################
#ploting my network details
fig = plt.figure(figsize=(20, 10))
plt.subplot(2,3,1)
plt.hist(geo['pore.diameter'], color='Black',edgecolor='w', bins = 20, label = 'Pore Diameter')
# plt.xlabel('Pore Diameter', fontsize = 20)
# plt.ylabel('No of Pore bodies', fontsize = 20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.legend(loc='upper right')
# plt.legend(('Pore Diameter'), loc='upper right')
plt.subplot(2,3,2)
plt.hist(geo['pore.volume'], color='Red',edgecolor='k', bins = 20, label = 'Pore Volume')
# plt.xlabel('Pore Volume', fontsize = 20)
# plt.ylabel('No of Pore bodies', fontsize = 20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.legend(loc='upper right')
plt.subplot(2,3,3)
plt.hist(geo['throat.diameter'], color='Green',edgecolor='k', bins = 20, label = 'Throat Diameter')
# plt.xlabel('Throat Diameter', fontsize = 20)
# plt.ylabel('No of Pore throats', fontsize = 20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.legend(loc='upper right')
plt.subplot(2,3,4)
plt.hist(geo['throat.length'], color='Orange',edgecolor='k', bins = 20, label = 'Throat Length')
# plt.xlabel('Throat Length', fontsize = 20)
# plt.ylabel('No of Pore throats', fontsize = 20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.legend(loc='upper right')
plt.subplot(2,3,5)
plt.hist(geo['throat.volume'], edgecolor='k', bins = 20, label = 'Throat Volume')
# plt.xlabel('Throat Volume', fontsize = 20)
# plt.ylabel('No of Pore throats', fontsize = 20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.legend(loc='upper right')
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.savefig("detailed plot.jpg")
# ###############################################################################
# #Defining phase of network
# water = op.phases.GenericPhase(network=pn, name = 'water')
# water['pore.temperature'] = 298.0
# water['pore.viscosity'] = 0.001
# #Defining physics
# phys_water = op.physics.GenericPhysics(network=pn, phase=water, geometry=geo)
# R = geo['throat.diameter']/2
# L = geo['throat.length']
# mu_w = 0.001
# phys_water['throat.hydraulic_conductance'] = 3.14159*R**4/(8*mu_w*L)
# #Definiing algorithm
# alg = op.algorithms.StokesFlow(network=pn)
# alg.setup(phase=water)
# #Defining boundaries
# BC1_pores = pn.pores('front')
# alg.set_value_BC(values=202650, pores=BC1_pores)
# BC2_pores = pn.pores('back') 
# alg.set_value_BC(values=101325, pores=BC2_pores)
# alg.run()
# Q = alg.rate(pores=pn.pores('front'))
# A = 0.001*19*2   # Cross-sectional area for flow
# L = 50*0.001    # Length of flow path
# del_P = 101325   # pressure difference
# K = Q*mu_w*L/(A*del_P)
# print("The average hydraulic conductivity using equation=",K)
# K = alg.calc_effective_permeability(domain_area=A, domain_length=L)
# print("The average hydraulic conductivity using in built function=",K)
# water.update(alg.results())
# g = water['throat.hydraulic_conductance'] #Accessing the results
# print(g)

###############################################################################
#Assigning water phase to network
water = op.phases.Water(network=pn, name = 'water')
water['throat.contact_angle'] = 110
water['throat.surface_tension'] = 0.072
water['pore.temperature'] = 298.0
water['pore.viscosity'] = 0.001
print(water)
#Defining Physics
phys = op.physics.Basic(network=pn, phase=water, geometry=geo)
flow = op.models.physics.hydraulic_conductance.hagen_poiseuille
phys.add_model(propname='throat.hydraulic_conductance',
               pore_viscosity='pore.viscosity',
               throat_viscosity='throat.viscosity',
               model=flow, regen_mode='normal')
print(phys)
# algorithms
#Pressure distribution is found by running  Stokes flow algorithm
sf = op.algorithms.StokesFlow(network=pn, phase=water)
sf.set_value_BC(pores=pn.pores('left'), values=101325)
sf.set_value_BC(pores=pn.pores('right'), values=0.00)
sf.settings['rxn_tolerance'] = 1e-12
sf.run()
water.update({'pore.pressure':sf['pore.pressure']})
#The results obtained from the StokesFlow algorthim must be attached to the water phase
water.update(sf.results())
#Predicting Dispersion coefficient
water['pore.diffusivity'] = 1e-9
def effective_pore_volume(target, throat_volume='throat.volume', pore_volume='pore.volume'):
    Pvol = pn['pore.volume']
    Tvol = pn['throat.volume']
    Vtot = Pvol.sum() + Tvol.sum()
    np.add.at(Pvol, pn.conns[:, 0], pn['throat.volume']/2)
    np.add.at(Pvol, pn.conns[:, 1], pn['throat.volume']/2)
    assert np.isclose(Pvol.sum(), Vtot)  # Ensure total volume has been added to Pvol
    return Pvol
pn.add_model(propname='pore.effective_volume', model=effective_pore_volume)
#Transient diffusion simulation performance
mod = op.models.physics.ad_dif_conductance.ad_dif
water.add_model(propname='throat.ad_dif_conductance', model=mod, s_scheme='powerlaw')
ad = op.algorithms.TransientAdvectionDiffusion(network=pn, phase=water)
ad.settings.update({'pore.volume' : 'pore.effective_volume'})
inlet  = pn.pores('left') 
outlet = pn.pores('right')
ad.set_value_BC(pores=inlet, values=1.0)
ad.set_outflow_BC(pores=outlet)
tspan = (0,100)
save_at = 5
ad.run(save_at)
#Break Through Curve (BTC) generation
Ps_right = pn.pores(['right'])
Ts_right = pn.find_neighbor_throats(pores=Ps_right, mode='xor') #selects throats that are connected to only one right-side pore body
steps = tspan[1]/save_at + 1
count = 0
c_avg = []
soln = ad.soln
for ti in soln['pore.concentration'].t:
    c_right = soln['pore.concentration'](ti)[Ps_right]
    q_right = sf.rate(throats=pn.Ts,mode='single')[Ts_right]
    c_avg.append((q_right*c_right).sum() / q_right.sum())

fig = plt.figure(figsize=(16, 8))
fig, ax = plt.subplots()    
ax.plot(soln['pore.concentration'].t, c_avg, "o-")
ax.legend(('simulation', 'fitted'))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
ax.set_xlabel('time (s)')
ax.set_ylabel('concentration');
plt.savefig("BTC.jpg")

# ################################################################################
# #Defining advection and diffusion physic to the network
# ADE = op.models.physics.ad_dif_conductance.ad_dif
# phys.add_model(propname='throat.ad_dif_conductance', model=ADE, s_scheme='powerlaw')

# #Defining advection-diffusion algorithm
# ad = op.algorithms.AdvectionDiffusion(network=pn, phase=water)
# #Boundary condition for diffusion 
# ad.set_value_BC(pores=pn.pores('left'), values=100.0)
# ad.set_value_BC(pores=pn.pores('right'), values=0.0)
# ad.run()

# #Post processing
# print(ad.settings)
##################################################################################
# #Pressure distribution is found by running  Stokes flow algorithm
# sf = op.algorithms.StokesFlow(network=pn, phase=water)
# sf.set_value_BC(pores=pn.pores('left'), values=101325*2)
# sf.set_value_BC(pores=pn.pores('right'), values=101325)
# sf.run()
# sf.settings['rxn_tolerance'] = 1e-12
# water.update({'pore.pressure':sf['pore.pressure']})
# #Predicting Dispersion coefficient
# water['pore.diffusivity'] = 1e-9
# def effective_pore_volume(target, throat_volume='throat.volume', pore_volume='pore.volume'):
#     Pvol = pn['pore.volume']
#     Tvol = pn['throat.volume']
#     Vtot = Pvol.sum() + Tvol.sum()
#     np.add.at(Pvol, pn.conns[:, 0], pn['throat.volume']/2)
#     np.add.at(Pvol, pn.conns[:, 1], pn['throat.volume']/2)
#     assert np.isclose(Pvol.sum(), Vtot)  # Ensure total volume has been added to Pvol
#     return Pvol
# pn.add_model(propname='pore.effective_volume', model=effective_pore_volume)
#################################################################################



##################################################################################
# #Transient diffusion
# water['pore.diffusivity'] = 2e-09
# fick_diff = op.models.physics.diffusive_conductance.ordinary_diffusion
# phys.add_model(propname='throat.diffusive_conductance', model=fick_diff, regen_mode='normal')
# #deifing algo for transientfickdiff
# fd = op.algorithms.TransientFickianDiffusion(network=pn, phase=water)
# fd.set_value_BC(pores=pn.pores('left'), values=0.5)
# fd.set_value_BC(pores=pn.pores('right'), values=0.2)
# fd.set_IC(0.2)
# fd.setup(t_scheme='cranknicolson', t_final=100, t_output=5, t_step=1, t_tolerance=1e-12)
# print(fd.settings)
# fd.run()
# print(fd)   
# water.update(fd.results())

# proj.export_data(phases=water , filename='ade_results', filetype='VTK')

# proj.export_data(phases=water , filename='ade_results', filetype='CSV')

# # #ploting hydraulic conductance for throats
# # fig = plt.figure(figsize=(16, 8))
# # plt.hist(water["throat.hydraulic_conductance"], edgecolor = 'k', bins = 50)
# # plt.savefig("throat.hydraulic_conductance.jpg")


# # ########################################################################################
# # #Next step
# # # phys_water = op.physics.GenericPhysics(network=pn, phase=water, geometry=geo)
# # # water['throat.viscosity'] = water['pore.viscosity'][0]
# # # #Defining pipe Flow 
# # # mod = op.models.physics.hydraulic_conductance.hagen_poiseuille
# # # phys_water.add_model(propname='throat.hydraulic_conductance',
# # #                      model=mod) #, viscosity='throat.viscosity'
# # # #Defining Ficks Diffusion
# # # geo['pore.area'] = sp.pi*(geo['pore.diameter']**2)/4.0
# # # # mod2 = op.models.physics.diffusive_conductance#.bulk_diffusion
# # # # phys_water.add_model(propname='throat.diffusive_conductance',
# # # #                      model=mod2) #, diffusivity='pore.diffusivity'

# # # phys_water.regenerate_models()

# # # inlet = pn.pores('back')  # pore inlet
# # # outlet = pn.pores('front')  # pore outlet

# # # # inlet2 = pn.pores('left')  # pore inlet2
# # # # outlet2 = pn.pores('right')  # pore outlet2

# # # #Defining ALGORITHMS
# # # alg1 = op.algorithms.StokesFlow(network=pn, phase=water)
# # # # alg1.set_boundary_conditions(pores = pn.pores('top'), bctype='Dirichlet', bcvalue = 5)
# # # # alg1.set_boundary_conditions(pores = pn.pores('bottom'), bctype='Dirichlet', bcvalue = 0)
# # # # alg1.set_dirichlet_BC(pores=inlet, values=5)
# # # # alg1.set_dirichlet_BC(pores=outlet, values=0)
# # # alg1.run()

# # # # alg1b = op.algorithms.TransientStokesFlow(network=pn, phase=water)
# # # # alg1b.set_IC(0)
# # # # alg1b.set_dirichlet_BC(pores=inlet, values=5)
# # # # alg1b.set_dirichlet_BC(pores=outlet, values=0)
# # # # alg1b.run()



# # # # geom['pore.area'] = sp.pi*(geom['pore.diameter']**2)/4.0
# # # # #phys_water.models.add(propname='throat.capillary_pressure', model= capillary_pressure.washburn)
# # # # #OP = op.algorithms.OrdinaryPercolation(network=pn, invading_phase=water)
# # # # #OP.run(inlets=pn.pores('bottom'))
# # # # #Define Ficks Diffusion algorithm
# # # # FD = op.algorithms.FickianDiffusion(network=pn, phase=water)
# # # # FD.set_boundary_conditions(pores = pn.pores('top'), bctype='Dirichlet', bcvalue = 0.5)
# # # # FD.set_boundary_conditions(pores = pn.pores('bottom'), bctype='Dirichlet', bcvalue = 0.1)
# # # # FD.run(conductance = 'throat.conductance')
# # # # ###Define Stokes flow algorithm
# # # # sf = op.algorithms.StokesFlow(network = fcc)
# # # # sf.setup(phase = air)
# # # # sf.set_value_BC(pores = inlet, values = 101328*2)
# # # # sf.set_value_BC(pores = outlet, values = 101328)
# # # # sf.run()
# # # # K = sf.calc_eff_permeability()
