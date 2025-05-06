import sys
import time
import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
matplotlib.rcParams['text.usetex'] = True
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import mpmath
from mpmath import *
import scipy
from scipy import integrate
import cmath
#from numba import jit
from scipy.integrate import ode
mp.dps = 45; mp.pretty = True

np.set_printoptions(threshold=sys.maxsize)
# get the start time
st = time.time()

##########################################################################
########################## Show initializing #############################
##########################################################################

print("==========================================================================")
print("================= Selectivity Population Dynamics in E2PA ================")
print("==========================================================================")

##########################################################################
###  Conventions for levels: ground = 0, intermediate = 1, excited = 2 ###
##########################################################################

##########################################################################
##########:::::::::::::::::::: Inputs :::::::::::::::::::::::::::#########
##########################################################################

##########################################################################
###### Number of modes and number of levels in each potential ############
##########################################################################

Nlevels = 50 #Number of levels for each potential
Modes = 1001 #Number of modes for the incoming field
lex = 18 #Level of resonance

print("Number of levels for each potential: ", Nlevels)
print("Number of modes for the field: ", Modes)
print("Level of resonance: ", lex)
print("==========================================================================")

##########################################################################

##########################################################################
################## Constants for the photons profile #####################
##########################################################################


#Value of the spectral width of the wave packet \sigma = \sigma _{r}^{-1} in Hz
sigma = 10e12

#Value of the spectral width of the wave packet \sigma = \sigma _{r}^{-1} in eV -> 1Hz = 4.135 58 x 10^-15 eV

sigmaev = sigma * 4.13558e-15

#Constant for entanglement

ss = 0.05
print("Constants for the photons profile : ")
print("sigma: ", sigma)
print("sigma in eV: ", sigmaev)
print("ss: ", ss)
print("==========================================================================")
##########################################################################

##########################################################################
################### Constants for the simulations ########################
##########################################################################

t0 = -5.0
t1 = 5.0
dt = 0.01

print("Simulation parameters: ")
print("t0: ", t0)
print("t1: ", t1)
print("dt: ", dt)
print("==========================================================================")

##########################################################################

##########################################################################
################### Constants for the potentials #########################
##########################################################################

#Value of the electron mass in eV
emass = 0.5110e6

#Value of the reduced mass of two nuclei in eV
mu = 19800 * emass

#Value of Bohr radius in eV^-1
a0 = 2.68e-4

##########################################################################

################# Minimum of the potential energy: epsilon ###############

#Value of epsilon_{g} in eV
epsilong = 0

#Value of epsilon_{m} in eV
epsilonm = 1.8201

#Value of epsilon_{e} in eV
epsilone = 3.7918

##########################################################################

################### Depth of the potential: D ############################

#Value of D_{g} in eV
dg = 0.7466 * 1

#Value of D_{m} in eV
dm = 1.0303 * 1

#Value of D_{e} in eV
de = 0.5718 * 1

##########################################################################

########### Range of the potential in terms of Bohr radius: a ############

ag = 2.2951

am = 3.6591

ae = 3.1226

##########################################################################

############ Equilibrium position x_{0} of the Morse Potential ###########

#Value of x_{0}^{(g)}
x0g = 5.82

#Value of x_{0}^{(m)}
x0m = 6.87

#Value of x_{0}^{(m)}
x0e = 7.08

##########################################################################

##########################################################################
################### Constants for the interactions #######################
##########################################################################

print("Constants for the interactions: ")

#Definition of the relaxation rate \gamma = \gamma _{m_{\nu}} = \gamma _{e _{\nu \nu'}} in Hz
gammar = 6e6
print("relaxation rate gamma: ", gammar)

#Definition of the relaxation rate \gamma = \gamma _{m_{\nu}} = \gamma _{e _{\nu \nu'}} in eV
gammarev = gammar * 4.13558e-15
print("relaxation rate gamma in eV: ", gammarev)
print("==========================================================================")
##########################################################################

###############################################################
###### Definition of the vibrational eigen-energies ###########
###############################################################

#General definition of epsilon_{\ell} in eV

def epsilonl(l):
    if l == 0:
       return epsilong
    if l == 1:
       return epsilonm
    if l == 2:
       return epsilone
    else:
       return 0 

#General definition of D_{\ell} in eV

def potdepth(l):
    if l == 0:
       return dg
    if l == 1:
       return dm
    if l == 2:
       return de
    else:
       return 0 

#General definition of $a_{\ell}$ in eV

def al(l):
    if l == 0:
       return ag * a0
    if l == 1:
       return am * a0
    if l == 2:
       return ae * a0
    else:
       return 0 

#Definition of \omega _{\ell}

def omegal(l):
    return np.sqrt((2*potdepth(l))/((al(l)**2) * mu))

#Definition of \chi _{\ell}

def chil(l):
    return 1/np.sqrt(8 * (al(l)**2) * potdepth(l) * mu)
    
    
# Definition of \omega _{\ell _{\nu}}
def omegalnu_func(l, nu):
    return epsilonl(l) + (omegal(l) * (nu + 0.5)) - (omegal(l) * chil(l) * (nu + 0.5) ** 2)
    
omegalnu = np.zeros([3, Nlevels])

for l in range(3):
    for nu in range(Nlevels):
        omegalnu[l, nu] = omegalnu_func(l, nu) 
    
    
#########################################################################
## Definition of the two-photon joint amplitude for correlated pairs  ###
#########################################################################

wlength = float(4*np.pi/omegalnu[2, lex]) #Wavelength in ev^-1
k0 = float(2*np.pi / wlength) #Wave number in ev -> k_{0}
sigmaev = float(sigmaev)
r0 = float(t0/sigmaev) #spatial center position of the wave packet at t0

print("==========================================================================")
print("=================   Parameters of the correlated pairs  ==================")
print("==========================================================================")

#Energy of incident field
print("Energy of incident field in eV: ")
print("k0: ", k0, "2k0: ", 2*k0, "r0: ", r0)

#Value of \delta k in Hz
deltak = 100 * (10**9)
print("deltak: ", deltak)

#Value of \delta k in eV
#deltakev = deltak * 4.13558e-15
#const = (Modes * deltak * 4.13558e-15)/sigmaev
const = 9
#Previous const = 9

deltakev = (const * sigmaev)/Modes
print("delta k in eV: ", deltakev)

kz = np.linspace(k0 - (const / 2)*sigmaev, k0 + (const / 2)*sigmaev, Modes) #Array of all k

kp = np.linspace(k0 - (const / 2)*sigmaev, k0 + (const / 2)*sigmaev, Modes) #Array of all k'

Kz, Kp = np.meshgrid(kz, kp) #Multilinear array of all possible combinations of k and k'

noren = np.sqrt(2 * np.pi * sigmaev * ss * sigmaev) #Normalization constant
#noren = np.sqrt(2 * np.pi * sigmaev**2)

#Definition of the two-photon joint amplitude of an entangled photon pair with energy anticorrelation
def Psi2pen(k, k1):
    return np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k1 - k0)**2)/(4 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)  
    
    
#np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k + k1 - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)     
    
#np.exp(-(((k - k1)*0.5)**2)/(4 * sigmaev**2)) * np.exp(-((k + k1 - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)       
    
#np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k1 - k0)**2)/(4 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)
    
##########################################################################
###########  Visualization of the correlated profile      ################
##########################################################################

fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '17'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(17)
img3 = ax1.contourf(Kz, Kp, np.abs(Psi2pen(Kz, Kp))**2,  20, cmap=cmr.rainforest)
ax1.set_xlabel(r"$k$")
ax1.set_ylabel(r"$k '$")

cbar = plt.colorbar(img3,cmap='seismic')
cbar.set_label(r"$|\psi _{2p}(k, k')|^{2}$",size=17)
plt.show()  


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Kz, Kp, np.abs(Psi2pen(Kz, Kp))**2, cmap=cmr.rainforest)
ax.grid(False)
ax.set_facecolor('white')
plt.show()

##########################################################################
########  Visualization of the correlated profile with k/k_{0}   #########
##########################################################################

k22 = (1/k0)*np.linspace(k0 - 4.5*sigmaev, k0 + 4.5*sigmaev, Modes)
k23 = (1/k0)*np.linspace(k0 - 4.5*sigmaev, k0 + 4.5*sigmaev, Modes)

K22, K23 = np.meshgrid(k22, k23)


def Psi2pennorm(x, x1):
    return 0.5 * ((np.exp(-((k0**2)*(x - 1)**2)/(4 * sigmaev**2)) * np.exp(-(k0**2)*((x + x1 - 2)**2)/(4*((ss*sigmaev)**2))) * (np.cos(k0*(x + x1)*r0) - 1j*np.sin(k0*(x + x1)*r0)) * (1/noren)) + (np.exp(-((k0**2)*(x1 - 1)**2)/(4 * sigmaev**2)) * np.exp(-(k0**2)*((x1 + x - 2)**2)/(4*((ss*sigmaev)**2))) * (np.cos(k0*(x1 + x)*r0) - 1j*np.sin(k0*(x1 + x)*r0)) * (1/noren)))  
    
#    np.exp(-((k0**2)*(x - 1)**2)/(4 * sigmaev**2)) * np.exp(-(k0**2)*((x + x1 - 2)**2)/(4*((ss*sigmaev)**2))) * (np.cos(k0*(x + x1)*r0) - 1j*np.sin(k0*(x + x1)*r0)) * (1/noren) 

# np.exp(-((k0**2)*(x - 1)**2)/(4 * sigmaev**2)) * np.exp(-((k0**2)*(x1 - 1)**2)/(4 * sigmaev**2)) * (np.cos(k0*(x + x1)*r0) - 1j*np.sin(k0*(x + x1)*r0)) * (1/noren) 
    
#    np.exp(-((k0**2)*(x*0.5 - x1*0.5)**2)/(4 * sigmaev**2)) * np.exp(-(k0**2)*((x + x1 - 2)**2)/(4*((ss*sigmaev)**2))) * (np.cos(k0*(x + x1)*r0) - 1j*np.sin(k0*(x + x1)*r0)) * (1/noren)

# 0.5 * ((np.exp(-((k0**2)*(x - 1)**2)/(4 * sigmaev**2)) * np.exp(-(k0**2)*((x + x1 - 2)**2)/(4*((ss*sigmaev)**2))) * (np.cos(k0*(x + x1)*r0) - 1j*np.sin(k0*(x + x1)*r0)) * (1/noren)) + (np.exp(-((k0**2)*(x1 - 1)**2)/(4 * sigmaev**2)) * np.exp(-(k0**2)*((x1 + x - 2)**2)/(4*((ss*sigmaev)**2))) * (np.cos(k0*(x1 + x)*r0) - 1j*np.sin(k0*(x1 + x)*r0)) * (1/noren)))  

fig, ax1 = plt.subplots(figsize=(12, 9))
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '61'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(61)
img4 = ax1.pcolormesh(K22, K23, np.abs(Psi2pennorm(K22, K23))**2, shading='gouraud', cmap = cmr.rainforest)


ax1.set_xlabel(r"$k/k_{0}$", fontsize=61, labelpad=0)
ax1.set_ylabel(r"$k '/k_{0}$", fontsize=61, labelpad=0)
ax1.text(0.95, 0.95, r'$f)$', transform=ax1.transAxes, fontsize=70, verticalalignment='top', horizontalalignment='right', color='white', bbox=None)
#ax1.set_title(r"$d)$")

cbar = plt.colorbar(img4,cmap=cmr.rainforest)
cbar.set_label(r"$|\psi _{sym}^{(2p)}(k, k')|^{2}$", size=61, labelpad=15)
#plt.savefig("/home/christiandavid/Desktop/UCL_PhD/corr_photons_001sigma.pdf", format="pdf", dpi=300)
ax1.set_yticks([k23[0], 1, k23[-1]])
plt.tight_layout(rect=[-0.07, -0.07, 1.04, 1.04]) 
#plt.axis('tight') 
plt.show()

##########################################################################
########:::::::::::::::  Franck-Condon Factors  :::::::::::::::::#########
##########################################################################

#Importing of files

#Franck-Condon Factors g->m
lista2 = []

nus = np.array(range(Nlevels))

lista2 = np.loadtxt('~/Ff_Factors_gm_63.txt')
lista2 = np.array(lista2, dtype=float)

lista22 = []

for i in range(Nlevels):
    lista22.append(lista2[i])
    
lista2 = lista22

lista2 = np.array(lista2, dtype=float)

fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '30'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(30)
plt.bar(range(Nlevels), lista2, color = 'black')
plt.xlabel(r"$\nu$", fontsize=30)
plt.ylabel(r"$F_{\nu}$", fontsize=30)
#plt.title(r"$b)$")
plt.subplots_adjust(left=0.2, bottom=0.18, right=0.979, top=0.95)
plt.show()

#Franck-Condon Factors m->e
Fnn = np.zeros((Nlevels, Nlevels))

Fnn = np.loadtxt('~/Ff_Factors_me_50_rev_1.txt')

Fnn1 = np.zeros((Nlevels, Nlevels))

for i in range(Nlevels):
    for j in range(Nlevels):
        Fnn1[i, j] = Fnn[j, i]
        
Fnn = Fnn1

fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '27'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(27)
img2 = ax1.imshow(Fnn,interpolation='nearest',
                    cmap = 'inferno',
                    origin='lower')
ax1.set_xlabel(r"$\nu$")
ax1.set_ylabel(r"$\alpha$")
#ax1.set_title(r"$d)$")

cbar = plt.colorbar(img2,cmap='inferno')
cbar.set_label(r"$F_{\nu \alpha}$",size=27)
plt.subplots_adjust(left=0.001, bottom=0.16, right=0.9, top=0.95)
plt.show()

def prod(alpha):
    listprod = np.zeros(Nlevels)
    for i in range(Nlevels):
        listprod[i] = Fnn[alpha, i] * lista2[i]
    sumr = np.sum(listprod)
    
    return sumr
    
arrprod = []

for i in range(Nlevels):
    arrprod.append(prod(i))
    
fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '17'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(17)
plt.bar(range(Nlevels), arrprod)
plt.xlabel(r"$\nu$", fontsize=17)
plt.ylabel(r"$F_{\nu, \alpha}F_{\nu}$", fontsize=17)

plt.show()            

print("------------------------------------------")
print("FC Factors: ")
print("F0 = ", lista2[0])
print("F12 = ", lista2[12])
print("F18 = ", lista2[18]) 
print("------------------------------------------")


##########################################################################
########################## Gaussian plots ################################
##########################################################################

listomegase = np.zeros(Nlevels)

for i in range(Nlevels):
        listomegase[i] = omegalnu_func(2, i)

def gaussiane(we, ss):
    a = np.pi * np.sqrt(np.pi)
    b = 2 * np.sqrt(2 * sigmaev * ss * sigmaev)
    pref = a/b
    c = 2 * k0 - we
    d = 4 * (sigmaev * ss)**2
    arggauss = (c**2)/d 
    return pref * np.exp(-arggauss)
    
x = np.linspace(omegalnu_func(2, 0), omegalnu_func(2, 45), 71)    
    
fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '17'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(21)
	
plt.plot(x, gaussiane(x, 1)**2, color = "slateblue", label = r"$\sigma _{s} = \sigma$", marker = "2")
plt.plot(x, gaussiane(x, 0.5)**2, color = "orange", label = r"$\sigma _{s} = 0.5\sigma$", marker = ".")
plt.plot(x, gaussiane(x, 0.25)**2, color = "green", label = r"$\sigma _{s} = 0.25\sigma$", linestyle = "dashed")
plt.plot(x, gaussiane(x, 0.1)**2, color = "blue", label = r"$\sigma _{s} = 0.1\sigma$", linestyle = "dashdot")
plt.plot(x, gaussiane(x, 0.05)**2, color = "red", label = r"$\sigma _{s} = 0.05\sigma$")
plt.xlabel(r"$\omega _{e_{\alpha}}$", fontsize=21)
plt.ylabel(r"$|\zeta_{\alpha}|^{2}$", fontsize=21)
#plt.title(r"$b)$")
plt.legend(bbox_to_anchor=(0.55, 0.9), loc="upper left", fontsize=16, fancybox=True, framealpha=1, borderpad=1, edgecolor="black")
plt.subplots_adjust(left=0.2, bottom=0.18, right=0.979, top=0.95)
plt.show()    


#######################################################################################
########################## Gaussian - FC factors plots ################################
#######################################################################################

def gaussianei(alpha, ss):
    a = np.pi * np.sqrt(np.pi)
    b = 2 * np.sqrt(2 * sigmaev * ss * sigmaev)
    pref = a/b
    c = 2 * k0 - omegalnu_func(2, alpha)
    d = 4 * (sigmaev * ss)**2
    arggauss = (c**2)/d 
    return pref * np.exp(-arggauss)

def expgw0wm(nu):
    am = k0 - omegalnu_func(1, nu)
    bm = 4 * sigmaev**2
    argaussm = (am**2)/bm
    return np.exp(-argaussm)

def expgwowewm(alpha, nu):
    aem = k0 - omegalnu_func(2, alpha) + omegalnu_func(1, nu)
    bem = 4 * sigmaev**2
    argaussem = (aem**2)/bem
    return np.exp(-argaussem)


Fnnexp1 = np.zeros((24, 24))

for nu in range(24):
    for alpha in range(24):
#        Fnnexp1[nu, alpha] =  np.sqrt(lista2[nu]/np.pi) * gaussianei(alpha, 0.25) * (expgw0wm(nu) + expgwowewm(alpha, nu)) * np.sqrt(Fnn1[nu, alpha]/np.pi)
        Fnnexp1[nu, alpha] = np.sqrt(lista2[nu]/np.pi) * expgw0wm(nu) * expgwowewm(alpha, nu) * np.sqrt(Fnn1[nu, alpha]/np.pi)
#        Fnnexp1[nu, alpha] = lista2[nu] * Fnn1[nu, alpha] * gaussianei(alpha, 0.05) * (expgw0wm(nu) + expgwowewm(alpha, nu))
        

fig, ax1 = plt.subplots(figsize=(12, 9))
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '61'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(61)
img2 = ax1.imshow(Fnnexp1.T,interpolation='nearest',
                    cmap = 'inferno',
                    origin='lower', extent=[0, 23, 0, 23], aspect='auto')
ax1.set_xlabel(r"$\nu$", fontsize=61, labelpad=0)
ax1.set_ylabel(r"$\alpha$", fontsize=61, labelpad=0)
#ax1.set_title(r"$e)$")
ax1.text(0.15, 0.95, r'$a)$', transform=ax1.transAxes, fontsize=70, verticalalignment='top', horizontalalignment='right', color='white', bbox=None)

ax1.set_xticks([0.0, 6.0, 12.0, 18.0, 23.0])
ax1.set_xticklabels([r'$0$', r'$6$', r'$12$', r'$18$', r'$23$'])
ax1.set_xlim([0, 23])

ax1.set_yticks([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0])
ax1.set_yticklabels([r'$3$', r'$6$', r'$9$', r'$12$', r'$15$', r'$18$', r'$21$'])
ax1.set_ylim([0, 23])

#ax1.axhline(y=12, color='white', linewidth=3)

cbar = plt.colorbar(img2,cmap='inferno')
cbar.set_label(r"$\Theta _{\nu \alpha}^{(u)}$", size=61, labelpad=15)
plt.tight_layout(rect=[-0.09, -0.09, 1.0, 1.09]) 
plt.show()

#cbar = plt.colorbar(img2,cmap='seismic')
#cbar.set_label(r"$F_{\nu}F_{\nu \alpha}\zeta _{\alpha \nu}^{(e)}$",size=17)
#plt.show()


##########################################################################
################## Set of differential equations #########################
##########################################################################

#Definition of the matrix ka + kb
sumk = np.zeros((Modes, Modes), dtype=complex)


for i in range(Modes):
    for j in range(Modes):
        sumk[i, j] = kz[i] + kp[j]

#print("Matrix of ka + kb: ")        
#print(sumk)
#print("==========================================================================")
sumk = sumk.flatten()

#Definition of the matrix ka + wmnu

sumkwm = np.zeros((Modes, Nlevels), dtype=complex)

for i in range(Modes):
    for n in range(Nlevels):
        sumkwm[i, n] = kz[i] + omegalnu[1, n]

#print("Matrix of ka + wmnu: ")        
#print(sumkwm)
#print("==========================================================================")
sumkwm = sumkwm.flatten()


#Definition of the vector gamma
gammanu = np.sqrt(gammarev * deltakev * (lista2 / np.pi)) 

#Definition of the matrix of sqrt(Fnn)
gammanunup = np.sqrt((gammarev / np.pi) * deltakev * Fnn)

def odes(t, psi):
    
    # Definition of the vector psi2p in terms of the components of psi
    psi2p = psi[0 : Modes ** 2]
        
    # Definition of the vector psi1p in terms of the components of psi
    psi1pm = psi[Modes ** 2 : Modes * (Modes + Nlevels)]
       
    # Definition of the vector psi0e in terms of the components of psi
    psi0e = psi[Modes * (Modes + Nlevels) : Modes * (Modes + Nlevels + 1)]

    ##################################################################
    ###############  Definition of array d/dt psi2p  #################
    ##################################################################
   
    psi1pm_mat = psi1pm.reshape((Modes, Nlevels))
   
    def f1(k):
        return np.dot(gammanu, psi1pm_mat[k, :])
    
    f1_list = np.fromiter((f1(k) for k in range(Modes)), dtype=np.float64, count=Modes)
    
    sumf1 = f1_list[:, np.newaxis] + f1_list
           
    dpsi2p = -1j * (1 / sigmaev) * sumk * psi2p - (1j / np.sqrt(2)) * (1 / sigmaev) * sumf1.flatten()

    ##################################################################
    ###############  Definition of array d/dt psi1p  #################
    ##################################################################
    
    psi2p_mat = psi2p.reshape((Modes, Modes))
    
    f21_list = np.sum(psi2p_mat[:Modes, :], axis=1)
    
    f3_list = np.dot(gammanunup[:Nlevels, :], psi0e)
   
    gammanu_broadcasted = gammanu[:Nlevels, np.newaxis].T
    sqrt_2_inv = 1 / np.sqrt(2)
    f2 = gammanu_broadcasted * f21_list[:, np.newaxis] + sqrt_2_inv * np.tile(f3_list, (Modes, 1))

    dpsi1pm = -1j * (1 / sigmaev) * sumkwm * psi1pm - 1j * (1 / sigmaev) * np.sqrt(2) * f2.flatten()    
       
    ##################################################################
    ###############  Definition of array d/dt psi0e  #################
    ##################################################################
    
    psi1pm_mat = psi1pm.reshape((Modes, Nlevels))

    f5_list = np.sum(psi1pm_mat, axis=0) 

    dpsi0e = -1j * (1 / sigmaev) * omegalnu[2, :] * psi0e - 1j * (1 / sigmaev) * np.dot(gammanunup, f5_list)
    
    #Definition of the vector dy which contains the derivatives    
    dy = np.concatenate((dpsi2p, dpsi1pm, dpsi0e)) 

    return dy

##########################################################################
####################  Initial conditions  ################################
##########################################################################

###### Initial conditions of psi2p ######
psi2p0 = np.zeros((Modes, Modes), dtype=complex)

for i in range(Modes):
    for j in range(Modes):
        psi2p0[i, j] = Psi2pen(kz[i], kp[j])

###### Initial conditions of psi1pm ######
psi1pm0 = np.zeros((Modes, Nlevels), dtype=complex)

###### Initial conditions of psie #######
psie0 = np.zeros((Nlevels, 1), dtype=complex)

##### Total initial conditions
psiv0 = np.array(np.append(np.append(psi2p0.flatten(), psi1pm0.flatten()), psie0.flatten()))

#print("==========================================================================")
#print("====================  Initial conditions  ================================")
#print("==========================================================================")

#print("Initial conditions: ")
#print(psiv0)

norinitstate = np.sum(np.abs(psiv0)**2)

#print("Normalization constant of inital state: ")
#print(norinitstate)

psiv0nor = psiv0/np.sqrt(norinitstate)

#print("Normalized initial conditions: ")
#print(psiv0nor)

#print("Sum: ", np.sum(np.abs(psiv0nor)**2))

##########################################################################

##########################################################################
############################  Solver  ####################################
##########################################################################

print("==========================================================================")
print("=======================  Solver started...  ==============================")
print("==========================================================================")


methodsolver = 'adams'

r = ode(odes)
r.set_integrator('zvode', method=methodsolver)
r.set_initial_value(psiv0nor, t0)

lista1 = []

while r.successful() and r.t < t1:
    print(r.t)
    lista1.append(r.integrate(r.t+dt)[Modes * (Modes + Nlevels) : Modes * (Modes + Nlevels + 1)])

#    listaprint = np.array(lista1)
#    tii = np.linspace(t0, t1, np.shape(listaprint)[0])
#    np.savetxt('/home/christiandavid/Desktop/UCL_PhD/solutions_MO_entangled_ss005_135modes_corrected_newsolver.txt', listaprint)
#    print("shape new list: ", np.shape(listaprint))
#    print("==========================================================================")
#    print(np.abs(listaprint[:, -Nlevels + 18])**2)
#    print("==========================================================================")



##########################################################################
##########################  Creation of the files ########################
##########################################################################

#Array which contains all the solutions
lista1 = np.array(lista1)

#Temporal array
tii = np.linspace(t0, t1, np.shape(lista1)[0])

#Array which contains the final row of the solutions - to be the initial conditions in the next running
flist = np.array(lista1[-1, :])

#Array which contains the solutions of the population of the 25 first excited levels
#norfinalstate = np.sum(np.abs(lista1[-1, :])**2)
norfinalstate = 1

list25excited = np.zeros((25, np.shape(lista1)[0]), dtype=complex)

for i in range(25):
    list25excited[i] = (1/norfinalstate) * np.abs(lista1[:, -Nlevels + i])**2

list25excited = np.array(list25excited)

#Definition of the name of the file which contains all the solutions
name_solutions = f'~/solutions_MO/solutions_MO_entangled_ss{ss}_{Modes}modes_{methodsolver}_{t0}to{t1}_dt{dt}_r0_nonsym_fsolverp.txt'

#Definition of the name of the file which contains the final row of the solutions
name_initial_conditions = f'~/initialconditions_MO/initial_conditions_MO_entangled_ss{ss}_{Modes}modes_{methodsolver}_{t0}to{t1}_dt{dt}_r0_nonsym_fsolverp.txt'

#Definition of the name of the file which contains the solution of the 25 first excited levels
name_populations = f'~/populations_MO/populations25_MO_entangled_ss{ss}_{Modes}modes_{methodsolver}_{t0}to{t1}_dt{dt}_r0_nonsym_fsolverp.txt'



#Save the total array of all solutions
np.savetxt(name_solutions, lista1)

#Save the final row of the solutions to be the initial conditions in the next running
np.savetxt(name_initial_conditions, flist)

#Save the 25 first excited levels
np.savetxt(name_populations, list25excited)

#print(np.shape(lista1))

#print('Final shape: ', np.shape(lista1))
#print("==========================================================================")
#print("Final vector: ")
#print(np.abs(lista1[:, -Nlevels + 18])**2)
#print("==========================================================================")
#print("Final data:")
#print("Number of levels for each potential: ", Nlevels)
#print("Number of modes for the field: ", Modes)
#print("Level of resonance: ", lex)
#print("ss: ", ss)
#print("Energy of incident field in eV: ")
#print("k0: ", k0, "2k0: ", 2*k0, "r0: ", r0)
#print("Simulation parameters: ")
#print("t0: ", t0)
#print("t1: ", t1)
#print("dt: ", dt)
#print("Final value of level 18: ", (1/norfinalstate) * (np.abs(lista1[:, -Nlevels + 18])**2)[-1])
#print("==========================================================================")



# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st

print('Execution time: ', elapsed_time, 'seconds')
print('Execution time: ', elapsed_time/60, 'minutes')
print('Execution time: ', elapsed_time/3600, 'hours')
print('Execution time: ', elapsed_time/86400, 'days')

print('norfinalstate: ', norfinalstate)

########################################
fig, ax1 = plt.subplots(figsize=(5, 10))
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '10'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(17)

plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 0])**2, color = "slateblue", label = r"$\nu = 0$")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 1])**2, color = "green", label = r"$\nu = 1$")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 2])**2, color = "darkcyan", label = r"$\nu = 2$")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 3])**2, color = "orange", label = r"$\nu = 3$")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 4])**2, color = "gold", label = r"$\nu = 4$")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 5])**2, color = "blue", label = r"$\nu = 5$")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 6])**2, color = "red", label = r"$\nu = 6$")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 7])**2, color = "black", label = r"$\nu = 7$")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 8])**2, color = "slateblue", label = r"$\nu = 8$", linestyle = "dashed")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 9])**2, color = "green", label = r"$\nu = 9$", linestyle = "dashed")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 10])**2, color = "darkcyan", label = r"$\nu = 10$", linestyle = "dashed")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 11])**2, color = "orange", label = r"$\nu = 11$", linestyle = "dashed")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 12])**2, color = "gold", label = r"$\nu = 12$", linestyle = "dashed")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 13])**2, color = "blue", label = r"$\nu = 13$", linestyle = "dashed")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 14])**2, color = "red", label = r"$\nu = 14$", linestyle = "dashed")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 15])**2, color = "black", label = r"$\nu = 15$", linestyle = "dashed")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 16])**2, color = "slateblue", label = r"$\nu = 16$", linestyle = "dashdot")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 17])**2, color = "green", label = r"$\nu = 17$", linestyle = "dashdot")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 18])**2, color = "deepskyblue", label = r"$\nu = 18$", linestyle = "dashdot")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 19])**2, color = "orange", label = r"$\nu = 19$", linestyle = "dashdot")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 20])**2, color = "gold", label = r"$\nu = 20$", linestyle = "dashdot")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 21])**2, color = "blue", label = r"$\nu = 21$", linestyle = "dashdot")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 22])**2, color = "red", label = r"$\nu = 22$", linestyle = "dashdot")
plt.plot(tii, (1/norfinalstate) * np.abs(lista1[:, -Nlevels + 23])**2, color = "black", label = r"$\nu = 23$", linestyle = "dashdot")
#plt.plot(tii, np.abs(lista1[:, -Nlevels + 24])**2, color = "purple", label = r"$\nu = 24$", linestyle = "dashdot")

plt.xlabel(r"$r \sigma $", fontsize=17)
plt.ylabel(r"$\langle e _{\nu} \rangle$", fontsize=17)
plt.legend(bbox_to_anchor=(0.1, 0.9), loc="upper left", fontsize=10, fancybox=True, framealpha=1, borderpad=1, edgecolor="black")
plt.subplots_adjust(left=0.13, bottom=0.05, right=0.979, top=0.97)
plt.show()
