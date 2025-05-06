import sys
sys.path.append('/home/ucapcdr/Scratch/ETPA')
import time
import numpy as np
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
Modes = 501 #Number of modes for the incoming field
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

ss = 11
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
    return ((np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k + k1 - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)) + (np.exp(-((k1 - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k1 + k - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k1 + k)*r0) - 1j*np.sin((k1 + k)*r0)) * (1/noren))) * 0.5


    
    
#np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k + k1 - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)     
    
#np.exp(-(((k - k1)*0.5)**2)/(4 * sigmaev**2)) * np.exp(-((k + k1 - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)       
    
#np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k1 - k0)**2)/(4 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)

#((np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k + k1 - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)) + (np.exp(-((k1 - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k1 + k - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k1 + k)*r0) - 1j*np.sin((k1 + k)*r0)) * (1/noren))) * 0.5
    

##########################################################################
########:::::::::::::::  Franck-Condon Factors  :::::::::::::::::#########
##########################################################################

#Importing of files

#Franck-Condon Factors g->m
lista2 = []

nus = np.array(range(Nlevels))

lista2 = np.loadtxt('/lustre/scratch/scratch/ucapcdr/ETPA/Ff_Factors_gm_63.txt')
lista2 = np.array(lista2, dtype=float)

lista22 = []

for i in range(Nlevels):
    lista22.append(lista2[i])
    
lista2 = lista22

lista2 = np.array(lista2, dtype=float)

#Franck-Condon Factors m->e
Fnn = np.zeros((Nlevels, Nlevels))

Fnn = np.loadtxt('/lustre/scratch/scratch/ucapcdr/ETPA/Ff_Factors_me_50_rev_1i.txt')

Fnn1 = np.zeros((Nlevels, Nlevels))

for i in range(Nlevels):
    for j in range(Nlevels):
        Fnn1[i, j] = Fnn[j, i]
        
Fnn = Fnn1


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
    lista1.append(r.integrate(r.t+dt))
#    lista1.append(r.integrate(r.t+dt)[Modes * (Modes + Nlevels) : Modes * (Modes + Nlevels + 1)])

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




#-----------------------------------------------------------------------#
#-------------------# Cut of the intermediate states #------------------#
#-----------------------------------------------------------------------#

psi1pm = lista1[:, Modes ** 2 : Modes * (Modes + Nlevels)]


print(np.shape(psi1pm))

# Calculate the indices to sum: [0, N+1, 2*(N+1), 3*(N+1), ..., M*(N+1)] for intermediate level N = 0
max_index = psi1pm.shape[1]  # The number of columns
indices_lvl0 = [(i * (Nlevels)) for i in range(Modes) if (i * (Nlevels)) < max_index]


# Now it is necessary to add an index to generalize to the other levels
def indiceslvl(l):
    indices_lvl = [(i * (Nlevels)) + l for i in range(Modes) if (i * (Nlevels)) + l < max_index]
    return indices_lvl


# Select the columns using the indices
def psi1pm_kn(l):
    psi1pm_knl = np.abs(psi1pm[:, indiceslvl(l)])**2
    return psi1pm_knl
# Sum the selected columns
def psi1pm_l(l):
    psi1pm_lv = np.sum(psi1pm_kn(l), axis=1)
    return psi1pm_lv
    
# Calculate the square modulus to obtain the probability
def psi1pm_l_mod(l):
    psi1pm_lv_mod = psi1pm_l(l)
    return psi1pm_lv_mod


#Definition of the name of the file which contains the part of lista1 corresponding to psi1pm
name_solutions = f'/lustre/scratch/scratch/ucapcdr/ETPA/solutions_MO/inter_solutions_MO_{Modes}modes_{t0}to{t1}_dt{dt}_ss{ss}.txt'

#Save the total array of all solutions
np.savetxt(name_solutions, psi1pm)

########################################
 






































