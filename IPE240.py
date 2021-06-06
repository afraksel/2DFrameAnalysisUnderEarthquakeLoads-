import numpy as np
import warnings

warnings.filterwarnings('ignore')
from numpy import linalg as lin
from scipy.linalg import eig
from matplotlib import pyplot as plot

# PROBLEM #18.19 (Tedesco, Page 679)

m1 = 23.9837253  # (kNs2 / m)
m2 = 35.4861242  # (kNs2 / m)
m3 = 35.4861242  # (kNs2 / m)
m4 = 35.4861242  # (kNs2 / m)
m5 = 35.4861242  # (kNs2 / m)


M = np.array([
    [m1, 0, 0, 0, 0],
    [0, m2, 0, 0, 0],
    [0, 0, m3, 0, 0],
    [0, 0, 0, m4, 0],
    [0, 0, 0, 0, m5]
])  # (kNs2 / m)

E = 200000000  # kN / m^2
A = 0.00391   #m^2
I_IPE240 = 0.00003892  # m ^ 4
cg = 0.12  #m


L_1 = 4    #m
L_floors = 3  #m

k1 = 2 * (12 * E * I_IPE240 / (L_floors ** 3))  # kN / m (Roof)
k2 = 2 * (12 * E * I_IPE240 / (L_floors ** 3))   # kN / m (4th floor)
k3 = 2 * (12 * E * I_IPE240 / (L_floors ** 3))   # kN / m (3rd floor)
k4 = 2 * (12 * E * I_IPE240 / (L_floors ** 3))   # kN / m (2nd floor)
k5 = 2 * (12 * E * I_IPE240 / (L_1 ** 3))   # kN / m (1st floor)

K = np.array([
    [k1 + k2,     -k2,      0,        0,       0],
    [-k2,      k2 + k3,    -k2,       0,       0],
    [0,          -k3,     k3 + k4,    -k4,     0],
    [0,           0,        -k4,    k4 + k5,  -k5],
    [0,           0,         0,      -k5,      k5]
])  # kN / m

# Generate the C (Damping) matrix
# Input  Ksi(r)  Values --> C = diag(2M_r*Ksi_r*w_r)

Ksi_values = [0.05, 0.05, 0.05, 0.05, 0.05]

# Calculate C = diag(2M_r*Ksi_r*w_r)

D, V = eig(K, M)
wi_values = np.sqrt(np.array(D))

wi_NonZero_Values = wi_values[wi_values != 0 & np.isfinite(wi_values)]

OmegaValues = np.sort(wi_NonZero_Values)

M2 = np.dot(2,M)
KO = Ksi_values * OmegaValues
M2xKO = np.dot(M2, KO)

C = np.diag(M2xKO, 0)
#print(C.shape)

# Earthquake Loading (Northridge N-S component)

eq_data = np.loadtxt("nridge.dat")

Ground_Acceleration_percent_of_g = np.reshape(eq_data, (1, 3000), order='C')
# print(Ground_Acceleration_percent_of_g.shape)


# Determine the Total Time (in seconds)

T = 60
dt = T / len(Ground_Acceleration_percent_of_g[0])
Time = np.arange(0, (len(Ground_Acceleration_percent_of_g[0]) * dt), dt)
# print(len(Time))
Gravity_constant = 9.80665 # m / s^2
Acceleration_Data = Ground_Acceleration_percent_of_g * Gravity_constant # m / s^2
# print(Acceleration_Data.shape)

# Effective Load Vector
R = np.dot(M, np.ones((M.shape[1], 1)))
# print(R.shape)

Force = np.dot(R, Acceleration_Data)
#print(Force.shape)



# How Many Number of Degrees of Freedom (NDOF) Are There?
NDOF = len(M)

# Input Inıtıal Conditions (x0 and xd0) and initial force values, P0

x0 = np.zeros((1, len(M)))
xd0 = np.zeros((1, len(M)))
# print(xd0.shape)

# Calculate the initial acceleration, xdd0
P0 = Force[:, 0]
#print(P0.shape)

inv_M = lin.inv(M)
transposexd0 = np.transpose(xd0)
transposex0 = np.transpose(x0)
Cxtranposexd0 = np.dot(C, transposexd0)
Kxtranposex0 = np.dot(K, transposex0)

xdd0 = np.dot(inv_M, ((P0.reshape((5, 1))) - Cxtranposexd0 - Kxtranposex0))
#print(xdd0.shape)

# Preallocate memory for X, Xd, and Xdd
X = np.zeros((NDOF, len(Time)))  # Displacement-Time Histories
Xd = np.zeros((NDOF, len(Time)))  # Velocity-Time Histories
Xdd = np.zeros((NDOF, len(Time)))  # Acceleration-Time Histories

X[:, 0] = x0
#print(X[:,0].shape)
Xd[:, 0] = xd0
Xdd[:, 0] = xdd0.reshape((5,))
#print(X[:,0].shape)

# The integration constants

a0 = 1 / (dt * dt)
a1 = 1 / (2 * dt)
a2 = 2 * a0
a3 = 1 / a2

# Generate effective mass matrix
M_hat = (a0 * M) + (a1 * C)

# Calculate Initial X Values

transposexdd0 = np.transpose(xdd0)
# print(transposexdd0.shape)

X_dt = x0 - (dt * xd0) + (a3 * transposexdd0)
# print(X_dt.shape)

transposeX_dt = np.transpose(X_dt)
# print(transposeX_dt.shape)

# SOLUTION STEPS
# Perform the First Step:

d = K - (a2 * M)
e = np.dot(d, X[:, 0])
# print(e.shape)

f = (a0 * M) - (a1 * C)
g = np.dot(f, transposeX_dt)
# print(g.shape)

# print(Force[:,0].shape)
F_Hat_First = (Force[:, 0].reshape((5, 1))) - (e.reshape((5, 1))) - g
# print(F_Hat_First.shape)

invM_Hat = lin.inv(M_hat)

X[:, 1] = np.dot(invM_Hat, F_Hat_First).reshape((5,))
# print(X[:,1].shape)
Xd[:, 1] = a1 * (-transposeX_dt.reshape((5,)) + X[:, 1])  # Not Verified Yet
# print(Xd[:,1].shape)
Xdd[:, 1] = a0 * (np.transpose(X_dt).reshape((5,)) - 2 * X[:, 0] + X[:, 1])  # Not Verified Yet
# print(Xdd[:,1].shape)

# Perform the Subsequet Steps

for cycle in range(2, len(Time + 1)):
    F_hat = Force[:, cycle - 1] - np.dot(K - a2 * M, X[:, cycle - 1]) - np.dot(a0 * M - a1 * C, X[:, cycle - 2])
    # print(F_hat.shape)

    X[:, cycle] = np.dot(invM_Hat, F_hat)
    # print( X[:, cycle].shape)
    Xd[:, cycle] = a1 * (-X[:, cycle - 2] + X[:, cycle])
    # print( Xd[:, cycle].shape)
    Xdd[:, cycle] = a0 * (X[:, cycle - 2] - 2 * X[:, cycle - 1] + X[:, cycle])
    # print(Xdd[:, cycle].shape)

# Fill in the Last Steps of Velocity and Acceleration

F_hat_son = Force[:, -1] - np.dot(K - a2 * M, X[:, -1]) - np.dot(a0 * M - a1 * C, X[:, -2])
# print(F_hat_son.shape)

extra_X = np.dot(invM_Hat, F_hat_son)
Xd[:, -1] = a1 * (-X[:, -2] + extra_X)
Xdd[:, -1] = a0 * (X[:, -2] - 2 * X[:, -1] + extra_X)

# Plot Displacement-Time

plot.plot(Time, X[0])
plot.ylabel('Disp1')
plot.show()

plot.plot(Time, X[1])
plot.ylabel('Disp2')
plot.show()

plot.plot(Time, X[2])
plot.ylabel('Disp3')
plot.show()

plot.plot(Time, X[3])
plot.ylabel('Disp4')
plot.show()

plot.plot(Time, X[4])
plot.ylabel('Disp5')
plot.show()

#To solve for Base Shear and Base Moments at each floor level

S = np.array([
             [1,   0,   0,  0,  0],  # Roof Floor
             [1,   1,   0,  0,  0],   # 4th Floor
             [1,   1,   1,  0,  0],   # 3rd Floor
             [1,   1,   1,  1,  0],    # 2nd Floor
             [1,   1,   1,  1,  1]     # 1st Floor
                              ])

H = np.array([
             [3,   0,   0,   0,  0],      # Roof Floor Height Only
             [3,   3,   0,   0,  0],     # Roof Floor Height and 4th Floor Height
             [3,   3,   3,   0,  0],     # Roof Floor Height and 3rd Floor Height
             [3,   3,   3,   3,  0],     # Roof Floor Height and 2nd Floor Height
             [3,   3,   3,   3,  4],     # Roof Floor Height and 1st Floor Height
                                     ])

Fs = np.dot(K, X)  # Earthquake Forces at each floor level
# print(Fs).shape


BaseShear = np.dot(S, Fs)  # Base Shears at each floor level
# print(BaseShear.shape)

# Plot BaseShear-Time

plot.plot(Time, BaseShear[0])
plot.ylabel('BaseShear1')
plot.show()

plot.plot(Time, BaseShear[1])
plot.ylabel('BaseShear2')
plot.show()

plot.plot(Time, BaseShear[2])
plot.ylabel('BaseShear3')
plot.show()

plot.plot(Time, BaseShear[3])
plot.ylabel('BaseShear4')
plot.show()

plot.plot(Time, BaseShear[4])
plot.ylabel('BaseShear5')
plot.show()




BaseMoment = (np.dot(H, BaseShear))   # Base Moments at each floor level

# Plot BaseMoment-Time

plot.plot(Time, BaseMoment[0])
plot.ylabel('BaseMoment1')
plot.show()

plot.plot(Time, BaseMoment[1])
plot.ylabel('BaseMoment2')
plot.show()

plot.plot(Time, BaseMoment[2])
plot.ylabel('BaseMoment3')
plot.show()

plot.plot(Time, BaseMoment[3])
plot.ylabel('BaseMoment4')
plot.show()

plot.plot(Time, BaseMoment[4])
plot.ylabel('BaseMoment5')
plot.show()

# Preallocate memory for Maximum Response Values

Max_X = np.zeros((NDOF, 1))

for ij in range(0, NDOF):
    #print(np.amax(X[ij, len(X)]))
    if np.amax(X[ij, :]) >= np.abs(np.amin(X[ij, :])):
        Max_X[ij] = np.amax(X[ij, :])
    else:
        Max_X[ij] = np.abs(np.amin(X[ij, :]))


Max_Fs = np.dot(K, Max_X)
Max_BaseShear = np.dot(S, Max_Fs)
Max_BaseMoment = np.dot(H, Max_BaseShear)


#print(X.shape)
#print(np.amin(X[1, :]))
#print(np.abs(np.amin(X[0, :])))


#Shear Stress

ShearStress = BaseShear/A     # stress = F / A

#print(ShearStress.shape)


# Plot ShearStress-Time

plot.plot(Time, ShearStress[0])
plot.ylabel('ShearStress1')
plot.show()

plot.plot(Time, ShearStress[1])
plot.ylabel('ShearStress2')
plot.show()

plot.plot(Time, ShearStress[2])
plot.ylabel('ShearStress3')
plot.show()

plot.plot(Time, ShearStress[3])
plot.ylabel('ShearStress4')
plot.show()

plot.plot(Time, ShearStress[4])
plot.ylabel('ShearStress5')
plot.show()


#Bending Stress

BendingStress = BaseMoment * cg / I_IPE240

#print(BendingStress)

# Plot ShearStress-Time

plot.plot(Time, BendingStress[0])
plot.ylabel('BendingStress1')
plot.show()

plot.plot(Time, BendingStress[1])
plot.ylabel('BendingStress2')
plot.show()

plot.plot(Time, BendingStress[2])
plot.ylabel('BendingStress3')
plot.show()

plot.plot(Time, BendingStress[3])
plot.ylabel('BendingStress4')
plot.show()

plot.plot(Time, BendingStress[4])
plot.ylabel('BendingStress5')
plot.show()


MaxBendingStress = Max_BaseMoment * cg / I_IPE240


#RESULTS

resultsIPE240 = f"KDOF 1 Max Displacement is {np.max(np.abs(X[0, :]))} , Max Fs is {np.amax(np.abs(Fs[0, :]))} kN, MaxStoryShear is {np.amax(np.abs(BaseShear[0, :]))} kN, MaxMoment is {np.amax(np.abs(BaseMoment[0, :]))} kNm, Max Shear Stress is {np.max(np.abs(ShearStress[0, :]))} kPa, Max Bending Stress is {np.max(np.abs(BendingStress[0, :]))} kPa \nKDOF 2 Max Displacement is {np.max(np.abs(X[1, :]))}, Max Fs is {np.amax(np.abs(Fs[1, :]))} kN, MaxStoryShear is {np.amax(np.abs(BaseShear[1, :]))} kN, MaxMoment is {np.amax(np.abs(BaseMoment[1, :]))} kNm,  Max Shear Stress is {np.max(np.abs(ShearStress[1, :]))} kPa, Max Bending Stress is {np.max(np.abs(BendingStress[1, :]))} kPa \nKDOF 3 Max Displacement is {np.max(np.abs(X[2, :]))}, Max Fs is {np.amax(np.abs(Fs[2, :]))} kN, MaxStoryShear is  {np.amax(np.abs(BaseShear[2, :]))} kN, MaxMoment is {np.amax(np.abs(BaseMoment[2, :]))} kNm,  Max Shear Stress is {np.max(np.abs(ShearStress[2, :]))} kPa, Max Bending Stress is {np.max(np.abs(BendingStress[2, :]))} kPa \nKDOF 4 Max Displacement is {np.max(np.abs(X[3, :]))}, Max Fs is {np.amax(np.abs(Fs[3, :]))} kN, MaxStoryShear is  {np.amax(np.abs(BaseShear[3, :]))} kN, MaxMoment is {np.amax(np.abs(BaseMoment[3, :]))} kNm, Max Shear Stress is {np.max(np.abs(ShearStress[3, :]))} kPa, Max Bending Stress is {np.max(np.abs(BendingStress[3, :]))} kPa \nKDOF 5 Max Displacement is {np.max(np.abs(X[4, :]))}, Max Fs is {np.amax(np.abs(Fs[4, :]))} kN, MaxStoryShear is  {np.amax(np.abs(BaseShear[4, :]))} kN, MaxMoment is {np.amax(np.abs(BaseMoment[4, :]))} kNm, Max Shear Stress is {np.max(np.abs(ShearStress[4, :]))} kPa, Max Bending Stress is {np.max(np.abs(BendingStress[4, :]))} kPa"

with open('resultsIPE240.txt', "w") as f:
    f.write(resultsIPE240)

