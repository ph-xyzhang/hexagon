import numpy as np
import matplotlib.pyplot as plt

# Define parameters 
a1 = 2.22
a2 = 2.24
alpha1 = 96.5 * (np.pi / 180)
beta = 71.96 * (np.pi / 180)
b1 =  2 * np.pi / 4.35
b2 =  2 * np.pi / 3.32
m = 3000

# Define constants for hopping energies 
t1 = -1.220
t2 = 3.665
t3 = -0.205
t4 = -0.105
t5 = -0.055

t1v = 0.295
t2v = 0.273
t3v = -0.151
t4v = -0.091

# Initialize arrays to store eigenvalues
E = np.zeros((m, 8), dtype=np.complex128)

# Initialize kx, ky arrays
kx = np.zeros(m,dtype=np.complex128)
ky = np.zeros(m,dtype=np.complex128)

# Define path in the Brillouin zone
k = np.linspace(0, 3000, 3000)

# path 1
kx[0:630] = np.linspace(0, 0, 630)
ky[0:630] = np.linspace(0, b1/2, 630)

# path 2
kx[630:1500] = np.linspace(0, b2/2, 870)
ky[630:1500] = np.linspace(b1/2, b1/2, 870)

# path 3
kx[1500:2130] = np.linspace(b2/2, b2/2, 630)
ky[1500:2130] = np.linspace(b1/2, 0, 630)

# path 4
kx[2130:3000] = np.linspace(b2/2, 0, 870)
ky[2130:3000] = np.linspace(0, 0, 870)


# Calculate eigenvalues for each path
for i in range(m):
    # Calculate elements of the Hamiltonian matrix for given kx[i] and ky[i]
    tAA = 0

    tAB = 2 * t1 * np.cos(a1 * np.sin(alpha1 / 2) * kx[i]) * np.exp(-1j * a1 * np.cos(alpha1 / 2) * ky[i]) + \
          2 * t3 * np.cos(a1 * np.sin(alpha1 / 2) * kx[i]) * np.exp(1j * (2 * a2 * np.cos(beta) + a1 * np.cos(alpha1 / 2)) * ky[i]) 
    
    ctAB = np.conjugate(tAB)

    tAC = t2 * np.exp(1j * a2 * np.cos(beta) * ky[i]) + \
          t5 * np.exp(-1j * (a2 * np.cos(beta) + 2 * a1 * np.cos(alpha1 / 2)) * ky[i]) 

    ctAC = np.conjugate(tAC)

    tAD = 4 * t4 * np.cos(a1 * np.sin(alpha1 / 2) * kx[i]) * np.cos((a1 * np.cos(alpha1 / 2) + a2 * np.cos(beta)) * ky[i])

    tACp = (2 * t1v * np.exp(1j * a2 * np.cos(beta) * ky[i]) + 2 * t4v * np.exp(-1j * (2 * a1 * np.cos(alpha1 / 2) +  a2 * np.cos(beta)) * ky[i])) * \
           np.cos(a1 * np.sin(alpha1 /2) * kx[i])
    
    ctACp = np.conjugate(tACp)

    tADp = (4 * t3v * np.cos(2 * a1 * np.sin(alpha1 /2) * kx[i]) + 2 * t2v) * np.cos((a1 * np.cos(alpha1 / 2) + a2 * np.cos(beta)) * ky[i])

    ctADp = np.conjugate(tADp)

    # Construct Hamiltonian matrix 
    matrix = np.array([[tAA,   tAB,   tAD,  tAC, 0,    0,   tADp,  tACp],
                       [ctAB,  tAA,   ctAC, tAD, 0,    0,   ctACp, tADp],
                       [tAD,   tAC,   tAA,  tAB, 0,    0,   0,     0],
                       [ctAC,  tAD,   ctAB, tAA, 0,    0,   0,     0],
                       [0,     0,     0,    0,   tAA,  tAB, tAD,   tAC],
                       [0,     0,     0,    0,   ctAB, tAA, ctAC,  tAD],
                       [ctADp, tACp,  0,    0,   tAD,  tAC, tAA,   tAB],
                       [ctACp, ctADp, 0,    0,   ctAC, tAD, ctAB,  tAA]])

    # Calculate eigenvalues
    eigenvalues, _ = np.linalg.eig(matrix)

    # Sort eigenvalues (not necessary for plotting the lowest band directly)
    eigenvalues.sort()

    # Store eigenvalues
    E[i, :] = eigenvalues

# Plotting
fig, ax = plt.subplots()
ax.plot(k, E[:, 0].real, color='blue') 
ax.plot(k, E[:, 1].real, color='blue')
ax.plot(k, E[:, 2].real, color='blue')
ax.plot(k, E[:, 3].real, color='blue')
ax.plot(k, E[:, 4].real, color='blue') 
ax.plot(k, E[:, 5].real, color='blue')
ax.plot(k, E[:, 6].real, color='blue')
ax.plot(k, E[:, 7].real, color='blue')
ax.set_xticks([0, 630, 1500, 2130, 3000])
ax.set_xlim(0, m)
ax.set_xticklabels(['$\Gamma$', '$X$', '$S$', '$Y$', '$\Gamma$'])
ax.set_ylabel('Energy')
plt.show()
