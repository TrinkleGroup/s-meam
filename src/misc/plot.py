"""Used for plotting timing results comparing LAMMPS / Python"""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("timing_tests.dat")

x = data[:,0]
lammpsy = data[:,1]
pythony = data[:,2]

lammps_setup_time = 1.6e-4
lammpsy -= x*lammps_setup_time

A = np.ones((len(data), 2))
A[:,1] = x.copy()

lammpsb, lammpsm = np.linalg.lstsq(A, lammpsy)[0]
pythonb, pythonm = np.linalg.lstsq(A, pythony)[0]

intersection = (pythonb - lammpsb)/(lammpsm - pythonm)

lammpsf = lambda x: lammpsb + lammpsm*x
pythonf = lambda x: pythonb + pythonm*x

xplotting = np.linspace(data[0,0], data[len(data)-1,0], num=1000)
lammpsfit = np.array(list(map(lammpsf, xplotting)))
pythonfit = np.array(list(map(pythonf, xplotting)))

plt.figure()
plt.title("Worker performance test (1 struct, N pots)")
plt.plot(xplotting, lammpsfit, '-', label='lammps')
plt.plot(xplotting, pythonfit, '-', label='python')
plt.plot((intersection, intersection), (0, lammpsf(intersection)), '--r')
plt.plot(intersection, lammpsf(intersection), 'ro', label=r'$N '
                                    r'\approx$ {0}'.format(int(intersection)))
#plt.plot(x, lammpsy, 'ro')
#plt.plot(x, pythony, 'bo')
plt.xlabel("Number of potentials")
plt.ylabel("Time (s)")
plt.legend()
plt.show()


# lammpsfit_normed = np.zeros(lammpsfit.shape)

# for i in range(len(lammpsfit_normed)):
#     lammmpsfit_normed = lammpsfit[i] / (xplotting[i]*0.02441)

# plt.figure()
# plt.plot(xplotting, lammpsfit_normed, label='lammps')
# plt.plot(xplotting, pythonfit/1.441, label='python')
# plt.legend()
# plt.show()

#diff = lammpsfit - pythonfit
#diffnormed = np.divide(diff,xplotting)

#plt.figure()
#plt.plot(xplotting, abs(diffnormed))
#plt.show()
