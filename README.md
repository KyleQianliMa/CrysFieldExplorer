# CrysFieldExplorer
CrisFieldExplorer is a fast-converging Python package for optimizing crystal field parameters.

The novalty of CrysFieldExplorer is it adopts a new loss function using theory of characteristic polynomials. By adopting this loss function it can globaly optimize the CEF hamiltonian with Neutron + any other experimental data and does not rely much on accurate starting value, which is usually estimated from point charge models.

A comparsion of the new loss and traditional $\chi^2$ loss has been displayed below.

The details of this program can be found at https://scripts.iucr.org/cgi-bin/paper?S1600576723005897.

A comparison of the new loss function $L_{spectrum}$ vs traditional $\chi^2$ loss along a random line in a 15 dimensional parameter space.
![alt text](https://raw.githubusercontent.com/KyleQianliMa/CrysFieldExplorer/blob/main/images/loss.jpg)
