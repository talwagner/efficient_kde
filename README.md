# Space and Time Efficient Kernel Density Estimation in High Dimensions

This is an implementation (by authors) of the algorithm from the paper:
"Space and Time Efficient Kernel Density Estimation in High Dimensions"
by Arturs Backurs, Piotr Indyk and Tal Wagner, published in NeurIPS 2019.

For background on locality-sensitive hashing based estimators for kernel density, see the paper:
"Hashing-Based-Estimators for Kernel Density in High Dimensions"
by Moses Charikar and Paris Siminelakis, published in FOCS 2017.

The algorithm is implemented for the l_1-Laplacian kernel, k(x,y) = exp(||x-y|| / sigma), where ||.|| is the l_1 norm and sigma is the bandwidth parameter. 
In the preprocessing stage, it gets a dataset of high-dimensional points x_1,...,x_n, the bandwidth parameter sigma, and a parameter L (see below), and constructs the data structure.
In the query stage, it gets a query point y and returns an estimate for the kernel density: (1/n) * k(x,y)=sum_i k(x_i,y).

The parameter L is the number of hash tables constructed in the preprocessing stage, and the number of averaged samples in the query stage. The larger it is, the more accurate are the returned estimates, but a cost of larger preprocessing time, query time, and space usage.
