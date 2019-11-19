# Dbl_SHADE_python

Implementation of the Dbl-SHADE algorithm in Python.

[Full paper](https://doi.org/10.1109/SSCI.2017.8280959)

## Usage
Sample use can also be seen at the end of the file __main.py__. 
```python
dim = 10 #dimension size of the optimized problem
NP = NP = 18 * dim #population size (recomended setting)
maxFEs = 5000 #maximum number of objective function evaluations
F = 0.5
CR = 0.8
H = 10 #archive size
minPopSize = 4

sphere = Sphere(dim) #defined test function
de = Dbl_SHADE(dim, maxFEs, sphere, H, NP, minPopSize) #initialize Dbl-SHADE
resp = de.run() #run the optimization
print(resp) #print the results
```
Output ``resp`` then includes optimized values ``features`` and value of objective function ``ofv``. Also, the ``id`` of particle is included.

## File descriptions
* __main.py__
  * The main file contains the main class Dbl_SHADE and one sample test function class Sphere.
