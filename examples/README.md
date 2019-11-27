# Examples
This directory contains the examples of distributed computing in PyTorch and the package `dist_stat`.

- General examples
    - `mcpi-mpi-pytorch.py`: Pure PyTorch example of distributed Monte Carlo estimation of pi. 
    - `simple_example.py`: A simple example usage of `dist_stat`.
    - `test_distmat.py`: Further examples of `distmat`, the data structure for distributed matrices.
    - `test_distmm.py`: Matrix multiplication examples of `distmat`. 

- Nonnegative matrix factorization (NMF)
    - `test_nmf.py`: Multiplicative algorithm for NMF.
    - `test_nmf_pg.py`: Alternating projected gradient algorithm for NMF.
    - `test_nmf_pg_ridge.py`: Alternating projected gradient for NMF, with nonzero ridge penalty.
- Positron emissionn tomography (PET), a test dataset is provided in [data](../data).
    - `test_pet.py`: PET with squared difference penalty.
    - `test_pet_l1.py`: PET with TV penalty.
- L1-regularized Cox regression
    - `test_cox.py`: The Cox example with simulated data.
    - `test_cox_breslow_realdata_200k.py`: Code used for the UK BioBank dataset (additionally requires `pandas_plink` and `dask` packages).
- Multidimensional Scaling (MDS)
    - `test_euclidean.py`: A small example of all pairwise Euclidean distance computation used in MDS.
    - `test_mds.py`: The MDS example.
    
## How to run
In general, the code can be executed using the `mpirun` command from the MPI installation. For example, one can use the command 
```sh
mpirun -np 4 python mcpi-mpi-pytorch.py
```
to run `mcpi-mpi-pytorch.py` with four processes. 

For files beginning with `test_`, there are two common command line arguments to select device and data type (single-precision or double-precision):

- `--gpu`: use GPU for computation. Otherwise, they use CPU.
- `--double`: use double-precision floating-point arithmetics. Otherwise, they use single-precision.

For NMF, PET, Cox and MDS examples, an additional argument `--nosubnormal` can be used to force [subnormal numbers](https://en.wikipedia.org/wiki/Denormal_number) to zero.
For example, 
```sh
mpirun -np 4 python test_nmf.py --gpu
```
runs the computation with GPU in single-precision, and 
```sh
mpirun -np 4 python test_nmf.py --double --nosubnormal
```
runs the computation on CPU in double-precision, forcing any subnormal numbers to zero. 

Each applications have further command-line options to configure tolerance, number of iterations, size of simulated data, etc. 
Users can check the list of command-line options using the flag `--help`. For example, 
```sh 
python test_nmf.py --help
```
prints out the list of options used in the multiplicative algorithm for NMF:
```
usage: test_nmf.py [-h] [--gpu] [--double] [--nosubnormal] [--tol TOL]
                   [--rows M] [--cols N] [--r R] [--iter ITER]
                   [--set_from_master]

nmf testing

optional arguments:
  -h, --help         show this help message and exit
  --gpu              whether to use gpu
  --double           use this flag for double precision. otherwise single
                     precision is used.
  --nosubnormal      use this flag to avoid subnormal number.
  --tol TOL          error tolerance
  --rows M           number of rows
  --cols N           number of cols
  --r R              internal dim
  --iter ITER        max iter
  --set_from_master  samples are generated from the CPU of root: for obtaining
                     identical dataset for different settings.
```

### On Sun Grid Engine
See job scripts in [jobs](../jobs).


