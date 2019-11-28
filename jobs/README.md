# jobs

Job scripts for submission to the Sun Grid Engine.
The number right before the extension `.job` denotes the number of nodes to be used by the job, when running on the virtual cluster created by AWS ParallelCluster as in [our configuration](../parallelcluster/config).

Example commands to submit the jobs:
```sh 
qsub mcpi-2.job
```
for Monte Carlo pi estimation, 

```sh
qsub nmf-10.job 200000 100000 20
```
for NMF with alternating projected gradient, 200,000 rows, 100,000 columns and the inner dimension of 20, 

```sh
qsub pet_v2-10.job ../data/pet_gen_v2_100_120.npz
```
for PET with TV penalization, with the dataset in [data](../data),

```sh
qsub cox-10.job 
```
for Cox regression with 200,000 samples on 100,000 dimensions, and

```sh
qsub mds-10.job 200000 100000 20
```
for MDS with 200,000 datapoints in 100,000 dimension, with the resulting dimension of 20.




