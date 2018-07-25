#!/bin/bash --login

#SBATCH --job-name=imstack_test
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:10:00
#SBATCH --account=pawsey0001
#SBATCH --partition=debugq
#SBATCH --export=MYSCRATCH

##
## Set the environment
##----------------------------------------
module unload gcc/4.8.5 sandybridge
module load broadwell gcc/4.8.5 python h5py h5py-cache astropy

##
## Setup working directory
##-------------------------
workdir=${MYSCRATCH}/imstack
if [ -d "${workdir}" ]; then
   rm -rf ${workdir}
fi
mkdir ${workdir} 
lfs setstripe -c 2 ${workdir}

cp make_imstack.py ${workdir}
cd ${workdir}
ln -s /group/mwasci/jmorgan/ips/make_imstack_test/*.fits .

##
## Run the simulation
##--------------------
time srun -N 1 -n 1 --export=ALL python ./make_imstack.py 1114303032 --suffixes=image --bands=118-129 

