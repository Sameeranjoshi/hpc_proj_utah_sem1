# hpc_proj_utah_sem1

# How to run this project

The final versions are in the folder `final_files`, various other folders have other scripts which we used for scrapping the machines which are working, scripts to test and run the various inputs.

The final optimized codes are in the folder `final_files` the way to run each of the questions is 

```
cd final_files/
gcc -O3 -fopenmp atb_main.c atb_par.c
gcc -O3 -fopenmp atbt_main.c atbt_par.c

module load cuda
nvcc -O3 atb_main.cu atb_par.cu
nvcc -O3 atbt_main.cu atbt_par.cu

```

To see the reports we have generated check `report` for OpenMP &  `report_gpu` for CUDA codes.

We have an [excel](https://docs.google.com/spreadsheets/d/1ir2cIa9oyMpzsyp6Z6ynQKn3QNsJmzfG10438fz-rAs/edit?usp=sharing) for this above reports to summarize it.

Please read the report file **HPC_Final_Project_u1418973_and_u1419864.pdf**
