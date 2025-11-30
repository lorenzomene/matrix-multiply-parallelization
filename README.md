# Matrix Multiplication Parallelization

This project implements matrix multiplication using **MPI** (Message Passing Interface) and **OpenMP** for hybrid parallelization.

## Prerequisites

* GCC Compiler
* MPI implementation (e.g., MPICH or OpenMPI)

## 1. Compilation

Use the following command to build the project. This applies aggressive optimization flags (`-O3`, `-ffast-math`) and enables OpenMP.

```bash
mpicc -O3 -march=native -mtune=native -ffast-math -fstrict-aliasing -fopenmp -funroll-loops matrix_multiply.c -o out
```

## 2. Execution
Run the compiled executable using `mpiexec`.
> Update -n (number of processes) and the -hostfile path to match your specific cluster configuration.

```bash
mpiexec -n 7 -hostfile <hostfile> -npernode 1 ./out
```

| Flag          | Description                                                           |
| ------------- | --------------------------------------------------------------------- |
| O3            | Maximize optimization for speed.                                      |
| march=native  | Optimize code for the local CPU architecture.                         |
| fopenmp       | Enable OpenMP multi-threading support.                                |
| ffast-math    | Allow aggressive floating-point optimizations (may reduce precision). |
| funroll-loops | Unroll loops to reduce branch prediction overhead.                    |
