void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k;
    int unroll_factor = 4;

    #pragma omp parallel for private(i,j,k)
    for (i = 0; i < Ni; i++)
    {
        for (k = 0; k < Nk; k++)
        {
            for (j = 0; j < Nj; j += unroll_factor)
            {
                C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
                C[i*Nj+j+1] += A[k*Ni+i] * B[k*Nj+j+1];
                C[i*Nj+j+2] += A[k*Ni+i] * B[k*Nj+j+2];
                C[i*Nj+j+3] += A[k*Ni+i] * B[k*Nj+j+3];
            }
            for (; j < Nj; j++)
            {
                C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
            }
        }
    }
}

