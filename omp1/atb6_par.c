void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k;
    int unroll_factor = 2;
    int tile_size = 64;

    #pragma omp parallel for private(i,j,k)
    for (i = 0; i < Ni; i += tile_size)
    {
        for (k = 0; k < Nk; k++)
        {
            for (j = 0; j < Nj; j += unroll_factor)
            {
                C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
                C[i*Nj+j+1] += A[k*Ni+i] * B[k*Nj+j+1];
            }
            for (; j < Nj; j++)
            {
                C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
            }
        }
    }
}

