void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k, it;
    int unroll_factor = 4;
    int tile=16;
    #pragma omp parallel for private(i,j,k)
    for(it = 0; it < Ni; it += tile){
        
    for (i = it; i < Ni && it + tile; i++)
    {
        for (k = 0; k < Nk; k++)
        {
            int r = Nj % 4;
            for (j=0; j < r; j++)
                C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
            for (j = r; j < Nj; j += unroll_factor)
            {
                C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
                C[i*Nj+j+1] += A[k*Ni+i] * B[k*Nj+j+1];
                C[i*Nj+j+2] += A[k*Ni+i] * B[k*Nj+j+2];
                C[i*Nj+j+3] += A[k*Ni+i] * B[k*Nj+j+3];
            }
        }
    }
}
}
