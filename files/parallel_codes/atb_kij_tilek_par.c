#define min(a,b) (((a)<(b))?(a):(b))
void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk) 
{
    int i, j, k;

    // tile size
    int tile_size = 16;

    #pragma omp parallel private(i,j,k)
    {   
        for (k = 0; k < Nk; k += tile_size)
        {
            // compute the actual size of the tile
            int tile_k = min(tile_size, Nk - k);

            #pragma omp for
            for (i = 0; i < Ni; i++)
            {   
                for (j = 0; j < Nj; j++)
                {
                    // initialize the sum to 0
                    double sum = 0.0;

                    // compute the sum for the current tile
                    for (int t = k; t < k + tile_k; t++)
                    {
                        sum += A[t*Ni+i] * B[t*Nj+j];
                    }

                    // update C[i][j] with the sum
                    C[i*Nj+j] += sum;
                }   
            }   
        }
    }   
}

