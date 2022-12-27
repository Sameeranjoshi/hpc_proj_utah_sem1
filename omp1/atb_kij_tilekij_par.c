 void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk) 
 {
     int i, j, k, i_tile, j_tile, k_tile;
     int tile_size = 32; 
#pragma omp parallel private(i,j,k,i_tile, j_tile, k_tile)
     { 
  for (k_tile = 0; k_tile < Nk; k_tile += tile_size)
    for (k = k_tile; k < k_tile + tile_size && k < Nk; k++)
#pragma omp for
     for (i_tile = 0; i_tile < Ni; i_tile += tile_size)
      for (j_tile = 0; j_tile < Nj; j_tile += tile_size)
        for (i = i_tile; i < i_tile + tile_size && i < Ni; i++)
         for (j = j_tile; j < j_tile + tile_size && j < Nj; j++)
            C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
}
 }
 
