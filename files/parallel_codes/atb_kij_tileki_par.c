 void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk) 
 {
     int i, j, k, i_tile, k_tile;
     int tile_size = 50; 
#pragma omp parallel private(i,j,k,i_tile,k_tile)
     { 
  for (k_tile = 0; k_tile < Nk; k_tile += tile_size)
    for (k = k_tile; k < k_tile + tile_size && k < Nk; k++)
#pragma omp for
     for (i_tile = 0; i_tile < Ni; i_tile += tile_size)
        for (i = i_tile; i < i_tile + tile_size && i < Ni; i++){
         //int r = Nj % 2;
         for (j = 0; j < Nj; j++)
            C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
         //for (j = r; j < Nj; j+=2){
           // C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
            //C[i*Nj+j+1] += A[k*Ni+i] * B[k*Nj+j+1];
         //}
        }
}
 }
 
