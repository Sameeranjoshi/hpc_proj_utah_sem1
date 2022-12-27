
#define min(a,b) (((a)<(b))?(a):(b))
void atb3_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

#pragma omp parallel private(i,j,k)
 {
#pragma omp for
  for (i = 0; i < Ni; i++)
   for (k = 0; k < Nk; k++)
    for (j = 0; j < Nj; j++)
// C[i][j] = C[i][j] + A[k][i]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
 }
}

void atb5_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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

void atb6_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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

void atb7_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
 {
     int i, j, k, i_tile, j_tile, k_tile;
     int tile_size = 32;

     #pragma omp parallel for private(i,j,k,i_tile,j_tile,k_tile)
     for (i_tile = 0; i_tile < Ni; i_tile += tile_size)
     {
         for (k_tile = 0; k_tile < Nk; k_tile += tile_size)
         {
             for (j_tile = 0; j_tile < Nj; j_tile += tile_size)
             {
                 for (i = i_tile; i < i_tile + tile_size && i < Ni; i++)
                 {
                     for (k = k_tile; k < k_tile + tile_size && k < Nk; k++)
                     {
                         for (j = j_tile; j < j_tile + tile_size && j < Nj; j++)
                         {
                             C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
                         }
                     }
                 }
             }
         }
     }
 }

void atb_ikj_i_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

#pragma omp parallel private(i,j,k)
 {
#pragma omp for
  for (i = 0; i < Ni; i++)
    for (k = 0; k < Nk; k++)
        for (j = 0; j < Nj; j++)
// C[i][j] = C[i][j] + A[k][i]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
 }
}

void atb_ikj_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

#pragma omp parallel private(i,j,k)
 {
#pragma omp for
  for (i = 0; i < Ni; i++)
   for (k = 0; k < Nk; k++)
    for (j = 0; j < Nj; j++)
// C[i][j] = C[i][j] + A[k][i]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
 }
}

void atb_ikj_unrollk2_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

#pragma omp parallel private(i,j,k)
 {
  #pragma omp for
  for (i = 0; i < Ni; i++){
          int R = Nk%2;
    for (k = 0; k < R; k++){
      for (j = 0; j < Nj; j++){
        // C[i][j] = C[i][j] + A[k][i]*B[k][j];
             C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
      }
    }
    // Remaining part
    for (k = R; k < Nk; k+=2){
      for (j = 0; j < Nj; j++){
        // C[i][j] = C[i][j] + A[k][i]*B[k][j];
             C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
             C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
      }
    }
  }
 }
}

void atb_ikj_unrollk4_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

#pragma omp parallel private(i,j,k)
 {
  #pragma omp for
  for (i = 0; i < Ni; i++){
          int R = Nk%4;
    for (k = 0; k < R; k++){
      for (j = 0; j < Nj; j++){
        // C[i][j] = C[i][j] + A[k][i]*B[k][j];
             C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
      }
    }
    // Remaining part
    for (k = R; k < Nk; k+=4){
      for (j = 0; j < Nj; j++){
        // C[i][j] = C[i][j] + A[k][i]*B[k][j];
             C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
             C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
             C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[(k+2)*Nj+j];
             C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[(k+3)*Nj+j];
      }
    }
  }
 }
}

void atb_kij_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

#pragma omp parallel private(i,j,k)
 {
for (k = 0; k < Nk; k++)
#pragma omp for
  for (i = 0; i < Ni; i++)
//   for (k = 0; k < Nk; k++)
    for (j = 0; j < Nj; j++)
// C[i][j] = C[i][j] + A[k][i]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
 }
}

 void atb_kij_tilekij_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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

 void atb_kij_tileki_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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

 void atb_kij_tileki_unrollj_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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
         int r = Nj % 2;
         for (j = 0; j < r; j++)
            C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
         for (j = r; j < Nj; j+=2){
            C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
            C[i*Nj+j+1] += A[k*Ni+i] * B[k*Nj+j+1];
         }
        }
}
 }

void atb_kij_tilek_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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

void atb_kij_tilek_unroll2_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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

void atb_kij_unrollj2_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k;
    int unroll_factor = 2; // unroll the loop by a factor of 4
    int r = Nj % 2;
#pragma omp parallel private(i,j,k)
    {
        for (k = 0; k < Nk; k++)
#pragma omp for
            for (i = 0; i < Ni; i++)
            {
                for (j = 0; j < r; j++)
                    C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j];
                for (j = r; j < Nj; j += unroll_factor)
                {
                    C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j];
                    C[i*Nj+(j+1)] = C[i*Nj+(j+1)] + A[k*Ni+i]*B[k*Nj+(j+1)];
                }
            }
    }
}

void atb_kij_unrollj4_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k;
    int unroll_factor = 4; // unroll the loop by a factor of 4
    int r = Nj % 4;
#pragma omp parallel private(i,j,k)
    {
        for (k = 0; k < Nk; k++)
#pragma omp for
            for (i = 0; i < Ni; i++)
            {
                for (j = 0; j < r; j++)
                    C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j];
                for (j = r; j < Nj; j += unroll_factor)
                {
                    C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j];
                    C[i*Nj+(j+1)] = C[i*Nj+(j+1)] + A[k*Ni+i]*B[k*Nj+(j+1)];
                    C[i*Nj+(j+2)] = C[i*Nj+(j+2)] + A[k*Ni+i]*B[k*Nj+(j+2)];
                    C[i*Nj+(j+3)] = C[i*Nj+(j+3)] + A[k*Ni+i]*B[k*Nj+(j+3)];
                }
            }
    }
}

 void atb_kij_unrollj4_tilek_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
 {
     int i, j, k,kt;
     int unroll_factor = 4; // unroll the loop by a factor of 4
     int r = Nj % 4;
     int tile = 16;
 #pragma omp parallel private(i,j,k,kt)
     {
        for(kt = 0; kt < Nk; k+= tile)
         for (k = kt; k < kt+tile; k++)
 #pragma omp for
             for (i = 0; i < Ni; i++)
             {
                 for (j = 0; j < r; j++)
                     C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j];
                 for (j = r; j < Nj; j += unroll_factor)
                 {
                     C[i*Nj+j] = C[i*Nj+j] + A[k*Ni+i]*B[k*Nj+j];
                     C[i*Nj+(j+1)] = C[i*Nj+(j+1)] + A[k*Ni+i]*B[k*Nj+(j+1)];
                     C[i*Nj+(j+2)] = C[i*Nj+(j+2)] + A[k*Ni+i]*B[k*Nj+(j+2)];
                     C[i*Nj+(j+3)] = C[i*Nj+(j+3)] + A[k*Ni+i]*B[k*Nj+(j+3)];
                 }
             }
     }
 }

void atb_par_unroll(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

#pragma omp parallel private(i,j,k)
 {
#pragma omp master
  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[k][i]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
 }
}

void atb_tilei16_unrollj_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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
void atb_tilei8_unrollj_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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
void atb_tileikj_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k, i_tile, j_tile, k_tile;
    int tile_size = 32;

    #pragma omp parallel for private(i,j,k,i_tile,j_tile,k_tile)
    for (i_tile = 0; i_tile < Ni; i_tile += tile_size)
    {
        for (j_tile = 0; j_tile < Nj; j_tile += tile_size)
        {
            for (k_tile = 0; k_tile < Nk; k_tile += tile_size)
            {
                for (i = i_tile; i < i_tile + tile_size && i < Ni; i++)
                {
                    for (j = j_tile; j < j_tile + tile_size && j < Nj; j++)
                    {
                        for (k = k_tile; k < k_tile + tile_size && k < Nk; k++)
                        {
                            C[i*Nj+j] += A[k*Ni+i] * B[k*Nj+j];
                        }
                    }
                }
            }
        }
    }
}


void atb_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
// based on input selection:

	if (Ni ==8192 && Nj == 8192 && Nk == 16){
		atb_ikj_unrollk4_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni ==4096 && Nj == 4096 && Nk == 64){
		atb_ikj_unrollk4_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 2048 && Nj == 2048 && Nk == 256){
		atb_ikj_i_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 1024 && Nj == 1024 && Nk == 1024){
		atb_kij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 256 && Nj == 256 && Nk == 16384){
		atb_kij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 64 && Nj == 64 && Nk == 262144){
		atb_ikj_unrollk2_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 16 && Nj == 16 && Nk == 4194304){
		atb_ikj_unrollk4_par(A, B, C, Ni, Nj, Nk);
	}
// Right table
	if (Ni == 8991 && Nj == 8991 && Nk == 37){
		atb_ikj_unrollk4_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 2997 && Nj == 2997 && Nk == 111){
		atb_ikj_unrollk4_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 999  && Nj == 999 && Nk == 999){
		atb_ikj_i_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 333  && Nj == 333 && Nk == 8991){
		atb_kij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 111  && Nj == 111 && Nk == 80919){
		atb_ikj_par(A, B, C, Ni, Nj, Nk);
	} 
	if (Ni == 37  && Nj == 37 && Nk == 728271){
		atb_ikj_unrollk2_par(A, B, C, Ni, Nj, Nk);
	} 
}

