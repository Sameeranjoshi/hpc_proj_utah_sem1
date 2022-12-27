
// 
void atbt_ijk_i_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel private(i,j,k)
 {
#pragma omp for
  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[k][i]*B[j][k];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
 }
}

void atbt_ikj_i_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel private(i,j,k)
 {
#pragma omp for
  for (i = 0; i < Ni; i++)
    for (k = 0; k < Nk; k++)
   for (j = 0; j < Nj; j++)
// C[i][j] = C[i][j] + A[k][i]*B[j][k];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
 }
}

void atbt_ikj_tilebase_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
  int R, UF = 4;
  const int tile_size = 128;

#pragma omp parallel private(i,j,k)
{
  #pragma omp for schedule(static, tile_size)
  for (i=0;i<Ni;i++)
  {
    for (k=0;k<Nk;k++)
    {
      for (j=0;j<Nj;j++)
      {
        C[i*Nj+j] += A[k*Ni+i]*B[j*Nk+k];
      }
    }
  }
}

}

void atbt_ikj_tileij_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k, i_tile, j_tile, k_tile;
  int R, UF = 4;
  const int tile_size = 128;

#pragma omp parallel private(i, j, k, i_tile, j_tile, k_tile)
{
  // loop over the tiles of the outermost loop
  for (int i_tile = 0; i_tile < Ni; i_tile += tile_size) {
    // determine the bounds of the tile
    int i_tile_end = i_tile + tile_size;
    i_tile_end = (i_tile_end > Ni) ? Ni : i_tile_end;

    // loop over the tiles of the innermost loop
    for (int j_tile = 0; j_tile < Nj; j_tile += tile_size) {
      // determine the bounds of the tile
      int j_tile_end = j_tile + tile_size;
      j_tile_end = (j_tile_end > Nj) ? Nj : j_tile_end;

      // perform the matrix multiplication within the tile
      #pragma omp for
      for (int i = i_tile; i < i_tile_end; ++i) {
        for (int k = 0; k < Nk; ++k) {
          for (int j = j_tile; j < j_tile_end; ++j) {
            C[i*Nj+j] += A[k*Ni+i]*B[j*Nk+k];
          }
        }
      }
    }
  }
}
}


void atbt_ikj_tileij_unrollk_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k, i_tile, j_tile, k_tile;
  int R, UF = 2;
  const int tile_size = 128;

#pragma omp parallel private(i, j, k, i_tile, j_tile, k_tile)
{
  // loop over the tiles of the outermost loop
  for (int i_tile = 0; i_tile < Ni; i_tile += tile_size) {
    // determine the bounds of the tile
    int i_tile_end = i_tile + tile_size;
    i_tile_end = (i_tile_end > Ni) ? Ni : i_tile_end;

    // loop over the tiles of the innermost loop
    for (int j_tile = 0; j_tile < Nj; j_tile += tile_size) {
      // determine the bounds of the tile
      int j_tile_end = j_tile + tile_size;
      j_tile_end = (j_tile_end > Nj) ? Nj : j_tile_end;

      // perform the matrix multiplication within the tile
      #pragma omp for
      for (int i = i_tile; i < i_tile_end; ++i) {
        for (int k = 0; k < Nk; ++k) {
		int R = (j_tile_end - j_tile)%UF;
          for (int j = j_tile; j < R + j_tile; ++j) {
            C[i*Nj+j] += A[k*Ni+i]*B[j*Nk+k];
          }
          for (int j = R + j_tile; j < j_tile_end; j = j + UF) {
            C[i*Nj+j] += A[k*Ni+i]*B[j*Nk+k];
            C[i*Nj+j+1] += A[k*Ni+i]*B[(j+1)*Nk+k];
          }

        }
      }
    }
  }
}
}

void atbt_ikj_unrollj_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
  int UF = 4;
  int  R = Nj%UF;
#pragma omp parallel private(i,j,k, UF, R)
 {
#pragma omp for
  for (i = 0; i < Ni; i++)
   for (k = 0; k < Nk; k++){
     for (j = 0; j < R; j++){
	     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
     }
     for (j = R; j < Nj; j = j + UF){
	     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
	     C[i*Nj+j+1]=C[i*Nj+j+1]+A[k*Ni+i]*B[(j+1)*Nk+k];
	     C[i*Nj+j+2]=C[i*Nj+j+2]+A[k*Ni+i]*B[(j+2)*Nk+k];
	     C[i*Nj+j+3]=C[i*Nj+j+3]+A[k*Ni+i]*B[(j+3)*Nk+k];
     }
   }
 }
}

void atbt_jik_j_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel private(i,j,k)
 {
#pragma omp for
   for (j = 0; j < Nj; j++)
    for (i = 0; i < Ni; i++){
     for (k = 0; k < Nk; k++){
     	C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
     }
    }
 }
 }

 void atbt_jik_unrolli_par(const double *__restrict__ A, const double *__restrict__ B, double
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
  int R, UF = 4;
#pragma omp parallel private(i,j,k)
 {
#pragma omp for
   for (j = 0; j < Nj; j++){
    int R = Ni %UF;
    for (i = 0; i < R; i++){
     for (k = 0; k < Nk; k++){
        C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
     }

    }
    for (i = R; i < Ni; i= i + UF){
     for (k = 0; k < Nk; k++){
        C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
        C[(i+1)*Nj+j]=C[(i+1)*Nj+j]+A[k*Ni+i+1]*B[j*Nk+k];
        C[(i+2)*Nj+j]=C[(i+2)*Nj+j]+A[k*Ni+i+2]*B[j*Nk+k];
        C[(i+3)*Nj+j]=C[(i+3)*Nj+j]+A[k*Ni+i+3]*B[j*Nk+k];
     }
    }
   }
 }
}

void atbt_jik_unrollk_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
  int UF = 4;
#pragma omp parallel private(i,j,k)
 {
#pragma omp for
   for (j = 0; j < Nj; j++)
    for (i = 0; i < Ni; i++){
  	int R = Nk%UF;
     for (k = 0; k < R; k++){
     	C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
     }
     for (k = R; k < Nk; k = k + UF){
	C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
	C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[j*Nk+k+1];
	C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[j*Nk+k+2];
	C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[j*Nk+k+3];
     }
   }
 }
 }


void atbt_jki_j_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel private(i,j,k)
 {
#pragma omp for
   for (j = 0; j < Nj; j++)
     for (k = 0; k < Nk; k++){
    	for (i = 0; i < Ni; i++){
     	C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
     }
    }
 }
 }


 void atbt_kij_i_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel private(i,j,k)
 {
    for (k = 0; k < Nk; k++)
    #pragma omp for
    for (i = 0; i < Ni; i++)
     for (j = 0; j < Nj; j++)
// C[i][j] = C[i][j] + A[k][i]*B[j][k];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
 }
}

void atbt_kji_j_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel private(i,j,k)
 {
    for (k = 0; k < Nk; k++)
#pragma omp for
   for (j = 0; j < Nj; j++)
  for (i = 0; i < Ni; i++)
// C[i][j] = C[i][j] + A[k][i]*B[j][k];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
 }
}



void atbt_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
// based on input selection:

	if (Ni ==8192 && Nj == 8192 && Nk == 16){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni ==4096 && Nj == 4096 && Nk == 64){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 2048 && Nj == 2048 && Nk == 256){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 1024 && Nj == 1024 && Nk == 1024){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 256 && Nj == 256 && Nk == 16384){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 64 && Nj == 64 && Nk == 262144){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 16 && Nj == 16 && Nk == 4194304){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
// Right table
	if (Ni == 8991 && Nj == 8991 && Nk == 37){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 2997 && Nj == 2997 && Nk == 111){
		atbt_ikj_tileij_unrollk_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 999  && Nj == 999 && Nk == 999){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 333  && Nj == 333 && Nk == 8991){
		atbt_ikj_i_par(A, B, C, Ni, Nj, Nk);
	}
	if (Ni == 111  && Nj == 111 && Nk == 80919){
		atbt_ikj_tileij_par(A, B, C, Ni, Nj, Nk);
	}
}

