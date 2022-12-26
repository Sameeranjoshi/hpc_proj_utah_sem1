void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
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

