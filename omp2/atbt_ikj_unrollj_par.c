void atbt_par(const double *__restrict__ A, const double *__restrict__ B, double 
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
