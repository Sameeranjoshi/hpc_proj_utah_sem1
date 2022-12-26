void atbt_par(const double *__restrict__ A, const double *__restrict__ B, double 
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
