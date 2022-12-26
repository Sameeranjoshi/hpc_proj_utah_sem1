void atbt_par(const double *__restrict__ A, const double *__restrict__ B, double
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

