void atbt_par(const double *__restrict__ A, const double *__restrict__ B, double 
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
