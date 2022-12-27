__global__ void atb(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {

    // Get the row and column indices of the matrix C element being processed
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    // Check if the indices are within the bounds of the matrix C
    if (row < Ni && col < Nj)
    {
        double value = 0;
        double value1 = 0;
        for (int k = 0; k < Nk; k++)
        {
            //value += A[row*Nk + k] * B[k*Nj + col];
            value += A[k*Ni + row] * B[k*Nj + col];
            value1 += A[k*Ni + row] * B[k*(Nj+1) + col];
    
        }
        C[row*Nj + col] = value;
        C[row*(Nj+1) + col] = value1;
        //C[col*Ni+row] = value;
    }
}
