__global__ void atbt(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {

    // Get the row and column indices of the matrix C element being processed
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    // Check if the indices are within the bounds of the matrix C
    if (row < Ni && col < Nj)
    {
        double value = 0;
        for (int k = 0; k < Nk; k+=2)
        {
            //value += A[row*Nk + k] * B[k*Nj + col];
            value += A[k*Ni + row] * B[col*Nk + k];
            value += A[(k+1)*Ni + row] * B[col*Nk + (k+1)];
        }
        C[row*Nj + col] = value;
        //C[col*Ni+row] = value;
    }
}
