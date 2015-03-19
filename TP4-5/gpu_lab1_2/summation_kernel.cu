
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	float res=0;
	float tmp=1.0;
	int debut = i*data_size;
	int fin = debut + data_size;

	for (k=debut; k<fin; k++){
		if ((k%2) == 1) {tmp = -1.0;}
		else tmp = 1.0;

		res += tmp / (k + 1);
		}
	
	data_out[i] = res;

}


