
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;

	/* Version de calcul par blocs continus (en descendant) */
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	float res=0;
	float tmp=1.0;
	int debut = i*data_size;
	int fin = debut + data_size;

	for (k=fin-1; k>=debut; k--){
		if ((k%2) == 1) {tmp = -1.0;}
		else tmp = 1.0;

		res += tmp / (k + 1);
	}
	
	sdata[tid] = res;
	__syncthreads();

	for(unsigned int s = 1; s < blockDim.x; s*=2){

		int index = 2*s*tid;

		if(index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	if(tid == 0){

		data_out[blockIdx.x] = sdata[0];
	}
	
	/* Version de calcul par rang (en descendant) */
	/*
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	float res=0.0;
	float tmp=1.0;

	int debut = i;
	int fin = debut+data_size*blockDim.x;

	for (k=fin-blockDim.x; k>=debut; k-=blockDim.x){

		if ((k%2) == 1) {tmp = -1.0;}
		else tmp = 1.0;
		
		res += tmp / (k + 1);
	}

	data_out[i] = res;
	*/
}

// GPU kernel
__global__ void second_summation_kernel(float *data, float *res)
{
	unsigned int i;
	float temp = 0.0;

	for(i=0; i<blockDim.x; i++){

		temp += data[i];
	}

	*res = temp;
}


