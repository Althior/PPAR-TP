__global__ void init_kernel(int * domain, int domain_x)
{
	// Dummy initialization
	domain[blockIdx.y * domain_x + blockIdx.x * blockDim.x + threadIdx.x]
		= (1664525ul * (blockIdx.x + threadIdx.y + threadIdx.x) + 1013904223ul) % 3;
}

// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    return source_domain[y * domain_x + x];
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain,
    int domain_x, int domain_y)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;
    
	// copier mémoire globale dans la mémoire partagée pour que les threads y accedent par la suite
	extern __shared__ int sdata[];

	// chaque thread lit sa case
	sdata[] = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);
	
	// initialiser 4 variables par thread: haut bas gauche droite
	// combinaison pour les diags
	
	// si en bordure il lit à l'extérieur
	if ((threadIdx.x % blockDim.x) == 0) {
		// lecture à gauche
	} 
	if (((threadIdx.x - 1) % blockDim.x) == 0) {
		// lecture à droite
	}
	if (threadIdx.x < blocDim.x) {
		// lecture en haut
	}
	if (threadIdx.x >= blocDim.x * (blocDim.y -1)) {
		// lecture en bas
	}
	
	
	
	/*
	sdata[threadIdx.x] = read_cell(source_domain, tx, ty, 0, -1, domain_x, domain_y);
	sdata[threadIdx.x+blockDim.x] = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);
	sdata[threadIdx.x+(2*blockDim.x)] = read_cell(source_domain, tx, ty, 0, 1, domain_x, domain_y);	
	*/
	__syncthreads();
	
    // Read cell
    int myself = read_cell(sdata, tx, ty, 0, 0, domain_x, 3);
    
    // Read the 8 neighbors and count number of blue and red
	int i;
	int count_blue = 0, count_red = 0, valTemp;
	
	// parcours des voisins
	for (i=-1; i<2; i++){
	
		valTemp = read_cell(sdata, tx, ty, i, -1, domain_x, 3);
		switch(valTemp){
			
			case 1:	
				count_red++;
				break;

			case 2:
				count_blue++;
				break;
		}
		
		valTemp = read_cell(sdata, tx, ty, i, 1, domain_x, 3);
		
		switch(valTemp){
			
			case 1:	
				count_red++;
				break;

			case 2:
				count_blue++;
				break;
		}
	}
	
	valTemp = read_cell(sdata, tx, ty, -1, 0, domain_x, 3);
	
	switch(valTemp){
			
		case 1:	
			count_red++;
			break;

		case 2:
			count_blue++;
			break;
	}
		
	valTemp = read_cell(sdata, tx, ty, 1, 0, domain_x, 3);
	
	switch(valTemp){
			
		case 1:	
			count_red++;
			break;

		case 2:
			count_blue++;
			break;
	}
	
	// Compute new value
	int new_cell=0;
	int num_nei = count_red + count_blue;
	
	switch (myself){
	
	case 0: // empty cell 
		if (num_nei == 3) { // neighbors == 3
			if (count_red < count_blue)	new_cell = 2;
			else new_cell = 1;
		}
		break;
	default: // cell survives if neighbors == 2|3
		if (num_nei == 2 || num_nei == 3){ new_cell = myself;}
		break;
	}
	
	// Write it in dest_domain	
	// sych et recopie
	 __syncthreads();

	dest_domain[ty * domain_x + tx] = new_cell;
}	

