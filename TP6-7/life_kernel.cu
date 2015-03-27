
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
    
	
	// TODO
	// copier mémoire globale dans la mémoire partagée pour que les threads y accedent par la suite
	
	
    // Read cell
    int myself = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);
    
    // Read the 8 neighbors and count number of blue and red
	int i;
	int count_cell[3];
	// init tab count_cell
	for (i=0; i<3; i++){count_cell[i]=0;}
	// parcours des voisins
	for (i=-1; i<2; i++){
		count_cell[read_cell(source_domain, tx, ty, i, -1, domain_x, domain_y)]+=1;
		count_cell[read_cell(source_domain, tx, ty, i, 1, domain_x, domain_y)]+=1;
	}
	count_cell[read_cell(source_domain, tx, ty, -1, 0, domain_x, domain_y)]+=1;
	count_cell[read_cell(source_domain, tx, ty, 1, 0, domain_x, domain_y)]+=1;
		
	// Compute new value
	int new_cell=0;
	int num_nei = count_cell[1] + count_cell[2];
	
	switch (myself){
	
	case 0: // empty cell 
		if (num_nei == 3) { // neighbors == 3
			if (count_cell[1] < count_cell[2])	new_cell = 2;
			else new_cell = 1;
		}
		break;
	default: // cell survives if neighbors == 2|3
		if (num_nei == 2 || num_nei == 3){ new_cell = myself;}
		break;
	}
	
	// Write it in dest_domain
	dest_domain[ty * domain_x + tx] = new_cell;
}	

