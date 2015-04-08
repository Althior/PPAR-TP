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
    
	// Copier depuis la mémoire globale vers la mémoire partagée pour que les threads y accèdent par la suite
	// sdata -> Bloc = 8*8 threads -> 100 lectures (10*10) -> |sdata| = 100
	extern __shared__ int sdata[];
	int sdataDim = 10; // Voir si on ne peut pas obtenir ça de manière auto
	
	// Chaque thread lit sa case
	int myself = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);
	
	/*
	*	Dans le cas d'un bloc 3x3:
	*	bloc: x² (carré pour optimiser lectures)
	*	 0 1 2
	*	 3 4 5
	*	 6 7 8
	*
	*	sdata: (x+2)² (sdataDim=5)
	*	 0  1  2  3  4
	*	 5| 6  7  8| 9
	*	10|11 12 13|14
	*	15|16 17 18|19
	*	20 21 22 23 24
	*/
	
	// Position y du thread dans la shared memory
	int decY, myloc;
	decY = (threadIdx.x / blockDim.x) + 1;
	//decY = (threadIdx.x / sdataDim) + 1; // Mieux ?
	
	// Position dans sdata (avec contours compris)
	myloc = sdataDim*decY + 1 + (threadIdx.x % blockDim.x); // +1 nécessaire ?
	sdata[myloc] = myself;
	
	/*
		Initialisation de 4 variables par thread : haut bas gauche droite
		Booléens indiquant si le thread est situé en bordure du carré de cellules traité
	*/
	int haut, bas, gauche, droite;
	haut   = threadIdx.x < blockDim.x;
	bas    = threadIdx.x >= blockDim.x * (blockDim.y -1);
	gauche = (threadIdx.x % blockDim.x) == 0;
	droite = ((threadIdx.x + 1) % blockDim.x) == 0;
	
	/* Lectures des bordures si nécessaire */
	
	// Lecture en haut
	if (haut) {
	
		sdata[myloc-sdataDim] = read_cell(source_domain, tx, ty, 0, -1, domain_x, domain_y);
		
		// Lecture haut-gauche
		if (gauche) {
			
			sdata[myloc-sdataDim-1] = read_cell(source_domain, tx, ty, -1, -1, domain_x, domain_y);
		}
		
		// Lecture haut-droite
		if (droite) {
			
			sdata[myloc-sdataDim+1] = read_cell(source_domain, tx, ty, 1, -1, domain_x, domain_y);
		}
	}
	
	// Lecture en bas
	if (bas) {
		
		sdata[myloc+sdataDim] = read_cell(source_domain, tx, ty, 0, 1, domain_x, domain_y);
		
		// Lecture bas-gauche
		if (gauche) {
			
			sdata[myloc+sdataDim-1] = read_cell(source_domain, tx, ty, -1, 1, domain_x, domain_y);
		}
		
		// Lecture bas-droite
		if (droite) {
			
			sdata[myloc+sdataDim+1] = read_cell(source_domain, tx, ty, 1, 1, domain_x, domain_y);
		}
	}
	
	// Lecture à gauche
	if (gauche) {
	
		sdata[myloc-1] = read_cell(source_domain, tx, ty, -1, 0, domain_x, domain_y);
	}
	
	// Lecture à droite
	if (droite) {
		
		sdata[myloc+1] = read_cell(source_domain, tx, ty, 1, 0, domain_x, domain_y);
	}	

	/*
	sdata[threadIdx.x] = read_cell(source_domain, tx, ty, 0, -1, domain_x, domain_y);
	sdata[threadIdx.x+blockDim.x] = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);
	sdata[threadIdx.x+(2*blockDim.x)] = read_cell(source_domain, tx, ty, 0, 1, domain_x, domain_y);	
	*/
	__syncthreads();
	
    // Read cell
    // int myself = read_cell(sdata, tx, ty, 0, 0, domain_x, 3);
    
    // Read the 8 neighbors and count number of blue and red
	int i;
	int count_blue = 0, count_red = 0, valTemp;
	
	// Parcours des voisins du dessus et du dessous
	for (i=-1; i<2; i++){
	
		valTemp = read_cell(sdata, tx, ty, i, -1, sdataDim, sdataDim);
		switch(valTemp){
			case 1:	
				count_red++;
				break;
			case 2:
				count_blue++;
				break;
		}
		
		valTemp = read_cell(sdata, tx, ty, i, 1, sdataDim, sdataDim);
		switch(valTemp){
			case 1:	
				count_red++;
				break;
			case 2:
				count_blue++;
				break;
		}
	}
	
	// Voisin gauche
	valTemp = read_cell(sdata, tx, ty, -1, 0, sdataDim, sdataDim);
	switch(valTemp){
		case 1:	
			count_red++;
			break;
		case 2:
			count_blue++;
			break;
	}
	
	// Voisin droit	
	valTemp = read_cell(sdata, tx, ty, 1, 0, sdataDim, sdataDim);
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
	
	case 0: // Cellule vide
		if (num_nei == 3) { // 3 voisins
			if (count_red < count_blue)	new_cell = 2;
			else new_cell = 1;
		}
		break;
	default: // Survie de la cellule si nbVoisins == 2||3
		if (num_nei == 2 || num_nei == 3){ new_cell = myself;}
		break;
	}
	
	// Write it in dest_domain	
	// sync et recopie
	 __syncthreads();

	dest_domain[ty * domain_x + tx] = new_cell;
}	

