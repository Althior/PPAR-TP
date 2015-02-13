#include <stdio.h>
#include <mpi.h>

// Taille des matrices pour le calcul
#define N 10

void afficher_matrice(char *nom, float mat[N][N]);

void afficher_matrice(char *nom, float mat[N][N]){

	int i,j;
	
	printf("%s :\n", nom);
	for(i=0; i<N; i++){
	
		for(j=0; j<N; j++){
		
			printf("%06.2f ", mat[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char ** argv){
	
	float a[N][N], b[N][N], revbuffer[N], revbuffer2[N];
	int i, j, rang;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rang);
	MPI_Comm_size(MPI_COMM_WORLD, &n);

	if (rang == 0) {
		// Initialisation matrice a
		for(i=0; i<N; i++){
	
			for(j=0; j<N; j++){
		
				a[i][j] = (float)(i-j);	
			}
		}
		afficher_matrice("a", a);	
	} 

	// envoi une ligne de a par processus
	MPI_Scatter(a, N, MPI_FLOAT, revbuffer, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	// transposition
	MPI_Alltoall(revbuffer, 1, MPI_FLOAT, revbuffer2, 1, MPI_FLOAT, MPI_COMM_WORLD);
	
	// fusion dans la matrice b
	MPI_Gather(revbuffer2, N, MPI_FLOAT, b, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
			
	
	if ( rang == 0) {
		afficher_matrice("b", b);
	}

	MPI_Finalize();
	return 0;
}
