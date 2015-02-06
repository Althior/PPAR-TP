#include <stdio.h>
#include <omp.h>
#include <sys/time.h>

// Taille des matrices pour le calcul
#define N 200

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

int main(){
	
	float a[N][N], b[N][N], c[N][N], res;
	double ecoule;
	int ligne, colonne, i, j;
	struct timeval tv0, tv1;
	
	gettimeofday(&tv0, 0);
	
	// Initialisation
	#pragma omp parallel for schedule(static) shared(a, b) private(i, j)
	for(i=0; i<N; i++){
	
		for(j=0; j<N; j++){
		
			a[i][j] = (float)(i*j);	
			b[i][j] = (float)(i*j);
		}
	}
	
	// Affichage des matrices
	//afficher_matrice("A", a);
	//afficher_matrice("B", b);
	
	// Calcul
	#pragma omp parallel for schedule(static) shared(a, b, c) private(ligne, colonne, i, res)
	for(ligne=0; ligne<N; ligne++){

		for(colonne=0; colonne<N; colonne++){
		
			res = 0.0;
			
			// On considère une matrice comme un carré
			for(i=0; i<N; i++){
			
				res += a[ligne][i]*b[i][colonne];
			}
			
			c[ligne][colonne] = res;
		}
	}

	//afficher_matrice("C", c);
	
	gettimeofday(&tv1, 0);
	ecoule = (double)((tv1.tv_sec-tv0.tv_sec)*10e6 + tv1.tv_usec - tv0.tv_usec) / 10e6;
	
	printf("Temps écoulé : %.6f secondes\n", ecoule);	
	return 0;
}
