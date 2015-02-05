#include <stdio.h>
#include <omp.h>

// Taille des matrices pour le calcul
#define N 5

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
	
	// c = a*b
	float a[N][N], b[N][N], c[N][N], res;
	int ligne, colonne, i, j;
	
	// Initialisation
	for(i=0; i<N; i++){
	
		for(j=0; j<N; j++){
		
			a[i][j] = (float)(i*j);	
			b[i][j] = (float)(i*j);
		}
	}
	
	// Affichage des matrices
	afficher_matrice("A", a);
	afficher_matrice("B", b);
	
	// Calcul
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

	afficher_matrice("C", c);
	
	return 0;
}
