#include <stdio.h>
#include <omp.h>

#define N 5

int main(){
	
	float a[N][N], b[N][N], c[N][N], res;
	int ligne, colonne, i, j;
	omp_set_num_threads(N);
	
	//Init. 
	for(i=0; i<N; i++){
	
		for(j=0; j<N; j++){
		
			a[i][j] = (float)(i*j);	
			b[i][j] = (float)(i*j);
		}
	}
	
	//Affichage
	printf("A :\n");
	for(i=0; i<N; i++){
	
		for(j=0; j<N; j++){
		
			printf("%f ", a[i][j]);
		}
		
		printf("\n");
	}
	
	printf("B :\n");
	for(i=0; i<N; i++){
	
		for(j=0; j<N; j++){
		
			printf("%f ", b[i][j]);
		}
		
		printf("\n");
	}
	
	#pragma omp parallel shared(a, b, c) private(ligne,res)
	{
		ligne = omp_get_thread_num();

		for(colonne=0; colonne<N; colonne++){
		
			res = 0.0;
			
			//Matrice carré
			for(i=0; i<N; i++){
			
				res += a[ligne][i]*b[i][colonne];
			}
			
			c[ligne][colonne] = res;
		}
	}
	
	//Affichage résultat
	printf("C :\n");
	for(i=0; i<N; i++){
	
		for(j=0; j<N; j++){
		
			printf("%f ", c[i][j]);
		}
		
		printf("\n");
	}
	
	return 0;
}
