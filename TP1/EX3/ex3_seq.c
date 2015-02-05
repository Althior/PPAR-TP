#include <stdio.h>
#include <omp.h>
#include <math.h>

#define N 120

int main(){

	int T[N], i, k, res=0;

	// Initialisation
	for(i=2; i<N; i++){
		T[i]=1;
	}

	// Crible
	for(k=2; k<=(int)sqrt(N); k++){
	
		// Si nombre premier => Élimination de ses multiples
		if(T[k]){
		
			for(i=k+1; i<N; i++){
		
				if (i%k == 0){
					T[i] = 0;
				}
			}
		}
	}

	// Nombre de premiers
	for(i=0; i<N; i++){

		if (T[i]){
			res+=1;
		}
	}

	printf("Nombre de premiers inférieur à %d : %d\n", N, res);

	// Affichage des nombres premiers
	for(i=0; i<N; i++){

		if (T[i]){
			printf("%d ", i);
		}
	}
	printf("\n");
	
	return 0;
}
