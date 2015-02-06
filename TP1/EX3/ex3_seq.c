#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>

#define N 1200

int main(){

	int T[N], i, k, res=0;
	double ecoule;
	struct timeval tv0, tv1;
	
	// Initialisation
	for(i=2; i<N; i++){
		T[i]=1;
	}

	gettimeofday(&tv0, 0);
	
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

	// Temps écoulé
	gettimeofday(&tv1, 0);
	ecoule = (double)((tv1.tv_sec-tv0.tv_sec)*10e6 + tv1.tv_usec - tv0.tv_usec) / 10e6;
	printf("Temps écoulé : %.6f secondes\n", ecoule);
	
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
