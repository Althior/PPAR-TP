#include <stdio.h>
#include <omp.h>
#include <math.h>

#define N 120

int main(){

int T[N];
int i, k;

// Init
for(i=2; i<N; i++){
	T[i]=1;
}


for(k=2; k<sqrt(N); k++){
	if(T[k] == 1) {
		for(i=k+1; i<N; i++){
			if (i%k == 0) {T[i]=0;}
		}
	}
}

//Affichage
for(i=0; i<N; i++){
	if (T[i]==1){printf("%d ", i);}
}

	
	return 0;
}
