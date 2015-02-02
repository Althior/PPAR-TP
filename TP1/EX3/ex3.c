#include <stdio.h>
#include <omp.h>
#include <math.h>

#define N 120

int main(){

int T[N];
int i, k;
int res=0;

// Init
for(i=2; i<N; i++){
	T[i]=1;
}

//Crible
#pragma omp parallel for shared(T) private(i,k)
for(k=2; k<=(int)sqrt(N); k++){
	if(T[k] == 1) {
		for(i=k+1; i<N; i++){
			if (i%k == 0) {T[i]=0;}
		}
	}
}

//Nb de premiers
#pragma omp parallel for shared(T) private(i) reduction(+:res)
for(i=0; i<N; i++){
	if (T[i]==1){res+=1;}
}

printf("Nb de premiers: %d \n", res);

for(i=0; i<N; i++){
	if (T[i]==1){printf("%d ", i);}
}
	
	return 0;
}
