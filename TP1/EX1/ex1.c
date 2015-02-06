#include <stdio.h>
#include <omp.h>

int main(){
	
	// Variables propre à chaque thread
	int nombreThreads, moi;
	
	#pragma omp parallel schedule(static) private(nombreThreads, moi)
	{
		nombreThreads = omp_get_num_threads();
		moi = omp_get_thread_num();
		
		printf("Salut du thread %d !\n", moi);
		
		if(moi == 0){
		
			printf("Nous sommes un groupe de %d threads.\n", nombreThreads);
		}
	}
	
	return 0;
}
