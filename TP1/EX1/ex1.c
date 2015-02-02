#include <stdio.h>
#include <omp.h>

int main(){
	
	int moi = 0;
	int nombreThreads = 4;
	omp_set_num_threads(nombreThreads);
	
	
	#pragma omp parallel private(nombreThreads, moi)
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
