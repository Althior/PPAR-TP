#include <stdio.h>
#include <omp.h>

#define N 45

int main(){

	int i;
	omp_lock_t forks[N];
	
	// Initialisation des locks
	#pragma omp parallel for schedule(static) private(i)
	for(i=0; i<N; i++){
	
		omp_init_lock(&forks[i]);
	}
	
	#pragma omp parallel for schedule(static) private(i)
	for(i=0; i<N; i++){
	
		printf("Le philosophe %d se met à table\n", i);
		
		// Philosophe impair : Gauche => Droite
		if(i%2){
			
			omp_set_lock(&forks[i]);
			printf("Le philosophe %d prend la fourchette à sa gauche\n", i);
			
			omp_set_lock(&forks[(i+1)%N]);
			printf("Le philosophe %d prend la fourchette à sa droite\n", i);
			
			printf("J'aime mangé (%d) !\n", i);
			
			omp_unset_lock(&forks[i]);
			printf("Le philosophe %d relache la fourchette à sa gauche\n", i);
			
			omp_unset_lock(&forks[(i+1)%N]);
			printf("Le philosophe %d relache la fourchette à sa droite\n", i);
		}
		// Philosophe pair : Droite => Gauche
		else{
			
			omp_set_lock(&forks[(i+1)%N]);
			printf("Le philosophe %d prend la fourchette à sa droite\n", i);
			
			omp_set_lock(&forks[i]);
			printf("Le philosophe %d prend la fourchette à sa gauche\n", i);
			
			printf("J'aime mangé (%d) !\n", i);
			
			omp_unset_lock(&forks[(i+1)%N]);
			printf("Le philosophe %d Prend la fourchette à sa droite\n", i);
			
			omp_unset_lock(&forks[i]);
			printf("Le philosophe %d relache la fourchette à sa gauche\n", i);
		}
	}
	
	printf("Tout le monde a mangé\n");
	return 0;
}
