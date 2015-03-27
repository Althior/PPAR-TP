#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include "ocean.h"

/* constants for the ocean */
#define N 40
#define M 20
#define WALL 100
#define STEP 150000
#define RATIO 30

/* Utiliser mminimum 4 processus pour faire fonctionner ce programme (sinon précédent = suivant et non-géré) */

void update_ocean_part(fish_t *ocean, int n, int m, int *ns_north, int *nt_north, int *ns_south, int *nt_south);
void inject_ocean(fish_t *ocean, int n, int m, int ns, int nt);

int rang, nbproc;

/* Injection océan "sûre" */
void inject_ocean(fish_t *ocean, int n, int m, int ns, int nt){

	int i, j; 

	//printf("%i requins et %i thons a injecter pour le processus %i\n", ns, nt, rang);

	for(i=0; i<n && (ns != 0 || nt != 0); i++){

		for(j=0;j<m  && (ns != 0 || nt != 0); j++){
	
			if(ns > 0){
			
				if(ocean[i*m+j].type == 'F' || ocean[i*m+j].type == 'T'){
				
					//printf("Injection d'un requin position (%i,%i) par le processus %i\n", i, j, rang);
					ocean[i*m+j].type = 'S';
					ocean[i*m+j].moved =1;
				  	ns--;
				}
			}
			else if(nt > 0){
			
				if(ocean[i*m+j].type == 'F'){
				
					//printf("Injection d'un thon en position (%i,%i) par le processus %i\n", i, j, rang);
					ocean[i*m+j].type = 'T';
					ocean[i*m+j].moved = 1;
				  	nt--;
				}
			}
		}
	}
}

/* Met a jour une section de l'océan général */
void update_ocean_part(fish_t *ocean, int n, int m, int *ns_north, int *nt_north, int *ns_south, int *nt_south)
{

  int i, j;
  int next_i, next_j;
  int rd;
  int estSorti;

  /* Reinitiate the moved values */
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      ocean[i*m+j].moved = 0;

  /* Init. des poissons a transmettre aux voisins */
  *ns_north = 0;
  *ns_south = 0;
  *nt_north = 0;
  *nt_south = 0;

  for (i = 0; i < n; i++) {
  
    for (j = 0; j < m; j++) {
    
      if (ocean[i*m+j].moved == 0) {

        estSorti = 0;

        /* compute the next position (systematically) */
        rd = rand() % 100;
        if (rd < 25) { /* -> N */
          next_i = i - 1;
          if (next_i == -1) { estSorti = 1; }
          next_j = j;
        }
        else if (rd < 50) { /* -> E */
          next_i = i;
          next_j = (j + 1) % m;
        }
        else if (rd < 75) { /* -> S */
          next_i = (i + 1);
          if (next_i == n) { estSorti = 1; }
          next_j = j;
        }
        else { /* -> W */
          next_i = i;
          next_j = (j-1) % m;
          if (next_j == -1) next_j = m - 1;
        }
		
		// Le poisson n'est plus dans la partie de l'océan
        if (estSorti && (ocean[i*m+j].type == 'S' || ocean[i*m+j].type == 'T')) {

          // Envoyer le poisson au voisin prédescesseur
          if (next_i == -1) {

		        switch (ocean[i*m+j].type){

		          case 'S':
		            *ns_north += 1;
		            break;

		          case 'T':
		            *nt_north += 1;
		            break;
		        }
          }
          // Envoyer le poisson au voisin successeur
          else if (next_i == n) {

		        switch (ocean[i*m+j].type) {

		          case 'S':
		            *ns_south += 1;
		            break;

		          case 'T':
		            *nt_south += 1;
		            break;
		        }
          }

          // L'ancienne case est dorénavant vide
          ocean[i*m+j].type = 'F';

        } else {
          /* if I am a shark -- I move if no sharks is already here
             and eats tuna if some (implicit) */
          if (ocean[i*m+j].type == 'S'){
          
          	if (ocean[next_i*m+next_j].type != 'S') {
            
              ocean[next_i*m+next_j].type = 'S';
              ocean[next_i*m+next_j].moved = 1;
              ocean[i*m+j].type = 'F';
            }
          } /* fi 'S' */
            /* If I am a tuna, I move whenever it's free */
          else if (ocean[i*m+j].type == 'T'){
            
            if (ocean[next_i*m+next_j].type == 'F'){
           
              ocean[next_i*m+next_j].type = 'T';
              ocean[next_i*m+next_j].moved = 1;
              ocean[i*m+j].type = 'F';
            }
          } /* fi 'T' */
        } /* fi estSorti */
        
        //printf("\n");
      } /* fi !moved */
    } /* for j */
  } /* for i */
}

int main (int argc, char * argv[])
{
	int i;
  	MPI_Status status;
	MPI_Request myRequest;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rang);
	MPI_Comm_size(MPI_COMM_WORLD, &nbproc);

	fish_t *ocean;

	if(rang == 0){
	
		printf("Lancement du programme avec %i processus\n", nbproc);
	}
	
	// Calcul processeur suivant/precedent
	int procprec = (rang-1)%nbproc;
	if (procprec == -1) procprec = nbproc - 1;

	int procsuivant = (rang+1)%nbproc;

	/* Gestion de l'océan en parallèle */
	fish_t *oceanrec = (fish_t *)malloc(N*M*sizeof(fish_t)/nbproc);
	if(oceanrec == NULL){

    fprintf(stderr, "Erreur lors de l'allocation d'oceanrec\n");
    
		MPI_Finalize();
		exit(1);
	}

	int nbS_Nord, nbT_Nord, nbS_Sud, nbT_Sud; // Nombre de poissons a injecter pour les voisins
	int injSN = 0, injSS = 0, injTN = 0, injTS = 0; // Nombres de poissons a injecter

	// Création et commit du type MPI_FISH
 	int tailleChamp[2] = {1,1};
	MPI_Aint decChamp[2] = {0,1};
	MPI_Datatype typeChamp[2] = {MPI_CHAR, MPI_CHAR};
	MPI_Datatype MPI_FISH;

	if (MPI_Type_create_struct(2, tailleChamp, decChamp, typeChamp, &MPI_FISH) != MPI_SUCCESS){

		fprintf(stderr, "Erreur lors de la création du type MPI_FISH\n");
		MPI_Finalize();
		exit(1);
	}
	
	if(MPI_Type_commit(&MPI_FISH) != MPI_SUCCESS){
	
		fprintf(stderr, "Erreur lors du commit du type MPI_FISH\n");
		MPI_Finalize();
		exit(1);
	}

	// Allocation de l'ocean & affichage de départ
	if (rang == 0){

 		ocean = (fish_t *)malloc(N*M*sizeof(fish_t));

		if(ocean == NULL){

		  fprintf(stderr, "Erreur lors de l'allocation d'ocean\n");
		  MPI_Finalize();
		  exit(1);
		}
		
	  	init_ocean(ocean, N, M, RATIO);
	  	printf(CLS "\n");
	  	display_ocean(ocean, N, M);
	}


  for (i = 0; i < WALL; i++) {

    /* On divise l'océan selon le nombre de processeurs, chaque processeur possède une "macro-ligne" */
	if(MPI_Scatter(ocean, (N/nbproc)*M, MPI_FISH, oceanrec, (N/nbproc)*M, MPI_FISH, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
	
		 fprintf(stderr, "Erreur lors du découpage de l'océan entre processus\n");
		 MPI_Finalize();
		 exit(1);
	}

    usleep(STEP);
    update_ocean_part(oceanrec, N/nbproc, M, &nbS_Nord, &nbT_Nord, &nbS_Sud, &nbT_Sud);

    // Envoi requins/thons à injecter pour les voisins
    if(MPI_Isend(&nbS_Nord, 1, MPI_INT, procprec, 0, MPI_COMM_WORLD, &myRequest) != MPI_SUCCESS){
    
    	fprintf(stderr, "Erreur lors de l'envoi du premier message\n");
		MPI_Finalize();
		exit(1);
    }
    
    if(MPI_Isend(&nbT_Nord, 1, MPI_INT, procprec, 0, MPI_COMM_WORLD, &myRequest) != MPI_SUCCESS){
    
    	fprintf(stderr, "Erreur lors de l'envoi du second message\n");
		MPI_Finalize();
		exit(1);
    }

	if(MPI_Isend(&nbS_Sud, 1, MPI_INT, procsuivant, 0, MPI_COMM_WORLD, &myRequest) != MPI_SUCCESS){
	
		fprintf(stderr, "Erreur lors de l'envoi du troisième message\n");
		MPI_Finalize();
		exit(1);
	}
	
	if(MPI_Isend(&nbT_Sud, 1, MPI_INT, procsuivant, 0, MPI_COMM_WORLD, &myRequest) != MPI_SUCCESS){
	
		fprintf(stderr, "Erreur lors de l'envoi du quatrième message\n");
		MPI_Finalize();
		exit(1);
	}
	
    // Reception
    if(MPI_Recv(&injSN, 1, MPI_INT, procprec, 0, MPI_COMM_WORLD, &status) != MPI_SUCCESS){
    
    	fprintf(stderr, "Erreur lors de la réception du premier message\n");
		MPI_Finalize();
		exit(1);
    }
    
    if(MPI_Recv(&injTN, 1, MPI_INT, procprec, 0, MPI_COMM_WORLD, &status) != MPI_SUCCESS){
    
    	fprintf(stderr, "Erreur lors de la réception du deuxième message\n");
		MPI_Finalize();
		exit(1);
    }

	if(MPI_Recv(&injSS, 1, MPI_INT, procsuivant, 0, MPI_COMM_WORLD, &status) != MPI_SUCCESS){
	
		fprintf(stderr, "Erreur lors de la réception du troisième message\n");
		MPI_Finalize();
		exit(1);
	}
	
	if(MPI_Recv(&injTS, 1, MPI_INT, procsuivant, 0, MPI_COMM_WORLD, &status) != MPI_SUCCESS){
	
		fprintf(stderr, "Erreur lors de la réception du quatrième message\n");
		MPI_Finalize();
		exit(1);
	}
	
    // Remise des poissons
<<<<<<< HEAD
	inject_ocean2(oceanrec, N/nbproc, M, injSN+injSS, injTN+injTS);
	
	// TODO Synchro si besoi
	
	if(MPI_Gather(oceanrec, N*M/nbproc, MPI_FISH, ocean, N*M/nbproc, MPI_FISH, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
	
		fprintf(stderr, "Erreur lors du rassemblement des parties de l'océan");
=======
	inject_ocean(oceanrec, N/nbproc, M, injSN+injSS, injTN+injTS);

	if(MPI_Gather(oceanrec, N*M/nbproc, MPI_FISH, ocean, N*M/nbproc, MPI_FISH, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
	
		fprintf(stderr, "Erreur lors de fusions des parties d'océan\n");
>>>>>>> 56c40fc8e95950b5c2276a5812784033038675eb
		MPI_Finalize();
		exit(1);
	}

    // Affichage
    if(rang == 0){

      printf(CLS "\n");
      display_ocean(ocean, N, M);
    }
  }

	
	if(rang == 0) {

		free(ocean);
	}

	free(oceanrec);
	MPI_Finalize();
  	return 0;
} /* main */

