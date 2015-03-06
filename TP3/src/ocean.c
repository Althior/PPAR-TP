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
#define WALL 3
#define STEP 150000
#define RATIO 50

void update_ocean_part(fish_t *ocean, int n, int m, int *ns_north, int *nt_north, int *ns_south, int *nt_south);
void inject_ocean(fish_t *ocean, int n, int m, int ns, int nt);
void inject_ocean2(fish_t *ocean, int n, int m, int ns, int nt);


/*
  Injecte un nombre de requins et thons dans un océan (sur des emplacements libres)
  On pourrait parcourir de manière itérative l'océan pour trouver un emplacement libre à coup sûr s'il en existe un.
  Cependant, pour éviter un placement systématique, on préfère utiliser un rand() avec un risque de ne pas trouver d'emplacement libre s'il en reste peu.
*/
void inject_ocean(fish_t *ocean, int n, int m, int ns, int nt){

  int rand_i, rand_j, nbTentatives = 0;

  // A partir de 500 tentatives infructeuses on considère qu'il n'y a plus de places
  while( (ns != 0 && nt != 0) && nbTentatives < 5000){

    rand_i = rand() % n;
    rand_j = rand() % m;

    // Emplacement libre, on remplace
    if(ocean[rand_i*m+rand_j].type == 'F'){

      if(nt){

        ocean[rand_i*m+rand_j].type = 'T';
        nt--;
      }
      else{

        ocean[rand_i*m+rand_j].type = 'S';
        ns--;
        printf("(%i,%i)\n", rand_i, rand_j);
        exit(0);
      }
      nbTentatives = 0;
    }
    else{

      nbTentatives++;
    }
  }
}

/* Injection océan "sûre" */
void inject_ocean2(fish_t *ocean, int n, int m, int ns, int nt){

	int i, j; 

	printf("%i s et %i t\n", ns, nt);
	for(i=0; i<n && (ns != 0 || nt != 0); i++){
	
		for(j=0;j<m && (ns != 0 || nt != 0); j++){
		
			// Emplacement libre, on remplace
			if(ocean[i*m+j].type == 'F'){
			
				if(nt){

				  ocean[i*m+j].type = 'T';
				  nt--;
				}
				else{

				  ocean[i*m+j].type = 'S';
				  ns--;
				}
			}
		}
	}
}

/* Met à jour une section de l'océan général */
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

  /* Init. des poissons à transmettre aux voisins */
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
          if (next_i < 0) {	estSorti = 1; }
          next_j = j;
        }
        else if (rd < 50) { /* -> E */
          next_i = i;
          next_j = (j + 1) % m;
        }
        else if (rd < 75) { /* -> S */
          next_i = (i + 1);
          if (next_i >= n) {	estSorti = 1; }
          next_j = j;
        }
        else { /* -> W */
          next_i = i;
          next_j = (j-1) % m;
          if (next_j == -1) next_j = m - 1;
        }

        if (estSorti) {

          // Envoyer le poisson au voisin prédescesseur
          if (next_i < 0) {

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
          else if (next_i > n) {

            switch (ocean[i*m+j].type) {

              case 'S':
                *ns_south += 1;
                break;

              case 'T':
                *nt_south += 1;
                break;
            }

            //ocean[i*m+j].moved = 1;
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
      } /* fi !moved */
    } /* for j */
  } /* for i */
}

int main (int argc, char * argv[])
{
	int rang, nbproc, i;
  	MPI_Status status;
	MPI_Request myRequest;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rang);
	MPI_Comm_size(MPI_COMM_WORLD, &nbproc);

	fish_t *ocean;

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

	int nbS_Nord, nbT_Nord, nbS_Sud, nbT_Sud; // Nombre de poissons à injecter pour les voisins
	int injSN, injSS, injTN, injTS; // Nombres de poissons à injecter

	// Création du type MPI_FISH
 	int tailleChamp[2] = {1,1};
	MPI_Aint decChamp[2] = {0,1};
	MPI_Datatype typeChamp[2] = {MPI_CHAR, MPI_CHAR};
	MPI_Datatype MPI_FISH;

	if (MPI_Type_create_struct(2, tailleChamp, decChamp, typeChamp, &MPI_FISH) != MPI_SUCCESS){

		fprintf(stderr, "Erreur lors de la création du type MPI_FISH\n");
		MPI_Finalize();
		exit(1);
	}
	
	MPI_Type_commit(&MPI_FISH);

	// Allocation de l'ocean & affichage
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
	  MPI_Scatter(ocean, (N*M)/nbproc, MPI_FISH, oceanrec, N*M/nbproc, MPI_FISH, 0, MPI_COMM_WORLD);

    usleep(STEP);
    update_ocean_part(oceanrec, N/nbproc, M, &nbS_Nord, &nbT_Nord, &nbS_Sud, &nbT_Sud);

    // Envoi requins/thons aux voisins
    MPI_Isend(&nbS_Nord, 1, MPI_INT, procprec, 0, MPI_COMM_WORLD, &myRequest);
    MPI_Isend(&nbT_Nord, 1, MPI_INT, procprec, 0, MPI_COMM_WORLD, &myRequest);
    MPI_Isend(&nbS_Sud, 1, MPI_INT, procsuivant, 0, MPI_COMM_WORLD, &myRequest);
    MPI_Isend(&nbT_Sud, 1, MPI_INT, procsuivant, 0, MPI_COMM_WORLD, &myRequest);

    // Reception
    MPI_Recv(&injSN, 1, MPI_INT, procprec, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&injTN, 1, MPI_INT, procprec, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&injSS, 1, MPI_INT, procsuivant, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&injTS, 1, MPI_INT, procsuivant, 0, MPI_COMM_WORLD, &status);

    // Remise des poissons
	inject_ocean2(oceanrec, N/nbproc, M, injSN+injSS, injTN+injTS);

	MPI_Gather(oceanrec, N*M/nbproc, MPI_FISH, ocean, N*M/nbproc, MPI_FISH, 0, MPI_COMM_WORLD);

    // Affichage
    if(rang == 0){

      printf(CLS "\n");
      display_ocean(ocean, N, M);
      printf("%i\n", i);
    }
  }

  if (rang == 0) {
  	free(ocean);
  }

	free(oceanrec);
	MPI_Finalize();
  return 0;
} /* main */

