#ifndef OCEAN_H
#define OCEAN_H

/* constants for display */
#define BLINK   "\x1b[1m"
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define BLUE    "\x1b[34m"
#define RESET   "\x1b[0m"
#define CLS     "\e[1;1H\e[2J"

/* The fish structure */
struct fish {
  char type;
  char moved;
};
typedef struct fish fish_t;


void init_ocean(fish_t *ocean, int n, int m, int ratio)
{
  srand(time(NULL));
  int i, j;
  int f;

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      {
	int f = rand();
	if (f % 2 == 0) {
	  if (rand() % ratio == 0)
	    ocean[i*m+j].type = 'S';
	  else
	    ocean[i*m+j].type = 'T';
	  ocean[i*m+j].moved = 0;
	}
	else /* Cell is free */
	  ocean[i*m+j].type = 'F';
      }
} /* init_ocean */


void display_ocean(fish_t *ocean, int n, int m) 
{
  int i,j;
  int ns = 0;
  int nt = 0;

  for (i = 0; i < 2 * m + 3; i++)
    printf(BLUE "-");
  printf("\n");
  for (i = 0; i < n; i++){
    printf("| ");
    for (j = 0; j < m; j++){
      if (ocean[i*m+j].type == 'S'){
	printf(BLINK RED "<");
	ns++;
      }
      else if (ocean[i*m+j].type == 'T'){
	printf(BLINK GREEN "~");
	nt++;
      }
      else
	printf(BLUE ".");
      printf(" ");
    }
    printf(BLUE "|\n");
  }
  for (i = 0; i < 2 * m + 3; i++)
    printf("-");
  printf("\n" RESET);
  printf (BLINK RED "\t\t\tSHARKS: %d" 
	  BLUE " ----- " 
	  BLINK GREEN "TUNAS: %d\n", ns, nt);
  printf(RESET);
} /* display_ocean */

void update_ocean(fish_t *ocean, int n, int m)
{

  int i,j;
  int next_i, next_j;
  int rd;

  /* Reinitiate the moved values */
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      ocean[i*m+j].moved = 0;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      if (ocean[i*m+j].moved == 0) {
	/* compute the next position (systematically) */
	rd = rand() % 100;
	if (rd < 25) { /* -> N */
	  next_i = i - 1;
	  if (next_i == -1) next_i = n - 1;
	  next_j = j;
	}
	else if (rd < 50) { /* -> E */
	  next_i = i;
	  next_j = (j + 1) % m;
	}
	else if (rd < 75) { /* -> S */
	  next_i = (i + 1) % n;
	  next_j = j;
	}
	else { /* -> W */
	  next_i = i;
	  next_j = (j-1) % m;
	  if (next_j == -1) next_j = m - 1;
	}

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
      } /* fi !moved */
    } /* for j */
  } /* for i */
} /* update_ocean */



#endif
