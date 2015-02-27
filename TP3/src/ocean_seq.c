#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "ocean.h"

/* constants for the ocean */
#define N 40
#define M 20
#define WALL 100
#define STEP 150000
#define RATIO 10

int main ()
{
  int i;
  fish_t *ocean = (fish_t *)malloc(N*M*sizeof(fish_t));
  
  init_ocean(ocean, N, M, RATIO);
  printf(CLS "\n");
  display_ocean(ocean, N, M);

  for (i = 0; i < WALL; i++) {
    usleep(STEP);
    printf(CLS "\n");
    update_ocean(ocean, N, M);
    display_ocean(ocean, N, M);
  }

  return 0;
} /* main */

