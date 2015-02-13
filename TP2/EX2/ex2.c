#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>



int main (int argc, char **argv) {

	// l chaine = 100
	char chaine[] = "Hocdinmaturofinteritudipsesquoquedsuidpertaesusdexcessitdedvitadaetatisdnonofannodatquedvicensimodcu";

	int i, res, rang, n	;
	int tabFreq[26], tabFreqbis[26];
	char revbuffer[10];
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rang);
	MPI_Comm_size(MPI_COMM_WORLD, &n);


	if (rang == 0){
		//init tabFreq
		for (i=0; i<26; i++){
			tabFreq[i] = 0;
		}	
	}
	
	//init tabFreqbis
	for (i=0; i<26; i++){
		tabFreqbis[i] = 0;
	}

	// envoi chaine aux processus
	MPI_Scatter(chaine, strlen(chaine)/n, MPI_CHAR, revbuffer, strlen(chaine)/n, MPI_CHAR, 0, MPI_COMM_WORLD);

	// Calcul occurences de chaque caractere
	for (i=0; i<strlen(chaine)/n; i++) {
		tabFreqbis[tolower(revbuffer[i]) - 'a'] += 1;
	}	

	// Reception / reduction
	MPI_Reduce(tabFreqbis,tabFreq,26,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

	if (rang == 0) {
		for (i=0,res=0; i<26; i++){
			printf("%c=%d ",i+'a', tabFreq[i]);
			res += tabFreq[i];
		}
			printf("\nnb total de caracteres = %d\n", res);
	}
	
	MPI_Finalize();
	return 0;
}
