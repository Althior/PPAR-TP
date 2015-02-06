#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>
#include <sys/time.h>

int isvowel(int ch)
{
 char c = (char) ch + 'a'; 
 return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y');
}

int main () {

	char chaine[] = "Hoc inmaturo interitu ipse quoque sui pertaesus excessit e vita aetatis nono anno atque vicensimo cum quadriennio imperasset natus apud Tuscos in Massa Veternensi patre Constantio Constantini fratre imperatoris matreque Galla sorore Rufini et Cerealis quos trabeae consulares nobilitarunt et praefecturaeHoc inmaturo interitu ipse quoque sui pertaesus excessit e vita aetatis nono anno atque vicensimo cum quadriennio imperasset natus apud Tuscos in Massa Veternensi patre Constantio Constantini fratre imperatoris matreque Galla sorore Rufini et Cerealis quos trabeae consulares nobilitarunt et praefecturaeHoc inmaturo interitu ipse quoque sui pertaesus excessit e vita aetatis nono anno atque vicensimo cum quadriennio imperasset natus apud Tuscos in Massa Veternensi patre Constantio Constantini fratre imperatoris matreque Galla sorore Rufini et Cerealis quos trabeae consulares nobilitarunt et praefecturaeHoc inmaturo interitu ipse quoque sui pertaesus excessit e vita aetatis nono anno atque vicensimo cum quadriennio imperasset natus apud Tuscos in Massa Veternensi patre Constantio Constantini fratre imperatoris matreque Galla sorore Rufini et Cerealis quos trabeae consulares nobilitarunt et praefecturaeHoc inmaturo interitu ipse quoque sui pertaesus excessit e vita aetatis nono anno atque vicensimo cum quadriennio imperasset natus apud Tuscos in Massa Veternensi patre Constantio Constantini fratre imperatoris matreque Galla sorore Rufini et Cerealis quos trabeae consulares nobilitarunt et praefecturaeHoc inmaturo interitu ipse quoque sui pertaesus excessit e vita aetatis nono anno atque vicensimo cum quadriennio imperasset natus apud Tuscos in Massa Veternensi patre Constantio Constantini fratre imperatoris matreque Galla sorore Rufini et Cerealis quos trabeae consulares nobilitarunt et praefecturaeHoc inmaturo interitu ipse quoque sui pertaesus excessit e vita aetatis nono anno atque vicensimo cum quadriennio imperasset natus apud Tuscos in Massa Veternensi patre Constantio Constantini fratre imperatoris matreque Galla sorore Rufini et Cerealis quos trabeae consulares nobilitarunt et praefecturae";

	int i;
	int nbvoy=0, nbcons=0, nbtotal=0;
	int tabFreq[26];

	double ecoule;
	struct timeval tv0, tv1;
	gettimeofday(&tv0, 0);

	//init tabFreq
	#pragma omp parallel for schedule(static) shared(tabFreq) private(i)
	for (i=0; i<26; i++){
		tabFreq[i] = 0;
	}

	// Calcul des fréquences
	#pragma omp parallel for schedule(static) shared(tabFreq) private(i)
	for (i=0; i<strlen(chaine); i++){
		#pragma omp atomic
		tabFreq[tolower(chaine[i]) - 'a'] += 1;
	}

	// Calcul du nombre de voyelles/consonnes
	#pragma omp parallel for schedule(static) shared(tabFreq) private(i) reduction (+:nbvoy, nbcons)
	for (i=0; i<26; i++){
	
		if (isvowel(i)) {
			nbvoy+=tabFreq[i];
		}
		else {
			nbcons+=tabFreq[i];
		}
	
	}

	nbtotal = nbcons + nbvoy;

	// Temps écoulé
	gettimeofday(&tv1, 0);
	ecoule = (double)((tv1.tv_sec-tv0.tv_sec)*10e6 + tv1.tv_usec - tv0.tv_usec) / 10e6;
	printf("Temps écoulé : %.6f secondes\n", ecoule);

	printf("Nombre voyelles: %d\nNombre consonnes: %d\nNombre lettres: %d\n", nbvoy, nbcons, nbtotal);
	return 0;
	}
