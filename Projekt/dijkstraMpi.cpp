#include <stdio.h>
#include <stdlib.h>

#include <limits.h>
#include <time.h>

#include <mpi.h>

/// funkcja znajduje minimum lokalne
/// - wyznacza minimum lokalne danego procesu
/// - znajdujemy wierzcholek z przedzialu na jakim dany proces pracuje o jak najmniejszej odleglosci danego wierzcholka od wierzcholka zrodlowego
void findLoclaMinimum(int *minimum, int *vertexes, int *path, int lastElement,
                      int firstElement)
{
    int i;
    int min = INT_MAX;
    minimum[0] = INT_MAX;

    for(i = lastElement; i <= firstElement; i++)
    {
        if(vertexes[i] == 0)
        {
            if(min >= path[i])
            {
                min = path[i];

                minimum[0] = min;
                minimum[1] = i;
            }
        }
    }
}

/// funckja uaktualniajaca odleglosci
/// - uaktualnia odleglosci wszystkich nierozpatrzonych wierzolkow sasiadujacych ze znalezionym wczensiej wierzcholkiem minimum
/// - gdy odleglosc rozpatrywanego wierzcholka < minimalnego --> odleglosc rozpatrywanego = odl minimalnego
void updateDistances(int *wektor_sciezek, int *globalne_minimum, int *graf, int *rozpatrzone_wierzcholki,
                     int wielkosc_grafu, int ostatni_element)
{
    int najmniejsza_odleglosc = globalne_minimum[0];
    int najmniejszy_wierzcholek = globalne_minimum[1];

    int tmp_odleglosc = 0;
    int i;
    for (i = 0; i < wielkosc_grafu; i++)
    {
        tmp_odleglosc = graf[najmniejszy_wierzcholek * wielkosc_grafu +i];

        if(rozpatrzone_wierzcholki[i + ostatni_element] == 0 && tmp_odleglosc > 0)
        {

            if(wektor_sciezek[i + ostatni_element] > najmniejsza_odleglosc + tmp_odleglosc)
            {
                wektor_sciezek[i + ostatni_element] = najmniejsza_odleglosc + tmp_odleglosc;
            }
        }
    }
}

/// funkcja odpowiedzialna za generowanie grafu
/// - generuje losowy kawalek macierzy dla procesu
/// - losowane są wartosci od 0 do 500
/// - to czy krawedz powstawnie jest zalezne od gestosci grafu
/// - - im wieksza gestosc tym wieksza ilosc krawedzi odchodzacych od kazdego wierzcholka
void generateGraph(int *mat, int wielkosc, int rank, int gestosc_grafu, int wielkosc_grafu)
{
    time_t t;
    srand((unsigned) time(&t) * rank + 1);

    int i,j;
    int vertex;
    for (i = 0; i < wielkosc; i++)
    {
        for (j = 0; j < wielkosc_grafu; j++)
        {
            if (i == j)
            {
                mat[i * wielkosc_grafu + j] = 0;
            }
            else
            {
                vertex = rand() % 500;

                if (vertex <= gestosc_grafu)
                    mat[i * wielkosc_grafu + j] = rand() % 500;
                else
                    mat[i * wielkosc_grafu + j] = 0;
            }
        }
    }
}

///  funkcja sprawdza czy wierzcholek zrodlowy miesci sie w ilosci wierzcholkow w grafie
int sprawdz_wierzcholek(int zrodlo_wiercholek, int wielkosc)
{
    if (zrodlo_wiercholek > wielkosc)
        return 0;
    else
        return 1;
}


/// main
int _main(int argc, char *argv[])
{
    int i,j;


    /// zmienne odpowiedzalne za przechowywanie czasu
    /// potrzebne do obliczen czasu wykonania programu
    double 	start_init, end_init,
            start_time, end_time;

    int rank,  				/// ID procesu
            ilosc_procesow, 	/// ilosc procesow
            wielkosc_grafu, 	/// wielkosc grafu
            gestosc_grafu;		/// gestosc grafu - im wieksza tym wiecej krawedzi odchodzacyh od grafu

    /// zrodlowy wierzcholek grafu
    /// - od niego startujemy
    int zrodlo_wiercholek;

    /// rozmiar macierzy przechowujacej koszt podrozy po sciezkach
    int wielkosc_macierzy;



    int ostatni_element = 0, 	/// przechowuje ostatni element wektora sciezek dla danego ID
            pierwszy_element = 0; 	/// przechowuje pierwszy element wektora sciezek dla danego ID

    MPI_Init(&argc, &argv); // inicjalizacja

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // ktory komputer
    MPI_Comm_size(MPI_COMM_WORLD, &ilosc_procesow); // ilosc procesow

    int length=80; char name[length];
    MPI_Get_processor_name(name,&length);

    if(rank == 0)
    {
        printf("Pogram: PROJEKT NA PRIR - Michal Franczyk, Bartlomiej Rupik\n");
        printf("Algorytm Dijkstry - znajdywanie najkrotszych sciezek w grafie.\n\n");
    }

    if ( argc > 3 )
    {
        if(rank == 0)
        {
            wielkosc_grafu	 	= atoi(argv[1]);
            gestosc_grafu		= atoi(argv[2]);
            if(sprawdz_wierzcholek(atoi(argv[3]),wielkosc_grafu*ilosc_procesow) == 0)
            {
                if(rank == 0)
                    printf("nie ma takiego wierzcholka\n wierzcholek ustawiony na ostatni mozliwy\n\n");
                zrodlo_wiercholek = wielkosc_grafu*ilosc_procesow - 1 ;
            }
            else
            {
                zrodlo_wiercholek 	= atoi(argv[3]);
            }
        }
        MPI_Bcast(&wielkosc_grafu, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
        MPI_Bcast(&gestosc_grafu, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
        MPI_Bcast(&zrodlo_wiercholek, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    }
    else
    {
        if(rank == 0)
        {
            printf("podaj wielskosc grafu = "); scanf("%d",&wielkosc_grafu);
            printf("podaj gestosc grafu grafu = "); scanf("%d",&gestosc_grafu);
            printf("podaj startowy wierzcholek = "); scanf("%d",&zrodlo_wiercholek);

            if(sprawdz_wierzcholek(zrodlo_wiercholek,wielkosc_grafu*ilosc_procesow) == 0)
            {
                if(rank == 0)
                    printf("nie ma takiego wierzcholka\n wierzcholek ustawiony na ostatni mozliwy\n\n");
                zrodlo_wiercholek = wielkosc_grafu*ilosc_procesow - 1 ;
            }

            MPI_Bcast(&wielkosc_grafu, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
            MPI_Bcast(&gestosc_grafu, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
            MPI_Bcast(&zrodlo_wiercholek, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Bcast(&wielkosc_grafu, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
            MPI_Bcast(&gestosc_grafu, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
            MPI_Bcast(&zrodlo_wiercholek, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
        }
    }

    /// obliczamy wielkosc macierzy
    /// - wielkosc macierzy jest ilosczynem wielkosci grafu oraz ilosci procesow
    /// - - pozwala nam to na optymalne podzielenie pracy na kazdy proces
    wielkosc_macierzy = wielkosc_grafu * ilosc_procesow;

    if (rank == 0)
    {
        printf("Kolumn dla procesu: %d \n", wielkosc_grafu);
        printf("Rozmiar grafu: %d\n",wielkosc_macierzy);
        printf("Gestosc grafu: %d ( im wieksza gestosc tym wiecej sciezek )\n", gestosc_grafu);
    }

    start_init			=	MPI_Wtime();

    /// ustawienie ostatniego i pierwszwgo elementu
    ostatni_element 		=	rank * wielkosc_grafu;
    pierwszy_element 		=	ostatni_element + wielkosc_grafu - 1;

    int graf[wielkosc_grafu * wielkosc_macierzy];
    generateGraph(graf, wielkosc_macierzy, rank, gestosc_grafu, wielkosc_grafu);

    int wektor_sciezek[wielkosc_macierzy];
    int rozpatrzone_wierzcholki[wielkosc_macierzy];

    int lokalne_minimum[2];
    int globalne_minimum[2];

    /// - wypelnienie wektra sciezek nieskonczonosciami
    /// - wyzerowanie tablicy rozpatrzonych wierzcholkow
    for(i = 0; i < wielkosc_macierzy; i++)
    {
        rozpatrzone_wierzcholki[i] = 0;
        wektor_sciezek[i] = INT_MAX;
    }

    wektor_sciezek[zrodlo_wiercholek] = 0;

    if(rank==0)
        printf("\n Inicjalizacja %d procesow:  \n", ilosc_procesow);

    for(i=0;i<=ilosc_procesow;++i)
    {
        if(rank==i)
            printf(" proces: %d  :: nazwa: %s \n",i,name);
    }

    end_init=MPI_Wtime();

    if(rank==0)
        printf("Czas przeznaczony na inicjalizacje %f sekund\n\n", end_init-start_init);

    start_time = MPI_Wtime();

    /// start glownego algorytmu
    /// - szukamy minimum 			: kazdy proces szuka swojego po czym nastepuje zlicznie globalnego minumim
    /// - MPI_Allreduce   			: zbiera dane z procesow o lokalnych minimach aby obliczyc globalne minimum
    /// - uaktualniamy odleglosci 	: po wylicznieniu globalnego minimum aktualizowane sa odlegosci w grafie od wierzcholka zrodlowego
    /// - ustawiamy ostatni rozpatrzony wierzcholek aby usprawnic obliczenia
    for(i=1; i < wielkosc_macierzy; ++i)
    {
        findLoclaMinimum(lokalne_minimum, rozpatrzone_wierzcholki, wektor_sciezek, ostatni_element, pierwszy_element);

        // globalne minimum
        MPI_Allreduce (lokalne_minimum, globalne_minimum, 2, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        updateDistances(wektor_sciezek, globalne_minimum, graf, rozpatrzone_wierzcholki, wielkosc_grafu,
                        ostatni_element);

        rozpatrzone_wierzcholki[globalne_minimum[1]] = 1;
    }

    int wynikowy_wektor_sciezek[wielkosc_macierzy];

    /// zbieramy dane aby otrzymać odleglosci od zrodlowego do kazdego wierzcholka
    /// - wektor_sciezek + ostatni_element  : sciagany jest bufor z danego procesu
    /// - wynikowy_wektor_sciezek			: zapisywany koszt sciezki do danego wierzcholka
    MPI_Gather(wektor_sciezek + ostatni_element, wielkosc_grafu, MPI_INT, wynikowy_wektor_sciezek, wielkosc_grafu, MPI_INT,0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    MPI_Finalize();

// 	wypisz_macierz_sciezek(wynikowy_wektor_sciezek, wielkosc_grafu);

    if (rank == 0)
    {
        printf("\n");
        for(int i = 0; i < wielkosc_macierzy; i++)
        {
            if( wynikowy_wektor_sciezek[i] == INT_MAX )
                printf("Odleglosc wierzcholka zrodlowego ( %d ) od wiercholka %d  =  BRAK POLACZENIA\n",zrodlo_wiercholek,i);
            else
                printf("Odleglosc wierzcholka zrodlowego ( %d ) od wiercholka %d  =  %d\n",zrodlo_wiercholek,i, wynikowy_wektor_sciezek[i]);
        }

        printf("\n\nAlgorytm wykonal sie w %f\n", end_time - start_time);
        printf("Czas wykonania sie calego programu %lf \n", end_time - start_init);
    }

    return 0;
}