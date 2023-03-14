#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/*
    Pour compiler gcc -fopenmp open.c -o open -lm (sur linux)
    Pour executer : ./open
    Avant d'executer , assurez-vous que les fichiers X.txt et y.txt se trouvent dans le même répertoire que l'exécutable 
*/

//? Taille des données 
#define ROW   8192  //? Taille du dataset
#define COL 784     //? Taille d'un exemple du dataset

//? Constantes de l'algorithme
#define Threshold 0.5 
#define LEARN_RATE 0.1  
#define NB_ITER 50
#define BATCH_SIZE 2048
#define ACCURACY  92 //? Arrét si on atteint cette précision 

//? Pour l'initialisation si RAND=1 initialisation des paramètres avec des valeurs aléatoires 
#define RAND 0  
#define Init_Value 0 //? Valeur initiale des paramètres 

//? Mettre DISPLAY à 1 pour afficher les paramètres avant et aprés l'apprentissage 
#define DISPLAY 0

//? Nombre de threads 
#define NB_THREADS 6

//? Structure d'une couche du réseau de neurones
typedef struct Layer Layer;
struct Layer{
    double** W;   
    double* b;    
    int nb_units;
    int input_size;
    char type[256];
    char name[256];
};

//? Un réseaux de neurons composé de deux couches 
Layer L1;
Layer L2;



//? Fonctions d'affichages 
void print_Layer(Layer L){
    printf("\nNom de la couche : %s \n",L.name);
    printf("Activation : %s \n",L.type);
    printf("Nb neurones : %d \n",L.nb_units);
    printf("Paramètres de la couche : \n");
    printf("\nParamètres W : \n");
    printf("--------------\n");
    for(int i=0;i<L.input_size;i++){
        printf(" ");
        for(int j=0;j<L.nb_units;j++){
            printf("%2.10f |",L.W[i][j]);
        }
        printf("\n");
    }
    printf("\nParamètre b : \n");
    printf("--------------\n");
    for(int j= 0;j<L.nb_units;j++){
        printf(" %2.10f \n",L.b[j]);
    }
    printf("===========================================\n");
}

//? Functions utilisées dans l'algorithme 
double sigmoid(double x){
    return (1/(1+exp(-x)));
}

double binary_cross_entropy(double y_hat,int y){
    if(y==1){
        return -log(y_hat);
    }else{
        return -log(1-y_hat);
    }
}


//? Apprentissage (Entrainement du réseau de neurones)

//? Initialisation des paramètres  
void init_parameters(){
        #pragma omp for 
        for(int i=0;i<L1.input_size;i++){
            for(int j=0;j<L1.nb_units;j++){
                if(RAND==0){
                    L1.W[i][j]=Init_Value;
                }else{
                    L1.W[i][j]=-1 + 2*((double)rand()/(double)(RAND_MAX));
                }
            }
        }
        #pragma omp for 
        for(int i=0;i<L1.nb_units;i++){
            if(RAND==0){
                L1.b[i]=Init_Value;
            }else{
                L1.b[i]=-1 + 2*((double)rand()/(double)(RAND_MAX));
            }
        }
        #pragma omp for 
        for(int i=0;i<L2.input_size;i++){
            for(int j=0;j<L2.nb_units;j++){
                if(RAND==0){
                    L2.W[i][j]=Init_Value;
                }else{
                    L2.W[i][j]=-1 + 2*((double)rand()/(double)(RAND_MAX));
                }
            }
        }
        #pragma omp single
        {
            if(RAND==0){
                L2.b[0]=Init_Value;
            }else{
                L2.b[0]=-1 + 2*((double)rand()/(double)(RAND_MAX));     
            }
        }
}

//? Variables globales 
//? Ces variables sont globales pour les rendre accessibles à tous les threads 

//? Sommes locales des dérivées de chaque thread
double*** grad_W1; 
double**  grad_b1;
double*** grad_W2;
double**  grad_b2;

double global_cumul_loss = 0;
int global_wrong_pred = 0;

int batch_count = 0;
int num_iter = 0;


void train_model(int nb_iter,double** X_train,int* y_train,int size_train,double learn_rate,int batch_size){

    double cumul_loss = 0;
    double loss = 0;

    int wrong_pred= 0;

    int nb_batch = size_train / batch_size;

    double tmp = 0;
    double y_hat = 0;
    double delta = 0;
    double t,start,end ;
    double time_iter;

    double* z1 = (double *) malloc(sizeof(double)*L1.nb_units);
    double* a1 = (double *) malloc(sizeof(double)*L1.nb_units);
    double* z2 = (double *) malloc(sizeof(double)*L2.nb_units);
    double* a2 = (double *) malloc(sizeof(double)*L2.nb_units);

    int tid = omp_get_thread_num();  //*le numéro du thread 

    int convergence = 0;
    #pragma omp master  //* Thread maitre s'occupe du calcul du temps d'execution
    {
         start=omp_get_wtime();
    }

    //*Initialisation des paramètres 
    init_parameters();

    #pragma omp single 
    {
        printf("\nApprentissage : \n");
        printf("\t-Taille du dataset : %d\n",size_train);
        printf("\t-Nombre d'itérations : %d\n",nb_iter);
        printf("\t-Nombre of batchs (Nombre de mise à jour des paramètres dans une itération) : %d\n",nb_batch);
        printf("\t-Taille d'un batch : %d\n",batch_size);
        printf("\t-Nombre de threads : %d\n\n",NB_THREADS);
    }

    while ( (convergence==0) && (num_iter<nb_iter) ){
        
        #pragma omp master  //*Le thread maitre s'occupe de l'affichage du temps
        {
            t = omp_get_wtime();
        }

        //! Sync des threads avant d'incrémenter le nombre d'itérations 
        #pragma omp barrier
        
        #pragma omp single
        {
            num_iter++;
            batch_count = 0;
            global_cumul_loss = 0;
            global_wrong_pred = 0;
            printf("Iter : %d [",num_iter);
        }

        cumul_loss = 0;
        wrong_pred = 0;
        
        //? Parcours des lots 
        while(batch_count<nb_batch){
            
            #pragma omp single
            {
                printf("=");
            }

            //*Initialization des dérivées locales au thread 
            for(int i=0;i<L1.input_size;i++){
                for(int j=0;j<L1.nb_units;j++){
                    grad_W1[tid][i][j]=0;
                }
            }
            for(int i=0;i<L1.nb_units;i++){
                grad_b1[tid][i]=0;
            }
            for(int i=0;i<L2.input_size;i++){
                for(int j=0;j<L2.nb_units;j++){
                    grad_W2[tid][i][j]=0;
                }
            }
            grad_b2[tid][0]=0;


            //* Parcour des éléments du Batch  (ensemble ou lot )
            #pragma omp for 
            for(int k=batch_count*batch_size;k<(batch_count+1)*batch_size;k++){
                
                //? Première passe : Forward propagation 
                //? Calcul de la sortie pour le k-ème élémenet 
                //*Activation de la première couche 
                for(int j=0;j<L1.nb_units;j++){
                    tmp = 0;
                    for(int i=0;i<L1.input_size;i++){
                        tmp+=X_train[k][i]*L1.W[i][j];
                    }
                    z1[j] = tmp + L1.b[j];
                    a1[j] = sigmoid(z1[j]);
                }

                //*Activation de la deuxième couche 
                for(int j=0;j<L2.nb_units;j++){
                    tmp = 0;
                    for(int i=0;i<L2.input_size;i++){
                        tmp+=a1[i]*L2.W[i][j];
                    }
                    z2[j] = tmp + L2.b[j];
                    a2[j] = sigmoid(z2[j]);
                }

                //? Calcul du cout pour le k-èmè élément 
                y_hat = a2[0];  //*P[élément soit un 1]
                loss = binary_cross_entropy(y_hat,y_train[k]);

                //? Cas d'une fausse prédiction 
                if((y_hat>=Threshold) && (y_train[k]==0) || (y_hat<Threshold) && (y_train[k]==1) ){
                    wrong_pred += 1;
                }
                
                //? Mise à jour du cout globale
                cumul_loss += loss;
                
                //? Deuxième pass : Backpropagation
                //? Calcul des dérivées pour les paramètres
                
                //* Delta
                delta = y_hat - y_train[k];

                //* Dérivées pour les paramètres de la première couche
                for(int i=0;i<L2.input_size;i++){
                    grad_W2[tid][i][0]+=delta * a1[i];
                }
                grad_b2[tid][0]+=delta;
                
                //* Dérivées pour les paramètres de la deuxième couche 
                for(int i=0;i<L1.input_size;i++){
                    for(int j=0;j<L1.nb_units;j++){
                        grad_W1[tid][i][j]+=delta*L2.W[j][0]*a1[j]*(1-a1[j])*X_train[k][i];
                    }
                }
                for(int j=0;j<L1.nb_units;j++){
                    grad_b1[tid][j]+=delta*L2.W[j][0]*a1[j]*(1-a1[j]);
                }
            }
            
            //! Synchronisation nécessaire : calcul de la somme globale doit être fait une fois que tous les threads ont calculé leur somme locale 
            #pragma omp barrier
            //* Chaque thread ajoute ses dérivées au dérivées globaux 
            
            //* Mise à jour des paramètres à la fin du batch  
            #pragma omp for nowait
            for(int i = 0;i<L2.input_size;i++){
                for(int k=0;k<NB_THREADS;k++){
                    L2.W[i][0]-= LEARN_RATE*(grad_W2[k][i][0] / batch_size);
                }
            }
            #pragma omp single nowait 
            {
                for(int k=0;k<NB_THREADS;k++){
                    L2.b[0] -= LEARN_RATE*(grad_b2[k][0]/batch_size);
                }
            }
            #pragma omp for nowait 
            for(int i=0;i<L1.input_size;i++){
                for(int j=0;j<L1.nb_units;j++){
                    for(int k=0;k<NB_THREADS;k++){
                        L1.W[i][j] -= LEARN_RATE*(grad_W1[k][i][j] / batch_size);
                    
                    }
                }
            }
            #pragma omp for nowait 
            for(int j =0;j<L1.nb_units;j++){
                for(int k=0;k<NB_THREADS;k++){
                    L1.b[j] -= LEARN_RATE*(grad_b1[k][j] / batch_size);
                }
            }
            
            #pragma omp single 
            {
                batch_count++;
            }

            //! Synchronisation nécessaire : les paramètres doivent être à jour avant d'entamer un nouveau ensemble  
            #pragma omp barrier   //*Sync a cause de la condition sur le nombre de batch 
        }
        
        //* Variables globales => Accès en exclusion mutuelle 
        #pragma omp critical 
        {
            global_cumul_loss += cumul_loss;
            global_wrong_pred += wrong_pred;
        }
        
        #pragma omp barrier //*Attente que tous les processus ont mis à jour les variables cumul_loss et wrong_pred
        
        #pragma omp master  //* Le maitre affiche le temps d'execution 
        {
            printf(">]\n");
            t =  omp_get_wtime() - t;
            time_iter = t * 1000;
            printf("Cout = %10f, Temps d'exécution = %5.2f (ms), Nombre de fausses prédictions = %4d \n\n",global_cumul_loss,time_iter,global_wrong_pred);
        }
        //*Condition de convergence 
        if(( ((float) global_wrong_pred) / ROW)*100 < 100-ACCURACY){
            #pragma omp single
            {
                printf("Modèle a atteint la précision voulue !!\n");
            }           
            convergence = 1;
        }
    }
    
    #pragma omp master
    {
        end = omp_get_wtime()-start;
        printf("Apprentissage du modèle terminé , Temps d'exécution = %f (s) , Précision du modèle %f pourcent \n\n",end, ( ((float) global_wrong_pred) / ROW)*100);  
    }

    //? Freeing the space 
    free(z1);      free(z2);
    free(a1);      free(a2);
    for(int i=0;i<L1.input_size;i++){
        free(grad_W1[tid][i]);
    }
    for(int i=0;i<L2.input_size;i++){
        free(grad_W2[tid][i]); 
    }
    free(grad_b1[tid]);    free(grad_b2[tid]);
    free(grad_W1[tid]);    free(grad_W2[tid]);
}


void main(){

    printf("\n-------------------> Projet HPC <------------------\n\n");
    printf("  -Apprentissage d'un réseau de neurones\n");
    printf("  -Version : Parallèle avec OpenMP\n\n");

    //? Temps d'exécution de tout le programme 
    double start,end;
    start = omp_get_wtime();

    //? Chargement des données 
    //* X : les entrées
    FILE *f = fopen("X.txt","r");
    double ** X =(double **) malloc(sizeof(double *)*ROW);
    double tmp;
    for(int i = 0;i<ROW;i++){ 
        X[i] = (double *)malloc(sizeof(double)*COL);
        for(int j=0;j<COL;j++){
            fscanf(f,"%lf",&tmp);
            X[i][j]=tmp;
        }
    }
    fclose(f);
    
    //* Y :  les sortie (0 ou 1)
    f = fopen("y.txt","r");
    int y[ROW];
    int tmp_y;
    for (int i = 0;i<ROW;i++){ 
        fscanf(f,"%d",&tmp_y);
        y[i]=tmp_y;        
    }
    fclose(f);

    printf("Données chargées en Mc , Nombre d'exemple pour l'apprentissage du modèle : %d \n",ROW);

    //? Création des couches 
    //* Couche 01 : Input layer 
    L1.nb_units = 15;
    L1.input_size = COL;
    strcpy(L1.name,"Input Layer");
    strcpy(L1.type,"sigmoid");
    L1.W = (double **) malloc(sizeof(double *)*L1.input_size);
    for(int i=0;i<L1.input_size;i++){
        L1.W[i] = (double *)malloc(sizeof(double)*L1.nb_units);
        for(int j=0;j<L1.nb_units;j++){
            L1.W[i][j]=Init_Value;
        }
    }
    L1.b= (double *)malloc(sizeof(double)*L1.nb_units);
    for(int i=0;i<L1.nb_units;i++){
        L1.b[i]=Init_Value;
    }

    //* Couche 02 : Output layer
    L2.nb_units = 1;
    L2.input_size = L1.nb_units;
    strcpy(L2.name,"Output Layer");
    strcpy(L2.type,"sigmoid");
    L2.W = (double **) malloc(sizeof(double *)*L2.input_size);
    for(int i=0;i<L2.input_size;i++){
        L2.W[i] = (double *)malloc(sizeof(double)*L2.nb_units);
        for(int j=0;j<L2.nb_units;j++){
            L2.W[i][j]=Init_Value;
        }
    }
    L2.b= (double *)malloc(sizeof(double)*L2.nb_units);
    L2.b[0]=Init_Value;




    //* Allocation de l'espace pour les sommes locales des dérivées 
    grad_W1 = (double ***) malloc(sizeof(double **)*NB_THREADS);
    for(int k=0;k<NB_THREADS;k++){
        grad_W1[k] = (double **)malloc(sizeof(double *)*L1.input_size);
        for(int i=0;i<L1.input_size;i++){
            grad_W1[k][i] = (double *) malloc(sizeof(double)*L1.nb_units);
        }
    }
    grad_W2 = (double ***) malloc(sizeof(double **)*NB_THREADS);
    for(int k=0;k<NB_THREADS;k++){
        grad_W2[k] = (double **)malloc(sizeof(double *)*L2.input_size);
        for(int i=0;i<L2.input_size;i++){
            grad_W2[k][i] = (double *) malloc(sizeof(double)*L2.nb_units);
        }
    }

    grad_b1 = (double **)malloc(sizeof(double *)*NB_THREADS);
    for(int k =0;k<NB_THREADS;k++){
        grad_b1[k] = (double *)malloc(sizeof(double )*L1.nb_units);
    }

    grad_b2 = (double **)malloc(sizeof(double *)*NB_THREADS);
    for(int k =0;k<NB_THREADS;k++){
        grad_b2[k] = (double *)malloc(sizeof(double )*L2.nb_units);
    }

    if(DISPLAY==1){
        printf("Paramètres Initiaux : \n");
        print_Layer(L1);
        print_Layer(L2);
    }
    
    //? Apprentissage du modèle 
    #pragma omp parallel num_threads(NB_THREADS)
    {
        train_model(NB_ITER,X,y,ROW,LEARN_RATE,BATCH_SIZE);
    }
    end = omp_get_wtime()-start;
    printf("Temps d'exécution de tout le programme (Chargement de données + Initialisation + Apprentissage) :  %f (s) \n", end);  

    if(DISPLAY==1){
        printf("Paramètres finaux : \n");
        print_Layer(L1);
        print_Layer(L2);
    }

    //? Libération de l'espace utilisé 
    for (int i = 0;i<ROW;i++){
        free(X[i]);
    }
    free(grad_b1); free(grad_b2); free(grad_W1); free(grad_W2);
}