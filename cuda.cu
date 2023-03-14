#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*
    Pour compiler nvcc cuda.cu -o cuda 
    Pour executer : ./cuda
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

//? Nombre de bloques et nombre de threads par bloque
#define NB_BLOC 32
#define NB_THREAD 800  //! Fixe 

//? Mettre DISPLAY à 1 pour afficher les paramètres avant et aprés l'apprentissage 
#define DISPLAY 0

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

//? Fonctions utilisées dans l'algorithme
__device__
double sigmoid_gpu(double x){
    return (1/(1+exp(-x)));
}

__device__
double binary_cross_entropy_gpu(double y_hat,int y){
    if(y==1){
        return -log(y_hat);
    }else{
        return -log(1-y_hat);
    }
}

//? Apprentissage (Entrainement du réseau de neurones)

//? Initialisation des paramètres  
void init_parameters(){
    if(RAND==0){ //*Initialisation avec une constante
        for(int i=0;i<L1.input_size;i++){
            for(int j=0;j<L1.nb_units;j++){
                L1.W[i][j]=Init_Value;
            }
        }
        for(int i=0;i<L1.nb_units;i++){
            L1.b[i]=Init_Value;
        }
        for(int i=0;i<L2.input_size;i++){
            for(int j=0;j<L2.nb_units;j++){
                L2.W[i][j]=Init_Value;
            }
        }
        L2.b[0]=Init_Value;
    }else{ //*Initialisation avec des valeurs aléatoires entre [-1,1]
        srand((unsigned int)time(NULL));
        for(int i=0;i<L1.input_size;i++){
            for(int j=0;j<L1.nb_units;j++){
                L1.W[i][j]=-1 + 2*((double)rand()/(double)(RAND_MAX));
            }
        }
        for(int i=0;i<L1.nb_units;i++){
            L1.b[i]=-1 + 2*((double)rand()/(double)(RAND_MAX));
        }
        for(int i=0;i<L2.input_size;i++){
            for(int j=0;j<L2.nb_units;j++){
                L2.W[i][j]=-1 + 2*((double)rand()/(double)(RAND_MAX));
            }
        }
        L2.b[0]=-1 + 2*((double)rand()/(double)(RAND_MAX));         
    }
}


//? Variables globales 

//* Pointeur vers les données dans le GPU
double** gpu_X;
int* gpu_y;

//* Pointeurs vers les couhes dans le GPU
Layer* gpu_L1;
Layer* gpu_L2;

//* Pointeurs vers les dérivées des paramètres dans le GPU
double ** gpu_W1;
double ** gpu_W2;
double* gpu_b1;
double* gpu_b2;

//* Pointeur vers un tableau (stocké dans GPU) ou chaque element 
//* contient un pointeur vers les dérivées calculées per un BLOC
double*** gpu_grad_W1; 
double**  gpu_grad_b1;
double*** gpu_grad_W2;
double**  gpu_grad_b2;

//? Fonnction transfert GPU (device) <-> CPU (host)
//* Copie des paramètres des deux couches du CPU vers le GPU
void copy_param_host_device(){
    //*W1
    for(int i=0;i<L1.input_size;i++){
        cudaMemcpy(gpu_W1[i],L1.W[i],sizeof(double)*L1.nb_units,cudaMemcpyHostToDevice);
    }
    //*b1
    cudaMemcpy(gpu_b1,L1.b,sizeof(double)*L1.nb_units,cudaMemcpyHostToDevice);
    //*W2
    for(int i=0;i<L2.input_size;i++){
        cudaMemcpy(gpu_W2[i],L2.W[i],sizeof(double)*L2.nb_units,cudaMemcpyHostToDevice);
    }
    //*b2
    cudaMemcpy(gpu_b2,L2.b,sizeof(double)*L2.nb_units,cudaMemcpyHostToDevice);
}
//* Copie des paramètres du GPU vers le CPU
void copy_param_device_host(){
     //*W1
    for(int i=0;i<L1.input_size;i++){
        cudaMemcpy(L1.W[i],gpu_W1[i],sizeof(double)*L1.nb_units,cudaMemcpyDeviceToHost);
    }
    //*b1
    cudaMemcpy(L1.b,gpu_b1,sizeof(double)*L1.nb_units,cudaMemcpyDeviceToHost);
    //*W2
    for(int i=0;i<L2.input_size;i++){
        cudaMemcpy(L2.W[i],gpu_W2[i],sizeof(double)*L2.nb_units,cudaMemcpyDeviceToHost);
    }
    //*b2
    cudaMemcpy(L2.b,gpu_b2,sizeof(double)*L2.nb_units,cudaMemcpyDeviceToHost);
}

//? Fonction du Device 

__global__ void update_param(Layer *L1,Layer *L2,
    double*** tab_grad_W1,double** tab_grad_b1,
    double*** tab_grad_W2,double** tab_grad_b2){
    //Sum W1
        if(threadIdx.x<L1->input_size ){
            for(int j=0;j<L1->nb_units;j++){
                L1->W[threadIdx.x][j] -= LEARN_RATE*(tab_grad_W1[0][threadIdx.x][j] / BATCH_SIZE);
            }
        }

    //Sum b1
        if(threadIdx.x < L1->nb_units){
            L1->b[threadIdx.x] -= LEARN_RATE*( tab_grad_b1[0][threadIdx.x]/BATCH_SIZE);
        }
    
    //Sum W2
        if(threadIdx.x<L2->input_size){
            for(int j=0;j<L2->nb_units;j++){
                L2->W[threadIdx.x][j] -= LEARN_RATE*( tab_grad_W2[0][threadIdx.x][j]/BATCH_SIZE);
            }
        }
    //Sum b2
        if(threadIdx.x<L2->nb_units){
            L2->b[threadIdx.x] -=LEARN_RATE*( tab_grad_b2[0][threadIdx.x]/BATCH_SIZE);
        }
}

//*Calucl de la somme des dérivées calculées par les bloques 
//* On la stocke dans la zone du premier bloque (bloque 0)
__global__ void sum_grad(Layer *L1,Layer *L2,
    double*** tab_grad_W1,double** tab_grad_b1,
    double*** tab_grad_W2,double** tab_grad_b2){
    //Sum W1
    for(int k=1;k<NB_BLOC;k++ ){
        if(threadIdx.x<L1->input_size && threadIdx.y<L1->nb_units){
            for(int j=0;j<L1->nb_units;j++){
                tab_grad_W1[0][threadIdx.x][j] += tab_grad_W1[k][threadIdx.x][j];
            }
        }
    }  
    //Sum b1
    for(int k=1;k<NB_BLOC;k++ ){
        if(threadIdx.x < L1->nb_units){
            tab_grad_b1[0][threadIdx.x] +=tab_grad_b1[k][threadIdx.x];
        }
    }
    //Sum W2
    for(int k=1;k<NB_BLOC;k++ ){
        if(threadIdx.x<L2->input_size){
            for(int j=0;j<L2->nb_units;j++){
                tab_grad_W2[0][threadIdx.x][j] += tab_grad_W2[k][threadIdx.x][j];
            }
        }
    }
    //Sum b2
    for(int k=1;k<NB_BLOC;k++ ){
        if(threadIdx.x<L2->nb_units){
            tab_grad_b2[0][threadIdx.x] += tab_grad_b2[k][threadIdx.x];
        }
    }
}


//? Fonction principale du device
//* Cette fonction sera exécutée par NB_BLOC bloques 
//* Chaque bloque va calculer une somme de dérivées locale 
//* Un bloque va traiter (BATCH_SIZE / NB_BLOC) élement du dataset 

__global__ void calc_grad(double** X_train,int *y_train,
    Layer *L1,Layer *L2,
    double*** tab_grad_W1,double** tab_grad_b1,
    double*** tab_grad_W2,double** tab_grad_b2,
    double* cumul_loss,int *start,int *wrong_pred
    ){
    
    __shared__ double z1[15];
    __shared__ double a1[15];   //*activation de la  première couche 
    __shared__ double z2[1];
    __shared__ double a2[1];    //*activation de la  deuxième couche
    __shared__ double y_hat;
    __shared__ double delta;
    __shared__ double loss;
    int portion_size = BATCH_SIZE / NB_BLOC;   //* Nombre de données traité par le BLOC
    int k = blockIdx.x* portion_size + *start ;   //* Indice du premier element traité par le bloc  
    
    for(int elem = 0;elem<portion_size;elem++){
        
        //? Première passe : Forward propagation 
        //? Calcul de la sortie pour le k-ème élémenet 
        
        //*Activation de la première couche 
        if(threadIdx.x<L1->nb_units){
            z1[threadIdx.x] = 0;
            for(int i=0;i<L1->input_size;i++){
                z1[threadIdx.x]+=X_train[k][i]*L1->W[i][threadIdx.x];
            }
            z1[threadIdx.x] += L1->b[threadIdx.x];
            a1[threadIdx.x]=sigmoid_gpu(z1[threadIdx.x]);
        }
        //*Activation de la deuxième couche 
        if(threadIdx.x<L2->nb_units){
            z2[threadIdx.x] = 0;
            for(int i=0;i<L2->input_size;i++){
                z2[threadIdx.x]+=a1[i]*L2->W[i][threadIdx.x];
            }
            z2[threadIdx.x] += L2->b[threadIdx.x];
            a2[threadIdx.x]=sigmoid_gpu(z2[threadIdx.x]);
        } 
        
        
        if(threadIdx.x==0){   
            y_hat = a2[0];  //* Résultat de la première passe (y_hat = Probabilité que le k-èmè élément soit un 1)
            if(elem==0){    //* Si c'est le premier élément traité par le bloque, on initialise le nombre de fausse prédictions à 0
                wrong_pred[blockIdx.x]=0;            
            }

            //? Cas d'une fausse prédiction 
            if((y_hat>=Threshold) && (y_train[k]==0) || (y_hat<Threshold) && (y_train[k]==1) ){
                wrong_pred[blockIdx.x]+=1;
            }

            //* Calcul du cout pour le k-ème élement  
            loss = binary_cross_entropy_gpu(y_hat,y_train[k]);
            
            //* Initialisation du cout globale du bloque 
            if(elem==0){
                cumul_loss[blockIdx.x]=0;
            }
            cumul_loss[blockIdx.x]+=loss;  //* Cout cumulé calculé par le bloque
            delta = y_hat-y_train[k];
            
        }
        //! Synchronisation nécessaire pour ne pas avoir des fausses valeurs de loss et delta 
        __syncthreads(); 
        
        //? Ajout des dérivées calculées par le k-émé élément  
        //* Dérivée pour la deuxième couche 
        if(threadIdx.x<L2->input_size){
            if(elem==0){
                tab_grad_W2[blockIdx.x][threadIdx.x][0] = 0;
            }
            tab_grad_W2[blockIdx.x][threadIdx.x][0]+=delta * a1[threadIdx.x];
        }

        //* Dérivée pour la première couche 
        if(threadIdx.x<L1->input_size){
            for(int j=0;j<L1->nb_units;j++){
                if(elem==0){
                    tab_grad_W1[blockIdx.x][threadIdx.x][j]=0;
                }
                tab_grad_W1[blockIdx.x][threadIdx.x][j]+=delta*L2->W[j][0]*a1[j]*(1-a1[j])*X_train[k][threadIdx.x];
            }
        }
        if(threadIdx.x < L1->nb_units){
            if(elem==0){
                tab_grad_b1[blockIdx.x][threadIdx.x]=0;
            }
            tab_grad_b1[blockIdx.x][threadIdx.x]+=delta*L2->W[threadIdx.x][0]*a1[threadIdx.x]*(1-a1[threadIdx.x]);
        }
        if(threadIdx.x==0){
            if(elem==0){
                tab_grad_b2[blockIdx.x][0] = 0;
            }
            tab_grad_b2[blockIdx.x][0]+=delta;
        }
        k++;        
    }
}

//? Fonction principale 
void train_model_GPU(int nb_iter,double** X_train,int* y_train,int size_train,double learn_rate,int batch_size){

    //*Pour synchronisation CPU/GPU
    cudaEvent_t  sync;
    cudaEventCreate(&sync);
    
    cudaError_t error;

    double cumul_loss_tab[NB_BLOC];
    int cumul_wrong_pred[NB_BLOC];

    //*Cout calculé par chaque bloque
    double* gpu_cumul_loss;
    cudaMalloc(&gpu_cumul_loss,sizeof(double)*NB_BLOC);
    
    //*Nombre de fausse prédicion calculé par chaque bloque 
    int* gpu_cumul_wrong_pred;
    cudaMalloc(&gpu_cumul_wrong_pred,sizeof(int)*NB_BLOC);
    
    //* Une variable pour transmettre au bloc l'indice du premier element du batch
    int* gpu_start_indx;
    cudaMalloc(&gpu_start_indx,sizeof(int));

    int start_indx;

    int convergence = 0;
    int num_iter = 0;
    double cumul_loss = 0;

    int wrong_pred= 0;

    int batch_count = 0;
    int nb_batch = size_train / batch_size;

    //* Variables pour la mesure du temps 
    time_t t,start,end ;
    double time_iter;
    start=clock(); //* Temps globale 

    //*Initialisation des paramètres  
    init_parameters();

    
    printf("\nApprentissage : \n");
    printf("\t-Taille du dataset : %d\n",size_train);
    printf("\t-Nombre d'itérations : %d\n",nb_iter);
    printf("\t-Nombre of batchs (Nombre de mise à jour des paramètres dans une itération) : %d\n",nb_batch);
    printf("\t-Taille d'un batch : %d\n\n",batch_size);

    
    //* Copie des paramètres vers le GPU
    copy_param_host_device();
    
    while ( (convergence==0) && (num_iter<nb_iter) ){
        num_iter++;
        cumul_loss = 0;
        wrong_pred = 0;
        t = clock(); //* Temps de l'itération 
        
        printf("Iter : %d [",num_iter);
        for (batch_count=0;batch_count<nb_batch;batch_count++){
            printf("=");

            //* Index du premier élément du batch 
            start_indx = batch_count*batch_size;
            cudaMemcpy(gpu_start_indx,&start_indx,sizeof(int),cudaMemcpyHostToDevice);
        
            calc_grad<<<NB_BLOC,COL>>>(gpu_X,gpu_y
                ,gpu_L1,gpu_L2
                ,gpu_grad_W1,gpu_grad_b1
                ,gpu_grad_W2,gpu_grad_b2
                ,gpu_cumul_loss,gpu_start_indx,gpu_cumul_wrong_pred
                );
            cudaEventRecord(sync,0);
            error = cudaGetLastError();
            if (error != cudaSuccess){
                printf("Erreur lors de l'appel du device dans la fonction calc_grad:%s\n",cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
            cudaEventSynchronize(sync); //* CPU attend que le GPU termine le calcul

            //* Calcul du cout cumulé de l'itération ainsi que le nombre d'itérations 
            cudaMemcpy(cumul_loss_tab,gpu_cumul_loss,sizeof(double)*NB_BLOC,cudaMemcpyDeviceToHost);
            cudaMemcpy(cumul_wrong_pred,gpu_cumul_wrong_pred,sizeof(int)*NB_BLOC,cudaMemcpyDeviceToHost);
            for(int k=0;k<NB_BLOC;k++){
                cumul_loss+=cumul_loss_tab[k];
                wrong_pred+=cumul_wrong_pred[k];
            }

            //* Calcul de la somme des dérivées de l'itération 
            sum_grad<<<1,COL>>>(gpu_L1,gpu_L2,gpu_grad_W1,gpu_grad_b1,gpu_grad_W2,gpu_grad_b2);
            cudaEventRecord(sync,0);
            error = cudaGetLastError();
            if (error != cudaSuccess){
                printf("Erreur lors de l'appel du device dans la fonction sum_grad:%s\n",cudaGetErrorString(error));
                // return EXIT_FAILURE;
                exit(EXIT_FAILURE);
            }
            cudaEventSynchronize(sync);

            //* Mise à jour des paramètres 
            update_param<<<1,COL>>>(gpu_L1,gpu_L2,gpu_grad_W1,gpu_grad_b1,gpu_grad_W2,gpu_grad_b2);
            cudaEventRecord(sync,0);
            error = cudaGetLastError();
            if (error != cudaSuccess){
                printf("Erreur lors de l'appel du device dans la fonction update_param:%s\n",cudaGetErrorString(error));
                // return EXIT_FAILURE;
                exit(EXIT_FAILURE);
            }
            cudaEventSynchronize(sync);
        }
        printf(">]\n");
        t = clock() - t; 
        time_iter = (((double) t )/CLOCKS_PER_SEC) * 1000 ;
        printf("Cout = %10f, Temps d'exécution = %5.2f (ms), Nombre de fausses prédictions = %4d \n\n",cumul_loss,time_iter,wrong_pred);
        if(( ((float) wrong_pred) / ROW)*100 < 100-ACCURACY){
            printf("Modèle a atteint la précision voulue !!\n");
            convergence = 1;
        }
    }

    //*Copie des paramètres du GPU vers le CPU
    copy_param_device_host();

    end = clock()-start; //* Temps globale 
    double time_taken = ((double)end)/CLOCKS_PER_SEC; //* Conversion en secondes
    printf("Apprentissage du modèle terminé , Temps d'exécution = %f (s) , Précision du modèle %f pourcent \n\n",time_taken, ( ((float) wrong_pred) / ROW)*100);  
}

int main(){

    printf("\n-------------------> Projet HPC <------------------\n\n");
    printf("  -Apprentissage d'un réseau de neurones\n");
    printf("  -Version : Parallèle avec Cuda C\n\n");

    //? Temps d'exécution de tout le programme 
    time_t start,end;
    start = clock();

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
    L2.b[0]=Init_Value ;

    

    //? Copie des données vers le GPU
    
    //*Allocation de l'espace sur le GPU
    cudaMalloc(&gpu_X,sizeof(double *)*ROW);
    cudaMalloc(&gpu_y,sizeof(int)* ROW);
    double* tmp_gpu_X[ROW];
    for(int i=0;i<ROW;i++){
        cudaMalloc(&tmp_gpu_X[i],sizeof(double)*COL);
        cudaMemcpy(tmp_gpu_X[i],X[i],sizeof(double)*COL,cudaMemcpyHostToDevice);
    }
    cudaMemcpy(gpu_X,tmp_gpu_X,sizeof(double *)*ROW,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y,y,sizeof(int)*ROW,cudaMemcpyHostToDevice);
    
    printf("Données copiées sur le GPU \n");
    
    //? Copie des couches vers le GPU

    //*Allocation de l'espace pour les couches 
    cudaMalloc(&gpu_L1,sizeof(Layer));
    cudaMalloc(&gpu_L2,sizeof(Layer));
    
    //! Première couche
    //*Optionnel 
    cudaMemcpy(&gpu_L1->nb_units,&L1.nb_units,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L1->input_size,&L1.input_size,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L1->type[0],L1.type,sizeof(char)*256,cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L1->name[0],L1.name,sizeof(char)*256,cudaMemcpyHostToDevice);
    
    //*Copie des paramètres vers le GPU
    double ** tmp_W1_ptr;
    gpu_W1=(double**)malloc(sizeof(double *)*L1.input_size);
    cudaMalloc(&tmp_W1_ptr,sizeof(double *)*L1.input_size);
    cudaMalloc(&gpu_b1,sizeof(double)*L1.nb_units);
    for(int i=0;i<L1.input_size;i++){
        cudaMalloc(&gpu_W1[i],sizeof(double)*L1.nb_units);
        cudaMemcpy(gpu_W1[i],L1.W[i],sizeof(double)*L1.nb_units,cudaMemcpyHostToDevice);
    }
    cudaMemcpy(tmp_W1_ptr,gpu_W1,sizeof(double *)*L1.input_size,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b1,L1.b,sizeof(double)*L1.nb_units,cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L1->W,&tmp_W1_ptr,sizeof(double **),cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L1->b,&gpu_b1,sizeof(double *),cudaMemcpyHostToDevice);
    
    //! Deuxième couche 
    //*Optionnel
    cudaMemcpy(&gpu_L2->nb_units,&L2.nb_units,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L2->input_size,&L2.input_size,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L2->type[0],L2.type,sizeof(char)*256,cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L2->name[0],L2.name,sizeof(char)*256,cudaMemcpyHostToDevice);

    //*Copie des paramètres vers le GPU
    double ** tmp_W2_ptr;
    gpu_W2=(double**)malloc(sizeof(double *)*L2.input_size);
    cudaMalloc(&tmp_W2_ptr,sizeof(double *)*L2.input_size);
    cudaMalloc(&gpu_b2,sizeof(double)*L2.nb_units);
    for(int i=0;i<L2.input_size;i++){
        cudaMalloc(&gpu_W2[i],sizeof(double)*L2.nb_units);
        cudaMemcpy(gpu_W2[i],L2.W[i],sizeof(double)*L2.nb_units,cudaMemcpyHostToDevice);
    }
    cudaMemcpy(tmp_W2_ptr,gpu_W2,sizeof(double *)*L2.input_size,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b2,L2.b,sizeof(double)*L2.nb_units,cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L2->W,&tmp_W2_ptr,sizeof(double **),cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_L2->b,&gpu_b2,sizeof(double *),cudaMemcpyHostToDevice);
    

    //? Allocation espace pour les dérivées sur le GPU
    double*** tmp_grad_W1;
    double** tmp_grad_b1;
    double*** tmp_grad_W2;
    double** tmp_grad_b2;

    //! W1 
    tmp_grad_W1 = (double ***)malloc(sizeof(double **)*NB_BLOC);
    cudaMalloc(&gpu_grad_W1,sizeof(double **)*NB_BLOC);
    for(int k=0;k<NB_BLOC;k++){
        cudaMalloc(&tmp_grad_W1[k],sizeof(double *)*L1.input_size);
        double** tmp_ptr_row =(double **) malloc(sizeof(double*)*L1.input_size) ; //goes to tmp_grad_W1
        for(int i=0;i<L1.input_size;i++){
            cudaMalloc(&tmp_ptr_row[i],sizeof(double)*L1.nb_units);
        }
        cudaMemcpy(tmp_grad_W1[k],tmp_ptr_row,sizeof(double *)*L1.input_size,cudaMemcpyHostToDevice);
        free(tmp_ptr_row);
    }
    cudaMemcpy(gpu_grad_W1,tmp_grad_W1,sizeof(double **)*NB_BLOC,cudaMemcpyHostToDevice);
    free(tmp_grad_W1);
    
    //! W2
    tmp_grad_W2 = (double ***)malloc(sizeof(double **)*NB_BLOC);
    cudaMalloc(&gpu_grad_W2,sizeof(double **)*NB_BLOC);
    for(int k=0;k<NB_BLOC;k++){
        cudaMalloc(&tmp_grad_W2[k],sizeof(double *)*L2.input_size);
        double** tmp_ptr_row =(double **) malloc(sizeof(double*)*L2.input_size) ; //goes to tmp_grad_W1
        for(int i=0;i<L2.input_size;i++){
            cudaMalloc(&tmp_ptr_row[i],sizeof(double)*L2.nb_units);
        }
        cudaMemcpy(tmp_grad_W2[k],tmp_ptr_row,sizeof(double *)*L2.input_size,cudaMemcpyHostToDevice);
        free(tmp_ptr_row);
    }
    cudaMemcpy(gpu_grad_W2,tmp_grad_W2,sizeof(double **)*NB_BLOC,cudaMemcpyHostToDevice);
    free(tmp_grad_W2);

    //! b1 
    tmp_grad_b1 = (double **)malloc(sizeof(double *)*NB_BLOC);
    cudaMalloc(&gpu_grad_b1,sizeof(double *)*NB_BLOC);
    for(int k=0;k<NB_BLOC;k++){
        cudaMalloc(&tmp_grad_b1[k],sizeof(double )*L1.nb_units);
    }
    cudaMemcpy(gpu_grad_b1,tmp_grad_b1,sizeof(double *)*NB_BLOC,cudaMemcpyHostToDevice);
    free(tmp_grad_b1);

    tmp_grad_b2 = (double **)malloc(sizeof(double *)*NB_BLOC);    
    //! b2 
    cudaMalloc(&gpu_grad_b2,sizeof(double *)*NB_BLOC);
    for(int k=0;k<NB_BLOC;k++){
        cudaMalloc(&tmp_grad_b2[k],sizeof(double )*L2.nb_units);
    }
    cudaMemcpy(gpu_grad_b2,tmp_grad_b2,sizeof(double *)*NB_BLOC,cudaMemcpyHostToDevice);
    free(tmp_grad_b2);

    printf("Espace requis alloué sur le GPU et le CPU \n");


    if(DISPLAY==1){
        printf("Paramètres Initiaux : \n");
        print_Layer(L1);
        print_Layer(L2);
    }

    //? Apprentissage du modèle 
    train_model_GPU(NB_ITER,X,y,ROW,LEARN_RATE,BATCH_SIZE);
    end = clock()-start;
    double time_taken = ((double)end)/CLOCKS_PER_SEC; // in seconds
    printf("Temps d'exécution de tout le programme (Chargement de données + Initialisation + Apprentissage) :  %f (s) \n", time_taken);  
    
    if(DISPLAY==1){
        printf("Paramètres finaux : \n");
        print_Layer(L1);
        print_Layer(L2);
    }

    printf("Libération de l'espace alloué \n");
    //? Libération de l'espace 
    for (int i = 0;i<ROW;i++){
        free(X[i]);
    }

    //? Freeing space on the GPU
    for(int i=0;i<ROW;i++){
        cudaFree(tmp_gpu_X[i]);
    }
    cudaFree(gpu_X);  cudaFree(gpu_y);
    cudaFree(gpu_L1); cudaFree(gpu_L2);
    cudaFree(gpu_W1);    cudaFree(gpu_W2);
    cudaFree(gpu_b1);    cudaFree(gpu_b2);
    cudaFree(gpu_grad_W1);    cudaFree(gpu_grad_W2);
    cudaFree(gpu_grad_b1);    cudaFree(gpu_grad_b2);
}