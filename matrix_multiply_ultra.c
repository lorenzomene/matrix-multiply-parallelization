#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define NUMERO 4000
#define BLOCK_SIZE 64

int *matriz1;    
int *matriz2_transposta;
int *resultado;

void gerar_matrizes(int local_rows, int rank) {
    srand(time(NULL) + rank);
    #pragma omp parallel for
    for (int i = 0; i < local_rows * NUMERO; i++) {
        matriz1[i] = rand() % 2;
    }
}

void transpor_matriz() {
    int *matriz2 = (int *)malloc(NUMERO * NUMERO * sizeof(int));
    srand(time(NULL));
    #pragma omp parallel for
    for (int i = 0; i < NUMERO * NUMERO; i++) {
        matriz2[i] = rand() % 2;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NUMERO; i++) {
        for (int j = 0; j < NUMERO; j++) {
            matriz2_transposta[j * NUMERO + i] = matriz2[i * NUMERO + j];
        }
    }
    free(matriz2);
}

void print_rank_ip(int rank){
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    struct addrinfo hints, *info;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;

    getaddrinfo(hostname, NULL, &hints, &info);

    char ip_str[INET_ADDRSTRLEN];
    struct sockaddr_in *addr = (struct sockaddr_in *)info->ai_addr;
    inet_ntop(AF_INET, &(addr->sin_addr), ip_str, INET_ADDRSTRLEN);

    printf("Rank %d is %s\n", rank, ip_str);
}

void multiplicar_matrizes(int local_rows, int linha_inicio) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int ii = 0; ii < local_rows; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < NUMERO; jj += BLOCK_SIZE) {
            int i_max = (ii + BLOCK_SIZE < local_rows) ? ii + BLOCK_SIZE : local_rows;
            int j_max = (jj + BLOCK_SIZE < NUMERO) ? jj + BLOCK_SIZE : NUMERO;

            for (int i = ii; i < i_max; i++) {
                for (int j = jj; j < j_max; j++) {
                    int soma = 0;
                    int *row1 = &matriz1[i * NUMERO];
                    int *row2 = &matriz2_transposta[j * NUMERO];

                    for (int k = 0; k < NUMERO; k += 4) {
                        soma += row1[k] * row2[k];
                        soma += row1[k+1] * row2[k+1];
                        soma += row1[k+2] * row2[k+2];
                        soma += row1[k+3] * row2[k+3];
                    }

                    resultado[i * NUMERO + j] = soma;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double inicio, fim;
    double tempo_geracao, tempo_transposicao, tempo_broadcast, tempo_multiplicacao, tempo_gather;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int linhas_por_processo = NUMERO / size;
    int resto = NUMERO % size;
    int local_rows = linhas_por_processo + (rank < resto ? 1 : 0);
    int linha_inicio = rank * linhas_por_processo + (rank < resto ? rank : resto);

    matriz1 = (int *)malloc(local_rows * NUMERO * sizeof(int));
    matriz2_transposta = (int *)malloc(NUMERO * NUMERO * sizeof(int));
    resultado = (int *)calloc(local_rows * NUMERO, sizeof(int));

    print_rank_ip(rank);

    if (rank == 0) {
        inicio = MPI_Wtime();
        transpor_matriz();
        fim = MPI_Wtime();
        tempo_transposicao = fim - inicio;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        printf("Etapa 1: Broadcast da matriz2_transposta\n");
    }

    inicio = MPI_Wtime();
    MPI_Bcast(matriz2_transposta, NUMERO * NUMERO, MPI_INT, 0, MPI_COMM_WORLD);
    fim = MPI_Wtime();
    tempo_broadcast = fim - inicio;

    inicio = MPI_Wtime();
    gerar_matrizes(local_rows, rank);
    fim = MPI_Wtime();
    tempo_geracao = fim - inicio;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Etapa 2: Multiplicando matriz\n");
    }

    inicio = MPI_Wtime();
    multiplicar_matrizes(local_rows, linha_inicio);
    MPI_Barrier(MPI_COMM_WORLD);
    fim = MPI_Wtime();
    tempo_multiplicacao = fim - inicio;

    // gather results
    int *recvcounts = NULL;
    int *displs = NULL;
    int *full_result = NULL;
    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows_i = linhas_por_processo + (i < resto ? 1 : 0);
            recvcounts[i] = rows_i * NUMERO;
            displs[i] = offset;
            offset += recvcounts[i];
        }

        full_result = (int *)malloc(NUMERO * NUMERO * sizeof(int));
    }

    inicio = MPI_Wtime();
    if (rank == 0) {
        MPI_Gatherv(resultado, local_rows * NUMERO, MPI_INT,
                    full_result, recvcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(resultado, local_rows * NUMERO, MPI_INT,
                    NULL, NULL, NULL, MPI_INT,
                    0, MPI_COMM_WORLD);
    }
    fim = MPI_Wtime();
    tempo_gather = fim - inicio;

    if(rank == 0){
        printf("Geração de matrizes:                               %.6f\n", tempo_geracao);
        printf("Transposição da matriz:                            %.6f\n", tempo_transposicao);
        printf("Broadcast MPI:                                     %.6f\n", tempo_broadcast);
        printf("Multiplicação paralela:                            %.6f\n", tempo_multiplicacao);
        printf("Coleta de resultados:                              %.6f\n", tempo_gather);
        printf("Tempo total execução:                              %.6f\n", 
               tempo_geracao + tempo_transposicao + tempo_broadcast + tempo_multiplicacao + tempo_gather);
        printf("Tempo Valido(broadcast+multiplicação+gather):      %.6f\n",
                tempo_broadcast + tempo_multiplicacao + tempo_gather); 
    }

    free(matriz1);
    free(matriz2_transposta);
    free(resultado);
    if(rank == 0){
        free(recvcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
