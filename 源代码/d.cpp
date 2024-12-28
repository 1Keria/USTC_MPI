#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const int N = 16; // 矩阵A, B的维度

void Print_Mat(int* A, int n) { // 打印矩阵
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4d ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int int_sqrt(int p) { // 计算p的根号，返回整数结果
    for (int i = 1; i <= p; i++) {
        if (i * i == p) {
            return i;
        }
    }
    return -1;
}

void Get_Block(int* A, int* a, int i, int j, int n) { // 获取A的第i, j个n*n方块
    for (int k = 0; k < n; k++) {
        for (int l = 0; l < n; l++) {
            a[k * n + l] = A[i * n * N + j * n + k * N + l];
        }
    }
}

void Multi_Mat(int* a, int* b, int* c, int n) { // 矩阵乘法
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

void Copy_Block(int* c, int* C, int i, int j, int n) { // 将方块c赋值给C
    for (int k = 0; k < n; k++) {
        for (int l = 0; l < n; l++) {
            C[i * n * N + j * n + k * N + l] = c[k * n + l];
        }
    }
}

void Fox(int* a, int* b, int* c, int sp, int n, int myrank) { // Fox 算法实现
    int* temp_a = (int*)malloc(n * n * sizeof(int));
    int* temp_b = (int*)malloc(n * n * sizeof(int));
    memset(c, 0, n * n * sizeof(int));

    int j = myrank % sp;
    int i = myrank / sp;
    int senddest, recvdest;

    double start_time = MPI_Wtime(); // 开始计时

    for (int round = 0; round < sp; round++) {
        if (i == (j + round) % sp) {
            for (int k = 0; k < sp; k++) {
                if (k != j) {
                    MPI_Send(a, n * n, MPI_INT, i * sp + k, 1, MPI_COMM_WORLD);
                }
                else {
                    memcpy(temp_a, a, n * n * sizeof(int));
                }
            }
        }
        else {
            recvdest = i * sp + (i - round + sp) % sp;
            MPI_Recv(temp_a, n * n, MPI_INT, recvdest, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        Multi_Mat(temp_a, b, c, n);

        senddest = (i - 1 + sp) % sp;
        recvdest = (i + 1) % sp;
        MPI_Sendrecv(b, n * n, MPI_INT, senddest * sp + j, 0, temp_b, n * n, MPI_INT, recvdest * sp + j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy(b, temp_b, n * n * sizeof(int));
    }

    double end_time = MPI_Wtime(); // 结束计时

    if (myrank == 0) {
        printf("Fox algorithm execution time: %lf seconds\n", end_time - start_time);
    }

    free(temp_a);
    free(temp_b);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int p, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int* A, * B, * C;
    int sp = int_sqrt(p);
    if (sp == -1) {
        if (myrank == 0) printf("Error: Number of processes must be a perfect square!\n");
        MPI_Finalize();
        return 0;
    }

    int n = N / sp;
    int* a = (int *)malloc(n * n * sizeof(int));
    int* b = (int*)malloc(n * n * sizeof(int));
    int* c = (int*)malloc(n * n * sizeof(int));

    if (myrank == 0) {
        A = (int*)malloc(N * N * sizeof(int));
        B = (int*)malloc(N * N * sizeof(int));
        C = (int*)malloc(N * N * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < N * N; i++) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
        printf("Matrix A:\n");
        Print_Mat(A, N);
        printf("Matrix B:\n");
        Print_Mat(B, N);

        for (int i = 0; i < sp; i++) {
            for (int j = 0; j < sp; j++) {
                if (i == 0 && j == 0) {
                    Get_Block(A, a, 0, 0, n);
                    Get_Block(B, b, 0, 0, n);
                }
                else {
                    Get_Block(A, a, i, j, n);
                    Get_Block(B, b, i, j, n);
                    MPI_Send(a, n * n, MPI_INT, i * sp + j, 0, MPI_COMM_WORLD);
                    MPI_Send(b, n * n, MPI_INT, i * sp + j, 1, MPI_COMM_WORLD);
                }
            }
        }
    }
    else {
        MPI_Recv(a, n * n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, n * n, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    Fox(a, b, c, sp, n, myrank);

    if (myrank == 0) {
        Copy_Block(c, C, 0, 0, n);
        for (int i = 0; i < sp; i++) {
            for (int j = 0; j < sp; j++) {
                if (i == 0 && j == 0) continue;
                MPI_Recv(c, n * n, MPI_INT, i * sp + j, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                Copy_Block(c, C, i, j, n);
            }
        }
        printf("Result Matrix C:\n");
        Print_Mat(C, N);
        free(A); free(B); free(C);
    }
    else {
        MPI_Send(c, n * n, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    free(a); free(b); free(c);
    MPI_Finalize();
    return 0;
}