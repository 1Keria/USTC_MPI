#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 500
#define IDX(i, j) (((i)*N) + (j))

void compute(double *A, double *B, int num)
{
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            B[IDX(i, j)] = (A[IDX(i - 1, j)] + A[IDX(i, j + 1)] + A[IDX(i + 1, j)] + A[IDX(i, j - 1)]) / 4.0;
        }
    }
}

void gen_rand_array(double *a, int num)
{
    srand(time(NULL));
    for (int i = 0; i < num; i++)
    {
        a[i] = rand() % 100;
    }
}

int check_ans(double *B, double *C)
{
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            if (fabs(B[IDX(i, j)] - C[IDX(i, j)]) >= 1e-4)
            {
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[])
{
    int id_procs, num_procs, num_1;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_procs);

    double *A, *B, *B2;
    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    B2 = (double *)malloc(N * N * sizeof(double));
    num_1 = num_procs - 1;

    // 全局计时器
    double start_time, end_time, compute_start_time, compute_end_time;
    double send_time_start, send_time_end, gather_time_start, gather_time_end;

    if (id_procs == 0)
    {
        start_time = MPI_Wtime();
    }

    // Proc#N-1 randomize the data
    if (id_procs == num_1)
    {
        gen_rand_array(A, N * N);
        compute(A, B2, N * N);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 数据发送计时
    send_time_start = MPI_Wtime();
    int ctn = 0;
    for (int i = 0; i < N - 2; i++)
    {
        if (id_procs == num_1)
        {
            int dest = i % num_1;
            int tag = i / num_1;
            MPI_Send(&A[IDX(i, 0)], N * 3, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < (N - 2) / num_1; i++)
    {
        if (id_procs != num_1)
        {
            MPI_Recv(&A[IDX(3 * ctn, 0)], 3 * N, MPI_DOUBLE, num_1, ctn, MPI_COMM_WORLD, &status);
            ctn++;
        }
    }

    if (id_procs < (N - 2) % num_1)
    {
        MPI_Recv(&A[IDX(ctn * 3, 0)], 3 * N, MPI_DOUBLE, num_1, ctn, MPI_COMM_WORLD, &status);
        ctn++;
    }
    send_time_end = MPI_Wtime();

    // 计算计时
    compute_start_time = MPI_Wtime();
    if (id_procs != num_1)
    {
        for (int i = 1; i <= 3 * ctn - 2; i += 3)
        {
            for (int j = 1; j < N - 1; j++)
            {
                B[IDX((i + 2) / 3, j)] = (A[IDX(i - 1, j)] + A[IDX(i, j + 1)] + A[IDX(i + 1, j)] + A[IDX(i, j - 1)]) / 4.0;
            }
        }
    }
    compute_end_time = MPI_Wtime();

    // Gather计时
    gather_time_start = MPI_Wtime();
    for (int i = 0; i < N - 2; i++)
    {
        if (id_procs == num_1)
        {
            int src = i % num_1;
            MPI_Recv(&B[IDX(i + 1, 1)], N - 2, MPI_DOUBLE, src, i / num_1 + N, MPI_COMM_WORLD, &status);
        }
        else
        {
            for (int j = 0; j < ctn; j++)
            {
                MPI_Send(&B[IDX(j + 1, 1)], N - 2, MPI_DOUBLE, num_1, j + N, MPI_COMM_WORLD);
            }
        }
    }
    gather_time_end = MPI_Wtime();

    // 检查结果
    if (id_procs == num_1)
    {
        if (check_ans(B, B2))
        {
            printf("Done. No Error\n");
        }
        else
        {
            printf("Error Occurred!\n");
        }
    }

    if (id_procs == 0)
    {
        end_time = MPI_Wtime();
        printf("Total execution time: %f seconds\n", end_time - start_time);
        printf("Data sending time: %f seconds\n", send_time_end - send_time_start);
        printf("Computation time: %f seconds\n", compute_end_time - compute_start_time);
        printf("Data gathering time: %f seconds\n", gather_time_end - gather_time_start);
    }

    free(A);
    free(B);
    free(B2);
    MPI_Finalize();
    return 0;
}
