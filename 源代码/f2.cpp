#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define N 8
#define IDX(i, j) (((i)*N) + (j))

// Function to generate random values for matrix
void gen_rand_array(double* a, int num)
{
    for (int i = 0; i < num; i++)
    {
        srand(time(NULL) + i); // Ensure different seeds for randomness
        a[i] = rand() % 10 + 1; // Random values between 1 and 10
    }
}

// Function to compute the average for internal elements
void compute(double* A, double* B, int a, int b)
{
    for (int i = 1; i <= a; i++)
    {
        for (int j = 1; j <= b; j++)
        {
            B[IDX(i, j)] = (A[IDX(i - 1, j)] + A[IDX(i, j + 1)] + A[IDX(i + 1, j)] + A[IDX(i, j - 1)]) / 4.0;
        }
    }
}

// Function to check the result matrix
int check_ans(double* B, double* A)
{
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            if (fabs(B[IDX(i, j)] - A[IDX(i, j)]) >= 1e-2)
            {
                return 0;
            }
        }
    }
    return 1;
}

// Function to print a matrix
void print_matrix(const char* name, double* mat, int rows, int cols)
{
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f ", mat[IDX(i, j)]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int id_procs, num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_procs);
    MPI_Status status;
    MPI_Datatype SubMat;

    // Timers
    double start_time, end_time;
    double init_start, init_end;
    double comm_start, comm_end;
    double compute_start, compute_end;
    double gather_start, gather_end;

    if (id_procs == 0)
        start_time = MPI_Wtime();

    // Initialization phase
    init_start = MPI_Wtime();
    int rows = sqrt(num_procs);
    int cols = num_procs / rows;
    int a = (N - 2 + rows - 1) / rows;
    int b = (N - 2 + cols - 1) / cols;
    int alloc_num = (a + 1) * (b + 1) * num_procs;
    double A[alloc_num];
    double B[alloc_num];
    double B2[alloc_num];

    if (id_procs == 0)
    {
        gen_rand_array(A, N * N);
        compute(A, B2, N - 2, N - 2);
        print_matrix("Initial Matrix A", A, N, N); // Output matrix A
    }
    init_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    // Communication phase (Broadcast)
    comm_start = MPI_Wtime();
    MPI_Type_vector(a + 2, b + 2, N, MPI_DOUBLE, &SubMat);
    MPI_Type_commit(&SubMat);

    if (id_procs == 0)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == 0 && j == 0)
                    continue;
                MPI_Send(A + i * a * N + b * j, 1, SubMat, j + cols * i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        MPI_Recv(A, 1, SubMat, 0, 0, MPI_COMM_WORLD, &status);
    }
    comm_end = MPI_Wtime();

    // Computation phase
    compute_start = MPI_Wtime();
    compute(A, B, a, b);
    print_matrix("Matrix B After Computation", B, a + 2, b + 2); // Output local matrix B
    compute_end = MPI_Wtime();

    // Gather result
    gather_start = MPI_Wtime();
    MPI_Datatype SubMat_B;
    MPI_Type_vector(a, b, N, MPI_DOUBLE, &SubMat_B);
    MPI_Type_commit(&SubMat_B);
    if (id_procs == 0)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == 0 && j == 0)
                    continue;
                MPI_Recv(&B[IDX(a * i + 1, b * j + 1)], 1, SubMat_B, i * cols + j, 1, MPI_COMM_WORLD, &status);
            }
        }
    }
    else
    {
        MPI_Send(&B[IDX(1, 1)], 1, SubMat_B, 0, 1, MPI_COMM_WORLD);
    }
    gather_end = MPI_Wtime();

    // Root process checks the result
    if (id_procs == 0)
    {
        print_matrix("Final Result Matrix B", B, N, N); // Output final matrix B
        if (check_ans(B, B2))
        {
            printf("Done. No Error\n");
        }
        else
        {
            printf("Error!\n");
        }
    }

    if (id_procs == 0)
    {
        end_time = MPI_Wtime();
        printf("\nTiming Results:\n");
        printf("Initialization time: %f seconds\n", init_end - init_start);
        printf("Communication time: %f seconds\n", comm_end - comm_start);
        printf("Computation time: %f seconds\n", compute_end - compute_start);
        printf("Gathering time: %f seconds\n", gather_end - gather_start);
        printf("Total execution time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
