#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_SERVERS 2  // Number of parameter server processes
#define NUM_WORKERS 4  // Number of worker processes
#define TOTAL_PROCESSES (NUM_SERVERS + NUM_WORKERS) // Total number of processes

void worker_process(int rank, int num_servers);
void parameter_server_process(int rank, int num_servers);

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != TOTAL_PROCESSES) {
        if (rank == 0)
            printf("Error: The total number of processes must be %d (NUM_SERVERS + NUM_WORKERS).\n", TOTAL_PROCESSES);
        MPI_Finalize();
        return -1;
    }

    if (rank < NUM_WORKERS) {
        // Worker process
        worker_process(rank, NUM_SERVERS);
    }
    else {
        // Parameter server process
        parameter_server_process(rank, NUM_SERVERS);
    }

    MPI_Finalize();
    return 0;
}

void worker_process(int rank, int num_servers) {
    int value, updated_value;
    int server_rank = (rank % num_servers) + NUM_WORKERS;  // Map to corresponding parameter server process

    srand(time(NULL) + rank);  // Initialize random seed

    // Generate a random number and send it to the corresponding server
    value = rand() % 100;
    printf("Worker %d: Sending value %d to parameter server %d\n", rank, value, server_rank);
    MPI_Send(&value, 1, MPI_INT, server_rank, 0, MPI_COMM_WORLD);

    // Receive the updated value from the parameter server
    MPI_Recv(&updated_value, 1, MPI_INT, server_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Worker %d: Received updated value %d\n", rank, updated_value);
}

void parameter_server_process(int rank, int num_servers) {
    int num_workers = NUM_WORKERS / num_servers;  // Number of workers assigned to this parameter server
    int* values = (int*)malloc(num_workers * sizeof(int));
    int total_value = 0, average_value = 0;

    // Receive data from corresponding worker processes
    for (int i = 0; i < num_workers; i++) {
        int worker_rank = i * num_servers + (rank - NUM_WORKERS);
        MPI_Recv(&values[i], 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Parameter server %d: Received value %d from worker %d\n", rank, values[i], worker_rank);
        total_value += values[i];
    }

    // Compute the average value
    average_value = total_value / num_workers;
    printf("Parameter server %d: Computed average value %d\n", rank, average_value);

    // Send the average value back to corresponding worker processes
    for (int i = 0; i < num_workers; i++) {
        int worker_rank = i * num_servers + (rank - NUM_WORKERS);
        MPI_Send(&average_value, 1, MPI_INT, worker_rank, 1, MPI_COMM_WORLD);
        printf("Parameter server %d: Sent average value %d to worker %d\n", rank, average_value, worker_rank);
    }

    free(values);
}
