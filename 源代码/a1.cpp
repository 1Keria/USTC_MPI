#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm odd_comm, even_comm; // 创建两个新的通信器，分别用于奇数和偶数进程

    // 初始化 MPI 环境
    MPI_Init(&argc, &argv);

    // 获取当前进程的 rank 和总进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 根据 rank 的奇偶性分组
    if (rank % 2 == 0) {
        // 偶数进程
        MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &even_comm);
    } else {
        // 奇数进程
        MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &odd_comm);
    }

    // 获取每个组的进程数
    int num_procs_in_comm;
    if (rank % 2 == 0) {
        MPI_Comm_size(even_comm, &num_procs_in_comm);
    } else {
        MPI_Comm_size(odd_comm, &num_procs_in_comm);
    }

    // 打印每个进程所在的组及其信息
    if (rank % 2 == 0) {
        printf("Even Process %d out of %d is in even group (Total procs in even group: %d)\n", rank, size, num_procs_in_comm);
    } else {
        printf("Odd Process %d out of %d is in odd group (Total procs in odd group: %d)\n", rank, size, num_procs_in_comm);
    }

    // 清理通信器
    if (rank % 2 == 0) {
        MPI_Comm_free(&even_comm);
    } else {
        MPI_Comm_free(&odd_comm);
    }

    // 结束 MPI 环境
    MPI_Finalize();

    return 0;
}