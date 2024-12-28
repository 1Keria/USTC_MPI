#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int *send_buf, *recv_buf; // 动态分配发送和接收缓冲区

    // 初始化 MPI 环境
    MPI_Init(&argc, &argv);

    // 获取当前进程的 rank 和总进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 动态分配发送和接收缓冲区
    send_buf = (int*)malloc(size * sizeof(int)); // 每个进程发送 size 个数据
    recv_buf = (int*)malloc(size * sizeof(int)); // 每个进程接收 size 个数据

    // 初始化发送缓冲区
    for (int i = 0; i < size; i++) {
        send_buf[i] = rank * 10 + i; // 示例数据：rank 值乘以 10 加索引
    }

    // 全局程序计时器
    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    // 使用 MPI_Alltoall 进行全对全数据交换
    MPI_Alltoall(send_buf, 1, MPI_INT, recv_buf, 1, MPI_INT, MPI_COMM_WORLD);

    // 结束计时
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Total program execution time: %f seconds\n", end_time - start_time);
    }

    // 每个进程打印接收到的数据
    printf("Process %d received data: ", rank);
    for (int i = 0; i < size; i++) {
        printf("%d ", recv_buf[i]);
    }
    printf("\n");

    // 释放动态分配的内存
    free(send_buf);
    free(recv_buf);

    // 结束 MPI 环境
    MPI_Finalize();

    return 0;
}
