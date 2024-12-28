#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int sendbuf, *recvbuf;

    // 初始化 MPI
    MPI_Init(&argc, &argv);

    // 获取当前进程的秩和进程总数
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 全局程序计时开始
    double program_start_time, program_end_time;
    program_start_time = MPI_Wtime();

    // 动态分配接收缓冲区
    recvbuf = (int*)malloc(size * sizeof(int));

    // 初始化发送缓冲区为当前进程的秩
    sendbuf = rank;

    // 初始化接收缓冲区
    for (int i = 0; i < size; i++) {
        recvbuf[i] = -1;  // 初始化为一个标记值
    }

    // 模拟 MPI_Alltoall，设置自通信数据
    recvbuf[rank] = sendbuf;

    // 使用非阻塞通信避免死锁
    MPI_Request* requests = (MPI_Request*)malloc(2 * size * sizeof(MPI_Request));
    int request_count = 0;

    // 发送数据
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            MPI_Isend(&sendbuf, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[request_count++]);
        }
    }

    // 接收数据
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            MPI_Irecv(&recvbuf[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[request_count++]);
        }
    }

    // 等待所有通信完成
    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

    // 打印每个进程接收到的数据
    printf("Rank %d received: ", rank);
    for (int i = 0; i < size; i++) {
        printf("%d ", recvbuf[i]);
    }
    printf("\n");

    // 释放动态分配的内存
    free(recvbuf);
    free(requests);

    // 全局程序计时结束
    program_end_time = MPI_Wtime();

    // 打印整个程序的运行时间
    if (rank == 0) {
        printf("Total program execution time: %f seconds\n", program_end_time - program_start_time);
    }

    // 结束 MPI
    MPI_Finalize();
    return 0;
}
