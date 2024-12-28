#include <iostream>
#include <mpi.h>

void group_wise_allreduce(int* data, int rank, int size, MPI_Comm comm) {
    int total_data = *data;

    // 轮次进行数据交换（模拟二进制树式相加）
    for (int step = 1; step < size; step *= 2) {
        int send_to = (rank + step) % size;
        int recv_from = (rank - step + size) % size;

        // 进程向其他进程发送数据，并接收数据
        int recv_data;
        MPI_Sendrecv(&total_data, 1, MPI_INT, send_to, 0, &recv_data, 1, MPI_INT, recv_from, 0, comm, MPI_STATUS_IGNORE);

        // 将接收到的数据与自己的数据相加
        total_data += recv_data;
    }

    // 最终的结果会在总和中，输出根进程结果
    if (rank == 0) {
        std::cout << "Final result from process 0: " << total_data << std::endl;
    }

    // 这里可以使用MPI_Gather将每个进程的最终结果发送到根进程
    MPI_Gather(&total_data, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);  // 初始化 MPI 环境
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 获取当前进程的 rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 获取总的进程数

    // 假设每个进程有一个数据值
    int data = rank ;  // 例如，rank 0 -> 0，rank 1 -> 10，rank 2 -> 20，rank 3 -> 30

    // 输出每个进程初始的数据
    std::cout << "Process " << rank << " sending data: " << data << std::endl;

    // 执行全和操作
    group_wise_allreduce(&data, rank, size, MPI_COMM_WORLD);

    MPI_Finalize();  // 结束 MPI 环境
    return 0;
}
