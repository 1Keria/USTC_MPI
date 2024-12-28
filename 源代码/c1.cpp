#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void butterfly_allreduce(int* data, int size, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);       // 获取当前进程的秩
    MPI_Comm_size(comm, &nprocs);     // 获取进程总数

    int partner;
    int step = 1;

    // 蝶式算法：每一步的通信步长翻倍
    while (step < nprocs) {
        // 计算 partner 的秩
        partner = rank ^ step;  // XOR 运算确定通信伙伴

        if (partner < nprocs) {
            int recv_data;
            // 发送和接收操作
            if (rank < partner) {
                MPI_Send(data, 1, MPI_INT, partner, 0, comm);
                MPI_Recv(&recv_data, 1, MPI_INT, partner, 0, comm, MPI_STATUS_IGNORE);
            }
            else {
                MPI_Recv(&recv_data, 1, MPI_INT, partner, 0, comm, MPI_STATUS_IGNORE);
                MPI_Send(data, 1, MPI_INT, partner, 0, comm);
            }

            // 执行归约操作（这里以加法为例）
            *data += recv_data;
        }

        step *= 2;  // 增加步长
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // 每个进程初始化一个数据值（可以根据需求更改）
    int data = rank ;

    // 调用蝶式全和函数
    butterfly_allreduce(&data, 1, MPI_COMM_WORLD);

    // 输出最终结果（每个进程上的数据应该是全体进程数据的和）
    printf("Rank %d, Final result: %d\n", rank, data);

    MPI_Finalize();
    return 0;
}
