import time
import numpy as np
from mpi4py import MPI
from src.worker import Worker


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    worker = Worker.from_pickle("/home/jvita/scripts/s-meam/data/fitting_databases/pinchao/structures/crowd_rnd15.Ti.pkl")

    np.random.seed(42)


    if rank == 0:
        pop = np.random.random((50000, 116))
        split_pop = np.array_split(pop, size)
    else: split_pop = None

    pop = comm.scatter(split_pop, root=0)

    u_ranges = [(-1, 1), (-1, 1)]

    if rank == 0:
        start = time.time()

    grad = worker.energy_gradient_wrt_pvec(pop, u_ranges)

    all_grad = comm.gather(grad, root=0)
    if rank == 0:
        print("Runtime with", size, "processors:", time.time() - start)

    time.sleep(10)

if __name__ == "__main__":
    main()
