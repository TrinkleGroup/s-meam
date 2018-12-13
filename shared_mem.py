import time
import numpy as np
from itertools import product
from multiprocessing import Pool
from multiprocessing.managers import BaseManager
from src.worker import Worker

class WorkerManager(BaseManager):
    pass

class WorkerWrap(Worker):
    def __init__(self):
        self.w = Worker.from_pickle("/home/jvita/scripts/s-meam/data/fitting_databases/pinchao/structures/crowd_rnd15.Ti.pkl")

    def compute_energy(self, x, u_ranges):
        return self.w.compute_energy(x, u_ranges)

    @profile
    def compute_e_grad(self, x, u_range):
        return self.w.energy_gradient_wrt_pvec(x, u_range)

# WorkerManager.register(
#     'Worker', Worker,
#     exposed=['from_pickle', 'compute_energy']
# )

WorkerManager.register(
    'WorkerWrap', WorkerWrap,
    exposed=['compute_energy', 'compute_e_grad']
)

def main():
    print("Starting manager ...", flush=True)
    manager = WorkerManager()
    manager.start()

    print("Initializing worker ...", flush=True)
    worker = manager.WorkerWrap()

    num_procs = 1
    pool = Pool(num_procs)

    print("Evaluating population ...", flush=True)
    np.random.seed(42)

    start = time.time()
    # for i in range(20):

    pop = np.random.random((50000, 116))
    split_pop = np.array_split(pop, num_procs)

    u_ranges = [[(-1, 1), (-1, 1)]] * num_procs

    pool.starmap(
        worker.compute_e_grad, zip(split_pop, u_ranges),
    )

    time.sleep(10)

    pool.starmap(
        worker.compute_e_grad, zip(split_pop, u_ranges),
    )

    # print(np.vstack(energies))
    print("Runtime with", num_procs, "processors:", time.time() - start)

if __name__ == "__main__":
    main()
