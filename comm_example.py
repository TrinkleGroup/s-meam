import numpy as np
import itertools
from mpi4py import MPI#, MPI_COMM_NULL

def main():

    # Define settings
    procs_per_node = 2
    nodes_per_manager = 2

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    world_rank = world_comm.Get_rank()

    rank_list = np.arange(world_size)

    # Create manager communicator
    world_group = world_comm.Get_group()

    manager_group = world_group.Incl(
            rank_list[::procs_per_node*nodes_per_manager]
    )

    manager_comm = world_comm.Create(manager_group)

    if manager_comm != MPI.COMM_NULL:
        manager_rank = manager_comm.Get_rank()
        print("Manager", manager_rank, "reporting for duty!", flush=True)

    manager_color = world_rank // (procs_per_node * nodes_per_manager)

    # Create node head communicator
    start = manager_color * procs_per_node * nodes_per_manager
    stop = start + (procs_per_node * nodes_per_manager)

    head_ranks = rank_list[start:stop:procs_per_node]
    head_group = world_group.Incl(head_ranks)
    head_comm = world_comm.Create(head_group)

    if manager_comm != MPI.COMM_NULL:
        print(
                "Manager", manager_rank, "has these node heads", head_ranks,
                flush=True
        )

    # Create node communicator (connecting processes on one node)
    node_color = world_rank // procs_per_node
    node_comm = world_comm.Split(node_color, world_rank)

    node_rank = node_comm.Get_rank()

    procs_on_node = node_comm.gather(world_rank, root=0)

    if head_comm != MPI.COMM_NULL:
        procs_on_manager = head_comm.gather(procs_on_node, root=0)

    if manager_comm != MPI.COMM_NULL:
        procs_on_manager = list(itertools.chain.from_iterable(procs_on_manager))
        print("Manager", manager_color, "has these processes:",
                procs_on_manager, flush=True)


if __name__ == "__main__":
    main()
