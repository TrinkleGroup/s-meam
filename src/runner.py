import os
import sys
sys.path.append('./')

import glob
import shutil
import pickle
import mpi4py
import logging
import numpy as np
import src.lammpsTools

from mpi4py import MPI

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 

import src.partools as partools
from src.database import Database
from src.nodemanager import NodeManager
from src.cmaes import CMAES
from src.como import COMO_CMAES
from src.potential_templates import Template

np.set_printoptions(linewidth=1000)

import random

seed = 42
seed = np.random.randint(10000)

np.random.seed(seed)
random.seed(seed)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

################################################################################

# TODO: have a script that checks the validity of an input script befor qsub

def main(config_name, template_file_name, procs_per_node_manager,
        procs_per_phys_node=32, names_file=None):
    world_comm = MPI.COMM_WORLD

    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    # read config and template files
    if is_master:
        print("Random seed:", seed)

        parameters = read_config(config_name)
        template = read_template(template_file_name)

    else:
        parameters = None
        template = None

    parameters = world_comm.bcast(parameters, root=0)
    template = world_comm.bcast(template, root=0)

    # convert types of inputs from str
    int_params = [
        'NUM_STRUCTS', 'POP_SIZE', 'NSTEPS', 'LMIN_FREQ',
        'INIT_NSTEPS', 'FINAL_NSTEPS', 'CHECKPOINT_FREQ',
        'ARCHIVE_SIZE', 'PROCS_PER_PHYS_NODE',
        'PROCS_PER_NODE_MANAGER', 'NUM_SOLVERS'
    ]

    float_params = [
        'MOVE_PROB', 'MOVE_SCALE', 'CMAES_STEP_SIZE', 'NI_PENALTY',
        'ENERGY_WEIGHT', 'FORCES_WEIGHT', 'STRESS_WEIGHT', 'HUBER_THRESHOLD'
    ]

    bool_params = [
        'DO_LMIN', 'DEBUG', 'OVERWRITE_OLD_FILES', 'PENALTY_ON'
    ]

    for key, val in parameters.items():
        if key in int_params:
            parameters[key] = int(val)
        elif key in float_params:
            parameters[key] = float(val)
        elif key in bool_params:
            parameters[key] = (val == 'True')

    # every PROCS_PER_NODE-th rank is a node "master"; note that a node master
    # may be in charge of multiple node heads (e.g. when multiple nodes are in
    # charge of the same collection of structures

    parameters['PROCS_PER_NODE_MANAGER'] = int(procs_per_node_manager)  # command-line arg
    parameters['PROCS_PER_PHYS_NODE'] = int(procs_per_phys_node)  # command-line arg

    if parameters['DO_GROW']:
        parameters['MAX_POP_SIZE'] = max(
            parameters['POP_SIZE'], max(parameters['GROW_SIZE'])
        )
    else:
        parameters['MAX_POP_SIZE'] = parameters['POP_SIZE']

    manager_ranks = np.arange(0, world_size, parameters['PROCS_PER_NODE_MANAGER'])

    world_group = world_comm.Get_group()

    # manager_comm connects all manager processes
    manager_group = world_group.Incl(manager_ranks)
    manager_comm = world_comm.Create(manager_group)

    # prepare save directories
    if os.path.isdir(parameters['SAVE_DIRECTORY']) and \
            not parameters['OVERWRITE_OLD_FILES']:

        if is_master:
            print("Renaming save directory to avoid overwrite\n")

        parameters['SAVE_DIRECTORY'] = parameters['SAVE_DIRECTORY'] + '-' +\
            str(np.random.randint(100000))

    parameters['NI_TRACE_FILE_NAME'] = os.path.join(
        parameters['SAVE_DIRECTORY'], 'ni_trace.dat'
    )

    parameters['COST_FILE_NAME'] = os.path.join(
        parameters['SAVE_DIRECTORY'], 'cost_trace.dat'
    )

    parameters['BEST_POT_FILE'] = os.path.join(
        parameters['SAVE_DIRECTORY'], 'best_pot_trace.dat'
    )

    parameters['BEST_FIT_FILE'] = os.path.join(
        parameters['SAVE_DIRECTORY'], 'best_fitnesses_trace.dat'
    )

    if is_master:

        print("MASTER: Preparing save directory/files ... ", flush=True)

        prepare_save_directory(parameters)

        f = open(parameters['NI_TRACE_FILE_NAME'], 'ab')
        f.close()

        f = open(parameters['COST_FILE_NAME'], 'ab')
        f.close()

        f = open(parameters['BEST_POT_FILE'], 'ab')
        f.close()

        f = open(parameters['BEST_FIT_FILE'], 'w')
        f.write("EFS weights: {}, {}, {}\n".format(
            parameters['ENERGY_WEIGHT'],
            parameters['FORCES_WEIGHT'],
            parameters['STRESS_WEIGHT']
        ))

        f.close()

        template_path = os.path.join(
            parameters['SAVE_DIRECTORY'], 'template.pkl'
        )

        pickle.dump(template, open(template_path, 'wb'))

        print("Saved template to:", template_path)

        print("Loading database ...", flush=True)

    if world_comm.Get_size() > 1:

        with Database(
            parameters['DATABASE_FILE'], 'r',
            template.pvec_len, template.types,
            knot_xcoords=template.knot_positions, x_indices=template.x_indices,
            cutoffs=template.cutoffs,
            driver='mpio', comm=world_comm
            # driver='mpio', comm=manager_comm
            ) as database:

            if is_master:
                print("Preparing node managers...", flush=True)

            node_manager = prepare_node_managers(
                database, template, parameters, manager_comm, is_master,
                names_file, world_comm
            )
    else:
        with Database(
            parameters['DATABASE_FILE'], 'r',
            template.pvec_len, template.types,
            knot_xcoords=template.knot_positions, x_indices=template.x_indices,
            cutoffs=template.cutoffs,
            ) as database:

            if is_master:
                print("Preparing node managers...", flush=True)

            node_manager = prepare_node_managers(
                database, template, parameters, manager_comm, is_master,
                names_file, world_comm
            )

    if is_master:
        print()

    world_comm.Barrier()

    # run the optimizer
    if parameters.get('DEBUG', False):
        if is_master:
            print("Running debug script:", parameters['DEBUG_FILE'], flush=True)
            print()

        # debug_module = __import__(parameters['DEBUG_FILE'])
        split_fname = parameters['DEBUG_FILE'].split('.')

        debug_module = __import__(
            '.'.join(split_fname), fromlist=[split_fname[-1]]
        )

        debug_module.main(parameters, template)

    elif parameters['OPT_TYPE'] == 'CMAES':
        if is_master:
            print("Running CMAES", flush=True)

        CMAES(parameters, template, node_manager, manager_comm)

    elif parameters['OPT_TYPE'] == 'COMO':
        COMO_CMAES(parameters, template, node_manager, manager_comm)

    else:
        if is_master:
            kill_and_write("Invalid optimization type (OPT_TYPE)")

def read_template(template_file_name):

    # TODO: make template reading more dynamic; ignore blanks and comments

    template_args = {}

    # read potential template
    if os.path.isfile(template_file_name):
        with open(template_file_name, 'r') as f:
            # atom types
            template_args['types'] = f.readline().strip().split(" ")
            template_args['ntypes'] = len(template_args['types'])

            # radial function cutoffs
            v1, v2 = f.readline().strip().split(" ")
            template_args['cutoffs'] = (float(v1), float(v2))

            # u domains
            if template_args['ntypes'] == 1:
                v1, v2 = f.readline().strip().split(" ")
                tmp = np.array([float(v1), float(v2)])
            elif template_args['ntypes'] == 2:
                v1, v2, v3, v4 = f.readline().strip().split(" ")
                tmp = np.array([float(v1), float(v2), float(v3), float(v4)])

            template_args['u_domains'] = np.split(tmp, len(tmp) // 2)

            f.readline()

            # read spline information
            nsplines = template_args['ntypes']*(template_args['ntypes'] + 4)

            knot_positions = []
            spline_ranges = []
            spline_npts = []

            for _ in range(nsplines):
                entries = f.readline().strip().split(" ")
                x_lo, x_hi, y_lo, y_hi, nknots = [np.float(e) for e in
                    entries]

                nknots = int(nknots)
                spline_npts.append(nknots)

                knot_positions.append(np.linspace(x_lo, x_hi, nknots))
                spline_ranges.append((y_lo, y_hi))

            knot_positions = np.concatenate(knot_positions)

            spline_npts = np.array(spline_npts)

            end_indices = np.cumsum([n + 2 for n in spline_npts])

            start_indices = end_indices - spline_npts - 2

            spline_indices = list(zip(start_indices, end_indices))

            scales = np.zeros(end_indices[-1])
            spline_tags = []

            index = 0
            for j, (rng, npts) in enumerate(zip(spline_ranges, spline_npts)):
                scales[index:index + npts] = rng[1] - rng[0]
                spline_tags.append(np.ones(npts + 2)*j)

                index += npts + 2

            f.readline()

            spline_tags = np.concatenate(spline_tags)

            # should a potential be loaded from a file? randomly generated?
            if f.readline().strip().split(" ")[-1] == "True":
                fname = f.readline().strip().split(" ")[-1]
                print(
                    "Loading mask and parameter vector from:",
                    template_file_name
                )

                data = np.genfromtxt(fname)

            else:
                print(
                    "Loading mask and parameter vector from:",
                    template_file_name
                )

                data = np.genfromtxt(
                    template_file_name, skip_header=8+nsplines
                )

            mask = data[:, 0].astype(int)
            knot_values = data[:, 1]

            nphi = template_args['ntypes']*(template_args['ntypes'] + 1) // 2
            nphi = int(nphi)

            # TODO: these indices have been hard-coded for a binary system...
            # TODO: they're also completely un-used

            rho_indices = np.where(
                np.logical_or(
                    spline_tags == nphi,
                    spline_tags == nphi + 1
                )
            )[0]

            g_indices = np.where(
                spline_tags >= nphi + 3*template_args['ntypes']
            )[0]

            f_indices = np.where(
                np.logical_and(
                    spline_tags >= nphi + 2*template_args['ntypes'],
                    spline_tags < nphi + 3*template_args['ntypes']
                )
            )[0]

            print('spline_ranges:', spline_ranges)

            template = Template(
                template_args['types'],
                pvec_len=len(knot_values),
                u_ranges=template_args['u_domains'],
                spline_ranges=spline_ranges,
                spline_indices=spline_indices
            )

            template.biggest_min = max(
                [el[0] for el in template.u_ranges]
            )

            template.biggest_max = max(
                [el[1] for el in template.u_ranges]
            )

            x_indices = np.concatenate([
                [0], np.cumsum(spline_npts)
            ])[:-1]

            # parameter order goes: N knots, 1 LHS deriv, 1 RHS deriv
            template.phi_lhs_deriv_indices = np.array([spline_npts[i]+2*i for i in range(nphi)])

            template.active_mask = mask
            template.pvec = knot_values
            template.scales = scales
            template.spline_tags = spline_tags
            template.rho_indices = rho_indices
            template.g_indices = g_indices
            template.f_indices = f_indices
            template.knot_positions = knot_positions
            template.x_indices = x_indices
            template.cutoffs = template_args['cutoffs']
    else:
        kill_and_write("Config file does not exist")

    return template

def read_config(config_name):
    parameters = {}
    # read parameters from config file
    if os.path.isfile(config_name):
        with open(config_name, 'r') as f:
                i = 0
                for line in f:
                    i += 1
                    stripped = line.strip()

                    if len(stripped) > 0: # ignore empty lines
                        try:
                            if stripped[0] != "#": # ignore comments

                                split = stripped.split(" ")

                                if 'GROW_' in split[0]:
                                    parameters[split[0]] = [int(el) for el in split[1:]]
                                else:
                                    p, v = split
                                    parameters[split[0]] = split[1]

                        except:
                            kill_and_write(
                                "Formatting issue with line "
                                "%d in config file" % i,
                                )
    else:
        kill_and_write("Config file does not exist")

    return parameters

def kill_and_write(msg):
    print(msg, flush=True)
    MPI.COMM_WORLD.Abort(1)

def prepare_managers(is_master, parameters, potential_template, database):
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    if is_master:
        all_struct_names = [s.encode('utf-8').strip().decode('utf-8') for s in
                            list(database.keys())]

        struct_natoms = [database[key].attrs['natoms'] for key in database]
        num_structs = len(all_struct_names)

        old_copy_names = list(all_struct_names)

        worker_ranks = partools.compute_procs_per_subset(
            struct_natoms, world_size
        )
    else:
        potential_template = None
        num_structs = None
        worker_ranks = None
        all_struct_names = None

    potential_template = world_comm.bcast(potential_template, root=0)
    num_structs = world_comm.bcast(num_structs, root=0)

    # each Manager is in charge of a single structure
    world_group = world_comm.Get_group()

    all_rank_lists = world_comm.bcast(worker_ranks, root=0)
    all_struct_names = world_comm.bcast(all_struct_names, root=0)

    # Tell workers which manager they are a part of
    worker_ranks = None
    manager_ranks = []
    for per_manager_ranks in all_rank_lists:
        manager_ranks.append(per_manager_ranks[0])

        if world_rank in per_manager_ranks:
            worker_ranks = per_manager_ranks

    # manager_comm connects all manager processes
    manager_group = world_group.Incl(manager_ranks)
    manager_comm = world_comm.Create(manager_group)

    is_manager = (manager_comm != MPI.COMM_NULL)

    # One manager per structure
    if is_manager:
        manager_rank = manager_comm.Get_rank()

        struct_name = manager_comm.scatter(list(all_struct_names), root=0)

        print(
            "Manager", manager_rank, "received structure", struct_name, "plus",
            len(worker_ranks), "processors for evaluation", flush=True
        )

    else:
        struct_name = None
        manager_rank = None

    if is_master:
        all_struct_names = list(old_copy_names)

    worker_group = world_group.Incl(worker_ranks)
    worker_comm = world_comm.Create(worker_group)

    struct_name = worker_comm.bcast(struct_name, root=0)
    manager_rank = worker_comm.bcast(manager_rank, root=0)

    manager = Manager(manager_rank, worker_comm, potential_template)

    manager.struct_name = struct_name
    manager.struct = manager.load_structure(
        manager.struct_name, parameters['STRUCTURE_DIRECTORY'] + "/"
    )

    manager.struct = manager.broadcast_struct(manager.struct)

    return is_manager, manager, manager_comm

def prepare_node_managers(database, template, parameters, manager_comm, is_master,
        names_file, world_comm):
    if is_master:

        ref_keys = [
            'Ground_state_crystal',
        ]

        if names_file:
            with open(names_file, 'r') as f:
                key_choices = f.readlines()
                key_choices = [l.strip() for l in key_choices]

            for i, key in enumerate(ref_keys):
                if key not in key_choices:
                    raise RuntimeError(
                        "Missing reference structure '{0}' "
                        "in input file '{1}'".format(key, names_file)
                    )

        else:
            key_choices = random.sample(
                list(database.keys()),
                parameters['NUM_STRUCTS']
            )

            for i, key in enumerate(ref_keys):
                if key not in key_choices:
                    print("Adding", key, "to key_choices")
                    key_choices[i] = key
                else:
                    print(key, "already in key_choices")

        key_choices = sorted(key_choices)

        split_struct_lists = np.array_split(
            key_choices, manager_comm.Get_size()
        )

    else:
        split_struct_lists = None

    is_manager = (world_comm.Get_rank() % parameters['PROCS_PER_NODE_MANAGER'] == 0)

    if is_manager:
        struct_list = manager_comm.scatter(split_struct_lists, root=0)

        print(
            "Node", manager_comm.Get_rank(), 'loading', len(struct_list), 'structs',
            flush=True
        )
    else:
        struct_list = None

    global_rank = world_comm.Get_rank()
    color   = global_rank // parameters['PROCS_PER_NODE_MANAGER']
    key     = global_rank % parameters['PROCS_PER_NODE_MANAGER']

    node_comm = MPI.Comm.Split(world_comm, color, key)

    struct_list = node_comm.bcast(struct_list, root=0)

    if len(struct_list) < 1:
        kill_and_write(
            'num_procs / procs_per_manager must be '
            '<= number of structures in the database.'
        )

    node_manager = NodeManager(
        color, template, node_comm,
        max_pop_size=parameters['MAX_POP_SIZE'],
        num_structs = len(struct_list),
        # can't have more than 32 processors on one node
        physical_cores_per_node=min(32, parameters['PROCS_PER_PHYS_NODE']),
    )

    node_manager.load_structures(struct_list, database, load_true=True)

    if node_manager.is_node_master:
        global_num_atoms = manager_comm.allreduce(
            node_manager.local_num_atoms, MPI.SUM
        )
    else:
        global_num_atoms = None

    node_manager.global_num_atoms = world_comm.bcast(global_num_atoms, root=0)

    return node_manager

def prepare_save_directory(parameters):
    """Creates directories to store results"""

    if os.path.isdir(parameters['SAVE_DIRECTORY']):
        shutil.rmtree(parameters['SAVE_DIRECTORY'])

    os.mkdir(parameters['SAVE_DIRECTORY'])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        kill_and_write("Must specify a config and template file")
    else:
        if len(sys.argv) > 5:  # names file included
            main(
                config_name=sys.argv[1], template_file_name=sys.argv[2],
                procs_per_node_manager=sys.argv[3],
                procs_per_phys_node=sys.argv[4], names_file=sys.argv[5]
            )
        else:
            main(
                config_name=sys.argv[1], template_file_name=sys.argv[2],
                procs_per_node_manager=sys.argv[3],
                procs_per_phys_node=sys.argv[4], names_file=None
            )
