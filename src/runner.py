import os
import sys
sys.path.append('./')
import glob
import mpi4py
import numpy as np
import src.lammpsTools
import logging
from mpi4py import MPI
from src.sa import sa
from src.ga import ga
from src.sgd import sgd
from src.mcmc import mcmc
from src.potential_templates import Template
import src.partools as partools
from src.database import Database
from src.nodemanager import NodeManager

np.set_printoptions(linewidth=1000)

import random

seed = 42
seed = np.random.randint(10000)

np.random.seed(seed)
random.seed(seed)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: have a script that checks the validity of an input script befor qsub

def main(config_name, template_file_name):
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()

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

    # TODO: don't have different 'NSTEP' params for each algorithm type

    # convert types of inputs from str
    int_params = [
        'NUM_STRUCTS', 'POP_SIZE', 'NSTEPS', 'LMIN_FREQ',
        'INIT_NSTEPS', 'LMIN_NSTEPS', 'FINAL_NSTEPS', 'CHECKPOINT_FREQ',
        'RESCALE_FREQ', 'RESCALE_STOP_STEP', 'U_NSTEPS',
        'MCMC_BLOCK_SIZE', 'SGD_BATCH_SIZE', 'SHIFT_FREQ',
        'TOGGLE_FREQ', 'TOGGLE_DURATION', 'MCMC_FREQ', 'MCMC_NSTEPS',
        'PROCS_PER_NODE'
    ]

    float_params = [
        'MUT_PB', 'COOLING_RATE', 'TMIN', 'TSTART', 'SGD_STEP_SIZE',
        'MOVE_PROB', 'MOVE_SCALE'
    ]

    bool_params = [
        'RUN_NEW_GA', 'DO_LMIN', 'DEBUG', 'DO_RESCALE', 'OVERWRITE_OLD_FILES',
        'DO_SHIFT', 'DO_TOGGLE', 'PENALTY_ON', 'DO_MCMC'
    ]

    for key, val in parameters.items():
        if key in int_params:
            parameters[key] = int(val)
        elif key in float_params:
            parameters[key] = float(val)
        elif key in bool_params:
            parameters[key] = (val == 'True')

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

    if is_master:
        f = open(parameters['NI_TRACE_FILE_NAME'], 'ab')
        f.close()

        f = open(parameters['COST_FILE_NAME'], 'ab')
        f.close()

        print("MASTER: Preparing save directory/files ... ", flush=True)
        partools.prepare_save_directory(parameters)

        print("Loading database ...", flush=True)

    database = Database(
        parameters['DATABASE_FILE'], 'a',
        template.pvec_len, template.types,
        knot_xcoords=template.knot_positions, x_indices=template.x_indices,
        cutoffs=template.cutoffs,
        driver='mpio', comm=world_comm
    )

    # for fname in glob.glob(os.path.join(parameters['LAMMPS_FOLDER'], "*")):

    #     struct_name = os.path.splitext(os.path.split(fname)[-1])[0]

    #     if is_master:
    #         print("\t", struct_name, flush=True)

    #     atoms = src.lammpsTools.atoms_from_file(fname, template.types)

    #     database.add_structure(struct_name, atoms)
    #     database.add_true_value(
    #         os.path.join(parameters['INFO_DIRECTORY'],'info.' + struct_name),
    #         "Ti48Mo80_type1_c18"
    #     )

    if is_master:
        print("Preparing node managers...", flush=True)

    node_manager = prepare_node_managers(
        database, template, parameters, world_comm, is_master
    )

    if is_master:
        print()

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
    elif parameters['OPT_TYPE'] == 'GA':
        if is_master:
            print("Running GA", flush=True)
            print()

        ga(parameters, template, node_manager)
    elif parameters['OPT_TYPE'] == 'SA':
        if is_master:
            print("Running SA", flush=True)
            print()
        sa(parameters, database, template, node_manager)
    elif parameters['OPT_TYPE'] == 'MCMC':
        if is_master:
            print("Running MCMC", flush=True)
            print()
        mcmc(parameters, database, template, node_manager)
    elif parameters['OPT_TYPE'] == 'SGD':
        if is_master:
            print("Running SGD", flush=True)
        sgd(parameters, database, template, node_manager)
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
                print("Loading mask and parameter vector from:", fname)

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

            nphi = template_args['ntypes']*(template_args['ntypes'] + 1) / 2

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

            template = Template(
                template_args['types'],
                pvec_len=len(knot_values),
                u_ranges=template_args['u_domains'],
                spline_ranges=spline_ranges,
                spline_indices=spline_indices
            )

            x_indices = np.concatenate([
                [0], np.cumsum(spline_npts)
                ])[:-1]

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
                                p, v = stripped.split(" ")
                                parameters[p] = v
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

def prepare_node_managers(database, template, parameters, comm, is_master):
    if is_master:
        key_choices = random.sample(
            list(database.keys()),
            parameters['NUM_STRUCTS']
        )

        # TODO: the database should store these itself
        # ref_keys = [
        #     'Ti48Mo80_type1_c18',
        #     'Ti80Mo48_SQS1_lattice',
        #     'Ti72Mo56_SQS1_lattice',
        #     'Ti64Mo64_SQS2_lattice_f',
        #     'Ti56Mo72_SQS1_lattice',
        #     'Ti48Mo80_SQS1_lattice',
        #     'B2',
        #     'B32',
        # ]
        # 
        # for i, key in enumerate(ref_keys):
        #     if key not in key_choices:
        #         print("Adding", key, "to key_choices")
        #         key_choices[i] = key

        key_choices = sorted(key_choices)

        split_struct_lists = np.array_split(
            key_choices, comm.Get_size()
        )
    else:
        split_struct_lists = None

    struct_list = comm.scatter(split_struct_lists, root=0)

    node_manager = NodeManager(comm.Get_rank(), template)
    node_manager.load_structures(struct_list, database, load_true=True)
    node_manager.start_pool(parameters['PROCS_PER_NODE'])

    return node_manager

if __name__ == "__main__":
    is_master = MPI.COMM_WORLD.Get_rank() == 0

    if len(sys.argv) < 3:
        if is_master:
            kill_and_write("Must specify a config and template file")
    else:
        main(sys.argv[1], sys.argv[2])
