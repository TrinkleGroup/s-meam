import os
import sys
import numpy as np
from mpi4py import MPI
# from src.ga import ga
from src.sa import sa
from src.ga import ga
from src.potential_templates import Template

np.set_printoptions(linewidth=1000)

# TODO: print settings before running anything
# TODO: have a script that checks the validity of an input script befor qsub

def main(config_name, template_file_name):
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    is_master = (world_rank == 0)

    if is_master:
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

                # knot_positions = []
                spline_ranges = []
                spline_npts = []

                for _ in range(nsplines):
                    entries = f.readline().strip().split(" ")
                    x_lo, x_hi, y_lo, y_hi, nknots = [np.float(e) for e in
                        entries]

                    nknots = int(nknots)
                    spline_npts.append(nknots)

                    # knot_positions.append(np.linspace(x_lo, x_hi, nknots))
                    spline_ranges.append((y_lo, y_hi))

                spline_npts = np.array(spline_npts)

                end_indices = np.cumsum([n + 2 for n in spline_npts])

                start_indices = end_indices - spline_npts - 2

                spline_indices = list(zip(start_indices, end_indices))

                f.readline()

                # should a potential be loaded from a file? randomly generated?
                if f.readline().strip().split(" ")[-1] == "True":
                    fname = f.readline().strip().split(" ")[-1]
                    print("Loading mask and parameter vector from:", fname)

                    data = np.genfromtxt(fname)
                else:
                    print("Loading mask and parameter vector from:",
                        template_file_name
                    )

                    data = np.genfromtxt(
                        template_file_name, skip_header=8+nsplines
                    )

                mask = data[:, 0]
                knot_values = data[:, 1]

                print("pvec_len:", len(knot_values))
                print("u_domains:", template_args['u_domains'])
                print("spline_ranges:", spline_ranges)
                print("spline_indices:", spline_indices)

                template = Template(
                    pvec_len = len(knot_values),
                    u_ranges = template_args['u_domains'],
                    spline_ranges = spline_ranges,
                    spline_indices = spline_indices
                )

                template.active_mask = mask
                template.pvec = knot_values

        else:
            kill_and_write("Config file does not exist")
    else:
        parameters = None
        template = None

    parameters = world_comm.bcast(parameters, root=0)
    template = world_comm.bcast(template, root=0)

    # convert types of inputs from str
    int_params = ['NUM_STRUCTS', 'POP_SIZE', 'GA_NSTEPS', 'LMIN_FREQ',
            'INIT_NSTEPS', 'LMIN_NSTEPS', 'FINAL_NSTEPS', 'CHECKPOINT_FREQ',
            'SA_NSTEPS', ]

    float_params = ['MUT_PB', 'COOLING_RATE', 'TMIN', 'TSTART']
    bool_params = ['RUN_NEW_GA', 'DO_LMIN']

    for key, val in parameters.items():
        if key in int_params:
            parameters[key] = int(val)
        elif key in float_params:
            parameters[key] = float(val)
        elif key in bool_params:
            parameters[key] = (val == 'True')

    if os.path.isdir(parameters['SAVE_DIRECTORY']):
        parameters['SAVE_DIRECTORY'] = parameters['SAVE_DIRECTORY'] + '-' +\
            str(np.random.randint(100000))

    # run the optimizer
    if parameters['OPT_TYPE'] == 'GA':
        if is_master:
            print("Running GA")

        ga(parameters, template)
    elif parameters['OPT_TYPE'] == 'SA':
        if is_master:
            print("Running SA")

        sa(parameters, template)
    else:
        if is_master:
            kill_and_write("Invalid optimization type (OPT_TYPE)")

def kill_and_write(msg):
    print(msg, flush=True)
    MPI.COMM_WORLD.Abort(1)


if __name__ == "__main__":
    is_master = MPI.COMM_WORLD.Get_rank() == 0

    if len(sys.argv) < 3:
        if is_master:
            kill_and_write("Must specify a config and template file")
    else:
        main(sys.argv[1], sys.argv[2])
