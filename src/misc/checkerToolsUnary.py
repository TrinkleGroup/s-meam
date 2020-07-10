import os
import sys
import glob
import subprocess
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from ase import Atoms

import numpy as np
import src.meam
from src.potential_templates import Template
from src.meam import MEAM

# points_per_spline = 7

elements = ['Cu', 'Ge', 'Li' , 'Mo', 'Ni', 'Si']
all_ic   = [2.2, 2.4, 2.4, 2.4, 2.2, 2.2]
all_oc   = [4.0, 5.3, 5.1, 5.2, 3.9, 5.0]
crystals = ['fcc', 'diamond', 'bcc', 'bcc', 'fcc', 'diamond']
expected_a = [3.621, 5.763, 3.427, 3.168, 3.508, 5.469]  # DFT values, not each potential's given value

ulim = [(-1, 1)]

def compute_evf(elem, pot_to_use):

    cryst = crystals[elements.index(elem)]
    a0 = expected_a[elements.index(elem)]

    os.chdir('/home/jvita/scripts/s-meam/VACANCY')

    command = ['lmp_serial', '<', 'in.formation', f'-var elem {elem}', '-var pot_name /tmp/spline.meam2', f'-var cryst {cryst}', f'-var ao {a0}']

    results = subprocess.run([' '.join(command)], stdout=subprocess.PIPE, shell=True)
    cleaned = results.stdout.decode('utf-8').split('\n')

    return float([l for l in cleaned if 'Vacancy formation' in l][0].split()[-1])

def compute_cij(elem, pot_to_use):
    # compute the Cij using the given potential
    pot_to_use.write_to_file('/tmp/spline.meam2')

    os.chdir('/home/jvita/scripts/s-meam/ELASTIC')

    command = ['lmp_serial', '<', 'in.elastic', f'-var elem {elem}', f'-var str_name {elem.lower()}.data', '-var pot_name /tmp/spline.meam2']

    results = subprocess.run([' '.join(command)], stdout=subprocess.PIPE, shell=True)
    cleaned = results.stdout.decode('utf-8').split('\n')

    c11 = float([l for l in cleaned if 'C11all' in l][0].split()[-2])
    c12 = float([l for l in cleaned if 'C12all' in l][0].split()[-2])
    c44 = float([l for l in cleaned if 'C44all' in l][0].split()[-2])
    bulk = float([l for l in cleaned if 'Bulk' in l][0].split()[-2])
    
    return c11, c12, c44, bulk

def compute_errors(elem, pot_to_use):
    types = [elem]

    # load true values (tv_oneref_pa) and compute reference energy
    database_base_path = f'/home/jvita/scripts/s-meam/data/fitting_databases/mlearn/data/{elem}/'

    info_path = os.path.join(database_base_path, 'tv_oneref_pa/')

    ref_struct_name = 'Ground_state_crystal'
    ref_struct_file = os.path.join(database_base_path, 'lammps', ref_struct_name + '.data')

    ref_struct = src.lammpsTools.atoms_from_file(ref_struct_file, types=types)

    comp_ref_results = pot_to_use.get_lammps_results(ref_struct)
    comp_r_energy = comp_ref_results['energy'] / len(ref_struct)
    
    names = []
    energy_errors = []
    forces_errors = []
    stress_errors = []
    skipped = []
    
    # compute EFS errors for all structures in the database
    struct_folder = os.path.join(database_base_path, 'structures', '*')

    for i, struct_fname in enumerate(sorted(glob.glob(struct_folder))):

        cell = np.genfromtxt(struct_fname, max_rows=3)
        positions = np.genfromtxt(struct_fname, skip_header=3)[:, 1:]

        db_atoms = Atoms(
            f'{positions.shape[0]}{elem}',
             positions=positions,
             cell=cell,
             pbc=[1, 1, 1]
        )

        volume = db_atoms.get_volume()
        cell = db_atoms.get_cell()
        positions = db_atoms.get_positions()

        new_cell = rotate_into_lammps(cell.T).T

        lammps_to_org = cell.T @ np.linalg.inv(new_cell.T)
        org_to_lammps = new_cell.T @ np.linalg.inv(cell.T)

        new_positions = (org_to_lammps @ positions.T).T

        struct = db_atoms

    #     struct = Atoms(
    #         f'{positions.shape[0]}{elem}',
    #          positions=new_positions,
    #          cell=new_cell,
    #          pbc=[1, 1, 1]
    #     )

    #     struct = src.lammpsTools.atoms_from_file(struct_fname, types=types)

        natoms = len(struct)

        short_name = os.path.splitext(os.path.split(struct_fname)[-1])[0]

        struct_info_file = os.path.join(info_path, "info." + short_name)

        true_ediff = np.genfromtxt(struct_info_file, max_rows=1)
        true_forces = np.genfromtxt(struct_info_file, skip_header=1, skip_footer=1)
        true_stress = np.genfromtxt(struct_info_file, skip_header=1+true_forces.shape[0])

        # Note: database values are originally ordered as xx, yy, zz, xy, yz, xz
        true_stress = [true_stress[0], true_stress[1], true_stress[2], true_stress[4], true_stress[5], true_stress[3]]
        true_stress = np.array(true_stress)

        try:
            comp_results = pot_to_use.get_lammps_results(struct)

            comp_s_energy = comp_results['energy'] / len(struct)
            comp_ediff = comp_s_energy - comp_r_energy

            comp_forces = comp_results['forces']

            eng_err = abs(true_ediff - comp_ediff)
            fcs_err = np.average(abs(comp_forces - true_forces))

            comp_stress = comp_results['stress']

            padded = np.array([
                [comp_stress[0], comp_stress[5], comp_stress[4]],
                [comp_stress[5], comp_stress[1], comp_stress[3]],
                [comp_stress[4], comp_stress[3], comp_stress[2]]
            ])

            rotated = np.einsum('im,jn,mn->ij', lammps_to_org, lammps_to_org, padded)
            rotated = np.array([rotated[0,0], rotated[1,1], rotated[2,2], rotated[1,2], rotated[0,2], rotated[0,1]])
            rotated = -rotated*160.217662/0.1  # NodeManager pressure units are eV/(A^3); database is kbar

            str_err = np.average(abs(rotated - true_stress))

            energy_errors.append(eng_err)
            forces_errors.append(fcs_err)
            stress_errors.append(str_err)

            print(short_name, eng_err)

            if 'Vacancy' in short_name:
                names.append(short_name[10:])
            elif 'Snapshot' in short_name:
                names.append(short_name[10:])
            else:
                names.append(short_name)
        except RuntimeError as e:
            print(e)

            skipped.append(short_name)
    return np.array(energy_errors), np.array(forces_errors), np.array(stress_errors)

def rotate_into_lammps(cell):
    # expects columns to be cell vectors
    
    first_along_x = (cell[1, 0] == 0) and (cell[2,0] == 0)
    second_in_xy = (cell[2,1] == 0)
    
    if first_along_x and second_in_xy:
        # already in LAMMPS format; avoid math
        return cell
    
    ax = np.sqrt(np.dot(cell[0], cell[0]))
    a_hat = cell[0] / ax
    
    bx = np.dot(cell[1], a_hat)
    by = np.sqrt(np.dot(cell[1], cell[1]) - bx**2)
    cx = np.dot(cell[2], a_hat)
    cy = (np.dot(cell[1], cell[2]) - bx*cx) / by
    cz = np.sqrt(np.dot(cell[2], cell[2]) - cx*cx - cy*cy)
    
    return np.array([
        [ax, bx, cx],
        [0, by, cy],
        [0, 0, cz],
    ])

def build_template(types, inner_cutoff=1.5, outer_cutoff=5.5, spline_nknots=[]):
    cumsum = [0] + np.cumsum(spline_nknots).tolist()

    potential_template = Template(
        pvec_len=cumsum[-1]+10,
        u_ranges=[(-1, 1), (-1, 1)],
        spline_ranges=[
            (-1, 1), (-1, 1),
            (-1, 1), (-1, 1),
            (-1, 1)
        ],
        spline_indices=[(x+2*i, cumsum[i+1]+2*(i+1)) for i,x in enumerate(cumsum[:-1])],
        types=types
    )

    potential_template.active_mask = np.ones(potential_template.pvec_len)
    
    return potential_template

def plot_splines(potential_template, guess_pvec, fig, ax, inner_cutoff, outer_cutoff, x_indices=None, x_pvec=None, ext=0, lines=None, points=None, u_dom=[(-1, 1)], title='', colors='k', alpha=1, output=False):

    if guess_pvec.shape[0] == 1:
        colors = [colors]
        
    labels = ['A']

    titles= [r"$\phi$",
             r"$\rho$",
             r"$U$",
             r"$f$",
             r"$g$"]
    
    points_per_spline = (potential_template.pvec_len // 5) - 2

#     split_indices = [el[0] for el in potential_template.x_indices[1:]]
    split_indices = x_indices[1:]

#     x_rng_tups = np.split(x_pvec, split_indices)
#     x_rng_tups = [(t[0], t[points_per_spline-1], points_per_spline) for t in x_rng_tups]
    x_rng_tups = [(x[0], x[-1], len(x)) for x in np.split(x_pvec, split_indices)]
    
    x_rngs = [np.linspace(t[0], t[1], t[2]) for t in x_rng_tups]
    x_plts = [np.linspace(t[0], t[1], 100) for t in x_rng_tups]
    
    for i in [0, 1, 3]:  # phi, rho, f
        x_plts[i] = np.linspace(x_plts[i][0] - ext*abs(x_plts[i][0]), x_plts[i][-1], 100)

    for i in [2]:  # U
        x_plts[i] = np.linspace(x_plts[i][0] - ext*abs(x_plts[i][0]), x_plts[i][-1] + ext*abs(x_plts[i][-1]), 100)

    dat_guess = np.split(guess_pvec, [x[0] for x in potential_template.spline_indices[1:]], axis=1)

    row_num = 0
    col_num = 0
    subplot_counter = 0

    if lines is None:
        new_lines = []
        new_points = []
    else:
        new_lines = lines
        new_points = points

    for i in range(len(titles)):

        row_num = i // len(ax[0])
        col_num = i % len(ax[0])
            
        for j in range(dat_guess[i].shape[0]):
            if output:
                print(i, j)

            y_guess, dy_guess = dat_guess[i][j, :-2], dat_guess[i][j, -2:]
            
            cs1 = CubicSpline(x_rngs[i], y_guess)
#             cs1 = CubicSpline(x_rngs[i], y_guess, bc_type=((1,dy_guess[0]), (1,dy_guess[1])))
            cs1 = CubicSpline(x_rngs[i], y_guess, bc_type=('natural', 'natural'))

            shown_label = shown_label_true = None

            if lines is None:
                l = ax[row_num, col_num].plot(x_plts[i], cs1(x_plts[i]), color=colors[j], alpha=alpha)
                dots = ax[row_num, col_num].plot(x_rngs[i], y_guess, 'o', alpha=alpha)
                plt.setp(dots, 'color', plt.getp(l[0], 'color'))
                new_lines.append(l[0])
                new_points.append(dots[0])
            else:
                lines[i].set_ydata(cs1(x_plts[i]))
                points[i].set_ydata(y_guess)
                plt.setp(points[i], 'color', plt.getp(lines[i], 'color'))
                       
        ax[row_num, col_num].set_autoscale_on(True)
        ax[row_num, col_num].relim()
        ax[row_num, col_num].autoscale_view(True, True, True)
        
        col_num += 1
        subplot_counter += 1
    
    ax[0][1].set_title(title)
    ax[-1][-1].axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    
    return fig, ax, new_lines, new_points

def plot_ni(ni_trace, fig, ax, start_index=0, end_index=-1, lines=None):
    ulim = [-1, 1]
    
    if end_index == -1:
        end_index = ni_trace.shape[0]
    
    ni_trace = ni_trace[start_index:end_index, :]

    n = ni_trace.shape[0]

    a_min_rolling_average = ni_trace[:, 0]    
    a_max_rolling_average = ni_trace[:, 1]
    avg_a_ni = ni_trace[:, 2]

    x = np.arange(start_index, end_index)
    
    if lines is None:
        new_lines = []
    else:
        for l in lines:
            l.set_xdata(x)
            
        new_lines = lines

    ax.set_title("Trace of min/max ni sampling")

    if lines is None:
        l1 = ax.plot(x, a_max_rolling_average, 'r', label="A min/max")
        l2 = ax.plot(x, a_min_rolling_average, 'r', alpha=0.5)
        l3 = ax.plot(x, avg_a_ni, '--r', alpha=0.5, label='A avg')

        new_lines += [l1[0], l2[0], l3[0]]
    else:
        lines[0].set_ydata(a_max_rolling_average)
        lines[1].set_ydata(a_min_rolling_average)
        lines[2].set_ydata(avg_a_ni)
    
    if ni_trace.shape[1] > 3: # rolling U domains were logged
        if lines is None:
            l1 = ax.plot(x, ni_trace[:, -2], '--b', alpha=0.5, label="U_A[]")
            l2 = ax.plot(x, ni_trace[:, -1], '--b', alpha=0.5)       
        
            new_lines += [l1[0], l2[0]]
        else:
            lines[3].set_ydata(ni_trace[:, -2])
            lines[4].set_ydata(ni_trace[:, -1])
    
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$n_i$")
                       
    ax.set_autoscale_on(True)
    ax.relim()
    ax.autoscale_view(True, True, True)
            
    fig.tight_layout()
    fig.canvas.draw()
    
    return fig, ax, new_lines

def plot_cost(cost, fig, ax, start_index=0, end_index=-1, line=None):
    
    if end_index == -1:
        end_index = len(cost)
    
    x = np.arange(start_index, end_index)
    
    if line is None:
        l = ax.semilogy(x, cost[start_index:end_index], 'b')
        
        line = l[0]
    else:
        line.set_xdata(x)
        line.set_ydata(np.log10(cost[start_index:end_index]))
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Cost")
                           
    ax.set_autoscale_on(True)
    ax.relim()
    ax.autoscale_view(True, True, True)
        
    fig.tight_layout()
    fig.canvas.draw()
    
    return fig, ax, line