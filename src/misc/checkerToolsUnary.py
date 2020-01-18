import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import numpy as np
import src.meam
from src.potential_templates import Template
from src.meam import MEAM

points_per_spline = 7

def build_template(types, inner_cutoff=1.5, outer_cutoff=5.5):
    x_pvec = np.concatenate([
        np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 2),
        np.tile(np.linspace(-1, 1, points_per_spline), 1),
        np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 1),
        np.tile(np.linspace(-1, 1, points_per_spline), 1)],
    )

    x_indices = range(0, points_per_spline * 12, points_per_spline)

    potential_template = Template(
        pvec_len=45,
        u_ranges=[(-1, 1), (-1, 1)],
        spline_ranges=[(-1, 1), (-1, 1),
                       (-1, 1), (-1, 1),
                       (-1, 1)],
        spline_indices=[(0, 9), (9, 18), (18, 27), (27, 36), (36, 45)],
        types=types
    )

    mask = np.ones(potential_template.pvec_len)

    potential_template.pvec[6] = 0; mask[6] = 0  # rhs value phi
    potential_template.pvec[8] = 0; mask[8] = 0  # rhs deriv phi

    potential_template.pvec[15] = 0; mask[15] = 0  # rhs value rho
    potential_template.pvec[17] = 0; mask[17] = 0  # rhs deriv rho

    potential_template.pvec[33] = 0; mask[33] = 0  # rhs value f
    potential_template.pvec[35] = 0; mask[35] = 0  # rhs deriv f

    potential_template.active_mask = mask
    
    return potential_template

def plot_splines(potential_template, guess_pvec, fig, ax, inner_cutoff, outer_cutoff, ext=0, lines=None, points=None, u_dom=[(-1, 1)], title=''):

    labels = ['A']

    titles= [r"$\phi$",
             r"$\rho$",
             r"$U$",
             r"$f$",
             r"$g$"]

    x_pvec = np.concatenate([
        np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 2),
        np.linspace(u_dom[0][0], u_dom[0][1], points_per_spline),
        np.tile(np.linspace(inner_cutoff, outer_cutoff, points_per_spline), 1),
        np.tile(np.linspace(-1, 1, points_per_spline), 1)]
    )

    x_rng_tups = np.split(x_pvec, 5)
    x_rng_tups = [(t[0], t[6], 7) for t in x_rng_tups]

    x_rngs = [np.linspace(t[0], t[1], t[2]) for t in x_rng_tups]
    x_plts = [np.linspace(t[0], t[1], 100) for t in x_rng_tups]
    
    for i in [0, 1, 3]:  # phi, rho, f
        x_plts[i] = np.linspace(x_plts[i][0] - ext*abs(x_plts[i][0]), x_plts[i][-1], 100)

    for i in [2]:  # U
        x_plts[i] = np.linspace(x_plts[i][0] - ext*abs(x_plts[i][0]), x_plts[i][-1] + ext*abs(x_plts[i][-1]), 100)

    split_indices = [el[0] for el in potential_template.spline_indices[1:]]

    dat_guess = np.split(guess_pvec, split_indices, axis=1)

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

            y_guess, dy_guess = dat_guess[i][j, :-2], dat_guess[i][j, -2:]

            cs1 = CubicSpline(x_rngs[i], y_guess)
            cs1 = CubicSpline(x_rngs[i], y_guess, bc_type=((1,dy_guess[0]), (1,dy_guess[1])))

            shown_label = shown_label_true = None

            if lines is None:
                l = ax[row_num, col_num].plot(x_plts[i], cs1(x_plts[i]))
                dots = ax[row_num, col_num].plot(x_rngs[i], y_guess, 'o')
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