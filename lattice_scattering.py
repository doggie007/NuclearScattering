#RUNS CRYSTAL LATTICE SCATTERING

#dependencies
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from functools import partial
from ase.lattice.cubic import FaceCenteredCubic
from collections import Counter
import random
#numba significantly speeds up computations by compiling function beforehand
from numba import jit
from multiprocessing import Pool
#stop runtime warnings
import warnings
warnings.filterwarnings('ignore')

#constants
charge_of_alpha = 2 * constants.e
mass_of_alpha = constants.physical_constants["alpha particle mass"][0]
coulomb_constant = 1 / (4 * math.pi * constants.epsilon_0)

#class for access to properties of an element
class Element():
    def __init__(self, symbol, atomic_number, atomic_mass):
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.charge = atomic_number * constants.e
        self.mass = atomic_mass * constants.physical_constants["unified atomic mass unit"][0]

#elements
gold = Element("Au", 79, 196.96657)
tin = Element("Sn", 50, 118.71)
silver = Element("Ag", 47, 107.8682)
copper = Element("Cu", 29, 63.546)
aluminium = Element("Al", 13, 26.981539)
all_elements = [gold, tin, silver, copper, aluminium]


def MeV_to_joules(MeV):
    return MeV * 1e6 * constants.e

def energy_to_velocity(MeV):
    energy_in_joules = MeV_to_joules(MeV)
    return math.sqrt(energy_in_joules * 2 / mass_of_alpha)

#simulation parameters
sd = 1e-8   #simulation domain: radius of measurement sphere from the origin
dt = 5e-22  #initial timestep
starting_x = 5e-9 #distance from the closest atom in x-axis to start simulation


@jit(nopython=True, fastmath=True)
def coulomb_force(alpha_position, atom_positions, element_charge):
    #params: position vector [x,y,z] of alpha particle, 2D array of the positions of the scattering atoms, charge of scattering atom
    #returns: 2D array of forces [[Fx, Fy, Fz], ...] on the alpha particle from each atom
    #vectors from atom positions to alpha particle position
    r_to_alpha = alpha_position - atom_positions 
    #|r| ^ 2
    sum_distances_squared = (r_to_alpha ** 2).sum(axis = 1)
    #1 / (|r| ^ 3)
    inv_r3 = sum_distances_squared ** (-1.5)
    #coulomb force calculation: each row in r_to_alpha matrix multiplied by corresponding scalar in inv_r3
    force = (r_to_alpha.T * inv_r3).T * (coulomb_constant * element_charge * charge_of_alpha)
    return force

@jit(nopython=True, fastmath=True)
def scattering_angle(initial_v, final_v):
    #returns the angle (in radians) between two vectors
    costheta = np.dot(final_v, initial_v)  / (np.linalg.norm(final_v) * np.linalg.norm(initial_v))
    theta = math.acos(costheta)
    return theta

@jit(nopython=True, fastmath=True, cache=True)
def reached_boundary(t, S):
    #terminates solve_ivp when alpha particle reaches simulation boundary which causes a sign-change in the function
    position, _ = np.array_split(S, 2)
    return sd - np.linalg.norm(position)
reached_boundary.terminal = True

#Rewrite solve function and rhs function for KDTree implementation
@jit(nopython=True)
def split_array(S):
    #Faster jitted splitting of array into alpha particle position and velocity components
    return np.array_split(S,2)

@jit(nopython=True, fastmath=True)
def rhs_fast_helper(alpha_position, alpha_velocity, element_charge, closest_positions):
    #Faster jitted helper function which returns the RHS
    force = coulomb_force(alpha_position, closest_positions, element_charge)
    acceleration = force.sum(axis = 0) / mass_of_alpha
    return np.hstack((alpha_velocity, acceleration))

def rhs_closest_positions(t, S, element_charge, atom_positions, tree):
    #rhs function that calculates force by finding nearest atoms (within a radius)
    #unable to jit as KDTree is incompatible with Numba so uses helper functions to maintain good speed
    force_radius = 1e-9
    num_closest = 4
    alpha_position, alpha_velocity = split_array(S)
    #find closest points within radius
    indices_closest_positions = tree.query_ball_point(alpha_position, force_radius)
    #find closest num_closest points if nothing is within radius or else solve_ivp will fail
    if len(indices_closest_positions) == 0:
        indices_closest_positions = tree.query(alpha_position, num_closest)[1]
    #returns the derivatives
    return rhs_fast_helper(alpha_position, alpha_velocity, element_charge, atom_positions[indices_closest_positions])

def runge_kutta_lattice(initial_pos, v0, element, atom_positions, method, tree, rtol=1e-4, atol=1e-6):
    #returns scattering angle by simulating a-particle trajectory through a lattice
    #uses KD tree to find closest points within radius of particle
    initial_v = np.array([v0, 0.0, 0.0])
    #cut-off time if it does not reach simulation boundary
    time_interval = sd * 2 / v0
    initial_S = np.hstack((initial_pos, initial_v))
    sol = solve_ivp(fun = lambda t,S: rhs_closest_positions(t, S, element.charge, atom_positions, tree), t_span = (0, time_interval), y0 = initial_S, events = reached_boundary, method = method, first_step = dt, rtol= rtol, atol=atol)
    if not sol.success:
        raise Exception("ODE solver failed to integrate")
    #number of time steps taken
    num_points = len(sol.t)
    #values of solution at each time step
    values = sol.y
    #retrieve final velocity
    final_v = values[:,num_points-1][3:6]
    theta = scattering_angle(initial_v, final_v)
    return theta

def random_position(radius):
    #initial position of alpha particle within a uniform circle of specified radius 
    radius = radius * math.sqrt(random.random())
    theta = 2 * math.pi * random.random()
    z = radius * math.cos(theta)
    y = radius * math.sin(theta)
    coordinates = np.array([-(x_max + starting_x), y, z])
    return coordinates

def rotate_y(theta):
        return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                          [ 0              , 1, 0           ],
                          [-math.sin(theta), 0, math.cos(theta)]])
    
def rotate_z(theta):
    return np.matrix([[ math.cos(theta), -math.sin(theta), 0],
                        [ math.sin(theta),  math.cos(theta), 0],
                        [ 0              , 0               , 1]])

if __name__ == "__main__":
    energy_of_alpha = 5.0
    v0 = energy_to_velocity(energy_of_alpha)  #starting velocity in x-axis
    element = gold
    atoms_thick = 10
    cross_section_atoms = 30 #number of atoms across y and z direction
    beam_radius = 1e-9
    num_alpha = int(1e5)
    file_name = "____"
    NUM_PROCESSES = 8  #number of process to happen concurrently

    #Generate FCC lattice object
    atoms = FaceCenteredCubic(element.symbol, size=(atoms_thick, cross_section_atoms, cross_section_atoms))
    #2D array of atom positions
    atom_positions = atoms.get_positions()
    #to angstrom scale
    atom_positions *= 1e-10
    #centre to origin
    x_max,y_max,z_max = atom_positions.max(axis = 0)
    atom_positions[:,0] -= x_max / 2
    atom_positions[:,1] -= y_max / 2
    atom_positions[:,2] -= z_max / 2
    
    """
    #rotate the atom positions in y and z
    atom_positions = np.matmul(atom_positions, rotate_y(math.radians(15)))
    atom_positions = np.matmul(atom_positions, rotate_z(math.radians(10)))
    
    #turn back from matrix to numpy array after matrix multiplication
    atom_positions = np.squeeze(np.asarray(atom_positions))
    """

    #reupdate the max atom positions
    x_max,y_max,z_max = atom_positions.max(axis = 0)
    #construct KD tree
    tree = cKDTree(atom_positions)

    list_of_starting_positions = [random_position(beam_radius) for _ in range(num_alpha)]
    #multiprocessing to get scattering angle given initial positions
    with Pool(NUM_PROCESSES) as p:
        thetas = p.map(partial(runge_kutta_lattice, tree=tree, v0=v0,element=element,atom_positions=atom_positions,method="LSODA",rtol=1e-4,atol=1e-6), list_of_starting_positions)
    thetas_in_deg = np.degrees(thetas)

    #saved to text file
    with open(f"scattering_data/{file_name}.txt", "a") as file:
        for theta in thetas_in_deg:
            file.write(f"{theta}\n")