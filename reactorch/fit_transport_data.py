#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:04:05 2020

@author: weiqi
"""

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import torch


poly_order = 6


def species_viscosities(gas, T_list, p, comp):

    species_viscosities_list = np.zeros((T_list.shape[0], gas.n_species))

    for i, T in enumerate(T_list):
        gas.TPX = T, p, comp
        species_viscosities_list[i, :] = gas.species_viscosities

    poly = np.polyfit(np.log(T_list), species_viscosities_list, deg=poly_order)

    return species_viscosities_list, poly


def binary_diff_coeffs(gas, T_list, p, comp):
    binary_diff_coeffs_list = np.zeros((T_list.shape[0], gas.n_species*gas.n_species))

    for i, T in enumerate(T_list):
        gas.TPX = T, p, comp
        binary_diff_coeffs_list[i, :] = gas.binary_diff_coeffs.flatten()

    poly = np.polyfit(np.log(T_list), binary_diff_coeffs_list, deg=poly_order)
    return binary_diff_coeffs_list, poly


def thermal_conductivity(gas, T_list, p, comp):
    thermal_conductivity_list = np.zeros((T_list.shape[0], gas.n_species))

    arr_thermal_cond = np.zeros(gas.n_species)

    for i, T in enumerate(T_list):
        gas.TPX = T, p, comp
        for j in range(gas.n_species):
            Y = np.zeros(gas.n_species)
            Y[j] = 1
            gas.set_unnormalized_mass_fractions(Y)
            arr_thermal_cond[j] = gas.thermal_conductivity

        thermal_conductivity_list[i, :] = arr_thermal_cond

    poly = np.polyfit(np.log(T_list), thermal_conductivity_list, deg=poly_order)
    return thermal_conductivity_list, poly

def fit_transport_data(mech_yaml, TPX):

    gas = ct.Solution(mech_yaml)
    
    T = TPX[0]
    p = TPX[1]
    comp = TPX[2]

    n_points = 2700
    T_min = 300
    T_max = 3500
    T_list = np.linspace(start=T_min, stop=T_max, num=n_points, endpoint=True)

    gas.TPX = 1000, p, comp

    species_viscosities_list, species_viscosities_poly = species_viscosities(gas, T_list, p, comp)
    binary_diff_coeffs_list, binary_diff_coeffs_poly = binary_diff_coeffs(gas, T_list, p, comp)
    thermal_conductivity_list, thermal_conductivity_poly = thermal_conductivity(gas, T_list, p, comp)

    np.save('mech/transfit/species_viscosities_poly', species_viscosities_poly)
    np.save('mech/transfit/binary_diff_coeffs_poly', binary_diff_coeffs_poly)
    np.save('mech/transfit/thermal_conductivity_poly', thermal_conductivity_poly)

    max_error_relative_list = np.zeros(gas.n_species)
    for i in range(gas.n_species):
        poly = np.poly1d(species_viscosities_poly[:, i])
        max_error_relative_list[i] = np.abs(poly(np.log(T_list)) / species_viscosities_list[:, i] - 1).max()


    np.set_printoptions(precision=3)
    print(gas.species_names)
    print(max_error_relative_list * 100)


    max_error_relative_list = np.zeros(gas.n_species)
    for i in range(gas.n_species):
        poly = np.poly1d(thermal_conductivity_poly[:, i])
        max_error_relative_list[i] = np.abs(poly(np.log(T_list)) / (thermal_conductivity_list[:, i] + 1e-12) - 1).max()

    print(gas.species_names)
    print(max_error_relative_list * 100)
