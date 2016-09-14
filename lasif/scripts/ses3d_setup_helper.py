#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script assisting in determining suitable SES3D settings.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import argparse
import colorama
import itertools
import numpy as np

from lasif.data import OneDimensionalModel


# Arbitrary number but it is probably unlikely to run SES3D on more than
# 3000 cores.
MAX_CORES = 3000


def get_primes(number):
    """
    Gets all prime numbers 2 <= x <= number.

    :param number: The maximum number for which to generate primes.
    :type number: int

    Adapted from http://stackoverflow.com/a/15347389/1657047
    """
    if number <= 2:
        return []
    sieve = [True] * (number + 1)
    for x in range(3, int(number ** 0.5) + 1, 2):
        for y in range(3, (number // x) + 1, 2):
            sieve[(x * y)] = False
    return [2] + [i for i in range(3, number, 2) if sieve[i]]


def get_factors_and_multiplicity(number):
    """
    Get prime factors and their multiplicity.

    :param number: The number for which to get factors and multiplicity.
    :type number: int

    Adapted from http://stackoverflow.com/a/24545330/1657047
    """
    factors = {}
    for prime in get_primes(number):
        n = number
        factor = 0
        while True:
            if n % prime == 0:
                factor += 1
                n /= prime
                factors[prime] = factor
            else:
                break
    return [(key, factors[key]) for key in sorted(factors.keys())]


def get_divisors(number):
    """
    Get all divisors for a certain number.

    :param number: The number for which to get divisors.
    :type number: int

    Adapted from http://stackoverflow.com/a/171784/1657047
    """
    factors = get_factors_and_multiplicity(number)
    nfactors = len(factors)
    f = [0] * nfactors
    divisors = []
    while True:
        divisors.append(reduce(
            lambda x, y: x * y,
            [factors[x][0] ** f[x] for x in range(nfactors)], 1))
        i = 0
        while True:
            f[i] += 1
            if f[i] <= factors[i][1]:
                break
            f[i] = 0
            i += 1
            if i >= nfactors:
                return sorted(divisors)
    return sorted(divisors)


def get_domain_decompositions(nx, ny, nz, max_recommendations=5):
    """
    Attempt to get reasonable domain decomposition recommendations.

    :param nx: Elements in x direction.
    :type nx: int
    :param ny: Elements in y direction.
    :type ny: int
    :param nz: Elements in z direction.
    :type nz: int
    :param max_recommendations: The maximum number of recommendations to
        return.
    :type max_recommendations: int
    """
    factors_x = get_divisors(nx)
    factors_y = get_divisors(ny)
    factors_z = get_divisors(nz)

    # Don't have less than 4 or more than 20 elements per core in one
    # direction.
    factors_x = [_i for _i in factors_x if 15 >= nx / _i >= 4]
    factors_y = [_i for _i in factors_y if 15 >= ny / _i >= 4]
    factors_z = [_i for _i in factors_z if 15 >= nz / _i >= 4]

    # Get all possible combinations and choose then one with the smallest
    # standard deviation.
    combinations = itertools.product(factors_x, factors_y, factors_z)
    combinations = [_i for _i in combinations
                    if np.array(_i).prod() <= MAX_CORES]
    # Sort by their "standard deviation" of the elements per CPU which is a bit
    # wild given its only three samples but it should prefer fairly equal
    # distributions.

    def sort_fct(x):
        a, b, c = map(float, x)
        # Get number of elements per core.
        a = nx / a
        b = ny / b
        c = nz / c
        # kind of a normalized standard deviation...
        return (1.0 - b / a) ** 2 + (1.0 - c / a) ** 2 + \
            (1.0 - a / b) ** 2 + (1.0 - c / b) ** 2 + \
            (1.0 - a / c) ** 2 + (1.0 - b / c) ** 2

    combinations = sorted(combinations, key=sort_fct)
    return sorted(combinations[:max_recommendations],
                  key=lambda x: np.array(x).prod())


def get_ses3d_settings(dx, dy, dz, nx, ny, nz, max_recommendations):
    print("SES3D Setup Assistant\n")

    print("All calculations are done quick and dirty so take them with a "
          "grain of salt.\n")

    decompositions = get_domain_decompositions(
        nx, ny, nz, max_recommendations=max_recommendations)
    if not decompositions:
        print("Could not calculate recommended domain decompositions.")
        return

    print("Possible recommended domain decompositions:\n")
    for comp in decompositions:
        print(colorama.Fore.RED +
              "Total CPU count: %5i; "
              "CPUs in X: %3i (%2i elements/core), "
              "CPUs in Y: %3i (%2i elements/core), "
              "CPUs in Z: %3i (%2i elements/core)" %
              (np.array(comp).prod(), comp[0], nx / comp[0],
               comp[1], ny / comp[1], comp[2], nz / comp[2]) +
              colorama.Style.RESET_ALL)

        x_extent = dx * 111.32
        y_extent = dy * 111.32
        z_extent = dz
        elem_size_x = x_extent / (nx + comp[0])
        elem_size_y = y_extent / (ny + comp[1])
        elem_size_z = z_extent / (nz + comp[2])

        print("  Extent in latitudinal  (X) direction: %7.1f km, "
              "%5.1f km/element, %4i elements" % (x_extent, elem_size_x,
                                                  nx + comp[0]))
        print("  Extent in longitudinal (Y) direction: %7.1f km, "
              "%5.1f km/element, %4i elements" % (y_extent, elem_size_y,
                                                  ny + comp[1]))
        print("  Extent in depth        (Z) direction: %7.1f km, "
              "%5.1f km/element, %4i elements" % (z_extent, elem_size_z,
                                                  nz + comp[2]))
        element_sizes = [elem_size_x, elem_size_y, elem_size_z]
        min_element_size = min(element_sizes)
        max_element_size = max(element_sizes)
        # Smallest GLL point distance.
        dx_min = 0.1717 * min_element_size

        # Get the Wave speeds at all depths and choose the biggest one.
        m = OneDimensionalModel("ak135-f")
        s_values = [m.get_value("vs", _i) for _i in
                    np.linspace(15, dz, 400)]
        p_values = [m.get_value("vp", _i) for _i in
                    np.linspace(15, dz, 400)]
        v_max = max(p_values)
        v_min = min(s_values)
        print("  Wave velocities range from %.1f km/s to %.1f km/s. The "
              "velocities of the top 15 km have not been analyzed to avoid "
              "very slow layers." % (v_min, v_max))

        # criterion approximately 0.3 for SES3D.
        dt = 0.3 * dx_min / v_max
        # Well...the value to use here is kind of hand-wavy..1.6 seems to
        # agree quite well with data.
        minimum_period = 1.6 * max_element_size / v_min

        print(colorama.Fore.GREEN +
              "  Maximal recommended time step: %.3f s" % dt)
        print("  Minimal resolvable period: %.1f s" % minimum_period +
              colorama.Style.RESET_ALL)

        print(colorama.Fore.YELLOW + "  SES3D Settings: nx_global: %i, "
              "ny_global: %i, nz_global: %i" % (nx, ny, nz))
        print("                  px: %i, py: %i, px: %i" % (
            comp[0], comp[1], comp[2]) + colorama.Style.RESET_ALL + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m lasif.scripts.ses3d_setup_helper",
        description="Script assisting in determining suitable SES3D settings.")
    parser.add_argument("dx", type=float, help="latitudinal extent in degrees")
    parser.add_argument("dy", type=float, help="longitudinal extent in "
                                               "degrees")
    parser.add_argument("dz", type=float, help="depth extent in km")
    parser.add_argument("nx", type=int, help="~ element count in "
                        "latitudinal dir. Same as nx_global.")
    parser.add_argument("ny", type=int, help="~ element count in "
                        "longitudinal dir. Same as ny_global.")
    parser.add_argument("nz", type=int, help="~ element count in depth "
                        "dir. Same as nz_global.")
    parser.add_argument("--recommendations", type=int, default=3,
                        help="number of recommendations to provide")
    args = parser.parse_args()
    get_ses3d_settings(dx=args.dx, dy=args.dy, dz=args.dz, nx=args.nx,
                       ny=args.ny, nz=args.nz,
                       max_recommendations=args.recommendations)


if __name__ == "__main__":
    main()
