#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of classes dealing with SES3D 4.0 Models

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch),
    Stefan Mauerberger (mauerberger@geophysik.uni-muenchen.de),
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2012 - 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import glob
import os
import math
import numpy as np
import re
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
import warnings

from lasif import rotations


# Pretty units for some components.
UNIT_DICT = {
    "vp": r"$\frac{\mathrm{km}}{\mathrm{s}}$",
    "vsv": r"$\frac{\mathrm{km}}{\mathrm{s}}$",
    "vsh": r"$\frac{\mathrm{km}}{\mathrm{s}}$",
    "rho": r"$\frac{\mathrm{kg}^3}{\mathrm{m}^3}$",
    "rhoinv": r"$\frac{\mathrm{m}^3}{\mathrm{kg}^3}$",
}


def _get_colormap(colors, colormap_name):
    """
    A simple helper function facilitating linear colormap creation.
    """
    # Sort and normalize from 0 to 1.
    indices = np.array(sorted(colors.iterkeys()))
    normalized_indices = (indices - indices.min()) / indices.ptp()

    # Create the colormap dictionary and return the colormap.
    cmap_dict = {"red": [], "green": [], "blue": []}
    for _i, index in enumerate(indices):
        color = colors[index]
        cmap_dict["red"].append((normalized_indices[_i], color[0], color[0]))
        cmap_dict["green"].append((normalized_indices[_i], color[1], color[1]))
        cmap_dict["blue"].append((normalized_indices[_i], color[2], color[2]))
    return LinearSegmentedColormap(colormap_name, cmap_dict)


# A pretty colormap for use in tomography.
tomo_colormap = _get_colormap({
    0.0: [0.1, 0.0, 0.0],  # Reddish black
    0.2: [0.8, 0.0, 0.0],
    0.3: [1.0, 0.7, 0.0],
    0.48: [0.92, 0.92, 0.92],
    0.5: [0.92, 0.92, 0.92],  # Light gray
    0.52: [0.92, 0.92, 0.92],
    0.7: [0.0, 0.6, 0.7],
    0.8: [0.0, 0.0, 0.8],
    1.0: [0.0, 0.0, 0.1]},
    "seismic_tomography")  # Blueish black


class RawSES3DModelHandler(object):
    """
    Class able to deal with a directory of raw SES3D 4.0 models.

    A directory is defined by containing:
        * boxfile              -> File describing the discretization
        * A0-A[XX]             -> Anisotropic parameters (one per CPU)
        * B0-B[XX]             -> Anisotropic parameters (one per CPU)
        * C0-C[XX]             -> Anisotropic parameters (one per CPU)
        * lambda0 - lambda[XX] -> Elastic parameters (one per CPU)
        * mu0 - mu[XX]         -> Elastic parameters (one per CPU)
        * rhoinv0 - rhoinv[XX] -> Elastic parameters (one per CPU)
        * Q0 - Q[XX]           -> Elastic parameters (one per CPU)
    """
    def __init__(self, directory, type="earth_model"):
        self.directory = directory
        self.boxfile = os.path.join(self.directory, "boxfile")
        if not os.path.exists(self.boxfile):
            msg = "boxfile not found in folder. Wrong directory?"
            raise ValueError(msg)

        # Read the boxfile.
        self.setup = self._parse_boxfile()

        self.rotation_axis = None
        self.rotation_angle_in_degree = None

        # Now check what different models are available in the directory. This
        # information is also used to infer the degree of the used lagrange
        # polynomial.
        components = ["A", "B", "C", "lambda", "mu", "rhoinv", "Q"]
        self.components = {}
        self.parsed_components = {}
        for component in components:
            files = glob.glob(os.path.join(directory, "%s[0-9]*" % component))
            if len(files) != len(self.setup["subdomains"]):
                continue
            # Check that the naming is continuous.
            all_good = True
            for _i in xrange(len(self.setup["subdomains"])):
                if os.path.join(directory, "%s%i" % (component, _i)) in files:
                    continue
                all_good = False
                break
            if all_good is False:
                msg = "Naming for component %s is off. It will be skipped."
                warnings.warn(msg)
                continue
            # They also all need to have the same size.
            if len(set([os.path.getsize(_i) for _i in files])) != 1:
                msg = ("Component %s has the right number of model files but "
                    "they are not of equal size") % component
                warnings.warn(msg)
                continue
            # Sort the files by ascending number.
            files.sort(key=lambda x: int(re.findall(r"\d+$",
                (os.path.basename(x)))[0]))
            self.components[component] = {"filenames": files}
        # All files for a single component have the same size. Now check that
        # all files have the same size.
        unique_filesizes = len(list(set([os.path.getsize(_i["filenames"][0])
            for _i in self.components.itervalues()])))
        if unique_filesizes != 1:
            msg = ("The different components in the folder do not have the "
                "same number of samples")
            raise ValueError

        # Now calculate the lagrange polynomial degree. All necessary
        # information is present.
        size = os.path.getsize(self.components.values()[0]["filenames"][0])
        sd = self.setup["subdomains"][0]
        x, y, z = sd["index_x_count"], sd["index_y_count"], sd["index_z_count"]
        self.lagrange_polynomial_degree = \
            int(round(((size * 0.25) / (x * y * z)) ** (1.0 / 3.0) - 1))

        self._calculate_final_dimensions()

        self.available_derived_components = ["vp", "vsh", "vsv", "rho"]

    def _read_single_box(self, component, file_number):
        """
        This function reads Ses3ds raw binary files, e.g. 3d velocity field
        snapshots, as well as model parameter files or sensitivity kernels. It
        returns the field as an array of rank 3 with shape (nx*lpd+1, ny*lpd+1,
        nz*lpd+1), discarding the duplicates by default.

        Parameters:
        -----------
        par: dictionary
            Dictionary with parameters from read_par_file()
        file_name: str
            Filename of Ses3d 3d raw-output
        """
        # Get the file and the corresponding domain.
        filename = self.components[component]["filenames"][file_number]
        domain = self.setup["subdomains"][file_number]

        lpd = self.lagrange_polynomial_degree

        shape = (domain["index_x_count"], domain["index_y_count"],
            domain["index_z_count"], lpd + 1, lpd + 1, lpd + 1)

        # Take care: The first and last four bytes in the arrays are invalid.
        with open(filename, "rb") as open_file:
            field = np.ndarray(shape, buffer=open_file.read()[4:-4],
                dtype="float32", order="F")

        # Calculate the new shape by multiplying every dimension with the lpd +
        # 1 value for every dimension.
        new_shape = [_i * _j for _i, _j in zip(shape[:3], shape[3:])]

        # Reorder the axes:
        # v[x, y, z, lpd+1 ,lpd+1, lpd+1] -> v[x, lpd+1, y, z, lpd+1, lpd+1] ->
        #   v[x, lpd+1, y, lpd+1, z, lpd+1]
        field = np.rollaxis(np.rollaxis(field, 3, 1), 3, 5)

        # Reshape the data:
        # v[nx,lpd,ny,lpd,nz,lpd] to v[nx*(lpd+1),ny*(lpd+1),nz*(lpd+1)]
        # XXX: Check how this depends on C and Fortran memory layout.
        field = field.reshape(new_shape, order="C")

        # XXX: Attempt to do this in one step.
        for axis, count in enumerate(new_shape):
            # Mask the duplicate values.
            mask = np.ones(count, dtype="bool")
            mask[::(lpd + 1)] = False
            mask[0] = True
            # Remove them by compressing the array.
            field = field.compress(mask, axis=axis)

        return field

    def parse_component(self, component):
        # If a real component,
        if component in self.components.keys():
            self._parse_component(component)
            return
        elif component not in self.available_derived_components:
            msg = "Component %s is unknown" % component
            raise ValueError(msg)

        if component == "vp":
            self._parse_component("lambda")
            self._parse_component("mu")
            self._parse_component("rhoinv")
            lambda_ = self.parsed_components["lambda"]
            mu = self.parsed_components["mu"]
            rhoinv = self.parsed_components["rhoinv"]
            self.parsed_components["vp"] = \
                np.sqrt(((lambda_ + 2.0 * mu) * rhoinv)) / 1000.0
        elif component == "vsh":
            self._parse_component("mu")
            self._parse_component("rhoinv")
            mu = self.parsed_components["mu"]
            rhoinv = self.parsed_components["rhoinv"]
            self.parsed_components["vsh"] = np.sqrt((mu * rhoinv)) / 1000.0
        elif component == "vsv":
            self._parse_component("mu")
            self._parse_component("rhoinv")
            self._parse_component("B")
            mu = self.parsed_components["mu"]
            rhoinv = self.parsed_components["rhoinv"]
            b = self.parsed_components["B"]
            self.parsed_components["vsv"] = \
                np.sqrt((mu + b) * rhoinv) / 1000.0
        elif component == "rho":
            self._parse_component("rhoinv")
            rhoinv = self.parsed_components["rhoinv"]
            self.parsed_components["rho"] = 1.0 / (1000.0 * rhoinv)

    def _parse_component(self, component):
        """
        Parses the specified component.
        """
        if component in self.parsed_components:
            return
        # Allocate empty array with the necessary dimensions.
        data = np.empty((self.setup["point_count_in_x"],
            self.setup["point_count_in_y"], self.setup["point_count_in_z"]),
            dtype="float32")

        for _i, domain in enumerate(self.setup["subdomains"]):
            x_min, x_max = domain["boundaries_x"]
            y_min, y_max = domain["boundaries_y"]
            z_min, z_max = domain["boundaries_z"]
            x_min, x_max, y_min, y_max, z_min, z_max = [(_j + 1) * 4
                for _j in (x_min, x_max, y_min, y_max, z_min, z_max)]
            x_min, y_min, z_min = [_j - 4 for _j in (x_min, y_min, z_min)]
            subdata = self._read_single_box(component, _i)
            # Merge into data.
            # XXX: Whacky z-indexing...put some more thought into this.
            z1 = data.shape[2] - z_min
            z2 = data.shape[2] - z_max
            data[x_min:x_max + 1, y_min:y_max + 1, z2 - 1: z1] = subdata

        self.parsed_components[component] = data

    def _calculate_final_dimensions(self):
        """
        Calculates the total number of elements and points and also the size of
        the final array per component.
        """
        # Elements in each directions.
        x, y, z = (
            self.setup["boundaries_x"][1] - self.setup["boundaries_x"][0] + 1,
            self.setup["boundaries_y"][1] - self.setup["boundaries_y"][0] + 1,
            self.setup["boundaries_z"][1] - self.setup["boundaries_z"][0] + 1)
        self.setup["total_element_count"] = x * y * z

        self.setup["point_count_in_x"] = \
            (x * self. lagrange_polynomial_degree + 1)
        self.setup["point_count_in_y"] = \
            (y * self. lagrange_polynomial_degree + 1)
        self.setup["point_count_in_z"] = \
            (z * self. lagrange_polynomial_degree + 1)
        self.setup["total_point_count"] = (
            self.setup["point_count_in_x"] *
            self.setup["point_count_in_y"] *
            self.setup["point_count_in_z"])

    def plot_depth_slice(self, component, depth_in_km):
        """
        Plots a depth slice.
        """
        lat_bounds = [rotations.colat2lat(_i)
            for _i in self.setup["physical_boundaries_x"][::-1]]
        lng_bounds = self.setup["physical_boundaries_y"]
        depth_bounds = [6371 - _i / 1000
            for _i in self.setup["physical_boundaries_z"]]

        data = self.parsed_components[component]

        available_depths = np.linspace(*depth_bounds, num=data.shape[2])[::-1]
        depth_index = np.argmin(np.abs(available_depths - depth_in_km))

        lon, lat = np.meshgrid(
            np.linspace(*lng_bounds, num=data.shape[1]),
            np.linspace(*lat_bounds, num=data.shape[0]))
        if self.rotation_axis and self.rotation_angle_in_degree:
            lon_shape = lon.shape
            lat_shape = lat.shape
            lon.shape = lon.size
            lat.shape = lat.size
            lat, lon = rotations.rotate_lat_lon(lat, lon, self.rotation_axis,
                self.rotation_angle_in_degree)
            lon.shape = lon_shape
            lat.shape = lat_shape

        # Get the center of the map.
        lon_0 = lon.min() + lon.ptp() / 2.0
        lat_0 = lat.min() + lat.ptp() / 2.0

        plt.figure(0)

        # Attempt to zoom into the region of interest.
        max_extend = max(lon.ptp(), lat.ptp())
        extend_used = max_extend / 180.0
        if extend_used < 0.5:
            x_buffer = 0.2 * lon.ptp()
            y_buffer = 0.2 * lat.ptp()

            m = Basemap(projection='merc', resolution="l",
                #lat_0=lat_0, lon_0=lon_0,
                llcrnrlon=lon.min() - x_buffer,
                urcrnrlon=lon.max() + x_buffer,
                llcrnrlat=lat.min() - y_buffer,
                urcrnrlat=lat.max() + y_buffer)
        else:
            m = Basemap(projection='ortho', lon_0=lon_0, lat_0=lat_0,
                resolution="c")


        m.drawcoastlines()
        m.fillcontinents("0.9", zorder=0)
        m.drawmapboundary(fill_color="white")
        m.drawparallels(np.arange(-80.0, 80.0, 10.0), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-170.0, 170.0, 10.0), labels=[0, 0, 0, 1])
        m.drawcountries()

        x, y = m(lon, lat)
        im = m.pcolormesh(x, y, data[::-1, :, depth_index],
            cmap=tomo_colormap)

        # Add colorbar and potentially unit.
        cm = m.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cm.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)

        plt.suptitle("Depth slice of %s at %i km" % (component,
            int(depth_in_km)), size="large")

        def _on_button_press(event):
            if event.button != 1 or not event.inaxes:
                return
            lon, lat = m(event.xdata, event.ydata, inverse=True)
            # Convert to colat to ease indexing.
            colat = rotations.lat2colat(lat)

            x_range = (self.setup["physical_boundaries_x"][1] -
                self.setup["physical_boundaries_x"][0])
            x_frac = (colat - self.setup["physical_boundaries_x"][0]) / x_range
            x_index = int(((self.setup["boundaries_x"][1] -
                self.setup["boundaries_x"][0]) * x_frac) +
                self.setup["boundaries_x"][0])
            y_range = (self.setup["physical_boundaries_y"][1] -
                self.setup["physical_boundaries_y"][0])
            y_frac = (lon - self.setup["physical_boundaries_y"][0]) / y_range
            y_index = int(((self.setup["boundaries_y"][1] -
                self.setup["boundaries_y"][0]) * y_frac) +
                self.setup["boundaries_y"][0])

            plt.figure(1, figsize=(3, 8))
            depths = available_depths
            values = data[x_index, y_index, :]
            plt.plot(values, depths)
            plt.grid()
            plt.ylim(depths[-1], depths[0])
            plt.show()
            plt.close()
            plt.figure(0)

        plt.gcf().canvas.mpl_connect('button_press_event', _on_button_press)

        plt.show()

    def __str__(self):
        """
        Eases interactive working.
        """
        ret_str = "Raw SES3D Model (split in %i parts)\n" % \
            (self.setup["total_cpu_count"])
        ret_str += "\tSetup:\n"
        ret_str += "\t\tLatitude: {:.2f} - {:.2f}\n".format(
            *[rotations.colat2lat(_i)
            for _i in self.setup["physical_boundaries_x"][::-1]])
        ret_str += "\t\tLongitude: %.2f - %.2f\n" % \
            self.setup["physical_boundaries_y"]
        ret_str += "\t\tDepth in km: {:.2f} - {:.2f}\n".format(
            *[6371 - _i / 1000
            for _i in self.setup["physical_boundaries_z"][::-1]])
        ret_str += "\t\tTotal element count: %i\n" % \
            self.setup["total_element_count"]
        ret_str += "\t\tTotal grid point count: %i\n" % \
            self.setup["total_point_count"]
        ret_str += "\tMemory requirement per component: %.1f MB\n" % \
            ((self.setup["total_point_count"] * 4) / (1024.0 ** 2))
        ret_str += "\tAvailable components: %s\n" % (", ".join(
            sorted(self.components.keys())))
        ret_str += "\tAvailable derived components: %s\n" % (", ".join(
            sorted(self.available_derived_components)))
        ret_str += "\tParsed components: %s" % (", ".join(
            sorted(self.parsed_components.keys())))
        return ret_str

    def _parse_boxfile(self):
        setup = {"subdomains": []}
        with open(self.boxfile, "rt") as open_file:
            # The first 14 lines denote the header
            lines = open_file.readlines()[14:]
            # Strip lines and remove empty lines.
            lines = [_i.strip() for _i in lines if _i.strip()]

            # The next 4 are the global CPU distribution.
            setup["total_cpu_count"] = int(lines.pop(0))
            setup["cpu_count_in_x_direction"] = int(lines.pop(0))
            setup["cpu_count_in_y_direction"] = int(lines.pop(0))
            setup["cpu_count_in_z_direction"] = int(lines.pop(0))

            if set(lines[0]) == set("-"):
                lines.pop(0)
            # Small sanity check.
            if setup["total_cpu_count"] != setup["cpu_count_in_x_direction"] *\
                    setup["cpu_count_in_y_direction"] * \
                    setup["cpu_count_in_z_direction"]:
                msg = ("Invalid boxfile. Total and individual processor "
                    "counts do not match.")
                raise ValueError(msg)

            # Now parse the rest of file which contains the subdomains.
            def subdomain_generator(data):
                while data:
                    subdom = {}
                    # Convert both indices to 0-based indices
                    subdom["single_index"] = int(data.pop(0)) - 1
                    subdom["multi_index"] = map(lambda x: int(x) - 1,
                        data.pop(0).split())
                    subdom["boundaries_x"] = map(int, data.pop(0).split())
                    subdom["boundaries_y"] = map(int, data.pop(0).split())
                    subdom["boundaries_z"] = map(int, data.pop(0).split())
                    # Convert radians to degree.
                    subdom["physical_boundaries_x"] = map(
                        lambda x: math.degrees(float(x)), data.pop(0).split())
                    subdom["physical_boundaries_y"] = map(
                        lambda x: math.degrees(float(x)), data.pop(0).split())
                    # z is in meter.
                    subdom["physical_boundaries_z"] = map(float,
                        data.pop(0).split())
                    for component in ("x", "y", "z"):
                        idx = "boundaries_%s" % component
                        subdom["index_%s_count" % component] = \
                            subdom[idx][1] - subdom[idx][0] + 1
                    # Remove seperator_line if existant.
                    if set(lines[0]) == set("-"):
                        lines.pop(0)
                    yield subdom
            # Sort them after with the single index.
            setup["subdomains"] = sorted(list(subdomain_generator(lines)),
                key=lambda x: x["single_index"])
            # Do some more sanity checks.
            if len(setup["subdomains"]) != setup["total_cpu_count"]:
                msg = ("Invalid boxfile. Number of processors and subdomains "
                    "to not match.")
                raise ValueError(msg)
            for component in ("x", "y", "z"):
                idx = "index_%s_count" % component
                if len(set([_i[idx] for _i in setup["subdomains"]])) != 1:
                    msg = ("Invalid boxfile. Unequal %s index count across "
                        "subdomains.") % component
                    raise ValueError(msg)

            # Now generate the absolute indices for the whole domains.
            for component in ("x", "y", "z"):
                setup["boundaries_%s" % component] = (min([_i["boundaries_%s" %
                    component][0] for _i in setup["subdomains"]]),
                    max([_i["boundaries_%s" %
                    component][1] for _i in setup["subdomains"]]))
                setup["physical_boundaries_%s" % component] = \
                    (min([_i["physical_boundaries_%s" % component][0] for
                        _i in setup["subdomains"]]),
                    max([_i["physical_boundaries_%s" % component][1] for _i in
                        setup["subdomains"]]))

            return setup


def get_lpd_sampling_points(lpd):
    """
    Returns the sampling points for to the n-th degree lagrange polynomial in
    an interval between -1 and 1.

    Parameters:
    -----------
    lpd : int
        Lagrange polynomial degree (between 2 to 7)
    """
    if lpd == 2:
        knots = np.array([-1.0, 0.0, 1.0])
    elif lpd == 3:
        knots = np.array([-1.0, -0.4472135954999579, 0.4472135954999579, 1.0])
    elif lpd == 4:
        knots = np.array([-1.0, -0.6546536707079772, 0.0,
           0.6546536707079772, 1.0])
    elif lpd == 5:
        knots = np.array([-1.0, -0.7650553239294647, -0.2852315164806451,
           0.2852315164806451, 0.7650553239294647, 1.0])
    elif lpd == 6:
        knots = np.array([-1.0, -0.8302238962785670, -0.4688487934707142,
           0.0, 0.4688487934707142, 0.8302238962785670, 1.0])
    elif lpd == 7:
        knots = np.array([-1.0, -0.8717401485096066, -0.5917001814331423,
            -0.2092992179024789, 0.2092992179024789, 0.5917001814331423,
            0.8717401485096066, 1.0])
    return knots


def _read_blockfile(path):
    """
    Helper function reading a single block file and returning a list of
    np.arrays, each containing one subdomain.

    A blockfile is a custom ASCII file format with the following specification:

    2       <-- Total number of subdomains
    10      <-- Number of entries in the first subdomain
    ...     <-- 10 lines of data for the first subdomain
    12      <-- Number of entries in the second subdomain
    ...     <- 12 lines of data for the second subdomain

    and so on.

    :param path: Path to the blockfile.
    """
    raw_data = np.loadtxt(path, dtype="float32")
    number_of_subdomains = int(raw_data[0])

    def subdomain_generator(data):
        """
        Simple generator yielding one subdomain at a time.
        """
        while len(data):
            npts = int(data[0])
            yield data[1:npts + 1]
            data = data[npts + 1:]

    contents = list(subdomain_generator(raw_data[1:]))

    # Sanity check.
    if len(contents) != number_of_subdomains:
        msg = ("Number of subdomains in blockfile (%i) does not correspond to "
            "the number specified in the header (%i)." % (len(contents),
            number_of_subdomains))
        warnings.warn(msg)

    return contents


class SES3D_Model(object):
    """
    Class for reading, writing, plotting and manipulating a SES3D model
    """
    def __init__(self, blockfile_directory, model_filename, rotation_axis=None,
            rotation_angle_in_degree=None):
        """
        Initiates a list of Submodels
        """
        self.blockfile_directory = blockfile_directory
        self.model_filename = model_filename

        self.subvolumes = []
        self.read()

    def __str__(self):
        """
        Pretty print some information about the model.
        """
        ret = "Contains %i subvolume%s:\n" % (len(self.subvolumes), "s" if
            len(self.subvolumes) > 1 else "")
        for subvol in self.subvolumes:
            ret += "\tLat: %.2f-%.2f | Lng: %.2f-%.2f | Dep: %.1f-%.1f\n" \
                % (subvol["latitudes"][-1], subvol["latitudes"][0],
                subvol["longitudes"][0], subvol["longitudes"][-1],
                subvol["depths_in_km"][0], subvol["depths_in_km"][-1])
        return ret

    def read(self):
        """
        Reads a SES3D model from a file.
        """
        # Read all block files
        folder = self.blockfile_directory
        blockfile_x = _read_blockfile(os.path.join(folder, "block_x"))
        blockfile_y = _read_blockfile(os.path.join(folder, "block_y"))
        blockfile_z = _read_blockfile(os.path.join(folder, "block_z"))
        blockfiles = (blockfile_x, blockfile_y, blockfile_z)

        # Sanity checking that they all have the same number of subdomains.
        unique_subdomain_counts = len(set([len(_i) for _i in blockfiles]))
        if unique_subdomain_counts != 1:
            msg = ("Invalid blockfiles. They do not contain equal amounts of "
                "subdomains.")
            raise ValueError(msg)

        # Create the subvolume. Currently uses dictionaries.
        for x, y, z in zip(*blockfiles):
            self.subvolumes.append({
                "latitudes": np.array(map(rotations.colat2lat, x)),
                "longitudes": y,
                "depths_in_km": z})

        #- read model volume
        with open(os.path.join(directory, filename), "rb") as open_file:

            v=np.array(open_file.read().strip().splitlines(), dtype=float)

        #- assign values ======================================================
        idx=1
        for k in np.arange(self.nsubvol):
            n=int(v[idx])
            nx=len(self.m[k].lat)-1
            ny=len(self.m[k].lon)-1
            nz=len(self.m[k].r)-1

            self.m[k].v=v[(idx+1):(idx+1+n)].reshape(nx,ny,nz)

            idx=idx+n+1