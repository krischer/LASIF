#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of classes dealing with SES3D 4.1 Models

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de),
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch),
    Stefan Mauerberger (mauerberger@geophysik.uni-muenchen.de), 2012 - 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import collections
import glob
import math
import os
import re
import warnings

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

from lasif import rotations


# Pretty units for some components.
UNIT_DICT = {
    "vp": r"$\frac{\mathrm{km}}{\mathrm{s}}$",
    "vsv": r"$\frac{\mathrm{km}}{\mathrm{s}}$",
    "vsh": r"$\frac{\mathrm{km}}{\mathrm{s}}$",
    "rho": r"$\frac{\mathrm{kg}}{\mathrm{m}^3}$",
    "rhoinv": r"$\frac{\mathrm{m}^3}{\mathrm{kg}}$",
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
    0.0: [0.1, 0.0, 0.0],     # Reddish black
    0.2: [0.8, 0.0, 0.0],
    0.3: [1.0, 0.7, 0.0],
    0.48: [0.92, 0.92, 0.92],
    0.5: [0.92, 0.92, 0.92],  # Light gray
    0.52: [0.92, 0.92, 0.92],
    0.7: [0.0, 0.6, 0.7],
    0.8: [0.0, 0.0, 0.8],
    1.0: [0.0, 0.0, 0.1]},
    "seismic_tomography")     # Blueish black


class RawSES3DModelHandler(object):
    """
    Class able to deal with a directory of raw SES3D 4.0 models.

    An earth model directory is defined by containing:
        * boxfile              -> File describing the discretization
        * A0-A[XX]             -> Anisotropic parameters (one per CPU)
        * B0-B[XX]             -> Anisotropic parameters (one per CPU)
        * C0-C[XX]             -> Anisotropic parameters (one per CPU)
        * lambda0 - lambda[XX] -> Elastic parameters (one per CPU)
        * mu0 - mu[XX]         -> Elastic parameters (one per CPU)
        * rhoinv0 - rhoinv[XX] -> Elastic parameters (one per CPU)
        * Q0 - Q[XX]           -> Elastic parameters (one per CPU)

    A kernel directory is defined by containing:
        * boxfile              -> File describing the discretization
        * grad_cp_[XX]_[XXXX]
        * grad_csh_[XX]_[XXXX]
        * grad_csv_[XX]_[XXXX]
        * grad_rho_[XX]_[XXXX]

    A wavefield directory contains the following files:
        * boxfile              -> File describing the discretization
        * vx_[xx]_[timestep]
        * vy_[xx]_[timestep]
        * vz_[xx]_[timestep]
    """
    def __init__(self, directory, model_type="earth_model"):
        """
        The init function.

        :param directory: The directory where the earth model or kernel is
            located.
        :param model_type: Determined the type of model loaded. Currently
            two are supported:
                * earth_model - The standard SES3D model files (default)
                * kernel - The kernels. Identifies by lots of grad_* files.
                * wavefield - The raw wavefields.
        """
        self.directory = directory
        self.boxfile = os.path.join(self.directory, "boxfile")
        if not os.path.exists(self.boxfile):
            msg = "boxfile not found. Wrong directory?"
            raise ValueError(msg)

        # Read the boxfile.
        self.setup = self._read_boxfile()

        self.rotation_axis = None
        self.rotation_angle_in_degree = None

        self.model_type = model_type

        if model_type == "earth_model":
            # Now check what different models are available in the directory.
            # This information is also used to infer the degree of the used
            # lagrange polynomial.
            components = ["A", "B", "C", "lambda", "mu", "rhoinv", "Q"]
            self.available_derived_components = ["vp", "vsh", "vsv", "rho"]
            self.components = {}
            self.parsed_components = {}
            for component in components:
                files = glob.glob(
                    os.path.join(directory, "%s[0-9]*" % component))
                if len(files) != len(self.setup["subdomains"]):
                    continue
                # Check that the naming is continuous.
                all_good = True
                for _i in xrange(len(self.setup["subdomains"])):
                    if os.path.join(directory,
                                    "%s%i" % (component, _i)) in files:
                        continue
                    all_good = False
                    break
                if all_good is False:
                    msg = "Naming for component %s is off. It will be skipped."
                    warnings.warn(msg)
                    continue
                # They also all need to have the same size.
                if len(set([os.path.getsize(_i) for _i in files])) != 1:
                    msg = ("Component %s has the right number of model files "
                           "but they are not of equal size") % component
                    warnings.warn(msg)
                    continue
                # Sort the files by ascending number.
                files.sort(key=lambda x: int(re.findall(r"\d+$",
                                             (os.path.basename(x)))[0]))
                self.components[component] = {"filenames": files}
        elif model_type == "wavefield":
            components = ["vz", "vx", "vy", "vz"]
            self.available_derived_components = []
            self.components = {}
            self.parsed_components = {}
            for component in components:
                files = glob.glob(os.path.join(directory, "%s_*_*" %
                                  component))
                if not files:
                    continue
                timesteps = collections.defaultdict(list)
                for filename in files:
                    timestep = int(os.path.basename(filename).split("_")[-1])
                    timesteps[timestep].append(filename)

                for timestep, filenames in timesteps.iteritems():
                    self.components["%s %s" % (component, timestep)] = \
                        {"filenames": sorted(
                            filenames,
                            key=lambda x: int(
                                os.path.basename(x).split("_")[1]))}
        elif model_type == "kernel":
            # Now check what different models are available in the directory.
            # This information is also used to infer the degree of the used
            # lagrange polynomial.
            components = ["grad_cp", "grad_csh", "grad_csv", "grad_rho"]
            self.available_derived_components = []
            self.components = {}
            self.parsed_components = {}
            for component in components:
                files = glob.glob(
                    os.path.join(directory, "%s_[0-9]*_[0-9]*" % component))
                length = int(files[0][files[0].rfind("_") + 1:])
                if len(files) != len(self.setup["subdomains"]):
                    continue
                    # Check that the naming is continuous.
                all_good = True
                for _i in xrange(len(self.setup["subdomains"])):
                    if os.path.join(directory, "%s_%i_%i" % (
                            component, _i, length)) in files:
                        continue
                    all_good = False
                    break
                if all_good is False:
                    msg = "Naming for component %s is off. It will be skipped."
                    warnings.warn(msg)
                    continue
                    # They also all need to have the same size.
                if len(set([os.path.getsize(_i) for _i in files])) != 1:
                    msg = ("Component %s has the right number of model files "
                           "but they are not of equal size") % component
                    warnings.warn(msg)
                    continue
                    # Sort the files by ascending number.
                files.sort(key=lambda x: int(
                    re.findall(r"\d+$",
                               (os.path.basename(x)))[0]))
                self.components[component] = {"filenames": files}
        else:
            msg = "model_type '%s' not known." % model_type
            raise ValueError(msg)

        # All files for a single component have the same size. Now check that
        # all files have the same size.
        unique_filesizes = len(list(set([
            os.path.getsize(_i["filenames"][0])
            for _i in self.components.itervalues()])))
        if unique_filesizes != 1:
            msg = ("The different components in the folder do not have the "
                   "same number of samples")
            raise ValueError(msg)

        # Now calculate the lagrange polynomial degree. All necessary
        # information is present.
        size = os.path.getsize(self.components.values()[0]["filenames"][0])
        sd = self.setup["subdomains"][0]
        x, y, z = sd["index_x_count"], sd["index_y_count"], sd["index_z_count"]
        self.lagrange_polynomial_degree = \
            int(round(((size * 0.25) / (x * y * z)) ** (1.0 / 3.0) - 1))

        self._calculate_final_dimensions()

    def _read_single_box(self, component, file_number):
        """
        This function reads Ses3ds raw binary files, e.g. 3d velocity field
        snapshots, as well as model parameter files or sensitivity kernels. It
        returns the field as an array of rank 3 with shape (nx*lpd+1, ny*lpd+1,
        nz*lpd+1), discarding the duplicates by default.
        """
        # Get the file and the corresponding domain.
        filename = self.components[component]["filenames"][file_number]
        domain = self.setup["subdomains"][file_number]

        lpd = self.lagrange_polynomial_degree

        shape = (domain["index_x_count"], domain["index_y_count"],
                 domain["index_z_count"], lpd + 1, lpd + 1, lpd + 1)

        # Take care: The first and last four bytes in the arrays are invalid
        #  due to them being written by Fortran.
        with open(filename, "rb") as open_file:
            field = np.ndarray(shape, buffer=open_file.read()[4:-4],
                               dtype="float32", order="F")
        # field = np.require(field, requirements="C")

        # Calculate the new shape by multiplying every dimension with lpd + 1
        # value for every dimension.
        new_shape = [_i * _j for _i, _j in zip(shape[:3], shape[3:])]

        # Reorder the axes:
        # v[x, y, z, lpd + 1 ,lpd + 1, lpd + 1] -> v[x, lpd + 1, y, z,
        # lpd + 1, lpd + 1] -> v[x, lpd + 1, y, lpd + 1, z, lpd + 1]
        field = np.rollaxis(np.rollaxis(field, 3, 1), 3, lpd + 1)

        # Reshape the data:
        # v[nx,lpd+1,ny,lpd+1,nz,lpd+1] to v[nx*(lpd+1),ny*(lpd+1),nz*(lpd+1)]
        field = field.reshape(new_shape, order="C")

        # XXX: Attempt to do this in one step.
        for axis, count in enumerate(new_shape):
            # Mask the duplicate values.
            mask = np.ones(count, dtype="bool")
            mask[::lpd + 1][1:] = False
            # Remove them by compressing the array.
            field = field.compress(mask, axis=axis)

        return field[:, :, ::-1]

    def parse_component(self, component):
        """
        Helper function parsing a whole component.

        :param component: The component name.
        :type component: basestring
        """
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
            self.parsed_components["rho"] = 1.0 / rhoinv

    def _parse_component(self, component):
        """
        Parses the specified component.
        """
        if component in self.parsed_components:
            return
        # Allocate empty array with the necessary dimensions.
        data = np.empty((
            self.setup["point_count_in_x"],
            self.setup["point_count_in_y"], self.setup["point_count_in_z"]),
            dtype="float32")

        for _i, domain in enumerate(self.setup["subdomains"]):
            x_min, x_max = domain["boundaries_x"]
            y_min, y_max = domain["boundaries_y"]
            z_min, z_max = domain["boundaries_z"]

            # Minimum indices
            x_min, y_min, z_min = [self.lagrange_polynomial_degree * _j
                                   for _j in (x_min, y_min, z_min)]
            # Maximum indices
            x_max, y_max, z_max = [self.lagrange_polynomial_degree * (_j + 1)
                                   for _j in (x_max, y_max, z_max)]

            # Merge into data.
            data[x_min: x_max + 1, y_min: y_max + 1, z_min: z_max + 1] = \
                self._read_single_box(component, _i)

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
        self.setup["total_point_count"] = (x * y * z) * \
            (self.lagrange_polynomial_degree + 1) ** 3

        # This is the actual point count with removed duplicates. Much
        # smaller because each element shared a large number of elements
        # with its neighbours.
        self.setup["point_count_in_x"] = \
            (x * self.lagrange_polynomial_degree + 1)
        self.setup["point_count_in_y"] = \
            (y * self.lagrange_polynomial_degree + 1)
        self.setup["point_count_in_z"] = \
            (z * self.lagrange_polynomial_degree + 1)
        self.setup["total_point_count_without_duplicates"] = (
            self.setup["point_count_in_x"] *
            self.setup["point_count_in_y"] *
            self.setup["point_count_in_z"])

    def _get_collocation_points_along_axis(self, min_value, max_value, count):
        """
        Calculates count collocation points from min_value to max_value as
        they would be used by a simulation. The border elements are not
        repeated.

        :param min_value: The minimum value.
        :param max_value: The maximum value.
        :param count: The number of values.
        """
        # Normalize from 0.0 to 1.0
        coll_points = get_lpd_sampling_points(self.lagrange_polynomial_degree)
        coll_points += 1.0
        coll_points /= 2.0

        # Some view and reshaping tricks to ensure performance. Probably not
        # needed.
        points = np.empty(count)
        lpd = self.lagrange_polynomial_degree
        p_view = points[:count - 1].view().reshape(((count - 1) / lpd, lpd))
        for _i in xrange(p_view.shape[0]):
            p_view[_i] = _i
        points[-1] = points[-2] + 1
        for _i, value in enumerate(coll_points[1:-1]):
            points[_i + 1::lpd] += value

        # Normalize from min_value to max_value.
        points /= points.max()
        points *= abs(max_value - min_value)
        points += min_value
        return points

    def plot_depth_slice(self, component, depth_in_km):
        """
        Plots a depth slice.

        :param component: The component to plot.
        :type component: basestring
        :param depth_in_km: The depth in km to plot. If the exact depth does
             not exists, the nearest neighbour will be plotted.
        :type depth_in_km: integer or float
        """
        lat_bounds = [rotations.colat2lat(_i)
                      for _i in self.setup["physical_boundaries_x"][::-1]]
        lng_bounds = self.setup["physical_boundaries_y"]
        depth_bounds = [6371 - _i / 1000.0
                        for _i in self.setup["physical_boundaries_z"]]

        data = self.parsed_components[component]

        available_depths = np.linspace(*depth_bounds, num=data.shape[2])
        depth_index = np.argmin(np.abs(available_depths - depth_in_km))
        actual_depth = available_depths[depth_index]

        lngs = self._get_collocation_points_along_axis(
            lng_bounds[0], lng_bounds[1], data.shape[1])
        lats = self._get_collocation_points_along_axis(
            lat_bounds[0], lat_bounds[1], data.shape[0])

        lon, lat = np.meshgrid(lngs, lats)
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

            # Calculate approximate width and height in meters.
            width = lon.ptp()
            height = lat.ptp()
            width *= 110000 * 1.1
            height *= 110000 * 1.1
            # Lambert azimuthal equal area projection. Equal area projections
            # are useful for interpreting features and this particular one also
            # does not distort features a lot on regional scales.
            m = Basemap(projection='laea', resolution="l",
                        width=width, height=height,
                        lat_0=lat_0, lon_0=lon_0)
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
        depth_data = data[::-1, :, depth_index]
        vmin, vmax = depth_data.min(), depth_data.max()
        vmedian = np.median(depth_data)
        offset = max(abs(vmax - vmedian), abs(vmedian - vmin))
        if offset <= 0.01:
            offset = 0.01

        vmin = vmedian - offset
        vmax = vmedian + offset

        im = m.pcolormesh(x, y, depth_data, cmap=tomo_colormap, vmin=vmin,
                          vmax=vmax)

        # Add colorbar and potentially unit.
        cm = m.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cm.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)

        plt.suptitle("Depth slice of %s at %i km" % (
            component, int(round(actual_depth))), size="large")

        border = rotations.get_border_latlng_list(
            rotations.colat2lat(self.setup["physical_boundaries_x"][0]),
            rotations.colat2lat(self.setup["physical_boundaries_x"][1]),
            self.setup["physical_boundaries_y"][0],
            self.setup["physical_boundaries_y"][1],
            rotation_axis=self.rotation_axis,
            rotation_angle_in_degree=self.rotation_angle_in_degree)
        border = np.array(border)
        lats = border[:, 0]
        lngs = border[:, 1]
        lngs, lats = m(lngs, lats)
        m.plot(lngs, lats, color="black", lw=2, path_effects=[
            PathEffects.withStroke(linewidth=4, foreground="white")])

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
        ret_str += "\t\tTotal collocation point count: %i (without " \
            "duplicates: %i)\n" % (
                self.setup["total_point_count"],
                self.setup["total_point_count_without_duplicates"])
        ret_str += "\tMemory requirement per component: %.1f MB\n" % \
            ((self.setup["total_point_count_without_duplicates"] * 4) /
                (1024.0 ** 2))
        ret_str += "\tAvailable components: %s\n" % (", ".join(
            sorted(self.components.keys())))
        ret_str += "\tAvailable derived components: %s\n" % (", ".join(
            sorted(self.available_derived_components)))
        ret_str += "\tParsed components: %s" % (", ".join(
            sorted(self.parsed_components.keys())))
        return ret_str

    def _read_boxfile(self):
        setup = {"subdomains": []}
        with open(self.boxfile, "rU") as fh:
            # The first 14 lines denote the header
            lines = fh.readlines()[14:]
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
                """
                Simple generator looping over each defined box and yielding
                a dictionary for each.

                :param data: The text.
                """
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
                    subdom["physical_boundaries_z"] = \
                        map(float, data.pop(0).split())
                    for component in ("x", "y", "z"):
                        idx = "boundaries_%s" % component
                        index_count = subdom[idx][1] - subdom[idx][0] + 1
                        subdom["index_%s_count" % component] = index_count
                        # The boxfiles are slightly awkward in that the indices
                        # are not really continuous. For example if one box
                        # has 22 as the last index, the first index of the next
                        # box will also be 22, even though it should be 23. The
                        # next snippet attempts to fix this deficiency.
                        offset = int(round(subdom[idx][0] /
                                     float(index_count - 1)))
                        subdom[idx][0] += offset
                        subdom[idx][1] += offset
                    # Remove separator_line if existent.
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
                setup["boundaries_%s" % component] = (
                    min([_i["boundaries_%s" % component][0]
                         for _i in setup["subdomains"]]),
                    max([_i["boundaries_%s" %
                         component][1] for _i in setup["subdomains"]]))
                setup["physical_boundaries_%s" % component] = (
                    min([_i["physical_boundaries_%s" % component][0] for
                         _i in setup["subdomains"]]),
                    max([_i["physical_boundaries_%s" % component][1] for _i in
                         setup["subdomains"]]))

            return setup


def get_lpd_sampling_points(lpd):
    """
    Returns the sampling points for to the n-th degree lagrange polynomial in
    an interval between -1 and 1.

    :param lpd: Lagrange polynomial degree (between 2 to 7)
    :type lpd: integer
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
        knots = np.array([
            -1.0, -0.8717401485096066, -0.5917001814331423,
            -0.2092992179024789, 0.2092992179024789, 0.5917001814331423,
            0.8717401485096066, 1.0])
    else:
        msg = "Invalid degree. It has to be between 2 and 7."
        raise ValueError(msg)
    return knots
