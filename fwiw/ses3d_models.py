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
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
#import colormaps as cm
import warnings

from fwiw import rotations


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
    def __init__(self, directory):
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

        # Setup the map.
        m = Basemap(projection='ortho', lon_0=lon_0, lat_0=lat_0,
            resolution="c")
        m.drawparallels(np.arange(-80.0, 80.0, 10.0))
        m.drawmeridians(np.arange(-170.0, 170.0, 10.0))
        m.drawcoastlines()
        m.drawmapboundary(fill_color="white")

        x, y = m(lon, lat)
        im = m.pcolormesh(x, y, data[::-1, :, depth_index],
            cmap=plt.cm.seismic_r)
        m.colorbar(im, "right", size="3%", pad='2%')
        plt.title("Depth slice of %s at %i km" % (component, int(depth_in_km)))

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

    #def __rmul__(self, factor):
        #"""
        #multiplication with a scalar
        #override left-multiplication of an ses3d model by a scalar factor
        #"""
        #res = ses3d_model()
        #res.nsubvol = self.nsubvol

        #for k in np.arange(self.nsubvol):
            #subvol = ses3d_submodel()
            #subvol.lat = self.m[k].lat
            #subvol.lon = self.m[k].lon
            #subvol.r = self.m[k].r
            #subvol.v = factor*self.m[k].v
            #res.m.append(subvol)
        #return res

    #def __add__(self, other_model):
        #"""
        #Add two models.

        #Override addition of two ses3d models
        #"""
        #res = ses3d_model()
        #res.nsubvol = self.nsubvol

        #for k in np.arange(self.nsubvol):
            #subvol = ses3d_submodel()
            #subvol.lat = self.m[k].lat
            #subvol.lon = self.m[k].lon
            #subvol.r = self.m[k].r
            #subvol.v = other_model.m[k].v+self.m[k].v
            #res.m.append(subvol)

        #return res


  #########################################################################
  #- write a 3D model to a file
  #########################################################################

  #def write(self,directory,filename,verbose=False):
    #"""
    #Write ses3d model to a file
    #"""

    #fid_m=open(directory+filename,'w')

    #if verbose==True:
      #print 'write to file '+directory+filename

    #fid_m.write(str(self.nsubvol)+'\n')

    #for k in np.arange(self.nsubvol):

      #nx=len(self.m[k].lat)-1
      #ny=len(self.m[k].lon)-1
      #nz=len(self.m[k].r)-1

      #fid_m.write(str(nx*ny*nz)+'\n')

      #for idx in np.arange(nx):
    #for idy in np.arange(ny):
      #for idz in np.arange(nz):

        #fid_m.write(str(self.m[k].v[idx,idy,idz])+'\n')

    #fid_m.close()

  ##########################################################################
  ##- CUt Depth LEvel
  ##########################################################################

  #def cudle(self,r_min,r_max,verbose=False):
    #"""
    #cut out a certain radius range, mostly in order to produce smaller models
    #for vtk
    #"""
    #m_new = ses3d_model()

    ## march through subvolumes
    #for n in np.arange(self.nsubvol):

      #if (np.min(self.m[n].r)<=r_min) & (np.max(self.m[n].r)>=r_max):

    #if verbose==True:
      #print 'subvolume '+str(n)+': r_min='+str(np.min(self.m[n].r))+' km, r_max='+str(np.max(self.m[n].r))+' km\n'

    #subvol=ses3d_submodel()
    #subvol.lat=self.m[n].lat
    #subvol.lon=self.m[n].lon

    #idr=(self.m[n].r>=r_min) & (self.m[n].r<=r_max)
    #subvol.r=self.m[n].r[idr]

    #idr=idr[1:(len(idr))]
    #subvol.v=self.m[n].v[:,:,idr]

    #m_new.m.append(subvol)
    #m_new.nsubvol=m_new.nsubvol+1

    #return m_new

  #########################################################################
  #- convert to vtk format
  #########################################################################

  #def convert_to_vtk(self,directory,filename,verbose=False):
    #""" convert ses3d model to vtk format for plotting with Paraview

    #convert_to_vtk(self,directory,filename,verbose=False):
    #"""

    ##- preparatory steps

    #nx=np.zeros(self.nsubvol,dtype=int)
    #ny=np.zeros(self.nsubvol,dtype=int)
    #nz=np.zeros(self.nsubvol,dtype=int)
    #N=0

    #for n in np.arange(self.nsubvol):
      #nx[n]=len(self.m[n].lat)
      #ny[n]=len(self.m[n].lon)
      #nz[n]=len(self.m[n].r)
      #N=N+nx[n]*ny[n]*nz[n]

    ##- open file and write header

    #fid=open(directory+filename,'w')

    #if verbose==True:
      #print 'write to file '+directory+filename

    #fid.write('# vtk DataFile Version 3.0\n')
    #fid.write('vtk output\n')
    #fid.write('ASCII\n')
    #fid.write('DATASET UNSTRUCTURED_GRID\n')

    ##- write grid points

    #fid.write('POINTS '+str(N)+' float\n')

    #for n in np.arange(self.nsubvol):

      #if verbose==True:
    #print 'writing grid points for subvolume '+str(n)

      #for i in np.arange(nx[n]):
    #for j in np.arange(ny[n]):
      #for k in np.arange(nz[n]):

        #theta=90.0-self.m[n].lat[i]
        #phi=self.m[n].lon[j]

        ##- rotate coordinate system

        #if self.phi!=0.0:
          #theta,phi=rot.rotate_coordinates(self.n,-self.phi,theta,phi)

        ##- transform to cartesian coordinates and write to file

        #theta=theta*np.pi/180.0
        #phi=phi*np.pi/180.0

        #r=self.m[n].r[k]
        #x=r*np.sin(theta)*np.cos(phi);
            #y=r*np.sin(theta)*np.sin(phi);
            #z=r*np.cos(theta);

        #fid.write(str(x)+' '+str(y)+' '+str(z)+'\n')

    ##- write connectivity

    #n_cells=0

    #for n in np.arange(self.nsubvol):
      #n_cells=n_cells+(nx[n]-1)*(ny[n]-1)*(nz[n]-1)

    #fid.write('\n')
    #fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n')

    #count=0

    #for n in np.arange(self.nsubvol):

      #if verbose==True:
    #print 'writing conectivity for subvolume '+str(n)

      #for i in np.arange(1,nx[n]):
    #for j in np.arange(1,ny[n]):
      #for k in np.arange(1,nz[n]):
                                ## i j k
        #a=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n]-1       # 0 0 0
        #b=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n]         # 0 0 1
        #c=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]-1         # 0 1 0
        #d=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]           # 0 1 1
        #e=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]-1         # 1 0 0
        #f=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]           # 1 0 1
        #g=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]-1           # 1 1 0
        #h=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]             # 1 1 1

        #fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')

      #count=count+nx[n]*ny[n]*nz[n]

    ##- write cell types

    #fid.write('\n')
    #fid.write('CELL_TYPES '+str(n_cells)+'\n')

    #for n in np.arange(self.nsubvol):

      #if verbose==True:
    #print 'writing cell types for subvolume '+str(n)

      #for i in np.arange(nx[n]-1):
    #for j in np.arange(ny[n]-1):
      #for k in np.arange(nz[n]-1):

        #fid.write('11\n')

    ##- write data

    #fid.write('\n')
    #fid.write('POINT_DATA '+str(N)+'\n')
    #fid.write('SCALARS scalars float\n')
    #fid.write('LOOKUP_TABLE mytable\n')

    #for n in np.arange(self.nsubvol):

      #if verbose==True:
    #print 'writing data for subvolume '+str(n)

      #idx=np.arange(nx[n])
      #idx[nx[n]-1]=nx[n]-2

      #idy=np.arange(ny[n])
      #idy[ny[n]-1]=ny[n]-2

      #idz=np.arange(nz[n])
      #idz[nz[n]-1]=nz[n]-2

      #for i in idx:
    #for j in idy:
      #for k in idz:

        #fid.write(str(self.m[n].v[i,j,k])+'\n')

    ##- clean up

    #fid.close()

  ##########################################################################
  ##- plot horizontal slices
  ##########################################################################

  #def plot_slice(self,depth,min_val_plot,max_val_plot,colormap='tomo',verbose=False):
    #""" plot horizontal slices through an ses3d model

    #plot_slice(depth,min_val_plot,max_val_plot,verbose=False):

    #depth=depth in km of the slice
    #min_val_plot, max_val_plot=minimum and maximum values of the colour scale
    #colormap='tomo','mono'
    #"""

    #radius=6371.0-depth

    ##- set up a map and colourmap

    #if global_regional=='regional':
      #m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
      #m.drawparallels(np.arange(lat_min,lat_max,d_lon),labels=[1,0,0,1])
      #m.drawmeridians(np.arange(lon_min,lon_max,d_lat),labels=[1,0,0,1])
    #elif global_regional=='global':
      #m=Basemap(projection='ortho',lon_0=lon_centre,lat_0=lat_centre,resolution=res)
      #m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
      #m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])

    #m.drawcoastlines()
    #m.drawcountries()

    #m.drawmapboundary(fill_color=[1.0,1.0,1.0])

    #if colormap=='tomo':
      #my_colormap=cm.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
    #elif colormap=='mono':
      #my_colormap=cm.make_colormap({0.0:[1.0,1.0,1.0], 0.15:[1.0,1.0,1.0], 0.85:[0.0,0.0,0.0], 1.0:[0.0,0.0,0.0]})

    ##- loop over subvolumes

    #for k in np.arange(self.nsubvol):

      #r=self.m[k].r

      ##- check if subvolume has values at target depth

      #if (max(r>=radius) & (min(r)<=radius)):
    #idz=min(np.where(min(np.abs(r-radius))==np.abs(r-radius))[0])

    #if verbose==True:
      #print 'true plotting depth: '+str(6371.0-r[idz])+' km'

    #nx=len(self.m[k].lat)
    #ny=len(self.m[k].lon)
    #nz=len(self.m[k].r)

    #lon,lat=np.meshgrid(self.m[k].lon[0:ny],self.m[k].lat[0:nx])

    ##- rotate coordinate system if necessary

    #if self.phi!=0.0:

      #lat_rot=np.zeros(np.shape(lon),dtype=float)
      #lon_rot=np.zeros(np.shape(lat),dtype=float)

      #for idx in np.arange(nx):
        #for idy in np.arange(ny):

          #colat=90.0-lat[idx,idy]

          #lat_rot[idx,idy],lon_rot[idx,idy]=rot.rotate_coordinates(self.n,-self.phi,colat,lon[idx,idy])
          #lat_rot[idx,idy]=90.0-lat_rot[idx,idy]

      #lon=lon_rot
      #lat=lat_rot

    ##- convert to map coordinates and plot

    #x,y=m(lon,lat)
    #im=m.pcolor(x,y,self.m[k].v[:,:,idz],cmap=my_colormap,vmin=min_val_plot,vmax=max_val_plot)

    #m.colorbar(im,"right", size="3%", pad='2%')
    #plt.title(str(depth)+' km')
    #plt.show()

  ##########################################################################
  ##- plot depth to a certain threshold value
  ##########################################################################

  #def plot_threshold(self,val,min_val_plot,max_val_plot,colormap='tomo',verbose=False):
    #""" plot depth to a certain threshold value 'val' in an ses3d model

    #plot_threshold(val,min_val_plot,max_val_plot,colormap='tomo',verbose=False):
    #val=threshold value
    #min_val_plot, max_val_plot=minimum and maximum values of the colour scale
    #colormap='tomo','mono'
    #"""

    ##- set up a map and colourmap

    #if global_regional=='regional':
      #m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
      #m.drawparallels(np.arange(lat_min,lat_max,d_lon),labels=[1,0,0,1])
      #m.drawmeridians(np.arange(lon_min,lon_max,d_lat),labels=[1,0,0,1])
    #elif global_regional=='global':
      #m=Basemap(projection='ortho',lon_0=lon_centre,lat_0=lat_centre,resolution=res)
      #m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
      #m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])

    #m.drawcoastlines()
    #m.drawcountries()

    #m.drawmapboundary(fill_color=[1.0,1.0,1.0])

    #if colormap=='tomo':
      #my_colormap=cm.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
    #elif colormap=='mono':
      #my_colormap=cm.make_colormap({0.0:[1.0,1.0,1.0], 0.15:[1.0,1.0,1.0], 0.85:[0.0,0.0,0.0], 1.0:[0.0,0.0,0.0]})

    ##- loop over subvolumes

    #for k in np.arange(self.nsubvol):

      #depth=np.zeros(np.shape(self.m[k].v[:,:,0]))

      #nx=len(self.m[k].lat)
      #ny=len(self.m[k].lon)

      ##- find depth

      #r=self.m[k].r
      #r=0.5*(r[0:len(r)-1]+r[1:len(r)])

      #for idx in np.arange(nx-1):
    #for idy in np.arange(ny-1):

      #n=self.m[k].v[idx,idy,:]>=val
      #depth[idx,idy]=6371.0-np.max(r[n])

      ##- rotate coordinate system if necessary

      #lon,lat=np.meshgrid(self.m[k].lon[0:ny],self.m[k].lat[0:nx])

      #if self.phi!=0.0:

    #lat_rot=np.zeros(np.shape(lon),dtype=float)
    #lon_rot=np.zeros(np.shape(lat),dtype=float)

    #for idx in np.arange(nx):
      #for idy in np.arange(ny):

        #colat=90.0-lat[idx,idy]

        #lat_rot[idx,idy],lon_rot[idx,idy]=rot.rotate_coordinates(self.n,-self.phi,colat,lon[idx,idy])
        #lat_rot[idx,idy]=90.0-lat_rot[idx,idy]

    #lon=lon_rot
    #lat=lat_rot

    ##- convert to map coordinates and plot

      #x,y=m(lon,lat)
      #im=m.pcolor(x,y,depth,cmap=my_colormap,vmin=min_val_plot,vmax=max_val_plot)

    #m.colorbar(im,"right", size="3%", pad='2%')
    #plt.title('depth to '+str(val)+' km/s [km]')
    #plt.show()
