#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A file info cache class. Must be subclassed.

The cache is able to monitor a bunch of (arbitrary) files and store some
indexes information about each file. This is useful for keeping track of a
large number of files and what is in them.

To subclass it you need to provide the values you want to index, the types of
files you want to index and functions to extract the indices and find the
files. Upon each call to the constructor it will check the existing database,
automatically remove any deleted files, reindex modified ones and add new ones.

This is much faster then reading the files every time but still provides a lot
of flexibility as the data can be managed by some other means.


Example implementation:


.. code-block:: python

    class ImageCache(object):
        def __init__(self, image_folder, cache_db_file):
            # The index values are a list of tuples. The first denotes the
            # name of the index and the second the type of the index. The types
            # have to correspond to SQLite types.
            self.index_values = [
                ("width", "INTEGER"),
                ("height", "INTEGER"),
                ("type", "TEXT")]
            # The types of files to index.
            self.filetypes = ["png", "jpeg"]

            # Subclass specific values
            self.image_folder = image_folder

            # Don't forget to call the parents __init__()!
            super(ImageCache, self).__init__(cache_db_file=cache_db_file)

        # Now you need to define one 'find files' and one 'index file'
        # methods for each filetype. The 'find files' method needs to be named
        # '_find_files_FILETYPE' and takes no arguments. The 'index file'
        # method has to be named '_extract_index_values_FILETYPE' and takes one
        # argument: the path to file. It needs to return a list of lists. Each
        # inner list contains the indexed values in the same order as specified
        # in self.index_values. It can return multiple sets of indices per
        # file. Useful for lots of filetypes, not necessarily images as in the
        # example here.

        def _find_files_png(self):
            return glob.glob(os.path.join("*.png"))

        def _find_files_jpeg(self):
            return glob.glob(os.path.join("*.png"))

        def _extract_index_values_png(self, filename):
            # Do somethings to get the values.
            return [[400, 300, "png"]]

        def _extract_index_values_jpeg(self, filename):
            # Do somethings to get the values.
            return [[400, 300, "jpeg"]]


:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

from binascii import crc32
from itertools import izip
import os
import progressbar
import sqlite3
import time
import warnings

from lasif import LASIFWarning


# Table definition of the 'files' table. Used for creating and validating
# the table.
FILES_TABLE_DEFINITION = (
    (u"id", u"INTEGER"),
    (u"filename", u"TEXT"),
    (u"filesize", u"INTEGER"),
    (u"last_modified", u"REAL"),
    (u"crc32_hash", u"INTEGER")
)


class FileInfoCache(object):
    """
    Object able to cache information about arbitrary files on the filesystem.

    Intended to be subclassed.
    """
    def __init__(self, cache_db_file, root_folder,
                 read_only, pretty_name, show_progress=True):
        self.cache_db_file = cache_db_file
        self.root_folder = root_folder
        self.read_only = read_only
        self.show_progress = show_progress
        self.pretty_name = pretty_name

        # Will be filled in _init_database() method.
        self.db_cursor = None
        self.db_conn = None

        # Will be filled once database has been updated.
        self.files = {}

        # Readonly mode disable updates and the creation of new databases.
        # This is useful for scenarios where multiple processes access the
        # same file cache database concurrently which otherwise can result
        # in locked databases. Of course the cache database must already be
        # built for that to work.
        if self.read_only is True:
            if not os.path.exists(self.cache_db_file):
                raise ValueError("Cache DB '%s' does not exists and cannot "
                                 "be created as it has been requested in "
                                 "read-only mode." % self.cache_db_file)
            # Open in read-only mode (seen in
            # http://stackoverflow.com/a/28302809/1657047)
            # I am not sure this actually works as it opens the file in
            # read-only mode but not necessarily the database. It will work
            # if SQLite3 notices the file pointer is in read-only mode and
            # adjusts accordingly which might very well be the case.
            self._fd = os.open(self.cache_db_file, os.O_RDONLY)
            self.db_conn = sqlite3.connect("/dev/fd/%d" % self._fd)
            self.db_cursor = self.db_conn.cursor()
            if self._validate_database() is not True:
                raise ValueError(
                    "Cache DB '%s' did not validate and cannot be created "
                    "anew as it has been requested in read-only mode." %
                    self.cache_db_file)
        elif self.read_only is False:
            self._init_database()
            self._update_indices()
            self.update()
        else:
            raise NotImplementedError

    def __del__(self):
        if self.db_conn:
            try:
                self.db_conn.close()
            except sqlite3.Error:
                pass
        if hasattr(self, "_fd"):
            try:
                os.close(self._fd)
            except sqlite3.Error:
                pass

    @property
    def file_count(self):
        """
        Returns number of files.
        """
        QUERY = "SELECT COUNT(*) FROM files;"
        return self.db_cursor.execute(QUERY).fetchone()[0]

    @property
    def index_count(self):
        """
        Returns number of indices.
        """
        QUERY = "SELECT COUNT(*) FROM indices;"
        return self.db_cursor.execute(QUERY).fetchone()[0]

    @property
    def total_size(self):
        """
        Returns the total file size in bytes.
        """
        QUERY = "SELECT SUM(filesize) FROM files;"
        return self.db_cursor.execute(QUERY).fetchone()[0]

    def _validate_database(self):
        """
        Validates the tables and database scheme.

        Useful for migrations in which the database is simply deleted and
        build anew.
        """
        query = "SELECT name FROM sqlite_master WHERE type = 'table';"
        tables = [_i[0] for _i in self.db_cursor.execute(query).fetchall()]
        if sorted(tables) != sorted(["files", "sqlite_sequence", "indices"]):
            return False

        # Check the indices table.
        i_t = self.db_cursor.execute("PRAGMA table_info(indices);").fetchall()
        i_t = [(_i[1], _i[2]) for _i in i_t]
        if i_t[1: -1] != self.index_values:
            return False

        # Check the files table.
        f_t = self.db_cursor.execute("PRAGMA table_info(files);").fetchall()
        f_t = tuple([(_i[1], _i[2]) for _i in f_t])
        if f_t != FILES_TABLE_DEFINITION:
            return False

        return True

    def _init_database(self):
        """
        Inits the database connects, turns on foreign key support and creates
        the tables if they do not already exist.
        """
        # Make sure the folder of the database file exists and otherwise
        # raise a descriptive error message.
        if not os.path.exists(os.path.dirname(self.cache_db_file)):
            raise ValueError(
                "The folder '%s' does not exist. Cannot create database in "
                "it." % os.path.dirname(self.cache_db_file))
        # Check if the file exists. If it exists, try to use it, otherwise
        # delete and create a new one. This should take care that a new
        # database is created in the case of DB corruption due to a power
        # failure.
        if os.path.exists(self.cache_db_file):
            try:
                self.db_conn = sqlite3.connect(self.cache_db_file)
                self.db_cursor = self.db_conn.cursor()
                # Make sure the database is still valid. This automatically
                # enables migrations to newer database schema definition in
                # newer LASIF versions.
                valid = self._validate_database()
                if valid is not True:
                    self.db_conn.close()
                    print("Cache '%s' is not valid anymore. This is most "
                          "likely due to some recent LASIF update. Don't "
                          "worry, LASIF will built it anew. Hang on..." %
                          self.cache_db_file)
                    try:
                        os.remove(self.cache_db_file)
                    except:
                        pass
                    self.db_conn = sqlite3.connect(self.cache_db_file)
                    self.db_cursor = self.db_conn.cursor()
            except sqlite3.Error:
                os.remove(self.cache_db_file)
                self.db_conn = sqlite3.connect(self.cache_db_file)
                self.db_cursor = self.db_conn.cursor()
        else:
            self.db_conn = sqlite3.connect(self.cache_db_file)
            self.db_cursor = self.db_conn.cursor()

        # Enable foreign key support.
        self.db_cursor.execute("PRAGMA foreign_keys = ON;")

        # Turn off synchronous writing. Much much faster inserts at the price
        # of risking corruption at power failure. Worth the risk as the
        # databases are just created from the data and can be recreated at any
        # time.
        self.db_cursor.execute("PRAGMA synchronous = OFF;")

        # This greatly speeds up deleting files. Data corruptions is again
        # not a really big issue.
        self.db_cursor.execute("PRAGMA SECURE_DELETE = OFF;")

        self.db_conn.commit()
        # Make sure that foreign key support has been turned on.
        if self.db_cursor.execute("PRAGMA foreign_keys;").fetchone()[0] != 1:
            try:
                self.db_conn.close()
            except sqlite3.Error:
                pass
            msg = ("Could not enable foreign key support for SQLite. Please "
                   "contact the LASIF developers.")
            raise ValueError(msg)

        # Create the tables.
        sql_create_files_table = """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                %s
            );
        """ % ",".join("%s %s" % (_i[0], _i[1]) for _i in
                       FILES_TABLE_DEFINITION[1:])
        self.db_cursor.execute(sql_create_files_table)

        sql_create_index_table = """
            CREATE TABLE IF NOT EXISTS indices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                %s,
                filepath_id INTEGER,
                FOREIGN KEY(filepath_id) REFERENCES files(id) ON DELETE CASCADE
            );
        """ % ",\n".join(["%s %s" % _i for _i in self.index_values])

        self.db_cursor.execute(sql_create_index_table)
        self.db_conn.commit()

    def _update_indices(self):
        if not hasattr(self, "indices") or not self.indices:
            return

        get_indices_query = """
            SELECT name FROM sqlite_master
            WHERE type='index';"""

        indices = [_i[0] for _i in
                   self.db_cursor.execute(get_indices_query).fetchall()]
        if indices == self.indices:
            return

        # Drop all indices no.
        for index in indices:
            if index in self.indices:
                continue
            query = "DROP INDEX %s;" % index
            self.db_conn.execute(query)
            self.db_conn.commit()

        for index in self.indices:
            if index in indices:
                continue
            query = "CREATE INDEX %s on indices(%s);" % (index, index)
            self.db_conn.execute(query)
            self.db_conn.commit()

    def _get_all_files_by_filename(self):
        """
        Find all files for all filetypes by filename.
        """
        self.files = {}
        for filetype in self.filetypes:
            get_file_fct = "_find_files_%s" % filetype
            # Paths are relative to the root folder.
            self.files[filetype] = [
                os.path.relpath(_i, self.root_folder) for _i in
                getattr(self, get_file_fct)()]

    def _get_all_files_from_database(self):
        """
        Find all files that actually have indexes by querying the database.

        This assumes self._get_all_files_by_filename() has already been
        called.
        """
        filenames = \
            self.db_cursor.execute("SELECT filename FROM files").fetchall()
        filenames = set(_i[0] for _i in filenames)

        # Filter to exclude all files not correctly indexed.
        for key, value in self.files.iteritems():
            self.files[key] = list(filenames.intersection(set(value)))

    def update(self):
        """
        Updates the database.
        """
        # Get all files first.
        self._get_all_files_by_filename()

        # Get all files currently in the database and reshape into a
        # dictionary. The dictionary key is the filename and the value a tuple
        # of (id, last_modified, crc32 hash).
        db_files = self.db_cursor.execute("SELECT * FROM files").fetchall()
        db_files = {_i[1]: (_i[0], _i[3], _i[4]) for _i in db_files}

        # Count all files
        filecount = 0
        for filetype in self.filetypes:
            filecount += len(self.files[filetype])

        # XXX: Check which files have the correct mtime outside the loop.
        # Then the case when the caches already exist should be much faster
        # (and the average case as well...)

        # Use a progressbar if the filecount is large so something appears on
        # screen.
        pbar = None
        update_interval = 1
        current_file_count = 0
        start_time = time.time()
        try:
            # Store the old working directory and change to the root folder
            # to get relative pathnames.
            org_directory = os.getcwd()
            os.chdir(self.root_folder)

            # Now update all filetypes separately.
            for filetype in self.filetypes:
                for filename in self.files[filetype]:
                    current_file_count += 1
                    # Only show the progressbar if more then 3.5 seconds have
                    # passed.
                    if not pbar and self.show_progress and \
                            (time.time() - start_time > 3.5):
                        widgets = [
                            "Updating %s: " % self.pretty_name,
                            progressbar.Percentage(),
                            progressbar.Bar(), "", progressbar.ETA()]
                        pbar = progressbar.ProgressBar(
                            widgets=widgets, maxval=filecount).start()
                        update_interval = max(int(filecount / 100), 1)
                        pbar.update(current_file_count)
                    if pbar and not current_file_count % update_interval:
                        pbar.update(current_file_count)
                    if filename in db_files:
                        # Delete the file from the list of files to keep
                        # track of files no longer available.
                        this_file = db_files[filename]
                        del db_files[filename]
                        abs_filename = os.path.abspath(filename)

                        last_modified = os.path.getmtime(abs_filename)
                        # If the last modified time is identical to a
                        # second, do nothing.
                        if abs(last_modified - this_file[1]) < 1.0:
                            continue
                        # Otherwise check the hash.
                        with open(abs_filename, "rb") as open_file:
                            hash_value = crc32(open_file.read())
                        if hash_value == this_file[2]:
                            # XXX: Update last modified times, otherwise it
                            # will hash again and again.
                            continue
                        self._update_file(abs_filename, filetype,
                                          this_file[0])
                    else:
                        self._update_file(os.path.abspath(filename),
                                          filetype)
        finally:
            os.chdir(org_directory)
        if pbar:
            pbar.finish()

        # Remove all files no longer part of the cache DB.
        if db_files:
            if len(db_files) > 100:
                print("Removing %i no longer existing files from the "
                      "cache database. This might take a while ..." %
                      len(db_files))
            query = "DELETE FROM files WHERE filename IN (%s);" % \
                ",".join(["'%s'" % _i for _i in db_files])
            self.db_cursor.execute(query)
        self.db_conn.commit()

        # Update the self.files dictionary, this time from the database.
        self._get_all_files_from_database()

    def get_values(self):
        """
        Returns a list of dictionaries containing all indexed values for every
        file together with the filename.
        """
        try:
            org_directory = os.getcwd()
            os.chdir(self.root_folder)

            # Assemble the query. Use a simple join statement.
            sql_query = """
            SELECT %s, files.filename
            FROM indices
            INNER JOIN files
            ON indices.filepath_id=files.id
            """ % ", ".join(["indices.%s" % _i[0] for _i in self.index_values])

            all_values = []
            indices = [_i[0] for _i in self.index_values]

            for _i in self.db_cursor.execute(sql_query):
                values = {key: value for (key, value) in izip(indices, _i)}
                values["filename"] = os.path.abspath(_i[-1])
                all_values.append(values)
        finally:
            os.chdir(org_directory)

        return all_values

    def get_details(self, filename):
        """
        Get the indexed information about one file.

        :param filename: The filename for which to request information.
        """
        filename = os.path.relpath(os.path.abspath(filename),
                                   self.root_folder)

        # Assemble the query. Use a simple join statement.
        sql_query = """
        SELECT %s, files.filename
        FROM indices
        INNER JOIN files
        ON indices.filepath_id=files.id
        WHERE files.filename='%s'
        """ % (", ".join(["indices.%s" % _i[0] for _i in self.index_values]),
               filename)

        all_values = []
        indices = [_i[0] for _i in self.index_values]

        for _i in self.db_cursor.execute(sql_query):
            values = {key: value for (key, value) in izip(indices, _i)}
            values["filename"] = os.path.abspath(os.path.join(
                self.root_folder, _i[-1]))
            all_values.append(values)

        return all_values

    def _update_file(self, filename, filetype, filepath_id=None):
        """
        Updates or creates a new entry for the given file. If id is given, it
        will be interpreted as an update, otherwise as a fresh record.
        """
        abs_filename = filename
        rel_filename = os.path.relpath(abs_filename, self.root_folder)
        # Remove all old indices for the file if it is an update.
        if filepath_id is not None:
            self.db_cursor.execute("DELETE FROM indices WHERE "
                                   "filepath_id = %i" % filepath_id)
            self.db_conn.commit()

        # Get all indices from the file.
        try:
            indices = getattr(self, "_extract_index_values_%s" %
                                    filetype)(abs_filename)
        except Exception as e:
            msg = "Failed to index '%s' of type '%s' due to: %s" % (
                abs_filename, filetype, str(e)
            )
            warnings.warn(msg, LASIFWarning)

            # If it is an update, also remove the file from the file list.
            if filepath_id is not None:
                self.db_cursor.execute(
                    "DELETE FROM files WHERE filename='%s';" % rel_filename)
                self.db_conn.commit()

            return

        if not indices:
            msg = ("Could not extract any index from file '%s' of type '%s'. "
                   "The file will be skipped." % (abs_filename, filetype))
            warnings.warn(msg, LASIFWarning)

            # If it is an update, also remove the file from the file list.
            if filepath_id is not None:
                self.db_cursor.execute(
                    "DELETE FROM files WHERE filename='%s';" % rel_filename)
                self.db_conn.commit()

            return

        # Get the hash
        with open(abs_filename, "rb") as open_file:
            filehash = crc32(open_file.read())

        # Add or update the file.
        if filepath_id is not None:
            self.db_cursor.execute(
                "UPDATE files SET last_modified=%f, "
                "filesize=%i, "
                "crc32_hash=%i WHERE id=%i;" % (os.path.getmtime(abs_filename),
                                                os.path.getsize(abs_filename),
                                                filehash, filepath_id))
            self.db_conn.commit()
        else:
            self.db_cursor.execute(
                "INSERT into files(filename, last_modified, filesize, "
                " crc32_hash) VALUES"
                "('%s', %f, %i, %i);" % (
                    rel_filename, os.path.getmtime(abs_filename),
                    os.path.getsize(abs_filename), filehash))
            self.db_conn.commit()
            filepath_id = self.db_cursor.lastrowid

        # Append the file's path id to every index.
        for index in indices:
            index.append(filepath_id)

        sql_insert_string = "INSERT INTO indices(%s, filepath_id) VALUES(%s);"

        self.db_conn.executemany(sql_insert_string % (
            ",".join([_i[0] for _i in self.index_values]),
            ",".join(["?"] * (len(indices[0])))),
            indices)

        self.db_conn.commit()
