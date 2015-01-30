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


class FileInfoCache(object):
    """
    Object able to cache information about arbitrary files on the filesystem.

    Intended to be subclassed.
    """
    def __init__(self, cache_db_file, show_progress=True):
        self.cache_db_file = cache_db_file
        self.show_progress = show_progress

        # Will be filled in _init_database() method.
        self.db_cursor = None
        self.db_conn = None

        # Will be filled once database has been updated.
        self.files = {}

        self._init_database()
        self._update_indices()
        self.update()

    def __del__(self):
        if self.db_conn:
            try:
                self.db_conn.close()
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

    def _init_database(self):
        """
        Inits the database connects, turns on foreign key support and creates
        the tables if they do not already exist.
        """
        # Check if the file exists. If it exists, try to use it, otherwise
        # delete and create a new one. This should take care that a new
        # database is created in the case of DB corruption due to a power
        # failure.
        if os.path.exists(self.cache_db_file):
            try:
                self.db_conn = sqlite3.connect(self.cache_db_file)
            except sqlite3.Error:
                os.remove(self.cache_db_file)
                self.db_conn = sqlite3.connect(self.cache_db_file)
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
                filename TEXT,
                last_modified REAL,
                crc32_hash INTEGER
            );
        """
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
            self.files[filetype] = getattr(self, get_file_fct)()

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
        db_files = {_i[1]: (_i[0], _i[2], _i[3]) for _i in db_files}

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
        # Now update all filetypes separately.
        for filetype in self.filetypes:
            for filename in self.files[filetype]:
                current_file_count += 1
                # Only show the progressbar if more then 5 seconds have passed.
                if not pbar and self.show_progress and \
                        (time.time() - start_time > 5):
                    widgets = ["Updating cache: ", progressbar.Percentage(),
                               progressbar.Bar(), "", progressbar.ETA()]
                    pbar = progressbar.ProgressBar(widgets=widgets,
                                                   maxval=filecount).start()
                    update_interval = max(int(filecount / 100), 1)
                    pbar.update(current_file_count)
                if pbar and not current_file_count % update_interval:
                    pbar.update(current_file_count)
                if filename in db_files:
                    # Delete the file from the list of files to keep track of
                    # files no longer available.
                    this_file = db_files[filename]
                    del db_files[filename]
                    last_modified = os.path.getmtime(filename)
                    # If the last modified time is identical to a tenth of
                    # second, nothing to do.
                    if int(round(last_modified * 10)) == \
                            int(round(this_file[1] * 10)):
                        continue
                    # Otherwise check the hash.
                    with open(filename, "rb") as open_file:
                        hash_value = crc32(open_file.read())
                    if hash_value == this_file[2]:
                        # XXX: Update last modified times, otherwise it will
                        # hash again and again.
                        continue
                    self._update_file(filename, filetype, this_file[0])
                else:
                    self._update_file(filename, filetype)
        if pbar:
            pbar.finish()

        # Remove all files no longer part of the cache DB.
        for filename in db_files:
            self.db_cursor.execute("DELETE FROM files WHERE filename='%s';" %
                                   filename)
        self.db_conn.commit()

        # Update the self.files dictionary, this fime from the database.
        self._get_all_files_from_database()

    def get_values(self):
        """
        Returns a list of dictionaries containing all indexed values for every
        file together with the filename.
        """
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
            values["filename"] = _i[-1]
            all_values.append(values)

        return all_values

    def get_details(self, filename):
        """
        Get the indexed information about one file.

        :param filename: The filename for which to request information.
        """
        filename = os.path.abspath(filename)

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
            values["filename"] = _i[-1]
            all_values.append(values)

        return all_values

    def _update_file(self, filename, filetype, filepath_id=None):
        """
        Updates or creates a new entry for the given file. If id is given, it
        will be interpreted as an update, otherwise as a fresh record.
        """
        # Remove all old indices for the file if it is an update.
        if filepath_id is not None:
            self.db_cursor.execute("DELETE FROM indices WHERE "
                                   "filepath_id = %i" % filepath_id)
            self.db_conn.commit()

        # Get all indices from the file.
        try:
            indices = getattr(self, "_extract_index_values_%s" %
                                    filetype)(filename)
        except Exception as e:
            msg = "Failed to index '%s' of type '%s' due to: %s" % (
                filename, filetype, str(e)
            )
            warnings.warn(msg, LASIFWarning)

            # If it is an update, also remove the file from the file list.
            if filepath_id is not None:
                self.db_cursor.execute(
                    "DELETE FROM files WHERE filename='%s';" % filename)
                self.db_conn.commit()

            return

        if not indices:
            msg = ("Could not extract any index from file '%s' of type '%s'. "
                   "The file will be skipped." % (filename, filetype))
            warnings.warn(msg, LASIFWarning)

            # If it is an update, also remove the file from the file list.
            if filepath_id is not None:
                self.db_cursor.execute(
                    "DELETE FROM files WHERE filename='%s';" % filename)
                self.db_conn.commit()

            return

        # Get the hash
        with open(filename, "rb") as open_file:
            filehash = crc32(open_file.read())

        # Add or update the file.
        if filepath_id is not None:
            self.db_cursor.execute(
                "UPDATE files SET last_modified=%f, "
                "crc32_hash=%i WHERE id=%i;" % (os.path.getmtime(filename),
                                                filehash, filepath_id))
            self.db_conn.commit()
        else:
            self.db_cursor.execute(
                "INSERT into files(filename, last_modified,"
                " crc32_hash) VALUES('%s', %f, %i);" % (
                    filename, os.path.getmtime(filename), filehash))
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
