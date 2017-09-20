import sqlite3
from contextlib import contextmanager
from obspy import UTCDateTime

# Register adapter and converter for obspy utc datetime objects
from lasif import LASIFNotFoundError


def obspy_adapter(utcdt):
    return utcdt.isoformat()


def obspy_converter(value):
    return UTCDateTime(value.decode())


sqlite3.register_adapter(UTCDateTime, obspy_adapter)
sqlite3.register_converter("utc_datetime", obspy_converter)


class WindowGroupManager(object):
    """
    Represents all the windows for one window set, windows are constant between
    iterations. This simplifies misfit comparisons. New window sets can still
    be selected after performing a number of iterations. This produces a
    new window set
    """

    def __init__(self, filename):
        self.filename = filename
        self.windows = []

    @contextmanager
    def sqlite_cursor(self):
        """
        DB - Design Plan:
        events - event_id, event_name (possible more in the future)
        traces - trace_id, channel_name, FK_event_id (possibly more in the
         future, i.e. lat and lon)
        windows - window_id, FK_trace_id, start_time, end_time, weight
        """
        # filename = Path.home() / Path(".salvus-flow-job-tracker.sqlite")
        # This filename could be added as a class variable!
        filename = self.filename
        conn = sqlite3.connect(str(filename),
                               detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()
        c.execute("PRAGMA foreign_keys = 1")
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT NOT NULL UNIQUE
                )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                channel_name TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES events(event_id),
                UNIQUE (event_id, channel_name)
                )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS windows (
                window_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id INTEGER NOT NULL,
                start_time utc_datetime NOT NULL,
                end_time utc_datetime NOT NULL,
                weight REAL NOT NULL,
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
                )
        """)
        yield c
        conn.commit()
        conn.close()

    def drop_all_tables(self):
        """Drop all tables from the DB"""
        with self.sqlite_cursor() as c:
            c.execute("""DROP TABLE windows""")
            c.execute("""DROP TABLE traces""")
            c.execute("""DROP TABLE events""")

    def event_in_db(self, event_name):
        """Check if event is available in the database"""
        with self.sqlite_cursor() as c:
            c.execute("SELECT EXISTS(SELECT event_id FROM events "
                      "WHERE event_name = ?)", (event_name,))
            return bool(c.fetchone()[0])

    def get_event_id(self, event_name):
        """get event_id from database for a given event_name"""
        if self.event_in_db(event_name):
            with self.sqlite_cursor() as c:
                c.execute("SELECT event_id FROM events "
                          "WHERE event_name = ?", (event_name,))
                return c.fetchone()[0]
        else:
            raise LASIFNotFoundError("Event: \"{}\" could not be"
                                     " found in database.".format(event_name))

    def add_event(self, event_name):
        """Add event to databse if it does not exist"""
        if not self.event_in_db(event_name):
            with self.sqlite_cursor() as c:
                c.execute("INSERT INTO events VALUES (NULL, ? )",
                          (event_name,))
        else:
            raise ValueError(event_name, " already exists in db")

    def remove_event(self, event_name):
        """Remove event from db"""
        # for now just remove the event
        if self.event_in_db(event_name):
            with self.sqlite_cursor() as c:
                c.execute("DELETE FROM events WHERE event_name=?",
                          (event_name,))
        else:
            raise LASIFNotFoundError("Event: \"{}\" could not be "
                                     "found in database.".format(event_name))

    def trace_in_db(self, event_name, channel_name):
        """Check if channel, event pair exists in db"""
        event_id = self.get_event_id(event_name)
        with self.sqlite_cursor() as c:
            # check if exists
            c.execute(
                "SELECT EXISTS(SELECT 1 FROM traces "
                "WHERE channel_name=? AND event_id=? LIMIT 1)",
                (channel_name, event_id))
            # if new add to DB
            return bool(c.fetchone()[0])

    def add_trace(self, event_name, channel_name):
        """
        Add trace to database, each trace is linked to an event, channel pair
        """
        event_id = self.get_event_id(event_name)
        if not self.trace_in_db(event_name, channel_name):
            with self.sqlite_cursor() as c:
                c.execute(" INSERT INTO traces VALUES (NULL, ?, ?)",
                          (event_id, channel_name))
        else:
            raise ValueError("Trace {} - {} ".format(event_name, channel_name))

    def get_trace_id(self, event_name, channel_name):
        """Get trace id"""
        event_id = self.get_event_id(event_name)
        if self.trace_in_db(event_name, channel_name):
            with self.sqlite_cursor() as c:
                c.execute("SELECT trace_id FROM traces "
                          "WHERE channel_name=? AND event_id=?",
                          (channel_name, event_id))
                return c.fetchone()[0]

    def remove_trace(self, event_name, channel_name):
        """Remove trace, maybe check if window exist for this trace"""
        event_id = self.get_event_id(event_name)
        if self.trace_in_db(event_name, channel_name):
            with self.sqlite_cursor() as c:
                c.execute("DELETE FROM traces WHERE channel_name=? "
                          "AND event_id", (event_name, event_id))
                return c.fetchone()[0]
        else:
            raise LASIFNotFoundError("Trace {} - {} not found - could not be"
                                     " removed"
                                     .format(event_name, channel_name))

    def add_window(self, trace_id, start_time, end_time, weight=1.0):
        """
        start_time and end_time given as timestamps since epoch

        """
        start_time = start_time.datetime
        end_time = end_time.datetime
        assert end_time > start_time, "end_time must be larger than start_time"
        with self.sqlite_cursor() as c:
            # delete overlapping windows if they exist
            c.execute("""
                DELETE FROM windows WHERE trace_id=?
                AND ((start_time BETWEEN ? AND ?)
                OR (end_time BETWEEN ? AND ?)
                OR (start_time >= ? AND end_time <= ?)
                OR (start_time <= ? AND end_time >= ?))
                """,
                      ("{}".format(trace_id), start_time, end_time, start_time,
                       end_time, start_time, end_time, start_time, end_time))

            # insert window
            c.execute("""
                INSERT INTO windows VALUES (
                    NULL, ?, ?, ?, ?
                )
            """, ("{}".format(trace_id), start_time, end_time, weight))

    def delete_window(self, trace_id, start_time, end_time):
        """
        start_time and end_time given as timestamps since epoch

        """
        assert end_time > start_time, "end_time must be larger than start_time"
        with self.sqlite_cursor() as c:
            # delete overlapping windows if they exist
            c.execute("""
                DELETE FROM windows WHERE trace_id=?
                AND ((start_time BETWEEN ? AND ?)
                OR (end_time BETWEEN ? AND ?)
                OR (start_time >= ? AND end_time <= ?)
                OR (start_time <= ? AND end_time >= ?))
                """, (trace_id, start_time, end_time, start_time,
                      end_time, start_time, end_time, start_time, end_time))

    def get_all_windows_for_trace(self, trace_id):
        """
        Get all windows for a trace -
        returns a list of tupes (start_time, end_time, weight)
        """
        with self.sqlite_cursor() as c:
            c.execute(
                "SELECT start_time, end_time, weight FROM windows "
                "WHERE trace_id=?", (trace_id,))
            return c.fetchall()

    def delete_all_windows_for_trace(self, trace_id):
        with self.sqlite_cursor() as c:
            c.execute(
                "DELETE FROM windows WHERE trace_id=?", (trace_id,))

    def get_all_windows_for_station(self, station):
        with self.sqlite_cursor() as c:
            # get all trace_ids
            c.execute("""SELECT * FROM windows WHERE windows.trace_id IN
                    (SELECT trace_id FROM traces WHERE channel_name LIKE ?)""",
                      ('%' + station + '%',))
            return c.fetchall()

    def write_windows(self, event_name, results):
        if not self.event_in_db(event_name):
            self.add_event(event_name)

        for station, channels in results.items():
            if channels is not None:
                for channel, windows in channels.items():
                    if not self.trace_in_db(event_name=event_name,
                                            channel_name=channel):
                        self.add_trace(event_name=event_name,
                                       channel_name=channel)
                    trace_id = self.get_trace_id(event_name=event_name,
                                                 channel_name=channel)
                    for window in windows:
                        self.add_window(trace_id=trace_id,
                                        start_time=window[0],
                                        end_time=window[1], weight=1.0)

    def get_all_windows_for_event(self, event_name):
        """
        Returns a dictionary with all list of windows for each channel
        for each station for an event
        """
        event_id = self.get_event_id(event_name)
        with self.sqlite_cursor() as c:
            c.execute("""SELECT event_traces.channel_name, windows.start_time,
                        windows.end_time, windows.weight
                        FROM (SELECT * FROM traces
                        WHERE event_id=?) event_traces, windows
                        WHERE event_traces.trace_id = windows.trace_id""",
                      (event_id,))
            rows = c.fetchall()

        # Build a dictionary with stations-channels-list of windows
        results = {}
        for row in rows:
            channel_name, start_time, end_time, weight = row
            net, sta, cha = channel_name.split(".", 2)
            station = net + "." + sta

            # Add station
            if station not in results.keys():
                results[station] = {}

            # Add channel to station
            if channel_name not in results[station].keys():
                results[station][channel_name] = []

            start_end = (start_time, end_time)
            results[station][channel_name].append(start_end)
        return results

    def get_all_windows_for_event_station(self, event_name, station):
        """
        Returns a dictionary with a list of windows for each channel
         for a given station and event
         """
        event_id = self.get_event_id(event_name)
        with self.sqlite_cursor() as c:
            c.execute("""SELECT * FROM
                        (SELECT event_traces.channel_name, windows.start_time,
                         windows.end_time, windows.weight
                        FROM (SELECT * FROM traces
                        WHERE event_id=?) event_traces, windows
                        WHERE event_traces.trace_id = windows.trace_id)
                        WHERE channel_name LIKE ?""",
                      (event_id, '%' + station + '%'))
            rows = c.fetchall()

        results = {}
        for row in rows:
            channel_name, start_time, end_time, weight = row

            # Add channel to station
            if channel_name not in results.keys():
                results[channel_name] = []

            start_end = (start_time, end_time)
            results[channel_name].append(start_end)
        return results

    def add_window_to_event_channel(self, event_name, channel_name, start_time,
                                    end_time, weight=1.0):
        if not self.event_in_db(event_name):
            self.add_event(event_name)
        if not self.trace_in_db(event_name, channel_name):
            self.add_trace(event_name, channel_name)
        trace_id = self.get_trace_id(event_name, channel_name)
        self.add_window(trace_id, start_time, end_time, weight)

    def del_all_windows_from_event_channel(self, event_name, channel_name):
        if not self.event_in_db(event_name):
            self.add_event(event_name)
        if not self.trace_in_db(event_name, channel_name):
            self.add_trace(event_name, channel_name)
        trace_id = self.get_trace_id(event_name, channel_name)
        self.delete_all_windows_for_trace(trace_id)

    def del_window_from_event_channel(self, event_name, channel_name,
                                      start_time, end_time):
        if not self.event_in_db(event_name):
            self.add_event(event_name)
        if not self.trace_in_db(event_name, channel_name):
            self.add_trace(event_name, channel_name)
        trace_id = self.get_trace_id(event_name, channel_name)
        self.delete_window(trace_id, start_time, end_time)
