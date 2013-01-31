from obspy import UTCDateTime
import os


def read_event_list(filename):
    """
    Reads a certain event list format.
    """
    if not os.path.exists(filename):
        msg = "Can not find file %s." % filename
        raise Exception(msg)
    events = {}
    with open(filename, "rt") as open_file:
        for line in open_file:
            line = line.strip()
            line = line.split()
            if len(line) < 14:
                continue
            if not line[0].isdigit():
                continue
            index, date, colat, lon, depth, exp, Mrr, Mtt, Mpp, Mrt, Mrp, \
                Mtp, time, Mw = line[:14]
            index, exp = map(int, (index, exp))
            colat, lon, depth, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, Mw = map(float,
                (colat, lon, depth, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, Mw))
            year, month, day = map(int, date.split("/"))
            split_time = time.split(":")
            if len(split_time) == 3:
                hour, minute, second = split_time
                second, microsecond = second.split(".")
            elif len(split_time) == 4:
                hour, minute, second, microsecond = split_time
            else:
                raise NotImplementedError
            microsecond = int(microsecond) * 10 ** (6 - len(microsecond))
            hour, minute, second = map(int, (hour, minute, second))
            event_time = UTCDateTime(year, month, day, hour, minute, second,
                microsecond)
            event = {
                "longitude": lon,
                "latitude": -1.0 * (colat - 90.0),
                "depth_in_km": depth,
                "time": event_time,
                "Mw": Mw,
                "Mrr": Mrr * 10 ** exp,
                "Mtt": Mtt * 10 ** exp,
                "Mpp": Mpp * 10 ** exp,
                "Mrt": Mrt * 10 ** exp,
                "Mrp": Mrp * 10 ** exp,
                "Mtp": Mtp * 10 ** exp}
            events[index] = event
    return events
