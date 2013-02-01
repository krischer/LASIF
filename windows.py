from obspy import UTCDateTime
from lxml import etree
from lxml.builder import E
import os

# A list of valid windows including options.
WINDOWS = {
    "cosine": {
        "percentage": {
            "description": ("Decimal percentage being tapered. Default is 0.1 "
                "resulting in 10% being tapered. 5% at the beginning, 5 % at "
                "the end."),
            "default_value": 0.1,
            "obspy_name": "p"
        }
    },
    "gaussian": {
        "standard-deviation": {
            "description": "The standard deviation for the gaussian function.",
            "obspy_name": "std"
        }
    }
}


class Window(object):
    def __init__(self, event_identifier, additional_identifier, event_time,
            window_offset, window_length, window_type, window_options,
            channel_id):
        self.event_identifier = event_identifier
        self.additional_identifier = additional_identifier
        self.event_time = event_time
        if window_offset < 0 or window_length <= 0:
            msg = "Offset and length must not be negative."
            raise ValueError(msg)
        self.window_offset = window_offset
        self.window_length = window_length
        # Get the window type and the allowed options.
        if window_type not in WINDOWS:
            msg = "Invalid window type. Valid types: %s" % ", ".join(WINDOWS)
            raise ValueError(msg)
        self.window_description = WINDOWS[window_type]
        self.options = {}
        for param, values in self.window_description.iteritems():
            if param in window_options:
                self.options[param] = window_options[param]
            elif "default_value" in values:
                self.options[param] = values["default_value"]
            else:
                msg = "Option %s needs to have a value." % param
                raise ValueError(msg)
        self.window_type = window_type

        try:
            net, sta, loc, chan = channel_id.split(".")
        except:
            msg = "channel_id needs to have the form 'net.sta.loc.chan'"
            raise ValueError(msg)
        if not net or not sta or not chan:
            msg = "channel_id needs to have the form 'net.sta.loc.chan'"
            raise ValueError(msg)
        # Channel needs to end with either N, E, or Z
        if chan[-1] not in ["N", "E", "Z"]:
            msg = "The channel id needs to end in 'N', 'E', or 'Z'"
            raise ValueError(msg)
        self.channel_id = channel_id

    def serialize_options(self):
        params = []
        for param, value in self.options.iteritems():
            params.append(E.parameter(getattr(E, param)(str(value))))
        return params

    def serialize(self):
        doc = (
            E.window(
                E.event_identifier(str(self.event_identifier)),
                E.additional_identifier(str(self.additional_identifier)),
                E.event_time(str(self.event_time)),
                E.window_offset(str(self.window_offset)),
                E.window_length(str(self.window_length)),
                E.window_type(str(self.window_type)),
                E.window_parameters(*self.serialize_options()),
                E.channel_id(self.channel_id)))
        return etree.tostring(doc, pretty_print=True, xml_declaration=True,
            encoding="utf-8")

    def _filename_generator(self):
        _i = 0
        while True:
            basename = "window_event_%s_%s" % (str(self.event_identifier),
                str(self.additional_identifier))
            if _i:
                basename += "_%i" % _i
            _i += 1
            yield basename + ".xml"

    def write(self, folder):
        """
        Writes the file to the given folder. This is a convenience function
        that also automatically determines the filename. If you want to take
        care of the filename yourself, use the serialize() function to get a
        string representation.
        """

        ################
        # DEBUGGING START
        import sys
        __o_std__ = sys.stdout
        sys.stdout = sys.__stdout__
        from IPython.core.debugger import Tracer
        Tracer(colors="Linux")()
        sys.stdout = __o_std__
        # DEBUGGING END
        ################

        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.isdir(folder):
            msg = "The given folder is not a folder."
            raise ValueError(msg)
        # Otherwise check if the filename exists.
        for filename in self._filename_generator():
            subfolder = self.event_identifier + "." + \
                self.additional_identifier
            filename = os.path.join(folder, subfolder, filename)
            if not os.path.exists(filename):
                break
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "w") as open_file:
            open_file.write(self.serialize())

    def apply(self, trace):
        """
        Apply the selected window to an obspy.core.Trace object.
        """
        starttime = trace.stats.starttime
        endtime = trace.stats.endtime

        window_starttime = self.event_time + self.window_offset
        window_endtime = window_starttime + self.window_length

        # Check that both times are available in the trace.
        if not (starttime <= window_starttime <= endtime) or \
            not (starttime <= window_endtime <= endtime):
            msg = "The trace does not have data for the window."
            raise ValueError(msg)

        # Map the options to the name used in obspy.
        options = {}
        for option, value in self.options.iteritems():
            options[WINDOWS[self.window_type][option]["obspy_name"]] = value

        trace.trim(window_starttime, window_endtime)
        trace.taper(self.window_type, **options)
        trace.trim(starttime, endtime, pad=True, fill_value=0.0)


if __name__ == "__main__":
    win = Window("1", "iteration_15.17s", UTCDateTime(), 123.3, 234.23,
        "cosine", window_options={"percentage": 0.1},
        channel_id="BW.FURT..N")
    win.write("OUTPUT")
