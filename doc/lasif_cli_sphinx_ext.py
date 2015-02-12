#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom sphinx directive to document LASIF's CLI interface.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import argparse
from docutils import nodes
from docutils.statemachine import ViewList
import itertools
from sphinx.util.compat import Directive
import textwrap
import warnings

from lasif.scripts import lasif_cli


def scan_programs(parser):
    """
    Based on sphinx-contrib autoprogram directive.

    A lot of credit goes to Hong Minhee. Thanks a lot.
    """
    # Positional arguments
    positional_arguments = []
    for arg in parser._actions:
        if arg.option_strings:
            continue
        name = (arg.metavar or arg.dest).lower()
        if arg.choices:
            choices = list(arg.choices)
            # Not sure if this can actually happen with positional arguments.
            # I guess not...
            if arg.default:
                new_choices = []
                for choice in choices:
                    if choice == arg.default:
                        choice += " (Default)"
                    new_choices.append(choice)
                choices=new_choices
            name += " {%s}" % ", ".join(choices)
        desc = (arg.help or '') % {'default': arg.default}
        positional_arguments.append(([name], desc))

    # Optional arguments
    optional_arguments = []
    for arg in parser._actions:
        if not arg.option_strings:
            continue
        additional_info = ""
        if arg.choices:
            choices = list(arg.choices)
            # Not sure if this can actually happen with positional arguments.
            # I guess not...
            if arg.default:
                new_choices = []
                for choice in choices:
                    if choice == arg.default:
                        choice += " (Default)"
                    new_choices.append(choice)
                choices=new_choices
            additional_info += " {%s}" % ", ".join(choices)
            # from IPython.core.debugger import Tracer; Tracer(colors="linux")()
        names = ['{0}{1}'.format(option_string, additional_info)
                 for option_string in arg.option_strings]
        desc = (arg.help or '') % {'default': arg.default}
        optional_arguments.append((names, desc))
    yield positional_arguments, optional_arguments, \
        parser.description or '', parser.format_usage()


class LasifCLIDirective(Directive):
    def run(self):
        fcts = lasif_cli._get_functions()

        # Group the functions. Functions with no group will be placed in the
        # group "Misc".
        fct_groups = {}
        for fct_name, fct in fcts.iteritems():
            group_name = fct.group_name \
                if hasattr(fct, "group_name") else "Misc"
            fct_groups.setdefault(group_name, {})
            fct_groups[group_name][fct_name] = fct

        all_nodes = []

        # Print in a grouped manner.
        for group_name, fcts in sorted(fct_groups.iteritems()):
            node = nodes.section()
            node.document = self.state.document
            result = ViewList()

            title = group_name + " Functions"

            result.append("", "<lasif_cli_list>")
            result.append("------------------", "<lasif_cli_list>")
            result.append("", "<lasif_cli_list>")
            result.append(title, "<lasif_cli_list>")
            result.append("-" * len(title), "<lasif_cli_list>")
            result.append("", "<lasif_cli_list>")


            if group_name in lasif_cli.COMMAND_GROUP_DOCS:
                result.append(".. admonition:: %s" % group_name,
                              "<lasif_cli_list>")
                result.append("", "<lasif_cli_list>")
                for line in lasif_cli.COMMAND_GROUP_DOCS[group_name]\
                        .splitlines():
                    result.append("    " + line,
                                  "<lasif_cli_list>")

            for fct_name, fct in fcts.iteritems():
                parser = lasif_cli._get_argument_parser(fct)

                # The parser receive all their options just before they are
                # being parsed. Therefore monkey patch the parse_args() method
                # to get a fully ready parser object.
                class TempException(Exception):
                    pass

                def wild_monkey_patch(*args, **kwargs):
                    raise TempException

                parser.parse_args = wild_monkey_patch

                try:
                    fct(parser, [])
                except TempException:
                    pass

                for i in scan_programs(parser):
                    cmd_name = "lasif %s" % fct_name

                    positional_args, optional_args, desc, usage = i

                    title = cmd_name
                    result.append("", "<lasif_cli_list>")
                    result.append(".. program:: " + title, "<lasif_cli_list>")
                    result.append("", "<lasif_cli_list>")

                    result.append(title, "<lasif_cli_list>")
                    result.append("*" * len(title), "<lasif_cli_list>")
                    result.append("", "<lasif_cli_list>")

                    if hasattr(fct, "_is_mpi_enabled") and fct._is_mpi_enabled:
                        result.append(
                            "**This function can be used with MPI**",
                            "<lasif_cli_list>")
                        result.append("", "<lasif_cli_list>")


                    result.append(" .. code-block:: none", "<lasif_cli_list>")
                    result.append("", "<lasif_cli_list>")
                    result.append("    " + "\n    ".join(usage.splitlines()),
                                  "<lasif_cli_list>")
                    result.append("", "<lasif_cli_list>")

                    for line in textwrap.dedent(fct.__doc__).splitlines():
                        result.append(line, "<lasif_cli_list>")
                    result.append("", "<lasif_cli_list>")

                    # Collect arguments in table and render it.
                    table = []

                    if positional_args:
                        table.append(("**Positional Arguments**",))

                    for option_strings, help_ in positional_args:
                        for i, j in itertools.izip_longest(
                                (", ".join(["``%s``" % _i
                                            for _i in option_strings]),),
                                textwrap.wrap(help_, 50),
                                fillvalue=""):
                            table.append((i, j))

                    if optional_args:
                        table.append(("**Optional Arguments**",))

                    for option_strings, help_ in optional_args:
                        for i, j in itertools.izip_longest(
                                (", ".join(["``%s``" % _i
                                            for _i in option_strings]),),
                                 textwrap.wrap(help_, 20),
                                 fillvalue=""):
                            table.append((i, j))

                    # Render table.
                    padding = 1
                    max_length = max(len(_i) for _i in table)
                    rows = []
                    for i in range(max_length):
                        max_i = 0
                        for row in table:
                            if len(row) < max_length:
                                continue
                            max_i = max(max_i, len(row[i]) + 2 * padding)
                        rows.append(max_i)

                    separator = "+" + "+".join("-" * _i for _i in rows) + "+"
                    final_table = [separator, ]

                    for row in table:
                        if len(row) == 1:
                            final_table.append(
                                "|%-{0}s|".format(sum(rows) + len(rows) - 1) %
                                (" " * padding + row[0]))
                        elif len(row) == max_length:
                            # Super special case handling for LASIF!
                            if row[0] == "":
                                final_table.pop(-1)
                            final_table.append("|" +
                                "|".join(
                                    "%-{0}s".format(length) %
                                    (" " * padding + _i)
                                    for _i, length in zip(row, rows)
                                ) + "|")
                        else:
                            warnings.warn("Table cannot be rendered!")
                        final_table.append(separator)

                    for line in final_table:
                        result.append(line, "<lasif_cli_list>")

            self.state.nested_parse(result, 0, node, match_titles=1)

            all_nodes.extend(node.children)

        return all_nodes


class LasifCLINode(nodes.General, nodes.Element):
    pass


class LasifMPICLIDirective(Directive):
    def run(self):
        fcts = lasif_cli._get_functions()

        all_nodes = []

        node = nodes.section()
        node.document = self.state.document
        result = ViewList()

        mpi_enabled = []
        # Find function that have MPI.
        for fct_name, fct in fcts.iteritems():
            if not hasattr(fct, "_is_mpi_enabled") or not fct._is_mpi_enabled:
                continue
            mpi_enabled.append(fct_name)
        for fct_name in sorted(mpi_enabled):
            result.append("* `lasif %s`_" % fct_name, "<lasif_cli_list>")

        self.state.nested_parse(result, 0, node, match_titles=1)
        all_nodes.extend(node.children)

        return all_nodes


def setup(app):
    app.add_node(LasifCLINode)
    app.add_directive("include_lasif_cli_commands", LasifCLIDirective)
    app.add_directive("include_lasif_mpi_cli_commands", LasifMPICLIDirective)
