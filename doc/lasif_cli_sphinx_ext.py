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
from sphinx.util.compat import Directive

from lasif.scripts import lasif_cli


def scan_programs(parser, command=[]):
    """
    Straight copy from the sphinx-contrib autoprogram directive.

    All credit goes to Hong Minhee. Thanks a lot.
    """
    options = []
    for arg in parser._actions:
        if not (arg.option_strings or
                isinstance(arg, argparse._SubParsersAction)):
            name = (arg.metavar or arg.dest).lower()
            desc = (arg.help or '') % {'default': arg.default}
            options.append(([name], desc))
    for arg in parser._actions:
        if arg.option_strings:
            metavar = (arg.metavar or arg.dest).lower()
            names = ['{0} <{1}>'.format(option_string, metavar)
                     for option_string in arg.option_strings]
            desc = (arg.help or '') % {'default': arg.default}
            options.append((names, desc))
    yield command, options, parser.description or ''
    if parser._subparsers:
        for cmd, sub in parser._subparsers._actions[-1].choices.items():
            if isinstance(sub, argparse.ArgumentParser):
                for program in scan_programs(sub, command + [cmd]):
                    yield program


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

            result.append(title, "<lasif_cli_list>")
            result.append("-" * len(title), "<lasif_cli_list>")

            if group_name in lasif_cli.COMMAND_GROUP_DOCS:
                result.append(lasif_cli.COMMAND_GROUP_DOCS[group_name],
                               "<lasif_cli_list>")

            #self.state.nested_parse(result, 0, node, match_titles=1)
            #all_nodes.extend(node.children)

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

                #node = nodes.section()
                #node.document = self.state.document
                #result = ViewList()

                for i in scan_programs(parser):
                    cmd_name = "lasif %s" % fct_name

                    _, options, desc = i

                    title = cmd_name
                    result.append("", "<lasif_cli_list>")
                    result.append(".. program:: " + title, "<lasif_cli_list>")
                    result.append("", "<lasif_cli_list>")

                    result.append(title, "<lasif_cli_list>")
                    result.append("^" * len(title), "<lasif_cli_list>")

                    result.append("", "<lasif_cli_list>")
                    result.append(desc, "<lasif_cli_list>")
                    result.append("", "<lasif_cli_list>")
                    for option_strings, help_ in options:
                        result.append(".. cmdoption:: {0}".format(
                            ", ".join(option_strings)), "<lasif_cli_list>")
                        result.append("", "<lasif_cli_list>")
                        result.append("   " + help_.replace("\n", "   \n"),
                                      "<lasif_cli_list>")
                        result.append("", "<lasif_cli_list>")
            self.state.nested_parse(result, 0, node, match_titles=1)

            all_nodes.extend(node.children)

        return all_nodes


class LasifCLINode(nodes.General, nodes.Element):
    pass


def setup(app):
    app.add_node(LasifCLINode)
    app.add_directive("include_lasif_cli_commands", LasifCLIDirective)
