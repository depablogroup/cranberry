from __future__ import annotations

import importlib
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive, directives


class ArgparseHelpDirective(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'module': directives.unchanged_required,
        'function': directives.unchanged_required,
        'command': directives.unchanged,
    }

    def run(self):
        module_name = self.options['module']
        function_name = self.options['function']
        command = self.options.get('command')

        module = importlib.import_module(module_name)
        factory = getattr(module, function_name)
        parser = factory()
        if command:
            parser = _find_subparser(parser, command)

        help_text = parser.format_help().rstrip()
        literal = nodes.literal_block(help_text, help_text)
        literal['language'] = 'text'
        return [literal]


def _find_subparser(parser: Any, command: str):
    for action in getattr(parser, '_actions', []):
        choices = getattr(action, 'choices', None)
        if choices and command in choices:
            return choices[command]
    raise ValueError(f'Could not find subparser {command!r}')


def setup(app):
    app.add_directive('argparse-help', ArgparseHelpDirective)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
