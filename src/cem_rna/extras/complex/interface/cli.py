# design rna sequences for the target structures
import argparse
import sys
from ..cem import optimize_many
from ....cli import make_parser, add_settings, read_args, config_options, run_command, cli_errors
from .config import Config, MODES, ENGINES, TRACKING, make_config


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    advanced = '--advanced-help' in argv
    parser = make_parser('Design RNA sequences from target secondary structures.')
    defaults = add_settings(parser, Config(), visible=None if advanced else ('samples', 'seed', 'verbose'))
    parser.add_argument('--engine', choices=ENGINES, default='lazyoutside', help='scoring engine (default: lazyoutside)')
    parser.add_argument('--tracking', choices=TRACKING, default='basic', help='basic: P/NED; elites: structural metrics; umfe: sample-wide uMFE')
    parser.add_argument('--advanced-help', action='help', help='show research settings and numerical options')
    help_text = lambda text: text if advanced else argparse.SUPPRESS
    parser.add_argument('--temperature', type=float, default=Config.temperature, help=help_text(f'default: {Config.temperature}'))
    parser.add_argument('--mode', choices=MODES, help=help_text('research override for the engine'))
    parser.add_argument('--track-sep', action='store_true', default=None, help=help_text('add structural metrics for final P/NED winners'))
    parser.add_argument('--track-umfe', action=argparse.BooleanOptionalAction, default=None, help=help_text('sample-wide exact uMFE checks; can be expensive'))
    parser.add_argument('--track-all', action='store_true', default=None, help=help_text('structural tracking on elites/finalists; adds folding cost'))
    args = read_args(parser, argv)
    with cli_errors(parser):
        settings = config_options(args, defaults)
        config = make_config(args.engine, args.tracking, **settings)
        run_command(args, config, optimize_many)


if __name__ == '__main__':
    main()
