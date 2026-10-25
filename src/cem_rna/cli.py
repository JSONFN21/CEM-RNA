# design rna with lazyoutside scores and 1-ned weights
from .parallel.parallel_scorer import _worker_environment
_worker_environment()

import argparse
from dataclasses import asdict
from contextlib import contextmanager
import json
import sys
from pathlib import Path
from time import perf_counter
from .cem import optimize_many
from .config import Config
from .checks import check_workers
from .structure_parser import parse_dot_bracket


def read_targets(path):
    targets = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) > 2:
                raise ValueError(f'{path}, line {number}: use a structure or an id and structure')
            targets.append((str(number), parts[0]) if len(parts) == 1 else tuple(parts))
    return targets


# attach the input labels in the same order as the target and seed jobs
def label_results(results, targets, seeds):
    labels = targets if seeds is None else [item for item in targets for _ in seeds]
    for (label, target), result in zip(labels, results):
        result.update(id=label, target=target)


# show the same small summary for the simple and extra optimizers
def print_summary(results, elapsed):
    warnings = dict.fromkeys(message for result in results for message in result.get('warnings', []))
    for message in warnings:
        print(message)
    width = max([8, *(len(r['sequence']) for r in results)])
    print(f'{"Target ID":>10} {"Seed":>6} {"Sequence":<{width}} {"P":>12} {"NED":>12} {"uMFE":>5} {"Time (s)":>10}')
    for result in results:
        print(f'{result["id"]:>10} {result["config"]["seed"]:6} {result["sequence"]:<{width}} '
              f'{result["probability"]:12.6g} {result["sequence_ned"]:12.6f} {str(result["is_umfe"]).lower():>5} {result["runtime_seconds"]:10.2f}')
    if results:
        print(f'\nTotal wall time: {elapsed:.2f} s · '
              f'Mean best P: {sum(r["probability"] for r in results)/len(results):.6g} · '
              f'Mean best NED: {sum(r["best_ned"] for r in results)/len(results):.6f}')
    if len(results) == 1:
        print(f'Best-P sequence:   {results[0]["sequence"]}')
        print(f'Best-NED sequence: {results[0]["best_ned_sequence"]}')


# keep existing results and use the next available number
def save_results(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    number = 0
    while True:
        candidate = path if number == 0 else path.with_name(f'{path.stem}{number}{path.suffix}')
        try:
            with candidate.open('x', encoding='utf-8') as output:
                output.write(text)
            return candidate
        except FileExistsError:
            number += 1


@contextmanager
def cli_errors(parser):
    try:
        yield
    except (ValueError, OSError) as error:
        parser.error(str(error))


# run the target and seed jobs, then keep their labels with the results
def run_targets(args, config, optimizer=optimize_many):
    targets = read_targets(args.input) if args.input else [('1', args.target)]
    check_workers(args.workers, args.concurrent_puzzles, args.seeds)
    if not targets:
        raise ValueError('input file contains no targets')
    for label, target in targets:
        try:
            parse_dot_bracket(target)
        except ValueError as error:
            raise ValueError(f'target {label}: {error}') from None
    jobs = len(targets) * (len(args.seeds) if args.seeds is not None else 1)
    active = min(jobs, args.concurrent_puzzles or args.workers * 2) if args.workers > 1 else 1
    if active > args.workers:
        print(f'Running {active} puzzles or seeds at once can use a lot of RAM. Lower --concurrent-puzzles if memory is tight.', file=sys.stderr)
    started = perf_counter()
    results = optimizer([target for _, target in targets], config=config, workers=args.workers, concurrent_puzzles=args.concurrent_puzzles, seeds=args.seeds)
    elapsed = perf_counter() - started
    label_results(results, targets, args.seeds)
    return results, elapsed


# build the common command options once for both optimizers
def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('target', nargs='?', help='target in dot-bracket notation')
    parser.add_argument('--input', type=Path, help='file of structures, optionally preceded by IDs')
    parser.add_argument('--output', type=Path, default=Path('results/results.json'), help='save JSON results (default: results/results.json)')
    parser.add_argument('--json', action='store_true', help='print only JSON, without the terminal summary')
    parser.add_argument('--workers', '--threads', type=int, default=1, help='shared scoring workers (default: 1)')
    parser.add_argument('--seeds', type=int, nargs='+', help='run each target with these seeds')
    parser.add_argument('--concurrent-puzzles', type=int, help='simultaneous optimizers; default: twice the workers')
    return parser


def add_settings(parser, config, visible=None):
    defaults = asdict(config)
    for name, value in defaults.items():
        if name not in Config.__dataclass_fields__:
            continue
        flag = '--' + name.replace('_', '-')
        help_text = 'include round history and search details' if isinstance(value, bool) else f'default: {value}'
        if name == 'max_requests':
            help_text = f'scoring request cap; default: {value:,}'
        if visible is not None and name not in visible:
            help_text = argparse.SUPPRESS
        if isinstance(value, bool):
            parser.add_argument(flag, action='store_true', default=value, help=help_text)
        else:
            parser.add_argument(flag, type=int if value is None else type(value), default=value, help=help_text)
    return defaults


def read_args(parser, argv=None):
    args = parser.parse_args(argv)
    if bool(args.target) == bool(args.input):
        parser.error('Give a target or --input, not both')
    return args


def config_options(args, defaults):
    return {name: getattr(args, name) for name in defaults if hasattr(args, name) and getattr(args, name) is not None}


# share execution, numbered saves, and terminal output
def run_command(args, config, optimizer=optimize_many, notice=None):
    results, elapsed = run_targets(args, config, optimizer)
    value = results if args.input or args.seeds else results[0]
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output = save_results(args.output, text)
    if args.json:
        print(text, end='')
    else:
        if notice:
            print(notice)
        print_summary(results, elapsed)
        if args.output:
            print(f'Saved: {args.output}')


def main(argv=None):
    parser = make_parser('Design RNA with LazyOutside scoring and 1-NED-weighted CEM.')
    defaults = add_settings(parser, Config())
    args = read_args(parser, argv)
    with cli_errors(parser):
        config = Config(**config_options(args, defaults))
        run_command(args, config, notice='LazyOutside search; final P, NED and uMFE validated with ViennaRNA.')


if __name__ == '__main__':
    main()
