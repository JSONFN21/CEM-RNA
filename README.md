# CEM-RNA

This project contains a simple optimizer that uses the Cross Entropy Method (CEM) for RNA design.

**sample → score → select elites → update → repeat**

## Requirements

The package needs Python 3.11 or later and a compiler with C++17 support

## Installation

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Example

```sh
cem-rna "(((...)))" --samples 100 --output results/example.json
```

Example output:

| Target ID | Seed | Sequence | P | NED | uMFE | Time (s) |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 706 | GCCUAUGGC | 0.861019 | 0.077727 | true | 0.02 |

Each run saves its results to `results/results.json` unless a different path is
set with `--output`, and existing files are kept by adding a number to the new
filename. The terminal shows the sequence with the highest P alongside its NED
and final Vienna uMFE status. The JSON file keeps this sequence
as well as the sequence with the lowest NED, so both are available when they differ.

If the output file already exists:

```text
results.json → results1.json → results2.json
```

Results are saved once all puzzles and seeds finish, so let the run complete
before closing the terminal as progress is not saved along the way.

To print only JSON in the terminal:

```sh
cem-rna "(((...)))" --samples 100 --json
```

## Run a dataset

`targets.txt`:

```text
hairpin (((...)))
short ((((...)))).
```

Or without IDs:

```text
(((...)))
((((...)))).
```

```sh
cem-rna --input targets.txt --samples 100 --output results/targets.json
```

Example output with IDs:

| Target ID | Seed | Sequence | P | NED | uMFE | Time (s) |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| hairpin | 706 | GCCUAUGGC | 0.861019 | 0.077727 | true | 0.02 |
| short | 706 | GGCCUAUGGCCA | 0.983827 | 0.002739 | true | 0.01 |

Running many puzzles at once can use a lot of RAM because each keeps its own
score cache, so start with fewer `--concurrent-puzzles` if memory is tight.

Eterna100 with 96 shared workers:

```sh
cem-rna --input data/eterna100.txt --samples 100 --workers 96 --output results/eterna100.json
```

Five seeds with round history:

```sh
cem-rna --input data/eterna100.txt --samples 100 --workers 96 --seeds 706 707 708 709 710 --verbose --output results/eterna100_5seeds.json
```

## Optimization options

**LazyOutside P/NED → select by P → weight by `1−NED` → final Vienna checks**

At the end of the search, Vienna checks the sequences with the best approximate
P and NED along with the final argmax. Since LazyOutside can overestimate P,
the extras engines provide exact scoring for the elites or for every sample.

| Command | How it scores samples | How it weights elites | Final results |
| --- | --- | --- | --- |
| **`cem-rna` · default** | Uses LazyOutside P and NED to pick elites by P | `1−NED` | Vienna checks at most three finalists |
| `cem-rna-extras --engine lazyoutside` | The same LazyOutside search, with extra tracking available | `1−NED` | Results checked with Vienna and optional structural metrics |
| `cem-rna-extras --engine hybrid` | LinearPartition picks a shortlist and Vienna scores the elites | Exact `1−NED` | Results checked with Vienna |
| `cem-rna-extras --engine vienna` | Vienna scores every unique sample | Exact `1−NED` | Exact Vienna results |

## Arguments

| Flag | Default | What it does |
| --- | --- | --- |
| `--input` | — | Target file |
| `--output` | `results/results.json` | Where to save JSON, adding a number when the file already exists |
| `--json` | Off | Print only JSON in the terminal |
| `--samples` | 500 | How many sequences to sample each round |
| `--workers` | 1 | How many scoring workers all puzzles and seeds share |
| `--concurrent-puzzles` | Up to `2 × workers` | How many puzzle/seed runs can be active at once |
| `--seed`, `--seeds` | 706, one seed | Use one random seed, or try several for each target |
| `--alpha` | 0.5 | How far to move toward the elite distribution |
| `--uniform` | 0.002 | Mix in a little variety when sampling: 0.2% by default |
| `--elite-fraction` | 0.1 | Keep the top 10% of samples, rounded up |
| `--initialization` | 0.75 | Favor A and CG/GC pairs |
| `--beam` | 250 | The beam width used for approximate folding |
| `--lazy-threshold` | 0.00005 | Controls outside pruning, with larger values pruning more |
| `--patience` | 25 | The recent average window and how long to wait for improvement |
| `--max-rounds` | 2,000 | Stop after this many rounds at most |
| `--max-requests` | 5,000,000 | How many scoring requests each run can make |
| `--probability-floor`, `--log-clip` | `1e-15`, `60` | Keep very small probabilities and exponentials well behaved |
| `--verbose` | Off | Include round history and search progress details in the saved results |
| `--help` | — | Show the command line options |
| Extras: `--engine` | `lazyoutside` | Choose `lazyoutside`, `hybrid`, or `vienna` |
| Extras: `--tracking` | `basic` | Save P and NED with `basic`, check elite structures with `elites`, or check samples for uMFE with `umfe` |

The search stops when both the recent average loss and best objective have gone
without improvement for `--patience` rounds, or when a round or request limit is
reached. Repeated samples still count toward the request limit even when their
scores are cached, while the final Vienna checks are outside that limit.

```sh
cem-rna --help
cem-rna-extras --advanced-help
```

## Eterna100 results

The following saved runs used one server with 96 physical cores:

| Method · Eterna100, all 100 puzzles | Samples / round | Seeds / puzzle | Total wall time | Mean final P ↑ | Mean final NED ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| LazyOutside `1−NED` · main optimizer | 100 | 1 | **7.16 min** | **0.58473** | **0.034090** |
| LP/Vienna `1−NED` · extras hybrid | 100 | 1 | 7.58 min | 0.58428 | 0.037733 |

| Run setting | Value |
| --- | --- |
| Server | Same server with 96 physical cores, separate runs |
| Concurrent puzzles | Up to 100, sharing 96 scoring workers |
| Seed | 706 |
| Alpha | 0.5 |
| Uniform mixing | 0.2% |
| P/NED | Final Vienna checks |
| Total wall time | All 100 puzzles, one seed each |

The hybrid result comes from using Vienna on the elites during the search,
with both runs using 100 samples per round instead of the default 500.
These times will vary with the hardware and settings.

## Extras

[More optimization and tracking options](src/cem_rna/extras/complex/README.md)

Vienna scoring with elite structural tracking:

```sh
cem-rna-extras "(((...)))" --engine vienna --tracking elites --samples 100 --output results/extras.json
```

Extra Vienna tracking can slow the search, especially for long RNA.
