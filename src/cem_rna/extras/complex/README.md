# Complex optimizer

This folder contains the complex optimizer with more tracking and optimization
options for trying different scoring engines and saving extra results.

| Scoring option | What it does |
| --- | --- |
| `--engine lazyoutside` · default | Uses approximate P and NED from LazyOutside, then checks the final sequences with Vienna |
| `--engine hybrid` | Uses LinearPartition to pick elites, then Vienna for their exact P and NED |
| `--engine vienna` | Uses Vienna to score every unique sample |

These engine options pick elites by P and use `1−NED` to weight their updates.

| Tracking option | What gets saved |
| --- | --- |
| `--tracking basic` · default | The sequences with the highest P and lowest NED, checked with Vienna |
| `--tracking elites` | Also tracks energy gap, structural distance, MFE and uMFE on elites and final sequences |
| `--tracking umfe` | Checks every unique sample for a uMFE solution until one is found, while the search continues improving P and NED |
| `--verbose` | Adds round history, progress counters and the steps where winners were found |

Extra Vienna tracking can slow the search, especially for long RNA,
whereas `--verbose` only adds saved details.

For other optimization choices, use `--mode`:

| Mode | How elites are picked | How updates are weighted |
| --- | --- | --- |
| `lazy-1ned` · default | Higher LazyOutside P | LazyOutside `1−NED` |
| `lazy-ned` | Lower LazyOutside NED | LazyOutside `1−NED` |
| `lp-log` | Higher LinearPartition P | Relative probability, adjusted by temperature |
| `lp-vienna-1ned` | Higher LinearPartition P, then reordered by Vienna P | Vienna `1−NED` |
| `lp-vienna-log` | Higher LinearPartition P, then reordered by Vienna P | Relative Vienna probability, adjusted by temperature |
| `vienna-1ned` | Higher Vienna P | Vienna `1−NED` |
| `vienna-ned` | Lower Vienna NED | Vienna `1−NED` |
| `vienna-log` | Higher Vienna P | Relative probability, adjusted by temperature |

| Other option | What it does |
| --- | --- |
| `--output` | Sets the JSON path, adding a number when the file already exists |
| `--json` | Prints only JSON in the terminal |
| `--temperature` | Adjusts the probability weights in log modes, with a default of 1 |
| `--track-sep` | Adds structural details for the final P and NED winners |
| `--track-all` | Tracks separate metric winners using search scores and Vienna checks on elites and finalists |
| `--track-umfe` | Checks samples for uMFE, or turns these checks off with `--no-track-umfe` |
| `--advanced-help` | Shows all settings, including sample size, alpha, uniform mixing and scoring limits |

## Command examples

Existing files are kept, so repeated saves use `results.json`, `results1.json`
and `results2.json`. Results are saved after all runs finish, so let the command
complete before closing the terminal as progress is not saved along the way.

Many concurrent puzzles can use a lot of RAM as each keeps a score cache, so
lower `--concurrent-puzzles` if memory is tight.

Vienna `1−NED`:

```sh
cem-rna-extras "(((...)))" --engine vienna --samples 100 --output results/vienna.json
```

Hybrid scoring with elite tracking:

```sh
cem-rna-extras --input data/eterna100.txt --engine hybrid --tracking elites --samples 500 --workers 96 --output results/eterna_hybrid.json
```

LazyOutside with five seeds, uMFE checks across samples and round history:

```sh
cem-rna-extras --input data/eterna100.txt --engine lazyoutside --tracking umfe --samples 500 --workers 96 --seeds 706 707 708 709 710 --verbose --output results/eterna_lazy.json
```

Vienna using NED only:

```sh
cem-rna-extras "(((...)))" --mode vienna-ned --samples 100 --output results/vienna_ned.json
```

Custom input file, `targets.txt`:

```text
hairpin (((...)))
short ((((...)))).
```

```sh
cem-rna-extras --input targets.txt --engine hybrid --samples 100 --output results/custom.json
```

Print only JSON in the terminal and also save it to `results/results.json`:

```sh
cem-rna-extras "(((...)))" --engine vienna --samples 100 --json
```

Each terminal row shows the sequence with the highest P alongside its NED and
final Vienna uMFE status, while the JSON file also keeps the sequence with the
lowest NED so both are available when they differ.
