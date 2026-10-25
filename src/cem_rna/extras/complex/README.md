# Complex optimizer

This folder contains the complex optimizer, with more ways to run the search
and track results along the way.

For installation and an overview of the scoring engines, see the [main README](../../../../README.md).

## Tracking options

| Tracking option | What gets saved |
| --- | --- |
| `--tracking basic` · default | Saves the sequence with the highest P and the sequence with the lowest NED after checking both with Vienna |
| `--tracking elites` | Adds energy gap, structural distance, MFE and uMFE checks for elites and final sequences |
| `--tracking umfe` | Checks each unique sample for uMFE until a solution is found, while the search keeps improving P and NED |

## Optimization modes

Use `--mode` to change how the search picks and weights elites:

| Mode | How elites are picked | How updates are weighted |
| --- | --- | --- |
| `lazy-1ned` · default | Picks sequences with higher LazyOutside P | Weights each elite by its LazyOutside `1−NED` |
| `lazy-ned` | Picks sequences with lower LazyOutside NED | Weights each elite by its LazyOutside `1−NED` |
| `lp-log` | Picks sequences with higher LinearPartition P | Gives more weight to elites with higher P, with temperature controlling how strongly they are favored |
| `lp-vienna-1ned` | Uses LinearPartition P to pick a shortlist, then ranks those sequences by Vienna P | Weights each elite by its Vienna `1−NED` |
| `lp-vienna-log` | Uses LinearPartition P to pick a shortlist, then ranks those sequences by Vienna P | Gives more weight to elites with higher Vienna P, with temperature controlling how strongly they are favored |
| `vienna-1ned` | Picks sequences with higher Vienna P | Weights each elite by its Vienna `1−NED` |
| `vienna-ned` | Picks sequences with lower Vienna NED | Weights each elite by its Vienna `1−NED` |
| `vienna-log` | Picks sequences with higher Vienna P | Gives more weight to elites with higher P, with temperature controlling how strongly they are favored |

## Other options

| Other option | What it does |
| --- | --- |
| `--temperature` | Changes how strongly probability weights favor the better elites in modes ending in `-log`. Default: 1 |
| `--track-sep` | Saves structural details for the final sequence with the highest P and the one with the lowest NED |
| `--track-all` | Keeps a separate best sequence for each metric, using search scores and Vienna checks on elites and final sequences |
| `--track-umfe` | Checks samples for uMFE, with `--no-track-umfe` available to turn these checks off |
| `--advanced-help` | Shows all settings, including sample size, alpha, uniform mixing and scoring limits |

## Command examples

Run with Vienna `1−NED`:

```sh
cem-rna-extras "(((...)))" --engine vienna --samples 100 --output results/vienna.json
```

Use hybrid scoring and track structural details for the elites:

```sh
cem-rna-extras --input data/eterna100.txt --engine hybrid --tracking elites --samples 500 --workers 96 --output results/eterna_hybrid.json
```

Run LazyOutside with five seeds, check samples for uMFE and save the round history:

```sh
cem-rna-extras --input data/eterna100.txt --engine lazyoutside --tracking umfe --samples 500 --workers 96 --seeds 706 707 708 709 710 --verbose --output results/eterna_lazy.json
```

Run with Vienna NED only:

```sh
cem-rna-extras "(((...)))" --mode vienna-ned --samples 100 --output results/vienna_ned.json
```
