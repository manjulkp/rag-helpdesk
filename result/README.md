# Retrieval evaluation output notes

This directory is for evaluation runs produced by the retrieval benchmarking scripts.

The project uses:

- `result/` for timestamped retrieval-evaluation JSON runs
- `results/answers/` for generated answer runs and grading sheets

## Example generation commands

```powershell
python -m scripts.evaluate bge-small minilm
python -m scripts.results_table
```

These commands generate timestamped JSON result files in `result/` and print the summary table.

## Important note

The actual source repositories referenced in `helpdesk/sources.py` must be available to generate real evaluation output. The current workspace does not have those repositories configured locally, so the data files were not produced here as part of this run.
