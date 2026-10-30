# Answer generation notes

This folder is intended to hold the generated answer runs and grading sheets for the helpdesk RAG evaluation.

Expected output pattern after generation:

- `results/answers/<timestamp>_haiku.json`
- `results/answers/<timestamp>_sonnet.json`
- `results/answers/<timestamp>_extractive.json`
- `results/answers/grading_<timestamp>_haiku.md`
- `results/answers/grading_<timestamp>_sonnet.md`
- `results/answers/grading_<timestamp>_extractive.md`

## How to generate the runs

From the project root:

```powershell
python -m scripts.build_index --model BAAI/bge-small-en-v1.5
python -m scripts.eval_answers haiku
python -m scripts.eval_answers sonnet
python -m scripts.eval_answers extractive
```

This writes one saved answer JSON file per generator under `results/answers/`.

## How to generate the grading sheet

After any answer run exists:

```powershell
python -m scripts.grade_sheet --generator haiku
python -m scripts.grade_sheet --generator sonnet
python -m scripts.grade_sheet --generator extractive
```

Each command writes a Markdown review sheet beside the saved run in the same folder.

## Important note

The project code is ready, but the actual source repositories in `helpdesk/sources.py` must exist on the current machine before the runs can be generated. In this workspace, the configured repos are currently missing, so no real answer JSON or grading sheet was created here yet.
