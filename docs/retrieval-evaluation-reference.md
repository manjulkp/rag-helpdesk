# Helpdesk RAG: retrieval evaluation reference

This document is a readable project reference transcribed from the PDF page
screenshots shared in the conversation. The original PDF file was not
available, so this is a best-effort transcription of the visible pages rather
than a pixel-identical copy. Benchmark figures below are the values reported
in those screenshots. They are historical reference values, not evaluation
results generated from the current workspace.

The PDF's commands and code snippets describe the same project workflow, but
the current workspace is authoritative where paths or implementation details
differ. In particular, the PDF writes runs to `results/`; this workspace uses
`result/`.

## Contents

- [Reading notes](#reading-notes)
- [Baseline and failure modes](#baseline-and-failure-modes)
- [Chunking decisions](#chunking-decisions)
- [Embeddings and Chroma](#embeddings-and-chroma)
- [Reported evaluation results](#reported-evaluation-results)
- [How to regenerate the results](#how-to-regenerate-the-results)
- [Answer generation and grading](#answer-generation-and-grading)
- [Implementation appendix](#implementation-appendix)

## Reading notes

### Chunking strategies

- **Fixed-size chunking** cuts every *N* tokens, often near the embedding
  model's context window. It is simple but can split in the middle of a
  sentence or ignore document structure.
- **Content-aware chunking** follows natural boundaries such as paragraphs,
  sentences, recursive text splits, Markdown/HTML headings, or semantic
  changes.
- The notes recommend testing roughly 128–256-token chunks against
  512–1024-token chunks, passing text that fits the embedding model's input
  window and never embedding truncated text.
- A useful rule of thumb: a chunk should make sense to a person without
  surrounding text.
- The notes discuss overlap as a trade-off: repeating boundary text can help
  when a relevant fact falls across chunks, but creates redundant chunks and
  increases indexing cost. The current chunker exposes overlap as a token
  budget option.
- Markdown and pytest files benefit from different boundaries. Markdown uses
  headings and self-contained sections. Tests are natural code units, but
  imports and fixtures may live outside a test function, so chunks need
  pointers to the relevant fixture or file rather than copying all fixture
  code into each chunk.

### Contextual retrieval

The notes describe contextual retrieval as prepending a short, generated
description of a chunk's place in the larger document before embedding and
BM25 indexing. For example, a heartbeat timeout sentence can be given context
that it belongs to the configuration engine's WebSocket heartbeat section.
The source notes cite an Anthropic experiment with retrieval failure rates:

| Setup | Failure rate | Reduction |
| --- | ---: | ---: |
| Baseline plain embeddings | 5.7% | — |
| Contextual embeddings | 3.7% | 35% |
| Contextual embeddings + contextual BM25 | 2.9% | 49% |
| Contextual embeddings + contextual BM25 + reranking | 1.9% | 67% |

The notes say the experiment passed the top 20 chunks onward and tested
retrieval depths of 5, 10, and 20; 20 worked best. These numbers are cited by
the PDF and are not measured by this project's test suite.

## Baseline and failure modes

The PDF describes the older prediction-markets RAG baseline (`pred-v1`) as:

- Sentence Transformers `all-MiniLM-L6-v2`, with 384-dimensional vectors
  and a 256-token input limit.
- A hard 3,000-character chunk cap, with no overlap.
- ChromaDB cosine-distance search in a collection named
  `prediction_markets_rag`.
- Top-8 vector-similarity retrieval, with no BM25, reranking, score cutoff,
  or chat history.
- Four chunker groups based on file type. The visible notes mention
  `.cpp/.py` splits on top-level structure, JSON scenario files rewritten as
  readable text, and Markdown/text split on headings and separators.
- Chunk metadata for path, filename, language/type, title, and line range;
  identifiers based on a hash of the file path and the beginning of the
  chunk text.

The document calls out two principal weaknesses:

1. **Silent embedding truncation.** A 3,000-character chunk can be around 838
   MiniLM tokens, while only the first 256 tokens are embedded. Later
   information in a chunk is then unsearchable. The PDF recommends splitting
   at the embedding model's token limit using natural boundaries and adding
   context.
2. **Vector-only retrieval without abstention.** Similarity search always
   returns candidates, even when no candidate answers the question. A
   minimum-score threshold or a separate reranking stage could help, but
   thresholds need to be calibrated on more negative questions.

The notes also mention coverage gaps in the older corpus: some documentation
and non-recursive source files were not indexed. In particular, only 9 of 18
`.cpp` files were reportedly included.

## Chunking decisions

The PDF's step-2 design notes and the current project decisions emphasize:

- A 400-token chunk budget for BGE-small. Its 512-token input limit leaves
  room for special tokens and a short contextual description.
- A one-line fixture pointer in pytest chunk text, with fixture names also
  stored as metadata. Resolve fixture paths within the same repository;
  do not duplicate fixture implementations into every test chunk.
- Stable chunk IDs derived from repository, relative path, title, and
  same-title occurrence. This avoids the older path/prefix hash's collisions
  when chunks start with boilerplate.
- Markdown heading-aware chunks. Small sections can be combined; oversized
  sections are split at paragraph, line, and word boundaries, with a small
  configurable overlap.
- Pytest chunks organized around module headers, fixtures, functions, and
  classes/methods. Keep comments and decorators with the code they describe.
  A syntax-error fallback uses paragraph/line splitting rather than treating
  arbitrary Python text as valid AST structure.

The current project's detailed decisions are in [NOTES.md](../NOTES.md), and
the implementation is in [chunker.py](../helpdesk/chunker.py).

## Embeddings and Chroma

The PDF's step-3 notes describe:

- One Chroma collection per embedding model under `data/chroma/`.
- Collection metadata containing the model name and chunk-token budget;
  search should reject an index built with a different model.
- Cosine distance as the Chroma HNSW space.
- Upserts in batches, followed by removal of stale IDs so rebuilding the
  collection reflects the current corpus.
- Fixture names stored as a comma-separated metadata string.
- BGE's query instruction is added to the query only, not to document text.
- A reported Chroma maximum batch size of 5,461 in the environment used for
  the PDF.

The PDF reports the following build timings for 3,292 chunks:

| Run | Embedding time | Total build time |
| --- | ---: | ---: |
| GPU fp32, first run | 39 s | 66 s |
| GPU fp32, rebuild | 70 s | 90 s |
| GPU fp16 | 71 s | 90 s |
| CPU | 643 s | 665 s |

These timings are hardware- and environment-specific. The PDF notes that the
GPU was a throttled laptop GPU and that fp16 was faster in a like-for-like
check, with cosine similarity around 0.9995 versus fp32.

It also records a five-question spot check for BGE-small with query
instructions: expected-file ranks were 3, 1, 1, 1, and 2. The notes caution
that an expected filename can be too narrow when another chunk from the same
document is the useful result.

## Reported evaluation results

The PDF says the golden set has 35 questions across five repositories:
10 concept, 10 test, 5 identifier, 5 multi-file, and 5 negative. Negative
questions have no expected file and are excluded from recall and MRR. The
tables below are transcribed from the screenshot of the report.

### Headline results

| Config | Answerable n | R@1 | R@3 | R@5 | R@8 | R@20 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bge-small | 30 | 0.60 | 0.87 | 0.93 | 0.93 | 1.00 | 0.739 |
| bge-small-noinstr | 30 | 0.57 | 0.77 | 0.83 | 0.97 | 1.00 | 0.704 |
| qwen3-0.6b | 30 | 0.53 | 0.77 | 0.90 | 0.97 | 1.00 | 0.679 |
| minilm | 30 | 0.30 | 0.67 | 0.77 | 0.80 | 0.97 | 0.507 |
| pred-v1 | 10 | — | — | — | — | — | 0.389 |

The notes interpret BGE-small with its query prefix as strongest overall,
while describing the no-instruction BGE result as a small, noisy difference.
Qwen3 is reported between the two BGE configurations; MiniLM is lowest on the
full set.

### Fair comparison: prediction-markets questions

The `pred-v1` baseline indexes only the prediction-markets repository, so the
PDF compares every configuration on the same 10 questions from that
repository:

| Config | n | R@1 | R@3 | R@5 | R@8 | R@20 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bge-small | 10 | 0.70 | 0.90 | 1.00 | 1.00 | 1.00 | 0.820 |
| bge-small-noinstr | 10 | 0.70 | 0.90 | 0.90 | 1.00 | 1.00 | 0.813 |
| qwen3-0.6b | 10 | 0.50 | 0.80 | 0.80 | 0.90 | 1.00 | 0.670 |
| minilm | 10 | 0.30 | 0.80 | 0.90 | 0.90 | 1.00 | 0.564 |
| pred-v1 | 10 | 0.30 | 0.50 | 0.60 | 0.70 | 0.70 | 0.389 |

### Breakdown by question type

| Type | n | R@1 | R@3 | R@8 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| concept | 10 | 0.70 | 0.90 | 1.00 | 0.808 |
| identifier | 5 | 0.60 | 1.00 | 1.00 | 0.767 |
| multi | 5 | 0.60 | 0.80 | 0.80 | 0.722 |
| test | 10 | 0.50 | 0.80 | 0.90 | 0.664 |

The PDF observes that test questions are difficult when the answer is in a
fixture or nearby README rather than in the test chunk itself.

### Breakdown by repository

Values are R@1 / MRR as shown in the PDF. The configs are ordered BGE-small,
BGE-small without instruction, Qwen3, then MiniLM.

| Repository | n | bge-small | bge-small-noinstr | qwen3-0.6b | minilm |
| --- | ---: | ---: | ---: | ---: | ---: |
| unity-platform | 6 | 0.67 / 0.806 | 0.50 / 0.644 | 0.67 / 0.792 | 0.50 / 0.657 |
| prediction-markets | 10 | 0.70 / 0.820 | 0.70 / 0.813 | 0.50 / 0.670 | 0.30 / 0.564 |
| option-pricer | 5 | 0.60 / 0.722 | 0.40 / 0.629 | 0.60 / 0.707 | 0.20 / 0.353 |
| forex-market-data-aggregator | 4 | 0.50 / 0.653 | 0.50 / 0.646 | 0.50 / 0.633 | 0.50 / 0.528 |
| commodities-pricer | 5 | 0.40 / 0.583 | 0.60 / 0.603 | 0.40 / 0.569 | 0.00 / 0.350 |

### Multi-file questions and negative scores

The multi-file metric is the share of multi-file questions for which **all**
expected files have appeared by the given depth:

| Config | @1 | @3 | @5 | @8 | @20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bge-small | 0.00 | 0.20 | 0.20 | 0.40 | 0.80 |
| bge-small-noinstr | 0.00 | 0.40 | 0.40 | 0.60 | 0.80 |
| qwen3-0.6b | 0.00 | 0.20 | 0.40 | 0.40 | 0.80 |
| minilm | 0.00 | 0.20 | 0.20 | 0.40 | 0.80 |
| pred-v1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

No configuration retrieves both expected files at rank 1. The screenshot notes
that BGE-small still misses one of five multi-file questions by depth 20.

The PDF also examines whether top-1 similarity scores can separate the five
negative questions from answerable questions. Its visible discussion reports
negative top-1 ranges of approximately 0.619–0.830 for BGE-small and
0.367–0.446 for MiniLM, with substantial overlap between negative and
answerable scores. A BGE threshold around 0.70 reportedly rejects all five
negatives but also rejects 13 of 30 answerable questions. The report concludes
that five negatives are not enough to calibrate a reliable cutoff and suggests
using a reranker and adding more negative examples.

### Misses and comparison notes

The report's worst-miss examples include:

- A multi-file option-pricing scenario question where an integration-test
  fixture/README ranked ahead of the expected files.
- A recorded-traffic comparison question where a helper utility outranked the
  test file.
- A prediction-market configuration question where a data-driven test module
  header ranked first.
- A commodities hand-written quality-operator question where legacy
  provenance documentation ranked ahead of the expected file.

The notes attribute misses to shared vocabulary, fixture context, broad
expected-file labels, and embedding similarity's difficulty with identifiers
or domain-specific terms. They caution that a narrow five-question category
can move by 0.1 when a single question changes rank.

The report's paired comparisons say:

- BGE-small versus MiniLM: MRR difference +0.23 on 30 questions; BGE wins on
  17, ties 11, loses 2. The report calls this the most trustworthy direction.
- BGE-small versus `pred-v1`: MRR difference +0.43 on the 10 common
  prediction-markets questions; BGE wins on 7, ties 2, loses 1. The report
  cautions that this also changes the chunking/indexing pipeline.
- Query instruction versus no instruction: MRR difference +0.035; the
  screenshot does not treat the result as conclusive.
- Qwen3 versus BGE-small: MRR difference -0.06; the report says no clear
  difference is shown.
- New MiniLM chunks versus `pred-v1`: MRR difference +0.17; chunking alone
  appears helpful, but the sample is small.

## How to regenerate the results

Run commands from the workspace root after installing dependencies. The source
repositories must exist at the paths in `helpdesk/sources.py` (or that file
must be updated), and every expected golden-set file must be present in the
indexed corpus.

Build one Chroma collection for each model:

```powershell
python -m scripts.build_index --model BAAI/bge-small-en-v1.5
python -m scripts.build_index --model sentence-transformers/all-MiniLM-L6-v2
python -m scripts.build_index --model Qwen/Qwen3-Embedding-0.6B
```

BGE-small with and without its query instruction uses the same collection.
Build the legacy prediction-markets baseline separately, with its source repo
and original RAG code available:

```powershell
python -m scripts.build_pred_index
```

Then evaluate and render tables:

```powershell
python -m scripts.evaluate bge-small bge-small-noinstr minilm qwen3-0.6b pred-v1
python -m scripts.results_table
```

Current-workspace outputs are written to `data/chroma/`,
`data/chroma_pred/`, and `result/`. Each evaluation creates a timestamped JSON
file; previous runs are retained. The PDF's original instructions refer to
`results/`, but the workspace scripts currently default to the singular
`result/` folder.

## Answer generation and grading

The answer-generation screenshots describe:

- `helpdesk/generate.py`: answer objects with citations, a Claude CLI generator,
  and an extractive QA baseline. Answers are grounded in retrieved passages;
  invalid citation numbers are removed, and an answer without a valid citation
  is treated as "not found."
- `scripts/eval_answers.py`: evaluates the golden questions with Haiku,
  Sonnet, and extractive generators. It stores one timestamped JSON run per
  generator, including retrieved source text, answers, citations, timing,
  optional cost, and summary metrics.
- `scripts/ask.py`: retrieves top-k chunks and answers one question, optionally
  printing the retrieved sources.
- `scripts/grade_sheet.py`: creates a manual faithfulness worksheet with
  answers, cited source text, and reviewer checkboxes.
- `tests/test_generate.py`: covers citation checking, failure behavior,
  extractive thresholds, metrics, and grading-sheet output.

In the current workspace, these scripts write answer runs and grading sheets
to `results/answers/`. After building the BGE-small index, run:

```powershell
python -m scripts.eval_answers
python -m scripts.grade_sheet --generator haiku
```

The default evaluation runs all three generators. Haiku and Sonnet require an
installed, authenticated `claude` CLI; extractive QA downloads its
question-answering model on first use. To ask one question:

```powershell
python -m scripts.ask "How does the retry policy work?" --generator haiku --show-sources
```

The screenshots show the intended implementation, but no answer-generation
runs are claimed for this workspace unless the commands above have been run
with a populated index and available model/CLI dependencies.

## Implementation appendix

The PDF's final pages show the embedding and index-building implementation.
In this workspace, use the source files rather than the photographed code
listing:

- [embedder.py](../helpdesk/embedder.py) wraps Sentence Transformers,
  normalizes document/query vectors, chooses a device, counts model tokens,
  and applies query instructions.
- [build_index.py](../scripts/build_index.py) loads configured repositories,
  chunks using the selected model's tokenizer, builds the model-specific
  Chroma collection, and prints counts and timings.
- [index.py](../helpdesk/index.py) implements collection naming, metadata,
  batched upserts, stale-ID removal, model checks, and cosine-ranked search.
- [evaluate.py](../scripts/evaluate.py) executes configured golden-set runs.
- [results_table.py](../scripts/results_table.py) reads saved runs and renders
  summary/detail tables.
- [generate.py](../helpdesk/generate.py) validates answer citations.
- [eval_answers.py](../scripts/eval_answers.py) writes one answer run per
  generator under `results/answers/`.
- [ask.py](../scripts/ask.py) answers one retrieved question.
- [grade_sheet.py](../scripts/grade_sheet.py) produces a manual review sheet.

The photographed appendix reflects the code state when that PDF was made. If
its exact code is required, use the original PDF; this Markdown reference
captures its visible design notes and results, not every character of the
source listings.
# Workspace implementation: BM25, hybrid, and paired runs

In addition to the historical screenshot-transcribed report below, this
workspace supports `vector`, `bm25`, and `hybrid` retrieval in the search,
retrieval-evaluation, answer-evaluation, and one-question CLI scripts.
BM25 is built in memory from the documents and metadata of the selected
model-specific Chroma collection. Hybrid retrieval fuses vector and BM25
candidates using reciprocal-rank fusion (default candidate depth 50 and RRF
constant 60). The options can be changed with `--candidates` and `--rrf-k`.

To run paired retrieval comparisons, save both evaluations with the same
golden file, then run:

```powershell
python -m scripts.compare_runs bge-small+hybrid bge-small
python -m scripts.compare_runs bge-small+hybrid bge-small --only-repo prediction-markets
```

The comparison pairs answerable questions by ID, reports wins/ties/losses, and
bootstraps the mean per-question reciprocal-rank difference with a 95%
confidence interval. `--boots` and `--seed` control the bootstrap. These are
workspace-generated comparisons only when the user runs the commands; no
corpus results are asserted here.
