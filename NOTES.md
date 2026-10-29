# Chunking decisions

For the full screenshot-derived retrieval-evaluation reference, including
reported benchmark tables and regeneration steps, see
[docs/retrieval-evaluation-reference.md](docs/retrieval-evaluation-reference.md).

- `max_tokens` is 400: BGE-small has a 512-token sequence limit, leaving about 112 tokens for the contextual description planned for step 9.
- Pytest chunks retain requested fixture names as metadata and add a short pointer line. Fixture pointers resolve only within the same repository; implementations are not copied into every test chunk.
- Chunk IDs use repository, relative path, title, and same-title occurrence. Editing chunk text does not change its ID; repeated titles in one file remain distinct.
- Token overlap defaults to 10% of the chunk budget, and tiny Markdown sections default to 1/16 of the budget. Both can be explicitly configured.
- The earlier RAG used 838-token chunks. The 400-token limit is about half that size and leaves room for added context without truncation.

## Step 4: results

The following is the benchmark summary transcribed from the shared PDF
screenshots. It describes the source report, not a run generated in this
workspace. The screenshot reports that the five embedding configurations were
evaluated on the same 35-question golden set, with five negative questions
excluded from recall and MRR. The `pred-v1` baseline was scored only on the
10 prediction-markets questions.

### Headline metrics

| Config | n | R@1 | R@3 | R@5 | R@8 | R@20 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bge-small | 30 | 0.60 | 0.87 | 0.93 | 0.93 | 1.00 | 0.739 |
| bge-small-noinstr | 30 | 0.57 | 0.77 | 0.83 | 0.97 | 1.00 | 0.704 |
| qwen3-0.6b | 30 | 0.53 | 0.77 | 0.90 | 0.97 | 1.00 | 0.679 |
| minilm | 30 | 0.30 | 0.67 | 0.77 | 0.80 | 0.97 | 0.507 |
| pred-v1 | 10 | — | — | — | — | — | 0.389 |

### BGE-small breakdown by question type

| Type | n | R@1 | R@3 | R@8 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| concept | 10 | 0.70 | 0.90 | 1.00 | 0.808 |
| identifier | 5 | 0.60 | 1.00 | 1.00 | 0.767 |
| multi | 5 | 0.60 | 0.80 | 0.80 | 0.722 |
| test | 10 | 0.50 | 0.80 | 0.90 | 0.664 |

### Negative-query score cutoff

The report says a BGE-small top-1 score cutoff of 0.75 rejects all five
negative queries, but also rejects 13 of the 30 answerable queries. The
negative and answerable top-1 score ranges overlap, so the report concludes
that this cutoff does not reliably identify "not found." It recommends
collecting more negative examples and calibrating abstention separately.

### Five worst BGE-small misses

These question IDs, ranks, and first-result summaries are from the report's
miss table. The explanations are the report author's hypotheses.

| ID | Rank | Question summary | What ranked first | Suspected reason |
| --- | ---: | --- | --- | --- |
| q29 | 9 | Base QC scenarios and tests built on them | Option-pricer integration-test README | Shared "scenarios"/"integration tests" vocabulary; expected file may be too narrow. |
| q18 | 9 | Test comparing option prices against recorded traffic | `tests/utils/compare_utils.py` | Helper logic is relevant, but the expected test file ranked lower. |
| q15 | 5 | Test for what the config engine stores for each prediction-market instrument | Prediction-markets `test_data_driven.py` module header | Shared vocabulary; the module header may not describe individual test contents. |
| q07 | 4 | Why the commodities cache has a hand-written equality operator | Commodities legacy-provenance document | Vocabulary/code-token mismatch between the question and source. |
| q03 | 3 | Switching source when one goes down | Unity config-engine data document | Plain-language question versus implementation terminology. |

The report also notes that q05, q06, and q27 are answered in documentation
that the `pred-v1` baseline does not index, while q12 is affected by the older
baseline's 3,000-character chunks being truncated by MiniLM's 256-token input
limit.

### Which differences to trust

The report compares paired per-question reciprocal ranks and gives bootstrap
95% confidence intervals. The sample sizes are small, especially for the
10-question prediction-markets-only comparisons.

| Comparison (MRR difference) | Bootstrap 95% CI | Wins / ties / losses | Report's interpretation |
| --- | --- | ---: | --- |
| BGE-small vs MiniLM (+0.23, n=30) | +0.12 to +0.34 | 17 / 11 / 2 | Strongest directional evidence; consistent across questions. |
| BGE-small vs pred-v1 (+0.43, n=10) | +0.11 to +0.74 | 7 / 2 / 1 | Direction looks positive, but chunking and indexing also differ. |
| BGE instruction vs no instruction (+0.035, n=30) | -0.04 to +0.12 | 6 / 22 / 2 | Interval includes zero; no reliable difference established. |
| Qwen3 vs BGE-small (-0.06, n=30) | -0.21 to +0.08 | 8 / 14 / 8 | No reliable difference established. |
| New MiniLM chunks vs pred-v1 (+0.17, n=10) | -0.07 to +0.43 | 5 / 2 / 3 | Chunking may help; the sample is too small for a firm conclusion. |

The report cautions that the 5-question type and 4–6-question repository
breakdowns are noisy: one question changes a rate by 0.1–0.25. It suggests
more questions and paired tests before making fine-grained claims.

For the full screenshot transcription, including the prediction-markets
comparison and multi-file table, see
[docs/retrieval-evaluation-reference.md](docs/retrieval-evaluation-reference.md).

## Generating the Chroma indexes and evaluation results

The retrieval-evaluation screenshots show one Chroma index per embedding model,
a separate index for the legacy `pred-v1` baseline, timestamped evaluation
JSON files, and Markdown tables generated from those JSON files. The commands
below describe how to produce those outputs; they have not been run as part of
this note.

### Prerequisites

- Place the source repositories in the locations configured in
  `helpdesk/sources.py`, or update `REPOS` there to point to their locations.
- Check `data/golden.json` against those repositories. Every answerable
  expected file must be present in the indexed corpus for its repository;
  negative questions must have an empty `expected` list.
- Activate the project virtual environment and install the project with its
  development dependencies as described in `README.md`.

### Build the indexes

Run these commands from the project root in PowerShell. Each model-specific
build uses the same repository corpus and writes a model-specific collection
under `data/chroma/`.

```powershell
python -m scripts.build_index --model BAAI/bge-small-en-v1.5
python -m scripts.build_index --model sentence-transformers/all-MiniLM-L6-v2
python -m scripts.build_index --model Qwen/Qwen3-Embedding-0.6B
```

The evaluation configurations `bge-small` and `bge-small-noinstr` share the
BGE index; they differ in whether the query instruction is added at search
time. `minilm` and `qwen3-0.6b` use their respective model collections.

To build the separate legacy baseline, make sure the prediction-markets
repository and its original RAG code are available at the paths expected by
`scripts/build_pred_index.py`, then run:

```powershell
python -m scripts.build_pred_index
```

That command writes the `pred-v1` collection under `data/chroma_pred/`.

### Run evaluations and print tables

After all required indexes are built, run:

```powershell
python -m scripts.evaluate bge-small bge-small-noinstr minilm qwen3-0.6b pred-v1
python -m scripts.results_table
```

Each completed configuration writes a new timestamped JSON run under
`result/`; prior runs are not overwritten. The results-table command reads
those saved runs and prints the overall metrics, the prediction-markets-only
comparison when applicable, and details for the full-set run with the highest
MRR. To choose a different run for the detail breakdown, use:

```powershell
python -m scripts.results_table --detail minilm
```

The screenshot's scores and chunk counts depend on its source repositories,
golden-set paths, model versions, and built indexes. They are not generated by
these instructions and should not be treated as results for this workspace
until the commands have been run successfully against the intended corpus.

## Step 9: contextual index variants (plain / header / llm)

Three bge-small indexes over the same 97 chunks (2 repos, 25 documents):
`plain` (chunk only), `header` ("repo / path / heading" + chunk, free) and
`llm` (Claude Haiku's one-or-two sentence situating context + chunk). Hits
still return the original chunk. 10 answerable questions, so treat every
difference as indicative only.

| Run | R@1 | R@5 | MRR | ansR@8 | ansR@20 |
|---|---|---|---|---|---|
| bge-small | 0.60 | 0.90 | 0.77 | 0.90 | 0.90 |
| bge-small@header | 0.60 | 0.90 | 0.73 | 0.70 | 0.90 |
| bge-small@llm | 0.70 | 1.00 | 0.85 | 0.80 | 1.00 |
| bge-small+bm25 | 0.60 | 0.90 | 0.70 | 0.60 | 0.80 |
| bge-small@header+bm25 | 0.60 | 0.90 | 0.72 | 0.80 | 0.90 |
| bge-small@llm+bm25 | 0.60 | 0.80 | 0.71 | 0.80 | 0.90 |
| bge-small+hybrid | 0.60 | 0.90 | 0.76 | 0.80 | 0.90 |
| bge-small@header+hybrid | 0.70 | 0.90 | 0.79 | 0.80 | 0.90 |
| bge-small@llm+hybrid | 0.70 | 1.00 | 0.82 | 0.90 | 1.00 |

### Paired bootstrap 95% CIs (`scripts.compare_runs`)

| A - B | MRR diff [CI] | ansR@20 diff [CI] | W/T/L (MRR) |
|---|---|---|---|
| header+hybrid - hybrid | +0.033 [-0.050, +0.149] | 0 [0, 0] | 1/7/2 |
| llm+hybrid - hybrid | +0.059 [-0.150, +0.310] | +0.10 [0, +0.30] | 2/5/3 |
| llm+hybrid - header+hybrid | +0.026 [-0.150, +0.262] | +0.10 [0, +0.30] | 1/7/2 |
| llm - plain (vector) | +0.083 [0.000, +0.200] | +0.10 [0, +0.30] | 2/8/0 |
| llm - header (vector) | +0.120 [-0.075, +0.295] | +0.10 [0, +0.30] | 4/5/1 |

Every CI touches or crosses 0. The `llm` gain is one question (the one answer
chunk none of plain/header found in the top 20) and two rank-1 improvements.

### Part-k

`scripts/part_k.py`: no labelled answer chunk sits in a section the chunker
split into several pieces, so there is nothing to measure on this golden set.
It needs questions whose answer is in a later piece of a long section.

### Negatives

Highest top-1 score of a negative question vs lowest top-1 score of an
answered question (vector):

| Variant | negatives max | answerable min | gap |
|---|---|---|---|
| plain | 0.648 | 0.688 | 0.040 |
| header | 0.650 | 0.689 | 0.039 |
| llm | 0.643 | 0.701 | 0.058 |

No negative passes a cut-off between the two in any variant; the llm context
widens the gap a little, header does not change it.

### Indexing cost and the caching diagnosis

- First trial (research-desk, 43 chunks, prompt v1): $0.024, $0.00056/call,
  only 16% of input tokens read from the prompt cache.
- Diagnosis (3 sequential + 4 parallel chunks of one document): every call
  read 524 tokens (the fixed system prompt) and wrote ~2,800. Sequential calls
  missed too, so parallelism was not the cause. The document and chunk were
  one user message, and the cache matches whole blocks, so the document was
  never reused.
- Fix (prompt v2): document in the system prompt, chunk alone in the user
  message, first chunk of each document before the rest. Later calls read
  2,476 and write ~800 tokens.
- Full run, 97 chunks, 4 at a time: 97 calls, 0 failures, 240 s, $0.036
  ($0.00037/call, 34% cheaper), 49% of input tokens read from cache (most
  documents here have only a few chunks, so the first-call write dominates).
- Re-indexing when one document changes: only chunks whose text changed get a
  new call (the cache key is the chunk), about $0.0004 each; re-embedding is
  local. A 12-chunk document rewritten in full is about $0.005. Unchanged
  chunks keep a context written against the old document.

### Verdict

On this corpus the LLM context is the best variant on every vector and hybrid
metric and costs under four cents to build, so it is worth keeping; but the
evidence is one extra answer found out of 10, with no CI clear of zero. The
free header helps BM25 (ansR@8 0.60 -> 0.80) and not vector search. Default
to `llm+hybrid`, and re-check once the golden set has more questions.

### Full pipeline on the best variant

`python -m scripts.eval_answers haiku --variant llm --retriever hybrid
--reranker gated-claude --candidates 20 -k 8` (cross-encoder gate, Claude
rerank of the top 20, Haiku answers from the top 8), all 15 questions:

| Answered (answerable) | Refused (negatives) | Citation validity | Citation recall | Citation precision | Latency | Answer cost |
|---|---|---|---|---|---|---|
| 10/10 | 5/5 | 1.00 | 0.92 | 0.57 | 11.3 s/q | $0.012 total |

Every answerable question is answered and every negative refused. Citations
always point at retrieved sources and cover 92% of the expected files;
precision is lower because Haiku also cites related files outside the golden
labels. The answer cost excludes the Claude rerank calls, which the reranker
does not record.

Against plain Haiku on plain vector search (`python -m scripts.eval_answers
haiku`, no rerank, run as the step 7/8 baseline):

| Pipeline | Answered | Refused | Cit. precision | Cit. recall | Answer cost |
|---|---|---|---|---|---|
| plain vector, Haiku | 10/10 | 5/5 | 0.59 | **1.00** | $0.012 |
| llm+hybrid, gated Claude rerank N20, Haiku | 10/10 | 5/5 | 0.57 | 0.92 | $0.012 + rerank |

On 15 questions the full pipeline adds latency and rerank calls and buys
nothing measurable; Haiku already answers and refuses perfectly from plain
retrieval.

### Step 9 follow-up checks

**llm vs header, question by question (9.7a).** Where they differ:

| Question | Winner | Why |
|---|---|---|
| identifier-01 (tickers for gold, Brent, gas) | **llm**, answer chunk rank 1-2 vs 14-19 | the LLM context says "gold (GC=F) ... natural gas (NG=F)", bridging names and tickers; a header can't |
| test-02, concept-04 (vector) | llm, rank 1-2 vs 2-4 | the context restates what the test or section checks in plain words |
| concept-01, multi-01, concept-03 | **header** | the section title ("Fixed Synthetic Answers", "Risk Rules") already matches the question; the LLM sentence dilutes it |
| multi-02 (vector) | llm 4 vs header 11; (bm25) header 8 vs llm 15 | mixed |

The LLM context earns its keep on vocabulary the chunk lacks (identifiers vs
names); the header wins when the heading is the best keyword. Some LLM
contexts are cut mid-sentence by the 100-token cap.

**Split sections (9.7b).** No labelled answer chunk is a later piece of a split
section, so `scripts/part_k.py` has nothing to measure on this golden set.

**Negatives through the gate (9.7c).** Best MiniLM cross-encoder score in the
top 30 hybrid candidates (cutoff 0.1, as gated-claude uses):

| Variant | gate scores | negatives max | answerable min | blocked answerable |
|---|---|---|---|---|
| plain | chunk | -10.59 | -2.21 | identifier-01, multi-01 |
| header | chunk | -10.59 | -1.60 | identifier-01, multi-01 |
| header | context + chunk | -10.71 | **+1.44** | none |
| llm | chunk | -10.59 | -1.60 | identifier-01, multi-01 |
| llm | context + chunk | -9.64 | -0.66 | concept-04, multi-01 |

No negative comes near the cutoff in any variant. The near misses are on the
answerable side: the gate scores only the bare chunk, so two real questions
fall under it and skip the Claude rerank. Scoring the header + chunk lifts every
answerable question over the cutoff and leaves negatives at -10: the header
makes near misses look answerable, the LLM context less so. Worth changing
GatedReranker to score `indexed_text` (not done yet).

**Re-index cost when one document changes (9.7e).** Worst case here:
`docs/interview-prep.md`, 12 chunks of a 1,301-token document. Re-writing all
its contexts is 12 Haiku calls, ~$0.0055 (first call ~$0.0009 writes the
document to the prompt cache, the rest ~$0.0004 each). Only chunks whose text
changed get a new call (the cache key is the chunk), so a one-paragraph edit
costs a fraction of a cent, but unchanged chunks keep a context written against
the old document. Embeddings are local; at the time `build_index` re-embedded
all 97 chunks (~6 s). Since step 12 it re-embeds only changed chunks (one
edited chunk: 0.08 s, see "Freshness" under Step 12).

**Against Anthropic's published numbers.** Anthropic's "Introducing Contextual
Retrieval" (19 Sep 2024; figures checked against the post) reports the top-20
retrieval failure rate falling from 5.7% to 3.7% (-35%) with contextual
embeddings, to 2.9% (-49%) with contextual embeddings + contextual BM25, and to
1.9% (-67%) with reranking added, at a one-time $1.02 per million document
tokens with prompt caching, for contexts of "usually 50-100 tokens". Here top-20
answer-chunk failures went from 1 of 10 to 0 of 10 for vector and hybrid: in
the same direction, but one question cannot measure a percentage. Our cost was
$0.036 for ~12,900 document tokens, about **$2.8 per million**, roughly 2.7x
Anthropic's, because:

- the documents are small (3.9 chunks each), so the first-call cache write is
  a large share and only 49% of input tokens were cache reads;
- the CLI adds a fixed ~524-token system prompt and a JSON schema to each call;
- output dominates: 22,870 output tokens, more than the corpus itself, at
  Haiku's higher output price; the contexts run to 100 tokens where
  Anthropic's prompt asks for 50-100.

**Verdict (9.8).** At under four cents to build and about half a cent to
re-index the worst document, the LLM context is cheap enough to keep, and it
fixes the one failure class a header cannot: vocabulary missing from the chunk
(identifier-01). But on this 97-chunk corpus with 10 answerable questions no
difference has a CI clear of zero, the header is free, better for BM25 and
best for the gate, and plain Haiku already answers everything. Default:
**llm+hybrid for retrieval with the header used for the gate**; revisit when
the golden set covers more identifier-style and split-section questions,
where the LLM context should matter most.

## Step 10: multi-turn conversation

Follow-up questions are searched in context. Golden set:
`data/golden_conversations.json`, 12 conversations (27 turns) over the two
repos: 3 pronoun, 2 ellipsis ("And for the trade desk?"), 2 drill-down,
2 topic switch, 2 correction, 1 unanswerable follow-up; every turn has
expected files and answer chunks (all titles checked against the index).
Retrieval is bge-small `@llm+hybrid`, depth 20. Retrieval history is the
earlier user questions; "sticky" sources are the previous turn's top 3 hits.
Run: `python -m scripts.eval_conversations --answer`.

### Answer-chunk recall@20 on follow-up turns (n = 12 answerable)

| Config | all | correction | drill-down | ellipsis | pronoun | topic switch |
|---|---|---|---|---|---|---|
| raw | 0.75 | 1.00 | 1.00 | 0.00 | 0.75 | 1.00 |
| concat | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| condense | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| condense + sticky | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Raw fails every ellipsis turn ("And for the trade desk?" has nothing to
search) and one pronoun turn. With a 97-chunk corpus and depth 20, concat and
condense both reach the answer chunk every time, so recall@20 alone cannot
separate them here; the topic-switch and prompt checks below can.

### Paired comparison (answerable follow-up turns, n = 12)

`python -m scripts.eval_conversations --from <saved run>` recomputes these
from a saved run without new LLM calls.

| Config | ansR@8 | ansR@20 | answer-chunk MRR |
|---|---|---|---|
| raw | 0.75 | 0.75 | 0.60 |
| concat | 0.92 | 1.00 | 0.46 |
| condense | 1.00 | 1.00 | 0.62 |
| condense + sticky | 1.00 | 1.00 | 0.62 |

| A - B (bootstrap 95% CI, W/T/L) | ansR@8 | ansR@20 | MRR |
|---|---|---|---|
| condense - raw | +0.25 [0.00, +0.50] 3/9/0 | +0.25 [0.00, +0.50] 3/9/0 | +0.03 [-0.25, +0.31] 5/4/3 |
| condense - concat | +0.08 [0.00, +0.25] 1/11/0 | 0 [0, 0] | +0.16 [-0.02, +0.35] 8/2/2 |
| condense+sticky - condense | 0 | 0 | 0 |

With 12 turns no CI is clear of zero; what holds is the direction. Condense
never loses to raw on recall (3 wins, 0 losses), and against concat it finds
the same answers but ranks them higher on 8 of 12 turns: concat's joined
history finds the chunk, the old topic's words push it down.

Sticky sources change no rank on any turn. `with_sticky` only adds a previous
source that is not already in the top 20, and puts it at the end (ranks
18-20), so it can never push the new answer out of the top 8; here the
previous sources were always already retrieved. A version that puts sticky
sources first would cost ansR@8 on drill-downs and topic switches, where the
answer is a new chunk, which is the drop seen elsewhere (0.60). Sticky belongs
behind the new hits, as rerank candidates, not in front of them.

### Topic switches

Share of the top 8 that comes from the previous turn's files ("drag"):
raw 0.00, **concat 0.25**, condense 0.06, condense + sticky 0.06. Concat drags
the old topic into a quarter of the results; condense returned switch-01
unchanged, but on switch-02 it appended "in the trade desk ..." from the old
topic (retrieval and answer were still right). The unanswerable follow-up has
the same near-zero top-1 score (0.03) in every config and was refused.

### Prompt tokens

One 10-turn conversation (estimated, ~4 chars/token), whole prompt at turn 10:
**3,130 with full history vs 1,916 summarised** (history 1,720 vs 506). With
summarising (last 4 turns in full, older turns one line each, 600-token
budget) the history stays flat at ~500 tokens from turn 5 on; without it, it
grows by ~150-200 tokens a turn. Measured Haiku prompts in the answer run:
2,466-4,171 input tokens per turn.

### Haiku answers (condense + sticky)

13 follow-ups, graded by hand in
`results/conversations/grading_20261028T183548Z.md`: 12 correct, 1 correctly
refused, 0 wrong; $0.022 for all 25 answers. The first run found two answers
that added an unprompted "corrections to my earlier answer" paragraph (the
history invited the model to re-check itself) and a 7,938-token prompt from
carrying that into the history; the system prompt now says to answer only the
current question, and the re-run had none.

### Bugs found on the way

- Windows encoding: `subprocess.run(..., text=True)` sent prompts in cp1252,
  so any non-cp1252 character (a byte-order mark from PowerShell piping, `−`,
  `×`) made the whole prompt vanish ("no stdin data received") and decoded
  Claude's UTF-8 replies as mojibake. Both Claude wrappers now use UTF-8.
- One Haiku call failed the answer schema after 5 retries; the evaluation now
  records a failed turn and carries on instead of stopping.

### Verdict

Use **condense** (with sticky sources in chat): it matches concat on recall,
drags the old topic in far less on a switch, and gives the answer model a
clean standalone question. Raw is not usable for follow-ups (0/2 on
ellipsis). Sticky adds nothing measurable at depth 20 on this corpus; keep it
for drill-downs in chat, where the reranker sees only a short list. Cost is one
Haiku condense call per follow-up (cached by history). Summarising history
keeps turn-10 prompts ~40% smaller with no loss on these conversations.
`python -m scripts.chat` handles the "Does it get rounded?" pronoun follow-up
(the corpus has no Mongo; same shape as "Does it retry?").

## Step 11: clarify, refuse, escalate

Before answering, `helpdesk/policy.py` decides what a turn needs: answer,
clarify, out_of_scope or escalate. Final design:

1. Rules, no LLM: asking for a person or saying the answer was wrong ->
   escalate; two refusals in a row -> escalate; prompt-injection or
   system-prompt requests -> out_of_scope; after a clarification -> answer
   (never a second one). Rules check what the user typed as well as the
   condensed question.
2. Scope, before ambiguity: a question naming a project is in scope; otherwise
   one cached Haiku call ({in_scope, reason}) decides. Out of scope -> refuse.
3. Ambiguity rule: no project named and the two projects' top hits score
   within 80% of each other -> clarify, offering only those projects.
4. Otherwise answer. Escalation for an in-scope question comes *after a
   refusal*: the gate finds nothing (score < 0.1, the generator is skipped)
   or the generator answers "not found".

Escalation (`helpdesk/escalate.py`) writes a
local ticket to `data/escalations/` with a suggested contact from `git log`
(most active recent non-merge author; a suggestion, not an owner). Nothing is
posted anywhere. The flow is `helpdesk/pipeline.py`, shared by
`scripts/chat.py` and `scripts/eval_policy.py`.

Golden set: `data/golden_policy.json`, 30 cases / 34 evaluated turns
(answer 8, clarify 6, out of scope 6 incl. 2 attacks, escalate 6, answer
after a clarification 4).

### Design decision (11.2b): who decides "is it answerable?"

The first versions let the policy LLM choose answer / out_of_scope /
escalate from the top 5 hit *titles*, before the reranker or generator had
read anything. That is the wrong place to judge answerability: titles rarely
show whether the answer is there, while the answer step reads the text and
is very good at refusing (5/5 negatives in steps 7-9). So the policy now
decides only scope (and clarify), and an in-scope question escalates after a
refusal. Scope also moved before the ambiguity check: an off-topic question
whose hits happen to span both projects must be refused, not clarified.

### Before and after (34 evaluated turns)

| Version | rules only | rules + Haiku | turns needing the LLM |
|---|---|---|---|
| v1: LLM picks answer/clarify/out_of_scope/escalate from hits | 25 | 31 | 16 |
| v2: spec order, typed-text rules, LLM picks answer/out_of_scope/escalate | 27 | 32 | 13 |
| **v3: LLM decides scope only; escalate after a refusal** | 29 | **32** | 14 |

Result files: `results/policy/20261028T190922Z.json` (v1),
`20261028T201014Z.json` (v2), `20261028T203954Z.json` (v3).

Final confusion matrix (v3, rules + Haiku scope, final reply action):

| expected \ predicted | answer | clarify | out_of_scope | escalate |
|---|---|---|---|---|
| answer (12) | **11** | 0 | 0 | 1 |
| clarify (10) | 0 | **10** | 0 | 0 |
| out_of_scope (6) | 0 | 0 | **5** | 1 |
| escalate (6) | 0 | 0 | 0 | **6** |

- v1's "This answer is wrong." miss was a bug, not tuning: the rule checked
  the condensed question, which condensing had rewritten.
- v3 fixes v2's two misses ("Should I buy gold?", "What's the weather?"):
  scope runs before ambiguity, so they are refused instead of clarified.
- v3's two misses: "Translate 'risk manager' into French" contains a project
  alias, so the "names a project, so in scope" shortcut skips the scope call
  and the empty gate escalates it. And "How do I run the tests for the research
  desk?" (after a clarification) escalates because retrieval misses the
  README's Run section; escalating is the right fallback for that.
- The three in-scope-but-undocumented questions (owners, SLA, releases) now
  escalate because the answer step finds nothing, not because a title guess
  said so.
- Over-clarifying: 0/12 answerable turns. Under-escalating: 0/6.
- Clarification options: 10/10 grounded in the hits and the expected pair.
  After a clarification, the condensed question names the chosen project.
- The ambiguity check looks at the fused hits *before* reranking: a reranker
  pulls its top 8 towards one project and hides the split a clarification is
  for (tested with a reranker that does exactly that). The eval runs without
  a reranker, so its numbers are unchanged.
- Live check (`scripts.chat`, final design): "How do I set up the Python
  environment?" -> clarify -> "The trade desk." -> cited answer; "Should I buy
  gold this week?" -> out of scope (the scope call notes the gold hits are
  fixtures); "Who is the on-call owner for the trade desk's production
  deployment?" -> the answer step finds nothing -> escalation ticket.
- Tickets end with "Drafted locally by the help desk; not sent anywhere."
- Both attacks refused by rule; the generator is never called (tested with a
  spy). The generator's system prompt holds only answering rules, nothing
  worth hiding.
- Suggested contacts: 0/6. Git isn't installed here and the source folders
  aren't git repositories, so there is no history to suggest from; the
  toy-repo test skips until git is available.

### What's wrong

- In the chat check, "How do I run the tests for the research desk?" (after
  the clarification) was answered "not found": the README's Run section isn't
  in the top 8, the word "tests" pulls in test-file module headers. The policy
  eval's "expected file retrieved 4/4" missed this because it checks files,
  not answer chunks.
- Escalation replies said "I couldn't find this in the docs" even when the
  user just asked for a person; fixed to match the reason.

### Verdict: before real users try it

1. Drop the "names a project, so in scope" shortcut, or only take it when the
   gate is above the cutoff: a project word inside an off-topic request
   ("translate 'risk manager'") shouldn't skip the scope check.
2. Fix the retrieval miss on "how do I run the tests" (test-file module
   headers crowd out README sections); every escalation of an answerable
   question costs a person's time.
3. Label policy cases with answer chunks, not just files, so a clarified
   follow-up that retrieves the wrong section counts as a miss.
4. Install git and point the repos at real clones, or replace git authorship
   with a maintained owners file; a hand-off with no contact is a dead end.
5. Measure on real traffic: 30 hand-written cases can't show the
   over-clarifying rate users will feel.

Run: `python -m scripts.eval_policy` (~8 minutes: it generates real Haiku
answers, since escalation now depends on the answer step); `python -m scripts.chat`.

## Step 12: access control, freshness, conflicting sources

### Access control

`data/acl.json`: three simulated users, `rd-analyst` (research desk only),
`td-risk` (trade desk only) and `admin` (everything), via groups.
`helpdesk/access.py` turns a user into a `{"repo": {"$in": [...]}}` filter,
applied to both vector (Chroma) and BM25 search, and `pipeline.respond`
filters every other path a chunk can take a second time: sticky sources, the
reranker's candidates, the gate, the policy's hits and clarification options,
the generator's sources, the hand-off ticket. Two layers, so one missed
`where` can't leak.

Decisions:
- A forbidden repo looks like a missing one: "I couldn't find this in the
  docs", never "you don't have access", because confirming that a document
  exists is itself a leak.
- Clarification options and the out-of-scope reply name only projects the
  user may see; the chat's policy/debug line (which can name any project) is
  off unless `--debug`.
- Sessions are bound to a user: loading someone else's raises
  `PermissionError`, and the chat gives the same "no session" message for
  someone else's session as for a missing one.

Result (`scripts/eval_access.py`, 15 golden questions x 3 users, full
pipeline with Haiku, sticky sources deliberately poisoned with the
unrestricted top 3 hits to exercise the second layer):

| User | Answerable from allowed repos | Questions from a forbidden repo | Negatives |
|---|---|---|---|
| admin | 9/10 answered | - | 5/5 refused |
| rd-analyst | 6/6 answered | 4/4 "couldn't find" + hand-off | 5/5 refused |
| td-risk | 4/4 answered | 6/6 "couldn't find" + hand-off | 5/5 refused |

- **Leaks: 0** (re-run with the corrected checker:
  `results/access/20261028T211538Z.json`). The first run's checker flagged 3, all false positives,
  inspected by hand: two hand-off tickets echoed the user's own question
  ("Which research-desk tests ..."), and one research-desk answer quoted its
  own README, which names the "Trader/RiskManager" project. The checker now
  looks only at the system-filled parts of tickets and ignores names the user
  typed or an allowed source contains. Leaks are also asserted in
  `tests/test_enterprise.py` (vector + BM25 + sticky filters, options, hand-off
  files, out-of-scope text, sessions).
- Allowed-repo answer rate matches steps 9/11, except admin's concept-04 ("When
  live market data is enabled...") got a clarifying question: it names no
  project and admin sees both. td-risk, who sees only the trade desk, got an
  answer. An over-clarification from the step 11 ambiguity rule, not a
  step 12 problem.
- What a user sees for a forbidden repo's question: the same "I couldn't find
  this" + hand-off as for any undocumented question. The hand-off goes to a
  person, who can decide whether to grant access.

### Freshness

- Citations show "last changed <date>" from the last git commit, in the
  sources Claude sees and in the chat. Untracked files show "unknown", not the
  file's modified time: copying or checking out a file resets that, so it would
  look like freshness it isn't. Here every file is "unknown" (no git).
- Incremental re-index (`build_index` stores a hash of each chunk's embedded
  text): only new or changed chunks are re-embedded; a chunk whose text is the
  same but whose lines moved gets a metadata-only update.
  `scripts/reindex_timing.py`, editing the middle of
  `docs/interview-prep.md` (12 chunks): full build 5.34 s (97 embedded);
  rebuild with nothing changed 0.01 s; after the edit 0.08 s, 1 chunk
  re-embedded and 5 metadata-only updates, about 1% of a full build.
- Not done: preferring newer chunks when scores are close. With every date
  "unknown" here there is nothing to prefer on.

### Conflicting sources

The generator's system prompt now says: when sources disagree, say so, cite
both with their dates, prefer the newer one and say that's why; if dates are
unknown or equal, say you can't tell; never silently pick one.

The corpus has no genuine conflicts: after reading the main docs, the one
candidate (research desk README: Python 3.12 target vs a 3.14 venv) is
explained in the README itself, so it became a no-conflict control. The
conflict cases are synthetic (`data/golden_conflicts.json`,
`scripts/eval_conflicts.py`): a real chunk plus a copy with one number
changed under another path and date.

| Case | Graded by hand |
|---|---|
| 3 older copies (thresholds, max loss, gold's move) | 3/3 flag the conflict, keep the current value, give the dates as the reason |
| 1 newer copy (turn limit six -> ten) | flags it and takes the newer value (ten), not the original |
| 1 with both dates unknown | flags it and says it can't tell which is current |
| 2 controls (no conflict) | 0 invented conflicts |

7/7 (`results/conflicts/grading_20261028T210739Z.md`). One note on grading: a
first pass at grading by hand marked synth-02 as "silently picked"; the flag
was in the answer's second paragraph, which the line filter used for reading
hadn't shown. Grade from the full answer.

PDF tables (12.4): not applicable, the corpus has no PDFs.

### Verdict: before indexing repos with real per-team permissions

1. Take permissions from the source of truth (repo/team membership in the code
   host or directory), not a hand-written `acl.json`, and re-check them per
   request: a cached decision outlives a revoked permission.
2. Consider a collection per tenant instead of one filtered collection: a
   metadata filter is one bug away from a leak, separate collections fail
   closed. The trade-off is index count and cross-team search.
3. Everything that stores text needs the same rules: the transform and policy
   caches, saved sessions, hand-off tickets and logs hold question text and
   project names from whichever user wrote them.
4. Real clones with git history, so freshness dates and suggested contacts
   exist; until then every source is "unknown".
5. Decide the "forbidden looks like missing" policy with security: hand-offs
   for a forbidden repo's questions tell a person what was asked.

## Step 13: FAISS index lab

Exact vs approximate nearest-neighbour search, on this repo's own bge-small
vectors (97 chunks x 384 dims) and 64 queries (every question in the golden,
conversation and policy sets, deduplicated). Scripts: `scripts/faiss_lab.py`
(13.1-13.2), `scripts/faiss_queries.py` + `scripts/faiss_scale.py` (13.3-13.5),
`scripts/latency_breakdown.py` (13.6). Results in `results/faiss/`. Tests in
`tests/test_faiss_lab.py`: ANN recall 1.0 / 0.5, HNSW with efSearch=256 equals
Flat on 200 random vectors, IVF with nprobe = nlist equals Flat, L2 and inner
product rank unit vectors the same.

One process-level finding: FAISS and PyTorch each ship their own OpenMP
runtime on Windows, and in one process the 97-vector sweep took ~10 minutes
(embedding the questions alone takes 7.6 s). The scale-up scripts therefore
never load PyTorch: questions and vectors are embedded once and saved.

### 13.1-13.2: the real corpus (N = 97)

Flat top 1 = Chroma top 1 for 64/64 questions. The collection itself uses
Chroma's HNSW defaults (`col.configuration`): cosine, M=16, ef_construction
100, ef_search 100, which the sweep shows is exact at this size.

| Index | Setting | recall@10 | p50 ms | memory MB |
|---|---|---|---|---|
| Flat | - | 1.000 | 0.013 | 0.149 |
| HNSW | M=16, efSearch 16 / 32 / 64 | 0.981 / 0.998 / 1.000 | 0.019-0.043 | 0.163 |
| HNSW | M=32, efSearch 16 | 1.000 | 0.022 | 0.175 |
| IVF | nlist=10, nprobe 1 / 4 / 8 / 10 | 0.372 / 0.863 / 0.994 / 1.000 | 0.009-0.016 | 0.165 |
| IVF-PQ | m=48, 8-bit codes | cannot train: needs >= 256 points per codebook | | |
| IVF-PQ | m=48, 4-bit codes, nprobe 8 | 0.750 | 0.026 | 0.043 |

- Flat is fastest or close to it: 97 dot products cost less than walking a
  graph or probing clusters, and it is exact. Every other index here only adds
  overhead and approximation.
- HNSW reaches 0.99 at efSearch=32 (M=16), or immediately with M=32.
- PQ needs training data the corpus doesn't have; with 4-bit codes it cuts
  memory 3.5x and caps recall at 0.75 (the codes can't separate close chunks).
- FAISS warns that IVF wants 39x more training points per centroid; left visible.

### 13.3: scale-up (synthetic, N = 250,000)

250k, not 500k: the machine had 1.7 GB of RAM free, and 500k x 384 float32 is
768 MB per copy, so 500k would have measured swapping (`--n 500000` re-runs
it). 97 real vectors + synthetic ones, each a real vector plus Gaussian noise,
re-normalised. Builds on 8 cores; each query timed alone on 1 thread; memory
is the index file's size.

| Index | sigma 0.05 (mean cos to source 0.71): recall@10, p50 ms | sigma 0.02 (cos 0.93): recall@10, p50 ms |
|---|---|---|
| Flat (exact, 384 MB) | 1.000, 22.7 | 1.000, 27.5 |
| HNSW M=16, efSearch 256 (420 MB, built in 6-14 s) | 0.594, 1.36 | 0.381, 0.50 |
| HNSW M=32, efSearch 256 (452 MB) | 0.834, 2.04 | 0.588, 0.72 |
| IVF nlist=500, nprobe 8 | 0.319, 0.59 | 0.958, 0.42 |
| IVF nlist=500, nprobe 32 / 128 | 0.772 / 1.000, 1.78 / 6.38 | 0.998 / 1.000, 1.35 / 5.07 |
| IVF-PQ nprobe 32 (15 MB, 25x smaller) | 0.637, 1.00 | 0.292, 0.83 |
| IVF-PQ + exact re-score of top 100 (399 MB) | 0.739, 1.06 | 0.738, 1.47 |

- Flat's latency grows with N: 0.013 ms -> 23-27 ms (about 2,000x for 2,600x
  the vectors), still fine for one user, not for 100M.
- The answers change because of how the data clusters, not only its size.
  With 97 real vectors, each has ~2,600 near-copies. HNSW's links then stay
  inside each cloud of near-copies and greedy search gets stuck in the wrong one
  (best 0.83, and worse the tighter the clouds); IVF's clusters match that
  structure exactly (0.958 at nprobe 8 for the tight clouds). The earlier
  setup, with 3,292 real vectors and ~150 copies each, saw HNSW win (0.993):
  same algorithms, different data, opposite verdict.
- PQ collapses on near-duplicates (0.29): the codes can't tell them apart, and
  re-scoring the top 100 helps only until the true neighbours aren't in it.
- Limitation: noise-based vectors don't cluster like real documents; treat
  these as indicative. At sigma 0.05 the true top 10 is 88% real chunks (the
  synthetic ones are distractors); at sigma 0.02 only 11%.

### 13.4: distance metrics

- On unit vectors, cosine, inner product and L2 give identical rankings: the
  same top-10 set and order for 64/64 questions. Proof in one line: for unit
  a, b, |a - b|^2 = |a|^2 + |b|^2 - 2a.b = 2 - 2cos(a, b); measured error 6.6e-7.
- With 20% of the vectors rescaled by 0.5-2x: cosine is unaffected (golden
  file R@8 1.00). Inner product fills 95% of its top 10 with the inflated
  vectors (16% of the corpus) and R@8 drops to 0.60. L2 favours the shrunk ones
  (27% of its top 10 from 4% of the corpus).
- The help desk wants cosine: relevance shouldn't depend on a vector's length.
  Gap found: `build_index` doesn't assert unit vectors; it relies on the
  embedder normalising, and Chroma's "cosine" space hides it today.

### 13.5: quantization (real vectors)

| Codes | bytes / vector | recall@10 vs float | golden R@8 | golden ansR@20 |
|---|---|---|---|---|
| float32 (Flat) | 1,536 | 1.000 | 1.000 | 0.900 |
| fp16 | 768 | 1.000 | 1.000 | 0.900 |
| int8 (SQ8) | 384 | 0.995 | 1.000 | 0.900 |
| binary (sign bits) | 48 | 0.627 | 1.000 | 0.900 |
| binary + float re-score of top 32 | 48 | 0.933 | 1.000 | 0.900 |

fp16 and int8 are free (2-4x less memory, same results). Binary loses a
third of the exact top 10, and a float re-score recovers most of it. None of
it changes what the help desk retrieves here, but only because at N=97 a top
20 is a fifth of the corpus; the re-score used the top 32 because the top 100
would be the whole corpus.

### 13.6: latency breakdown (one follow-up turn, caches off)

| Stage | median ms | share |
|---|---|---|
| condense (Haiku) | 10,262 | 33.7% |
| embed query | 44 | 0.1% |
| vector search | 15 | 0.0% |
| BM25 + fusion | 1 | 0.0% |
| gate (cross-encoder) | 202 | 0.7% |
| policy scope check (Haiku) | 8,930 | 29.4% |
| answer (Haiku) | 10,970 | 36.1% |
| **total** | **30,400** | |
| Claude rerank, if turned on | 8,643 | +28% |

- 99% of a turn is three LLM calls; vector search is 0.05%. "Which vector
  database?" is a question of operations (filters, updates, tenancy,
  durability), not of speed, at this size.
- The quickest wins: run the scope check in parallel with retrieval (it only
  needs hit titles as a hint, or none), and skip condensing for a follow-up
  that is already self-contained; together about a third of a turn.
- Open issue: one condense call took 35 minutes of wall time despite the
  CLI's 120 s timeout. Either the laptop slept mid-call, or the timeout didn't
  hold: on Windows `subprocess.run` kills only the `claude` process, and if its
  child processes keep the pipes open, Python waits for them. Fix: kill the
  whole process tree on timeout. Not fixed yet.

### Verdict: which index

- **This corpus (97 chunks, or a few thousand):** Flat, or Chroma's default
  HNSW, which is exact at this size. Index choice doesn't matter; LLM calls do.
- **100M vectors** (154 GB as float32): IVF with an HNSW coarse quantizer over
  PQ or SQ8 codes (about 4-40 GB), the exact vectors on disk for re-scoring the
  top 100, sharded by team, which the access control wants anyway.
- **1B vectors:** OPQ + IVF + PQ with re-ranking, sharded across machines (or a
  disk-based graph index like DiskANN), tuned on real queries: 13.3 shows the
  right index depends on how your data clusters, so measure on your own data.
