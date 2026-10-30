# Helpdesk RAG

A local help desk that answers engineers' questions about two project
repositories (their docs and tests) with cited answers. It handles follow-up
questions, asks which project you mean when it's ambiguous, refuses
off-topic requests, hands unanswerable ones to a person, and only shows each
user the repos they're allowed to see.

How it works, with the evidence for each design choice:
[docs/helpdesk.html](docs/helpdesk.html), a single self-contained page
(GitLab shows HTML files as source, so download it and open it in a browser,
or serve it with GitLab Pages). The full lab notebook is [NOTES.md](NOTES.md);
interview notes are in [docs/interview-prep.md](docs/interview-prep.md).

## Setup

Python 3.10 or newer, on Windows PowerShell (macOS/Linux: use
`.venv/bin/activate`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Answers, follow-up rewriting, the scope check and LLM contexts call Claude
through the `claude` CLI, which must be installed and logged in. Retrieval,
reranking and the tests run fully offline. The first run downloads the
embedding model (`BAAI/bge-small-en-v1.5`) and the cross-encoder.

The repos to index are set in `helpdesk/sources.py` (here `C:\research-desk`
and `C:\two-agent-trader-riskmanger`). Point them at your copies. Git is
optional: with it, citations show each file's last commit date and hand-offs
suggest a contact; without it, both show "unknown"/none.

## Build the index

```powershell
python -m scripts.build_index --variant llm      # what the chat uses (~4 min, ~$0.04 of Haiku calls)
python -m scripts.build_index                    # plain: chunks only, free, ~5 s
python -m scripts.build_index --variant header   # "repo / path / heading" + chunk, free
```

Rebuilds are incremental: only chunks whose text changed are re-embedded, and
LLM contexts are cached in `data/context_cache.jsonl`.

## Chat

```powershell
python -m scripts.chat                       # as admin (sees every repo)
python -m scripts.chat --user rd-analyst     # sees only the research desk
python -m scripts.chat --debug               # also show each policy decision and gate score
```

Commands: `/new`, `/history`, `/sources`, `/quit`. Sessions are saved in
`data/sessions/` and belong to their user (`--resume <id>`). Users and groups
are in `data/acl.json`. Hand-offs to a person are written as local tickets in
`data/escalations/`; nothing is sent anywhere.

One-off search and answers without the conversation layer:

```powershell
python -m scripts.search "How does the guardrail check a stop-loss?" --variant llm --retriever hybrid
python -m scripts.ask "How does the guardrail check a stop-loss?" --generator haiku --show-sources
```

## Tests

```powershell
python -m pytest -q
```

The tests run offline with fake embedders and fake LLMs (no model download,
no Claude login): 135 pass. Four of them need git (three loader metadata
tests and the toy-repo suggested-contact test) and are skipped without it.

## Evaluations

Each writes timestamped results and prints a summary. Golden sets are in
`data/`; all were written by hand for these two repos and are small (see
"Limits" in docs/helpdesk.html).

| What | Command | Golden set | Results |
|---|---|---|---|
| Retrieval (recall, MRR, answer-chunk recall) | `python -m scripts.evaluate bge-small --variant llm --retriever hybrid` | `golden.json` (15 q), `golden_chunks.json` | `result/` |
| Paired comparison with bootstrap CI | `python -m scripts.compare_runs bge-small@llm+hybrid bge-small+hybrid --metric answer20` | | |
| Cited answers | `python -m scripts.eval_answers haiku --variant llm --retriever hybrid` | `golden.json` | `results/answers/` |
| Follow-up questions | `python -m scripts.eval_conversations --answer` | `golden_conversations.json` (12 conversations) | `results/conversations/` |
| Clarify / refuse / escalate | `python -m scripts.eval_policy` | `golden_policy.json` (30 cases) | `results/policy/` |
| Access control (leaks) | `python -m scripts.eval_access` | `golden.json` x 3 users | `results/access/` |
| Conflicting sources | `python -m scripts.eval_conflicts` | `golden_conflicts.json` | `results/conflicts/` |
| Incremental re-index timing | `python -m scripts.reindex_timing` | | printed |
| LLM-context cost estimate | `python -m scripts.contextualize --repo research-desk` | | printed |
| FAISS: exact vs approximate indexes | `python -m scripts.faiss_lab` | all golden questions | `results/faiss/` |
| FAISS: scale-up, metrics, quantization | `python -m scripts.faiss_queries`, then `python -m scripts.faiss_scale scale --n 250000` / `metrics` / `quant` | | `results/faiss/` |
| Latency per chat stage | `python -m scripts.latency_breakdown` | conversation follow-ups | `results/faiss/` |

The Claude-calling evals take 2-10 minutes and cost a few cents each.

## Folder map

```
helpdesk/                 the library
  loader.py chunker.py      read repos; split Markdown by heading, pytest files by test
  embedder.py index.py      bge-small embeddings; Chroma collections per model and variant
  bm25.py hybrid.py         keyword search; reciprocal-rank fusion of vector + BM25
  contextualize.py          header and LLM contexts for chunks (cached, prompt-cache friendly)
  query_transform.py        HyDE / rewrite transforms; CondenseQuestion for follow-ups; Claude CLI wrapper
  rerank.py                 cross-encoder and Claude rerankers
  conversation.py           turns, sessions, history budget, follow-up search modes
  policy.py                 answer / clarify / out_of_scope / escalate
  escalate.py               suggested contact from git log; local hand-off tickets
  access.py                 users, groups, per-user repo filter
  pipeline.py               one turn: condense -> retrieve -> gate -> policy -> reply
  generate.py               cited answers (Claude, extractive) with source dates
  evaluate.py               retrieval metrics
scripts/                  command-line entry points (chat, build, search, evals)
tests/                    offline tests
data/                     golden sets, acl.json, indexes (chroma/), caches, sessions, tickets
result/, results/         evaluation outputs
docs/helpdesk.html        architecture and evidence
docs/interview-prep.md    pitch, walkthrough, likely questions
NOTES.md                  lab notebook, step by step
```
