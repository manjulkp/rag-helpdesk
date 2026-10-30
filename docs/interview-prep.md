# Interview prep: Helpdesk RAG

Every number below is from this repo's own runs (NOTES.md, docs/helpdesk.html).
Practise the pitch and walkthrough out loud until you can give them without notes.

## 30-second pitch

I built a help desk that answers engineers' questions about two project repos
with cited answers. It searches the docs and tests with a hybrid of vector and
keyword search, adds a one-line Haiku context to every chunk so a question
about "gold" finds a test that only says `GC=F`, rewrites follow-ups like
"does it retry?" into standalone questions, and decides per turn whether to
answer, ask which project you mean, refuse, or hand off to a person. Users
only ever see repos they're allowed to. I measured every step against
hand-written golden sets with paired bootstrap intervals, and kept or dropped
each change on that evidence.

## 2-minute walkthrough

1. **Indexing.** Markdown is split by heading and pytest files by test, into
   400-token chunks so a context still fits bge-small's 512 tokens. Each chunk
   gets a Haiku sentence situating it in its document; I fixed a prompt-caching
   bug that made those calls a third more expensive (the document has to be in
   its own cached block). Rebuilds are incremental by content hash: editing one
   document re-embeds one chunk in 0.08 s instead of 5.3 s.
2. **Retrieval.** bge-small vectors over context + chunk, plus BM25, fused with
   reciprocal-rank fusion. That combination was the only one that found every
   answer chunk in the top 20.
3. **Follow-ups.** A cached Haiku call condenses history + follow-up into one
   question. Searching the follow-up as typed fails every "And for the trade
   desk?"; concatenating history finds the answer but ranks it lower and drags
   the old topic in.
4. **Policy.** Rules first (asking for a person, two refusals in a row,
   injection attempts, never a second clarification), then one Haiku call that
   only judges scope. Whether something is answerable is decided by the
   evidence: a cross-encoder gate and the answer step's own "not found", either
   of which escalates to a local hand-off ticket.
5. **Answers.** Haiku answers only from numbered sources, cites lines and dates,
   and flags sources that disagree, preferring the newer one.
6. **Access.** A per-user repo filter on both searches, and again on every other
   path a chunk can take. 0 leaks in 45 user x question runs.
7. **Honest limits.** Two repos, 97 chunks, 15 golden questions: most
   differences are one question and no CI is clear of zero.

## Likely questions

**1. Why hybrid search?** Vector search misses exact identifiers and BM25 misses
paraphrases. On its own, hybrid on plain chunks didn't beat vector here (MRR
-0.009); it paid off combined with LLM context, where llm+hybrid was the only
setup with answer-chunk recall@20 of 1.00. BM25 also gained most from the free
header (answer recall@8 0.60 -> 0.80).

**2. How did you know the reranker helped?** It didn't, here. The gated Claude
rerank kept answers and refusals at 10/10 and 5/5 but cut citation recall from
1.00 to 0.92 and added calls and latency, so it's off by default. With 97
chunks the top 8 is already good; I'd re-test on a bigger corpus.

**3. Why didn't contextual retrieval help as much as Anthropic's results
suggest?** It did help in the same direction (top-20 answer failures 1/10 ->
0/10), but one question can't show a 35-67% reduction. Anthropic measured on
large corpora where chunks really lose their context; my documents average 3.9
chunks, and no golden answer sits in the middle of a split section, the case
it's designed for. It also cost about 2.7x per token, because tiny documents
make the first cache write dominate.

**4. How do you stop leaks across teams?** Two layers: a `repo $in` filter on
vector and BM25 search, then a second filter on sticky sources, gate, policy,
generator and hand-off tickets. Tested with deliberately poisoned sticky
sources: 0 leaks. Clarification options and refusal text name only allowed
projects, and a forbidden repo looks exactly like a missing one.

**5. What happens when docs contradict each other?** Each source shows its last
commit date; the prompt says to name the conflict, cite both with dates, prefer
the newer and say why, or say it can't tell when dates are unknown. 7/7 on
synthetic cases graded by hand, including a newer copy beating the original and
no invented conflicts on controls. The corpus itself had no real conflicts.

**6. How would this scale to 10,000 documents?** Indexing is incremental, so
day-to-day cost follows edits, not corpus size; LLM contexts are about $2.8 per
million document tokens here and should get cheaper with longer documents.
Retrieval stays local and fast: a turn is 30 s, and vector search is 15 ms of
it (99% is the three Claude calls), so the index isn't the bottleneck until
far beyond this. In the FAISS lab, exact search at 250k vectors was still
23 ms; which approximate index wins at scale depended on how the data
clusters (HNSW won in one setup, IVF in mine), so I'd tune it on real queries. What changes: the reranker probably starts to
pay off, the ambiguity rule needs more than two projects, and permissions should
move to one collection per team so a filter bug fails closed.

**7. What would you change for production?** Permissions from the directory,
checked per request; an owners map instead of git authorship; real traffic
metrics (clarifying rate, escalations, refusals users dispute); red-teaming of
injection inside documents; and the same access rules on caches, sessions and
tickets.

**8. How do you evaluate without a big labelled set?** Small golden sets with
both file and answer-chunk labels, paired comparisons on the same questions,
bootstrap CIs, and hand-graded answer sheets. I report direction when the CI
touches zero rather than claiming a size.

**9. How does it handle follow-up questions?** Condensing: 0.75 -> 1.00
follow-up recall vs searching as typed, and old-topic drag 0.06 vs 0.25 for
concatenation. History in the answer prompt is the last 4 turns in full and
older turns one line each, which keeps turn-10 prompts at 1,916 tokens instead
of 3,130.

**10. When does it ask a clarifying question?** Only when the question names no
project and both projects' top hits score within 80% of each other, checked on
the hits before reranking (a reranker hides the split). Never twice in a row,
and the options come only from the hits. 0 clarifications on 12 answerable
questions.

**11. When does it hand off to a person?** When asked, when the user says the
answer was wrong, after two refusals in a row, or when an in-scope question gets
nothing from the gate or the answer step. The ticket has the question,
conversation summary, what was searched, candidate files and a suggested
contact. It's local; sending it anywhere would need approval.

**12. What was the hardest bug?** On Windows, `subprocess` used the cp1252 code
page, so one non-cp1252 character (a byte-order mark from piped input) made the
whole prompt vanish; the CLI just said "no stdin data". Found by narrowing it to
a one-character repro; fixed by forcing UTF-8.

**13. What did you decide not to ship, and why?** Query transforms (not
evaluated here), the Claude rerank by default (no gain), and a policy that
judged answerability from titles (moved that decision to the evidence). Sticky
sources showed no effect and stay only as low-priority extras.

**14. What are the biggest risks in your results?** Small sets written by the
same person who built the system, so they share blind spots; and two known
misses: a retrieval miss on "how do I run the tests" and a scope shortcut that
lets "translate 'risk manager'" through.

**15. How much did it cost to build and run?** The LLM-context index was $0.036
for 97 chunks; a cited answer is about $0.0008 and 11 s; every evaluation in the
notebook cost a few cents.

## Ownership note

Edit this to match exactly what happened before you use it.

The design questions, evaluation plan and step-by-step specs came from my
mentor's plan, and I chose between options using the measurements. Most of the
code for steps 8-12 (query transforms, contextual variants, conversation,
policy, escalation, access control) and the evaluation scripts were written by
a Claude Code session working from those specs, which I directed and checked
against the tests and evaluation runs. Interviewers will probe those parts, so
I need to be able to explain every module myself, especially
`policy.decide()`, `pipeline.respond()` and the access filter.
