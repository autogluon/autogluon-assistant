---
name: mlzero
description: Disciplined end-to-end workflow for machine learning and deep learning work driven by the mlzero engine — inspect the data to establish what one unit is and where its truth lives, seal a hold-out, define one number of record, run the engine, and report only what a printed number licenses. Use this whenever someone wants a model or an ML solution built, trained, tuned, or improved from their data, whatever that data is and whatever is being predicted; whenever someone hands over a body of files and says what they want out of it, or asks for something to be done to their data automatically; whenever someone asks whether an existing ML result is trustworthy, reproducible, or leaking; and whenever someone asks which of several approaches is actually better. It applies even when the request names only the outcome wanted and no modeling step at all. It does not apply when the request is satisfied by describing or summarizing data already in hand and makes no claim about units not yet seen.
---

# MLZero: a result you can defend

## What this is

A way of reasoning about a machine learning or deep learning job, plus the discipline that makes its
result defensible. It assumes nothing about what the data is, how it is stored, or what is being
predicted. Its authority is the five questions below, answered by looking at the actual data — never a
category the request seems to fall into. If you catch yourself deciding what to do because the job
resembles a familiar kind of job, stop: the resemblance is a guess, and the questions are cheap to
answer properly.

This skill ships no code and no templates: every proof named here is a check you write for the data in
front of you, run, and quote, and it must print the values it compared — a check that prints only a status
word tells you nothing about what it actually ran on.

## Which mode you are in

**Solving** — a solution, a number, or both are being asked for. The rest of this file.

**Judging a result someone else produced** — do not re-solve it. Decide what the evidence in front of you
licenses, keeping what you could confirm apart from what you could not; `references/comparing.md` puts the
questions in the order that reaches a verdict for the least effort.

**Comparing approaches** — a difference between numbers is a difference between approaches only if
everything else was frozen before the first one ran, and only as far as you can state what else differed.
`references/comparing.md` says what to freeze, and what to do with an approach that cannot conform.

## What makes a result worth having

A number attached to a solution means something only when all of these hold: it came from data the
solution never influenced; it was produced by one scorer used identically everywhere a number was
recorded; predictions were matched to truth by identity rather than by position or order; and there was
a stated boundary the model could not see across, derived from the data's own structure. Break any one
and the number stops measuring the solution and starts measuring the leak. Everything below exists to
keep those four properties true and to make each of them provable rather than merely asserted.

## The five questions

Answer all five by inspection and record how you established each; if you cannot say how you know, you
have assumed — go look. When the request names only an outcome, what is actually being claimed is one of
the things you must establish rather than infer: where two objectives this data could
support would give different answers to questions 1, 2, 4 or 5, name them, ask which is wanted before
you partition anything, and record the answer and who gave it. Asking is always legal; choosing silently
is not.

**1. What is one unit?** The thing for which a single prediction is requested. Settle it together with
question 4, because the unit is whatever the number of record is computed per; if question 4 forces a
different granularity, revise this answer and log the revision. An answer is wrong if two things you
counted as separate units cannot be scored apart from each other, or if one thing you called a unit is
really several things that would be requested separately. Things that must be predicted together are one
unit, and so are predictions whose quality depends on the other predictions made about the same thing.
How many separately-scoreable pieces one prediction breaks into does not decide the unit: the internal
structure of a single prediction belongs to question 3, and those pieces are not units. Where the data
cannot settle the granularity because both a coarser and a finer answer would satisfy the request, the
granularity is part of the claim and not a property of the data — decide it by the grain at which the
result will actually be consumed, confirm that with whoever asked, and where no answer comes record the
choice as a declared judgment and a stated limit. Each choice carries a consequence: a unit finer than
the thing that produced it makes that producing thing a boundary key for question 5, and a unit coarser
than the pieces inside it makes how those pieces combine part of question 4's number.

Give every unit a stable identifier that survives being written to a file and read back unchanged, and
that you did not invent from position — an identifier derived from ordering silently becomes wrong the
first time anything is reordered. If none exists in the data, construct one from something intrinsic to
the unit, record how, and print the count of distinct identifiers against the count of units: a
collision is a finding about repeated content, so decide and record whether colliding things are one
unit or several, and if several, what separates them and that the separator is not a join key. If the
only thing binding a unit to its truth is position across two sources, do that bind once at ingestion,
print the counts on both sides as the only evidence you will ever have for it, materialise an intrinsic
identifier immediately and use it everywhere afterwards, and log the positional bind as an assumption no
check can verify, which enters the report as a limit.

**2. Where does the truth for one unit live?** Settle it by opening things, not by deciding in advance.
Establish whether truth for a unit is present wherever the unit is, including in how or where the unit
is stored; or has to be fetched from elsewhere, in which case the key it is fetched by is part of the
answer; or has to be assembled from more than one place, in which case the units that fail to assemble
are part of the answer too; or exists only as someone's judgement; or does not exist at all. Then count:
how many units truth was actually found for against the total, because a source covering most units is a
different situation from one covering all and you cannot tell which you have without counting. Where a
unit's truth covers only part of what a prediction for that unit must cover, count that too, and declare
how the uncovered part is treated rather than leaving the treatment implicit inside the scorer. Two
removals with opposite scopes fall out of this answer; `references/isolation.md` derives each in its
section 2 and says in section 5 what each proof must print. Truth itself must be absent from the
inputs of every held-back body, wherever that body carries it, while information that will not exist
at the moment a prediction is required must be absent from every body including the one you fit on. Do
not merge them — a check aimed at one passes while the other channel stands open.

Establish also *how* the units that have truth came to have it. If someone chose which units to
adjudicate, every number you produce describes the units that got chosen and not the population: compare
the units with truth against the rest on whatever attributes you can compute for both, print both rates,
and name in the report the population the number actually describes. If the truth you find answers a
narrower or different question than the one that was asked, what you have is a proxy rather than the
answer — see the next section, and ask whoever made the request when the gap is theirs to settle.

**3. What must a prediction for one unit look like once serialized?** Two parties must agree: whatever
produces predictions has to be able to write it, and a scorer has to be able to read it back and align
it to truth by identifier. Decide the concrete written form now, including where the identifier lives
in it, because a form that cannot carry the identifier forces alignment by position later. An answer is
wrong if it cannot represent everything the truth distinguishes, or if it can only be interpreted by
knowing the order things were written in. The engine's own prompt hardcodes a narrow artifact shape
before any wording of yours reaches the code that writes predictions, so settle this answer against
`references/engine.md` section 5 and log any revision it forces as a revision.

**4. Which single number ranks two solutions, and which direction is better?** Choose the one number
that would make a practitioner deciding whatever this model informs prefer one solution over another.
State the direction explicitly, and justify the choice against that decision rather than against
convenience. One number is of record for the whole job, used identically everywhere a number is
recorded; you may compute others for insight, but they never break a tie or crown a winner. See
`references/scoring.md` for defining and gating it.

**5. What must never straddle a partition boundary?** Determine what structure in the data would let a
model see at training time something it will not have at prediction time, and partition so that
structure cannot cross the boundary. Derive it from how this data is actually built, not from the kind of
task this looks like. Write down what you derived and the reasoning that produced it, because that
reasoning is the part a reviewer must be able to challenge, and it is the one part of the discipline no
check can validate for you. `references/isolation.md` is the derivation procedure and the proofs.

## When the questions cannot be answered

These are legal outcomes, not failures, and the only wrong move is to disguise one as a measured result.

**No truth exists, or the truth you hold answers a different question.** You can still deliver something
useful: build an explicitly-labelled proxy — a different, answerable question whose answer is evidence
about the real one — and report it under the proxy's own name, stating plainly what it does and does not
stand in for, and what it cannot speak to at all. A proxy is a new task, so the five questions are
answered again against it and everything below runs under the proxy's name: its number is a number of
record and carries the whole discipline. Two requirements are specific to manufactured truth. Its truth
must be provably absent from everything the solution receives, proven by the printed-count evidence
`references/isolation.md` requires for information that must not be visible — a proxy whose answer can be
recovered from the inputs is scored perfectly by a solution that learned nothing. And the proxy must
stay defined and non-degenerate under the boundary you derived; where it does not, either the proxy or
the claim changes, and you say which.

Or deliver a human-inspectable artifact that lets someone else judge quality directly, and report no
number at all. That drops the scoring, ranking and hold-out steps and the invariants serving them, and
keeps everything else: the five answers, the boundary and its proofs, the run log, and the separation of
what was requested from what was verified.

**Truth is only a judgement.** Turn the judgement into a written procedure that a second person could
apply to get the same verdict — and then find out whether it does. Where any units were judged more than
once, print the rate of agreement and report it beside the number as a ceiling on any score achievable
here, since a solution cannot agree with truth more closely than truth agrees with itself. Say what you
did with the disputed units — kept, excluded, or reported separately — and record that whichever you
chose changed the population the number describes. Where the procedure exists but was never applied
twice, its reproducibility is unmeasured and is logged as requested-not-verified. A judgement applied by
whoever happens to be looking is not a number of record; if it cannot be written down, treat it as the
case above.

**There is too little data to partition honestly.** If holding out enough units to resolve a difference
would leave too few to fit anything, the honest finding is that this data cannot support a trustworthy
claim about generalization. Say that as the headline. Then deliver what the data does support — a
working solution, the checks that passed, the units it was fit on, and an explicit statement that no
generalization number is being claimed — rather than a number whose confidence interval you cannot
justify.

## The procedure

Each step names what must be true when it is finished. Steps do not become true because you intended
them; they become true when a check prints numbers that say so.

**1. Record the five answers.** True at the end: all five are in the run log with how each was
established, dated before anything else happened.

**2. Derive the boundary and seal the partitions.** Read `references/isolation.md`. Four roles must stay
distinct wherever your procedure uses them, because they answer four different questions: `train`,
`engine_val`, `select` and `holdout`. Its section 3 defines each, says what collapsing any pair costs, what
a role you leave out forbids you to claim, and which forces decide how large each is — sizing is a genuine
judgement in both directions, so decide it deliberately and record the reasoning rather than adopting a
habitual figure. True at the end: the sealing and partition invariants below have each printed the values
they compared.

Settle one thing here, because step 5 depends on it. The *truth* for `select` and `holdout` must be
unreachable from everything the engine is given; that is the hard, checkable half of sealing. Their
*inputs* are a separate decision. Handing those over so that predictions can be produced for them is
legal — it is how `engine_val` already works — but a body whose inputs the fitting process could reach is
independent by request rather than by construction: the report says so in those words, and the ledger
records it that way instead of ticking the invariant.

**3. Define and gate the number of record.** Read `references/scoring.md`. Build the degenerate
artifacts the gates need, and whatever a reference point is derived from, out of `train`'s truth, so that
gating spends no read of `select` or `holdout`. True at the end: one scorer exists, its content hash is
recorded, and the gating checks have printed numbers showing it discriminates, fails closed, is
deterministic, and joins by identity.

**4. Brief and invoke the engine.** Read `references/engine.md` for how to invoke it, what actually
reaches the agent that writes the code, and the exact wire contract for the scorer hook. Write the brief
in terms of your five answers: what one unit is, what artifact to produce, where to write it, and which
identifier it must carry — wording that has to survive beside the shape constraint question 3 already made
you settle. The brief must also cause every candidate to emit an artifact covering every body of data you
will later score it on, with the truth for those bodies withheld. That travels through the instruction
channel and never through the engine's own path arguments (`references/engine.md` section 5), and it
leaves those bodies independent by request rather than by construction, in step 2's words. Where you
cannot get it, step 5 obliges you to re-apply finished candidates yourself, and you decide that now
rather than discovering it after the run. True at the end: the engine ran, and the run log distinguishes
what you *requested* of it from what you later *verified* — the engine cannot be compelled, so every
instruction you gave it is a request until a check on its output says otherwise.

**5. Rank the candidates on data the engine never saw.** Score every candidate that qualifies with the
same scorer, on `select`, and record the table. Predictions for a body the engine was not pointed at do
not exist unless you arranged for them: either the artifact you specified in step 4 already covers those
units, or you re-apply each finished candidate to `select` yourself afterwards from what it left behind
(`references/engine.md` section 5 for what a node leaves, how to re-apply it, and what each route costs).
Re-applying carries a hard condition: what you rank must be the state the engine scored, so a re-run that
refits ranks a refit, and the ranking stands only if the report says so plainly. A candidate you cannot
re-apply is recorded as unable to compete, with the reason; it is not a poor score. Never present the
engine's own per-node score summary as this ranking: those numbers came from the body the engine was
given, and passing them off as a ranking on `select` is the `engine_val`-doing-`select`'s-job collapse in
`references/isolation.md` section 3. Beside every candidate's number, print how many units it rests on and
how much the ordering moves under the resampling protocol you fixed before you saw any number
(`references/comparing.md` for what that protocol must resample and which variation its spread does and
does not cover). True at the end: a ranking with real numbers in it, or the explicit finding that no
candidate was scoreable. Where the leaders are not distinguishable under that spread, write that they are
not distinguishable, and either decline to spend the hold-out or state the non-quality rule that picked
the winner.

**6. Spend the hold-out exactly once, on the single winner.** One read event, in which the winner's
number, every reference point it will be reported against, and any context numbers you declared in
advance are all computed; nothing undeclared may be added after the read. The winner's predictions on
`holdout` come to exist the same way step 5's did, under the same condition. True at the end: one number,
from `holdout`, for one candidate, logged with the time it was read and why.

`select` and `holdout` are different bodies of data because choosing among candidates on a body spends
that body's independence; `references/isolation.md` section 3 gives the mechanism and how far it reaches.
If you read `holdout` before the winner is fixed, or read it again after seeing the number, you no longer
have a hold-out — say so rather than pretending.

**7. Hand over what was asked for.** A measured number does not by itself answer a request phrased as an
outcome. Preserve the winning candidate's code and whatever it saved, record where they are and what
they need in order to run, and state how the solution is applied to data that arrives later. Where the
deliverable is an artifact meant to be reused rather than a set of predictions, establish that it does
what it is claimed to do: locate it, load it, apply it to units it never saw, and print both what it
produced and the contract its inputs must satisfy. Where the units to be delivered lie outside every
body you partitioned and therefore have no truth, name that set, state that it is unmeasured, and state
what makes the measured situation resemble the delivered one — that resemblance is a judgment a reader
must be able to attack, never a result.

**8. Report.** See *Reporting honestly* below.

## Prove it, do not assert it

Work through these before you report; each is claimable only on the evidence named. The universal rule:
the check prints the values it compared, and the report may claim only what a printed value licenses.
This list is a minimum, not the whole of it — the boundary you derived carries proof obligations of its
own, proven as `references/isolation.md` section 5 requires and entered in the same ledger.

Sealing and partitioning — procedures in `references/isolation.md`:

1. **SEALED-INPUT** — nothing reachable from the directory you hand the engine resolves outside that
   directory, with reachable carrying the full meaning `references/isolation.md` section 4 gives it; and
   no truth for a unit held back for `select` or `holdout`, and no record of which body a unit was
   assigned to, is reachable from within it. *Evidence:* printed count of locations walked, printed count
   resolving outside, printed count of held-back units whose truth or assignment was found inside —
   counted per held-back body rather than once for the whole tree, and searched wherever
   `references/isolation.md` section 2 says truth can ride rather than only where a truth value is stored.
   Sealing a unit's *inputs* is a separate and weaker claim: where you handed them over (step 2), say so
   here rather than ticking this. Where held-back units can be constructed from what must stay readable,
   the units themselves cannot be sealed at all, and a printed zero for identifier strings does not say
   otherwise — report the construction instead of the zero.
2. **SPLIT-DISJOINT** — every pair of bodies has an empty identifier intersection, none is empty, and no
   value of a key you declared non-straddling appears in two of them. *Evidence:* printed size of each
   body in units and in distinct values of every key that constrains it, printed intersection size for
   each pair, per declared key the count of values occurring in more than one body, and the per-body
   composition of truth `references/isolation.md` section 5 requires — a body can be non-empty and still
   not support the number of record for everything in it, and that file's section 3 says why the count of
   key values rather than of units is what sets a body's resolution. A key declared instead as *spanning*
   is not tested here but supplies the groups the other checks are computed within; section 2 there owns
   both declarations and what each obliges you to record.
3. **ORDER-BARRIER** — if you declared an ordering, the bodies occupy non-overlapping stretches of it,
   and each sits on the side of the boundary the prediction claim requires. Which body sits where is
   derived from that claim rather than adopted by convention, and `references/isolation.md` section 2
   derives it. Computed across everything and again within every value of a key you declared spanning.
   *Evidence:* the declared sequence with its reason, plus printed extreme ordering values per body,
   overall and per group. This invariant and the one above it can be jointly unsatisfiable, one requiring
   a set of units to stay whole while the other requires a cut that runs through it; that is a finding
   whose size is printed and whose resolution the same section owns, where the requirement the claim does
   not need is recorded as UNMET together with the claim it withdraws rather than quietly relaxed.
4. **REFERENCES-RESOLVE** — every location recorded in a body's own index opens as readable when resolved
   the way the process that will consume it resolves paths, which is not from inside that body's root:
   the code the engine runs does not stand in your data (`references/engine.md` section 2). *Evidence:*
   printed count checked, printed count that failed to open, and the directory the check resolved from
   printed beside the directory the consuming process will run in.
5. **SEALED-WRITE-ONCE** — truth for `engine_val` lives outside the tree of data the engine is given, and
   truth for `select` and `holdout` lives where nothing disclosed to the engine leads: not inside, and
   not beside or above, any path the engine receives on any argument — including a ground-truth path
   handed to a scorer hook, whose directory becomes public the moment it is passed. *Evidence:* the
   resolved parent of every path handed to the engine printed beside the sealed root, plus a printed
   listing of what a walk outward from each of those parents actually reached.
6. **LOOK-BUDGET** — `train` and `engine_val` may be read freely; `select` is read at most once per
   candidate; `holdout` exactly once, on the winner. A read made to build something counts as a read, not
   only a read made to measure something (`references/scoring.md` section 5). *Evidence:* the log entries
   themselves, one per read, each naming its purpose. This one is accounting, not a barrier — see *What
   cannot be proven*.

The number of record — procedures in `references/scoring.md`:

7. **ONE-METRIC-OF-RECORD** — either the number the engine optimized against, the number that ranked
   candidates, and the reported number all came from the same scorer at the same content hash
   (*evidence:* the hash printed at each of the three uses), or the search's reward was deliberately not
   your number, in which case this invariant covers the ranking and the reported number only and the
   report states plainly that the search optimized a different quantity. That statement, not a third
   hash, is what discharges it; `references/scoring.md` owns the choice and its price.
8. **SCORER-DISCRIMINATES** — it scores a perfect prediction artifact strictly better than a trivial one,
   and strictly better than one holding the same predictions attached to the wrong units. *Evidence:* the
   three numbers printed, with the inequalities visible in the direction the wire convention requires.
   Rewriting an artifact without disturbing which prediction belongs to which unit is a different test
   with the opposite requirement — the number must come back identical — and its evidence belongs to
   invariant 11, not here; `references/scoring.md` section 5 builds both artifacts and says what each
   failure means.
9. **SCORER-FAILS-CLOSED** — an artifact missing a unit, an artifact carrying a unit that should not be
   there, and a request for a number the implementation does not provide each end in a non-zero exit and
   emit no score. *Evidence:* printed exit status and printed evidence that no score line was emitted,
   for each case.
10. **SCORER-DETERMINISTIC** — the same truth and the same artifact produce an identical number on two
    consecutive runs. *Evidence:* both numbers printed.
11. **JOIN-BY-IDENTITY** — predictions are matched to truth by identifier under exact set equality,
    never by position, order, or a fallback, and never by quietly discarding the excess on either side.
    *Evidence:* printed sizes of both identifier sets and the printed result of the equality test, which
    establishes coverage and nothing more; plus the number recomputed from the same artifact rewritten in
    a different entry order, printed beside the original and shown identical, which is the part that
    distinguishes an identifier join from a positional one. Where the artifact you actually receive
    carries no identifier, this invariant cannot be satisfied on that path: report the alignment as
    requested-not-verified and record the decision that put you there (`references/scoring.md`,
    `references/engine.md`).

Reading a result out — procedures in `references/engine.md`:

12. **CANDIDACY** — a candidate is a node whose newest verdict says it succeeded, whose declared
    prediction artifact exists, and whose artifact opens into something your scorer can consume;
    existence licenses nothing on its own. A candidate whose score is missing or non-finite cannot win —
    that is a rule you apply to your own ranking, not a property of the engine, which will crown such a
    node (`references/engine.md` section 4); where it did, report that it did and that you set the node
    aside. *Evidence:* printed verdict per node, what the scorer printed when it opened that node's
    artifact, and the score.
13. **PROVENANCE-MATCHES-EVIDENCE** — your account of how the winner was chosen is licensed only by a
    non-empty ranking containing real numbers. Without one, the honest account is that no candidate was
    scoreable and the engine's own pick was taken unvalidated.
14. **LEDGER-QUOTES-NUMBERS** — every line of the final ledger quotes a value some check printed. A line
    with no quoted value is reported as UNVERIFIED, not as a tick.

## What cannot be proven

State these in the report as limits, in your own words, rather than leaving a reader to assume the
guarantee is stronger than it is:

- Whether the hold-out was read only once is an accounting discipline. Nothing physically prevented a
  second read, so the claim rests on the log being complete and honest.
- Whether the structure you declared must not straddle the boundary is the *right* structure is
  unverifiable. The checks prove only that the thing you declared does not straddle; a leak through
  structure you did not think of passes every one of them.
- A correct, monotone, perfectly-wired implementation of the *wrong* number passes every gate. Gating
  establishes wiring, never appropriateness — that rests on the justification you wrote for question 4.

## Keep a run log

Maintain one human-readable log in the run workspace, appended as you go. It must let a reader audit the
claim without rerunning anything and without access to you: the five answers and how each was
established; the declared boundary and the reasoning that produced it; the size of each body of data and
the printed output of each proof; the number of record, its direction, and its scorer's content hash
with the gate output; every read of `select` or `holdout` with its purpose and when it happened; the
candidate ranking with the unit count each number rests on; every reference point; the final number; and
the limits above. Where an intention was never checked, log it as requested-not-verified in those words,
so it cannot later be mistaken for a result.

Choose your own layout. What matters is that a claim in the report can be traced to a line in the log,
and a line in the log to a value something printed.

## Reporting honestly

Lead with the number of record: its value, which direction is better, which body of data produced it,
how many units it rests on, every reference point it was measured against and what each one withheld
(`references/scoring.md` for what a reference point is and which ones a number needs), and how the winner
was chosen. The unit count and the reference points travel with that number everywhere it is stated, not
only in the lead. Then separate two sections that are never merged: what was **verified**, each line
quoting the value a check printed, and what was **requested but not verified**, each line naming what
would have to be checked to promote it. Finish with the limits above.

If nothing was scoreable, that is the headline. Report it as a run that produced no measured result,
name the reason, and offer the best-looking artifact only as an unvalidated artifact — never as a
winner, and never with a number that came from anywhere other than the scorer of record.
