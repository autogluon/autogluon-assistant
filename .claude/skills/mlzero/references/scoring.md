# The one number of record

Open this when you are about to decide how any two solutions will be compared, and again before you write
any number anywhere. It covers choosing that number, defining the artifact a prediction is serialized into,
joining predictions to truth, wiring the number to the engine, gating the implementation, and what gating
cannot tell you.

This builds on the questions in `SKILL.md` — what one unit is, where its truth lives, what a serialized
prediction must look like. Answer those by inspection first; nothing here can be decided without them. The
boundary the number must be measured across is `references/isolation.md`. The engine's wire contract is
`references/engine.md`.

If the truth's structure permits no number at all, stop here and read what `SKILL.md` says about delivering
an explicitly-labelled proxy or a human-inspectable artifact. A number invented so that something could be
ranked is worse than no number, because it will be quoted.

## 1. Choosing the number

You need exactly one number that orders any two solutions. Derive it from this task; do not recall one.

Ask, in this order:

- **What decision does this solution inform, and who acts on it?** The number is a stand-in for how much
  that person regrets the solution's mistakes. If you cannot say whose regret it approximates, you are not
  yet choosing a number, you are choosing a habit.
- **Which mistakes are not interchangeable?** If some errors cost more than others in that decision, a
  number that averages all errors uniformly hides exactly the thing that matters.
- **What does the truth you actually hold permit computing?** Truth that is partial, coarser than the
  prediction, uncertain, or only valid under conditions must not be scored as though it were complete and
  exact. Establish what you have by looking at it.
- **What of this number's value could no solution change?** Establish which properties of *this data*, rather
  than of a solution's behaviour, the number responds to, and how much of its value is already fixed before
  any solution acts. If a property of the data dominates, decide whether the number must be computed within
  groups and then combined, and state how you combined.
- **Which direction is better?** State it in words. It will not be re-derivable from the number later, and
  every downstream comparison depends on it.

**If the user named the number, adopt it.** Their decision context outranks your inference, and their prior
work is comparable only under their number. Record that it was given rather than chosen. If you think it is
a poor fit, say so once with the reason and use it anyway; silently substituting a "better" number makes
every comparison the user cares about meaningless.

**If nobody named one, choose it and justify it** in a sentence or two against the decision it informs, and
mark it in the report as your judgment so a reader knows it is open to challenge.

Other numbers may be reported as context — cost, time, coverage, whatever a reader needs. Exactly one ranks.
Write down which one, its direction, and the justification **before anything is trained.** A number chosen
after seeing results is chosen to flatter them, and the ranking it produces means only "this wins under the
number picked because it wins".

Write down with it the rule by which a prediction counts as right, or as a given distance from right, when
being identical to the truth is not the test (section 3). That rule is part of the number, it moves the number
as much as the choice of number does, and choosing it after seeing results flatters them the same way.

## 2. Defining the prediction artifact

A prediction has to leave the solution and arrive at a scorer. Four requirements, all structural, none of
them dependent on what a prediction is:

- **Coverage.** Every unit that must be predicted appears, holding exactly what it is owed and nothing more —
  none missing, none duplicated. Checkable against the identifier set of the truth at whatever granularity the
  artifact keys, which is not always one entry per unit: see *When one unit's prediction has internal
  structure* below.
- **Identity, not position.** Each prediction carries the stable unit identifier. Position in the artifact
  carries no meaning, because nothing preserves it: not the code that writes it, not the format, not the
  reader. Where the thing being predicted is itself an ordering, or otherwise carries sequence, that sequence
  is part of the predicted value and is written out explicitly as a value — never left implicit in the order
  records happen to appear. That is this same rule applied, not an exception to it.
- **Round trip.** What is written must read back as what was meant. Identifiers change under parsing more
  often than anything else — type coercion, case, precision, trimmed characters — and nested structure
  flattens, and separators occur inside content. Prove the round trip against your own truth before relying
  on it: serialize the truth into the artifact's own format, read it back, and print the number of units
  whose read-back value differs from the original.
- **Producibility.** It can be written by code holding only what exists at prediction time. If producing it
  requires anything only available with truth in hand, the artifact definition is wrong, not the solution.

**When one unit's prediction has internal structure.** If it holds several parts, or occupies more than one
record, the four requirements above apply at two levels and the definition must name both: exact set equality
is required at the granularity the artifact keys, and coverage is additionally required at the unit
granularity — every unit present, holding exactly the parts it is owed and no others. Print both counts.
Correspondence between the parts inside one unit's prediction and the matching parts of its truth is settled
here and nowhere else; it may be structural rather than keyed, and if it is, the definition states the structure
and the gates in section 5 exercise it instead of assuming it holds.

**The side you hand over.** Whatever enumerates the units to be predicted is yours to construct; where the data
does not already list its units anywhere, that enumeration does not exist until you build it. Nothing else
determines what the prediction artifact can contain, so build it to enumerate exactly the units of that body and
no others, to carry the identifiers of that body's truth in the same form the artifact will carry them, and to
fix the order that any ordering requirement in the engine's prompt can only be referring to. It must carry no
truth anywhere — not in a field, not in a name, not in a path — and that is something to check and print, not to
intend. Whatever it omits cannot be predicted; whatever order it fixes is the only order a positional reader
could ever be right about.

The artifact may be one file, many files, or a directory tree; the engine imposes no format and never opens
it. Two things constrain your choice anyway: your scorer must read it, and when the engine's external scorer
hook is active the prompt reaching the code-writing agent hardcodes a narrow artifact shape and an ordering
requirement, stated exactly in `references/engine.md` section 5. Read that before fixing the shape — if the
number of record cannot be computed from what that prompt demands, you have the decision in section 4 to make.

Note also from `references/engine.md` section 4 that the engine's only test on the artifact path is an existence
check. Presence is therefore not evidence of anything. Your scorer is what establishes that an artifact is
real, which is why it must fail on an empty or unreadable one rather than treating existence as success.

## 3. Joining truth to predictions

This is where honest-looking code lies, because a wrong join usually produces a plausible number rather than
an error.

Match by identifier at the granularity the artifact keys (section 2), and require **exact set equality**
between the truth's identifiers at that granularity and the artifact's. Print both counts and the size of the
difference in each direction, every run. Where that granularity is finer than a unit, print the same three
figures a second time at unit granularity: the two are independent, and either can match while the other does
not. Then, without exception:

- Never join by position, by order of appearance, or by the order files were encountered.
- Never fall back to whichever part of the artifact comes last, and never infer which part holds the
  prediction. If the artifact's shape has to be guessed, the definition in section 2 failed; fix the
  definition rather than the reader.
- Never truncate either side to the length of the other, and never quietly intersect the two identifier
  sets.
- Never compare leniently by accident, and do not confine that vigilance to identifiers. Any step that
  changes either side before the two meet, and any step that decides what counts as equal when being
  byte-identical is not the test, is part of the number rather than an implementation detail. Name each such
  step before anything is trained; apply one implementation of it to both sides; and print what it did — how
  many identifiers and how many values it altered on each side, how many items the truth side held before and
  after it ran, and how many units' verdicts differ from a comparison with no such step at all. It belongs
  inside the content hash of section 6 and beside the number's justification in the report, because two
  defensible conventions applied to identical data yield different numbers and a reader cannot otherwise see
  which one produced this one.

These rules govern which unit a prediction belongs to. Correspondence *inside* one unit's prediction is fixed
by the artifact definition in section 2 and may legitimately be structural — that is not a positional join,
but it is only legitimate when the definition states the structure and a gate exercises it.

A mismatch either way is a defect to surface, not to repair. An identifier the truth holds and the artifact
does not means part of the job was left undone; an identifier the artifact holds and the truth does not means
something was answered that you never asked for, or fabricated. Neither reading depends on one unit yielding
one entry: where a unit is owed several, a unit present with too few of them is the first case and a unit
present with too many is the second, and the count that tells you which is the unit-granularity count above.
Both change what the number would mean, and intersecting the sets yields a number measuring the solution's
easiest subset. Exit non-zero and print what was on each side.

## 4. Wiring the number to the engine

`references/engine.md` section 4 holds the wire contract — which flags activate the hook, how the scorer is
invoked, how a score is read from its output, the direction the engine assumes, and what happens on failure.
Read it there; do not re-derive it here.

The decision that belongs to you is one question in two parts, and both parts must be yes: **can the number of
record be computed from the artifact the engine's code-writing prompt will actually cause to be produced — and
can each value in that artifact be attributed to a unit by something other than its position?** Compare the
hardcoded constraint in `references/engine.md` against your section 2 definition for the first part. Answer the
second by asking what a scorer would have to read in order to know which unit a value is about.

Take that constraint as it comes and the second answer is no: nothing on the prediction side carries an
identifier, so the exact set equality of section 3 cannot be evaluated at all, and a scorer written against it
joins by position however it is dressed — while still computing a number, passing every gate in section 5, and
clearing any reference point. Computability alone is therefore not the test, and this is the point at which the
identity requirement gets abandoned silently if you let it.

**Both parts yes.** Wire your scorer in. The search's internal reward is then the number of record, which is
the outcome to aim for, because the search spends its effort improving the thing you will report.

**Either part no.** Change one of three things, each at a price:

- *Change the artifact* so the number is computable from the constrained shape. Cheapest when the number
  survives the reshaping; impossible when the number needs structure that shape cannot carry.
- *Override the template that carries the constraint.* `references/engine.md` says how, and says what else
  travels in the same block — you become responsible for restating that too, and an omission there is
  silent.
- *Leave the hook inactive and rank finished candidates yourself* with the same scorer afterwards. The price
  is that the search's internal reward is disconnected from the number of record: the engine optimizes its
  own notion of a successful run while you rank on yours. The ranking stays trustworthy, but the candidates
  in it were not selected to be good under your number, so expect to need more of them for the same quality
  of winner.

If you take none of the three and score the constrained shape positionally anyway, that goes into the report as
a limit and never into a claim: alignment of values to units was requested and never verified, the
join-by-identity invariant is unsatisfiable by construction rather than satisfied, and a misaligned artifact
under such a scorer is indistinguishable from a good one — a plausible number, no error, a zero exit status.
Never record that invariant as verified, because the evidence it asks for cannot exist here.

Whichever you choose, one thing must hold: every number that ranks anything came from the same scorer. A
ranking assembled from two implementations, or from the engine's internal notion for some candidates and
yours for others, is not a ranking. And if the search's reward was not your number, the report says so
plainly instead of implying the search was steered by the number reported.

**Direction.** The engine consumes a single scalar in one fixed direction (`references/engine.md`). If your
number is better in the opposite direction, what goes on the wire is a strictly monotone, one-to-one
transform of it — so no ordering is lost — and you record the transform. The report shows the natural number
in its natural direction, with the direction named. A sign error here inverts the entire search silently,
which is why the discrimination gate below is run in the wire direction rather than the natural one.

## 5. Gating the implementation

The scorer arbitrates everything and is typically written once and never reviewed. Gate it before it decides
anything, and gate it again whenever it changes — earlier gate output does not cover new code.

Build the degenerate artifacts yourself, out of truth you may read without limit — never out of the truth held
back for ranking candidates or for the number you will report (`SKILL.md` sets those budgets). Gating therefore
happens before either of those bodies is opened at all. Reading held-back truth to build a fixture is a read
like any other: it spends what it spends, and it is logged as one. The fixtures cost almost nothing and they are
the only cheap way to catch the errors that invert or flatten a ranking without erroring. Every gate must print
the values it compared; a gate that prints a verdict proves nothing.

**Discrimination.** Score two artifacts derived from truth: one exactly right, one deliberately poor but still
well-formed. Require, in the wire direction, that the right one wins strictly, and print both numbers. If it
does not, the number, its direction, or the transform onto the wire is wrong.

**Order invariance.** Write the right artifact again with its entries in a different order and every identifier
still attached to its own prediction. Require the number to be identical to the first. Any change at all proves
the number depends on where an entry sits — which means it is scoring whatever ordering happened to hold while
you tested, and will keep emitting confident numbers once that ordering changes.

**Misattribution.** Write it again with the identifier-to-prediction attachment permuted and the entry order
left as a positional reader would still accept. Require the right artifact to beat this one strictly. A tie
means the number does not depend on which unit received which prediction: it is measuring the distribution of
the predictions and nothing about their assignment. Where one unit is owed several entries, permute inside a
unit as well as across units, because a scorer that pools a unit's entries before comparing them is blind to
the inner permutation while passing the outer one. That inner test is how the structural correspondence
section 2 declared gets exercised rather than assumed.

Those last two test the join only when the artifact carries identifiers. If it carries none, neither artifact
can even be built, the join cannot be gated, and the honest record is that it is unverifiable rather than
verified (section 4). And no check written on the scorer side can detect a producer that attached identifiers
positionally in the first place; that remains a request about the producer, not a property of the number.

**Fails closed.** Three malformed inputs, one defect each: a unit absent from the artifact, an extra unit
present, and a request for a number the implementation does not provide. Each must exit non-zero and emit no
score line. Print the exit status and the whole output for each. A scorer that returns a number for
malformed input is more dangerous than one that crashes, because the number will be believed.

**Determinism.** Score the same truth and same artifact twice and compare the two outputs as exact text, not
approximately. A difference means something unordered, random, concurrent, or time-dependent is inside the
number, and every candidate comparison then carries that noise. Find it rather than averaging over it.

**Floor.** A number means something only against a reference point that withholds exactly what your solution
claims credit for, and nothing else. So state in words what the solution claims to contribute; then score the
*strongest* predictor that withholds that. Everything else already known when a prediction is made is fair for
it to use — a unit's own earlier truth if that will be in hand then, and whatever the unit's own input already
states — so long as it fits nothing on the relationship the solution claims to have learned. Derive it by asking
what someone would answer using only what they will already know about this unit, with no model at all. Its
parameters come only from truth you may read without limit: a reference point estimated on the very body it is
then compared against is not independent of that body, and the margin it shows is a consequence of that rather
than a result.

Where the data records the decisions something already in place was making — the collection captured them — that
record is a second reference point, and usually a far more demanding one. Score it with the same implementation
over the same units. A number that clears a reference point derived from the shape of the truth alone, while
failing to beat what was already being decided, does not answer a request to do better than that.

Report every reference point you computed beside the number of record, always, and name what each one withheld,
so the reader sees the demanding margin rather than the flattering one. Reporting only the weakest overstates a
result without writing a false number. A reference point that lands at or near the worst value the number can
express licenses no margin claim at all — say that, instead of quoting a margin against it. A winner that does
not clearly exceed the strongest reference point is a finding, not a result: it says the pipeline ran, not that
anything was learned. How much margin counts as clear depends on how many units the number rests on and how much
it varies; state that count wherever you state the number, and see `references/comparing.md` for deciding when a
difference is distinguishable from noise. When a reference point must sit beside a number computed on a body you
may read only once, compute it inside that same single read rather than opening the body again.

**Invocation path.** Before the run, invoke the scorer exactly as the engine will: the interpreter the engine's
own resolution rule yields, the same argument form, absolute paths throughout — all three stated in
`references/engine.md`. Print the interpreter's own identity beside the score line, and confirm the line matches
what the engine extracts with. A scorer that cannot start under that interpreter, or that cannot import what it
needs there, is invisible: the failure is a warning the run continues past, and the run ends with no numbers at
all. The environment the engine was handed is mutated while it runs, so repeat this gate whenever that
environment changes, and afterwards count how many nodes actually received a number rather than assuming they
did.

## 6. Identity of the number

Every recorded number must be attributable to one implementation in one state. Record a content hash of the
scoring implementation — and of anything you wrote that it imports — beside every number you write down: in
the run log, in the candidate ranking, in the final report, and for whatever the engine consumed.

The number the engine consumed you never compute yourself, so make the implementation print its own hash on
every invocation. Then the hash travels with the score inside whatever record of the scorer's output the engine
retains; `references/engine.md` says where that lands and how to read it back.

Numbers under different hashes are incomparable. If the implementation changes for any reason, including a
bug fix that makes it more correct, every number recorded under the old hash is void rather than adjustable.
Re-score whatever you still intend to compare and discard the rest. Converting an old number into its
"equivalent" under new code is indistinguishable from inventing it.

This is also the practical way to show that the search's reward, the candidate ranking, and the final number
came from one scorer: they carry one hash and a reader can check it. If they do not, the report states which
number came from which implementation and does not place them in a single ordering.

## 7. What gating does not establish

State this flatly here and again in the report: **a perfectly wired, deterministic, strictly monotone
implementation of the wrong number passes every gate above.** The gates establish that the implementation
does what it claims — discriminates, refuses malformed input, repeats itself, and joins by identity when the
artifact carries one. They establish nothing about whether the number is the right question to ask about this
solution.

The only defense is the justification from section 1, and it is a judgment rather than a measurement. Carry
it into the report as a judgment: the number, its direction, why it was chosen or who chose it, the rule by
which a prediction was judged right, and what it does not capture that someone acting on this solution might
care about. A reader who disagrees can then re-rank; a reader never told what was optimized cannot.

Two further limits belong in the report rather than being quietly assumed away. A flawless scorer computed
on contaminated data produces a confident wrong number — that risk is `references/isolation.md`'s subject,
not scoring's, and gating cannot see it. And a reference point cleared by a wide margin is strong evidence,
not proof; it bounds how trivial the solution is, not how good.
