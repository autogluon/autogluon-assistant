# Judging a result, and comparing approaches

Open this file when the request is not to build a solution but to decide whether to believe one, or to place
several side by side. Both modes rest on a single idea: a verdict or a comparison is only as good as the basis
it rests on — which units, held back how, scored by which implementation, in which direction. The basis is the
first thing to establish and the last thing anyone writes down.

For the method these modes serve, see `SKILL.md`. For the machinery they lean on, see `references/isolation.md`
(deriving, sealing and proving a boundary), `references/scoring.md` (defining, wiring and gating the one number
of record) and `references/engine.md` (what the engine does and what its output actually says).

## 1. Judging a result someone else produced

You are handed some combination of a claimed number, a directory of run output, and a report. Your job is not
to redo the work; it is to decide what the evidence in front of you licenses. Work the questions below in
order — roughly the order in which claims die — so you reach a verdict for the least effort. If one question
settles it, stop and say which one did.

**What body of data produced the number, and could the solution have influenced it?** A number is a claim about
units the solution did not shape. Ask which units it was computed over, how they were held back, and what
touched them: fitting, choosing among options, deciding when to stop, setting a preprocessing decision — or a
person who looked at the number and changed something. Influence does not have to be automated; a human
iterating on a number is a fitting procedure with a slow loop. If nobody can say which units produced the
number, there is no claim to assess, and the verdict is already decided.

**How many units does the number rest on, and what was removed on the way to it?** Ask for the count of units
the claim is about, the count the number was actually computed over, and what accounts for the difference,
stage by stage, since every stage between the two can drop something quietly. A number computed over whatever
survived is a claim about the survivors and inherits their easiness, so ask whether the removed units differ
from the kept ones on anything computable for both — removals that track difficulty flatter a number further
than almost any other defect while leaving every individual step looking reasonable. Where nobody can produce
those two counts, the population the number describes is undetermined, and every question after this one is
about a quantity of unknown scope.

**Was the number that chose among candidates the same number, over the same units, that got reported?** If it
was, the reported number is optimistically biased and the verdict must say so; the mechanism is in
`references/isolation.md` section 3, as the collapse where one body of data both chooses and reports. What you
have to extract here are the two counts that bound the overstatement: how many candidates were compared, and
how many units the number rests on. Ask as well what was compared informally and then dropped, because a
comparison nobody wrote down inflates the number exactly as a recorded one does while leaving no trace in the
evidence. This is the most common way a careful, honest process still ends up quoting an inflated number.

**Is the number attributable to one scoring implementation at one state?** Numbers computed by different
implementations, or by the same implementation before and after a change, are different quantities that share a
name. Ask what produced each number and whether that state was recorded; a number nobody can attribute is not
evidence. `references/scoring.md` covers what recording it requires.

**Can it be produced again from what is here?** Two questions hide inside that one word, and they license very
different claims. Ask first whether the *number* recomputes: with the retained truth, the retained prediction
artifact and the recorded implementation, run it yourself and print your value beside theirs. That is cheap,
and it is the one question in this list that can carry a claim all the way to contradicted rather than merely
unverified. Ask second whether the *result* would arise again if the work were redone, which is a far stronger
claim and often not demonstrable at all: a fitting procedure carries its own randomness, and a shared
environment drifts underneath a run rather than being pinned by it (`references/engine.md` section 7). Where
too little was retained to recompute anything, reproducibility is undemonstrated rather than refuted. Say
which of the two you established, and never let the cheap one be read as the strong one.

**Are predictions joined to truth by identity, or by position?** Ask for the check, and require that it print
the count of identifiers on each side, the count present in one side and absent from the other in both
directions, and the count actually scored. Positional alignment can be right by accident and wrong in silence.
A number produced by truncating to whichever side is shorter, or by aligning on the order in which entries
happened to be written, is not measuring what the report says it measured.

**What was declared as the thing that must not straddle the boundary, and was it proven not to straddle?**
These are two questions, and the second is answered only by output that printed the values it compared. The
first is unverifiable in principle — whether the declared thing was the right thing is a judgment — so ask for
the reasoning and assess it as reasoning, then record it as a judgment rather than as a check that passed.
`references/isolation.md` section 2 gives the derivation you are assessing against.

**Is there a floor beside the number?** Without the score of a trivial predictor over the same units by the same
implementation — `references/scoring.md` says what a floor is and how to build one — a number is
uninterpretable: you cannot separate what the solution contributed from what the task gives away. A floor is
usually cheap to establish from the truth alone, so if none was recorded, establish one and report both numbers.
Ask also what the *strongest* comparison available was, and in particular whether anything already in place was
scored on the same units by the same implementation, wherever the data records what it decided
(`references/scoring.md` section 5 on which reference points a number needs). Where only the weakest was
reported, the verdict records the margin against the demanding one as unestablished rather than crediting the
margin that was shown.

**How many times was the finally-reported body of data read?** Ask for the accounting. Nothing in a filesystem
records this, so it is discipline rather than a guarantee, and a report claiming a single spend without an
accounting is claiming something it cannot demonstrate. Carry it into the verdict as exactly that — neither
proven nor refuted.

**Does every claim in the report quote a value that a check printed?** Read the report against the evidence,
never against itself. A line asserting that a property holds, with no number beside it, records that the
property was requested. A check whose entire output was a verdict proves nothing about the values it compared.
In particular, a claim about how a winner was chosen is licensed only by a non-empty ranking containing real
numbers; with no such ranking, the claim is unverified no matter how confidently it is phrased.

### Judging a run the engine produced

`references/engine.md` describes the output tree and how to read a result out of it honestly. The traps that
most often become an unwarranted claim:

- Executing successfully and being measured are different facts. A candidate whose code ran to completion but
  was never scored can never be a validated winner, however plausible its output looks.
- When nothing was ever scored the engine still records a winner, chosen by a rule that has nothing to do with
  quality; `references/engine.md` section 6 says how to recognise that case in the output and what reason it
  records. A report presenting that winner as the best solution is contradicted by the evidence, not merely
  unverified.
- The engine's test for a prediction artifact is an existence check. Presence is not readability, and
  readability is not conformance to what the scoring implementation needs.
- Run-level state can hold a ranking for the whole run rather than a record of what any single candidate used.
  Do not attribute a per-candidate property to a run-level file.
- The search's internal reward and the number of record coincide only if someone wired them together. If they
  were not wired, the search improved one quantity while the report quotes another, and every comparison the
  run made among its own candidates means nothing.

### What the verdict may say

Sort every claim into exactly three states and use these words. **Verified** — a printed value licenses it;
quote the value. **Requested but never checked** — it appears as an instruction or an intention with no
evidence that it holds; it carries no weight, and a report that presents it as a property is overclaiming.
**Contradicted** — printed evidence disagrees with the claim. For every item that is not verified, name the
specific check that would upgrade it and what that check would have to print; a verdict that only withholds
credit is far less useful than one that says how to earn it. Where no check could produce that evidence, because
the setup that produced the claim makes it unobtainable, say that instead of naming a check: a gap that can be
closed and a gap that is closed off by construction are different findings, and a reader who cannot tell them
apart keeps asking for work nobody can do.

Lead with the number of record, its direction, which units produced it and how many there were. If the
evidence cannot support a grade, say that the result cannot be assessed from what is here and list what is
missing — that is a legitimate answer and frequently the correct one, whereas manufacturing a grade to look
decisive is the failure this mode exists to prevent. If you measure anything yourself, report it as your
measurement on your basis and keep it separate from the claim you were asked to judge; a repaired number is a
new number, not a corrected one.

Measuring anything yourself also spends something, so decide what before you read. A number you compute on the
body they already reported on is no more independent than theirs was, and it cannot rescue the claim. A number
you compute on a body nothing has touched spends that body's independence for everyone who comes after you,
including any later attempt to do the job properly — so log the read as `SKILL.md` requires and say in the
verdict what it cost. And where the verdict is that nothing here can be assessed while what was actually wanted
is a number worth having, say exactly that and start the solving path from its first question, rather than
repairing someone else's basis in place.

## 2. Comparing approaches

A comparison asserts that a difference between numbers is a difference between approaches. That holds only if
everything other than the approach was held fixed. The work is therefore mostly in freezing a basis and then
refusing to unfreeze it.

**Freeze the basis before anything runs.** Establish and write down, once: which units are held back and how
each is identified; where their truth lives, sealed so that no approach can reach it; the single number of
record, its direction, and the state of the implementation that computes it; what each approach must emit for
that implementation to read; the full list of approaches that will be entered; the rule that settles a tie,
which cannot be the number itself; and the resource bound every approach receives. Derive the boundary and the
sealing with `references/isolation.md`, and define and gate the number with `references/scoring.md`, before the
first approach starts. Once one has run, none of this may change: altering the basis mid-comparison silently
redefines what is being compared, and numbers from before and after the change are not the same quantity.

**A ranking is not a level.** Choosing the best of several approaches by a number computed on one body of data
and then quoting the winner's number from that same body reports the choosing along with the approach;
`references/isolation.md` section 3 gives the mechanism. So decide, before anything runs, which of two things
this comparison produces. Either it produces an ordering, in which case you report the ordering with every
number in it labelled as computed on the body the choice was made on, and quote no performance level for the
winner. Or it produces a level as well, in which case a further body that takes part in no choice at all is
sealed at the same time as the rest of the basis and spent once on the winner, the way `SKILL.md`'s procedure
spends one. This is not a decision available to you afterwards: once the ranking exists, no body is left that
took part in no choice.

**An approach that cannot conform does not get a different scorer.** If an approach cannot be made to emit an
artifact the one implementation can read, record it as unable to compete, with the reason and what conforming
would require. Scoring it by some other means and placing the result in the same ranking fabricates a
comparison out of two unrelated quantities.

**An approach you build yourself is built inside the frozen basis.** Where one competitor is work you do rather
than work you were handed, the solving path runs under the basis already fixed: it derives no boundary of its
own, defines no number of its own, and receives no read the others did not receive. Anything about it that had
to be settled by looking at held-back truth disqualifies its number rather than the comparison, and it is then
reported as unable to compete. Where instead a competitor runs somewhere you do not control, you cannot prove
it never reached the held-back truth: that isolation is requested but never checked, and the report says so in
those words for that competitor alone rather than for the comparison as a whole.

**Put the floor in the comparison.** Enter the trivial predictor of `references/scoring.md` as one of the
competitors, measured identically to the rest. A ranking among elaborate approaches that all fail to beat it
reads as a clean result while actually being a finding about the task, and without the floor beside them the
ranking hides that. The floor also gives you a first sense of whether the differences you are trying to detect
are larger than the number's own spread.

**Whatever is already in place is a competitor, not a backdrop.** Where the data records the decisions something
already running was making, those decisions are a prediction artifact you did not have to produce, and they
enter the comparison scored by the same implementation over the same units as everything else —
`references/scoring.md` section 5 says why that is also the demanding reference point when you are solving
rather than comparing. It competes on unequal terms and the report names them: it received no resource bound,
nothing about it can be repeated because it cannot be re-run, and it may have decided with information the
other competitors are denied, or have influenced the very units all of them are scored on. Establish what it
had access to at the moment it decided. Where that cannot be established, it stays in the comparison with that
stated as a limit on its number, which is far more informative than a ranking of only the things you built.

**Run them one at a time, under a bound, and record how each ended.** Heavy work sharing one machine contends
for memory, compute and storage bandwidth; the interference distorts every number, and the host may terminate a
run outright. Serial execution costs wall clock and buys interpretability. Record, per approach, whether it
finished, hit the bound, or was terminated, along with what it consumed. An approach that was truncated or
killed is recorded as truncated or killed — never as a poor score, because folding a resource event into a
quality ranking states something the evidence does not support. The bound is a genuine tradeoff: too tight and
you are comparing what fits the budget rather than what the approaches can do; too loose and the comparison may
never finish. Choose it, apply it identically to every approach, and report it beside the numbers. If the
budgets could not be equalized, then budget is part of what you compared, and the report says so. Running
serially also stretches the comparison over time, and anything shared can move while it does — an environment
that resolves differently for a later approach than for an earlier one is the ordinary case rather than the
exception (`references/engine.md` section 7). Record the state of everything shared at the moment each approach
ran. If it moved, that movement is part of what you compared, and no approach may be credited for having run
under a state another one never saw.

**Say what the number can and cannot separate.** One measurement over a limited body of held-back data carries
a spread, and a difference inside that spread is not a difference between approaches. Wherever you print a
ranking, print how many units each number rests on and how many distinct values of each declared boundary key
those units span; `references/isolation.md` section 3 says why the second count is the one that sets the
resolution. Quantify the spread by whatever the setup allows, and be exact about which variation your protocol
moves: repeating what is genuinely repeatable moves an approach's own randomness while the held-back data stays
fixed, and resampling the held-back data moves the basis while each approach's single fit stays fixed. Neither
covers the other, so name the one you did and the one you did not. Resample at the level of the declared
boundary key, taking whole groups of units that share a value rather than units one at a time — resampling
through a group reports a spread narrower than the truth and an ordering steadier than it is. Fix the protocol
before you see any number, because deciding how much to repeat after seeing results is selecting an outcome.
More repetition sharpens the estimate at proportional cost and adds no data. When two approaches are not
distinguishable, write that they are not distinguishable and settle the choice by the tie rule you froze; an
ordering you cannot support is worse than an honest tie.

**Name what differed besides the approach.** The claim a comparison makes is that the approach accounts for the
difference between the numbers, so everything else that was not identical belongs in the report beside them.
Attention counts: an approach you iterated on placed against one you ran once compares the effort as much as
the approaches, and the only honest resolutions are to equalize the effort or to state the asymmetry and refuse
to read the gap as an approach effect. So does anything at all one competitor received and another did not.
Write these down as they arise; once the numbers exist, the difference that would undermine the conclusion is
the easiest one to forget.

**Say what the comparison licenses.** It licenses a statement about the competitors that were entered, in the
states they were in, at the budgets they received, on this basis, ranked by this number. It licenses nothing
about an approach nobody entered, nothing about any competitor given more room than the bound allowed, and
nothing about a basis it was never measured on. State that scope wherever the result travels, because a ranking
read as a general verdict outlives the conditions that produced it.

**Aggregating across several bodies of work.** Report every per-basis number alongside any summary, and state
how the summary was formed and which direction is better. A summary alone hides which basis drove it: numbers
from different bases occupy different ranges and rest on different numbers of units, so any pooling is
dominated by whichever basis varies most, and one missing entry can reorder the whole thing. State, per
approach, which bases it actually competed on — an approach that competed on fewer of them is not comparable on
the summary at all.

## 3. What neither mode may do

- Do not move a boundary after seeing a number. Re-partitioning to rescue a comparison makes the boundary a
  function of the result you wanted, which leaves every number on both sides of the change uninterpretable.
- Do not adjust a number after seeing it. If the implementation that produced it is wrong, fix the
  implementation, void every number produced under the old state, and re-measure all of them; a correction
  applied to one number and not its peers is a new bias in place of the old one.
- Do not remove a competitor, a candidate, or a body of data from the record after seeing its number. What is
  reported is what was declared; an entry withdrawn once its number existed makes the winner's margin and the
  spread properties of your withdrawal, and nothing in the output reveals that it happened. An entry that could
  not compete is reported as unable to compete, with the reason, and one that competed badly is reported with
  its number.
- Do not compare numbers produced by different scoring implementations or over different held-back data,
  however alike their names. The honest output is to say they are incomparable and then either re-measure
  everything on one basis or report them separately and decline the ranking.
- Do not report a request as a property. Anything asked for but never demonstrated is unverified here exactly
  as it is when solving.
- Do not let the verdict or the ranking be the first place a decision gets written down. Both modes are
  auditable only if the basis, every read of held-back data, and the printed evidence were recorded as they
  happened — keep them in the run log described in `SKILL.md`.
