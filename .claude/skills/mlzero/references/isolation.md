# Isolation: deriving the boundary and proving it holds

Read this when you are about to divide data into parts, and when you must judge whether a division someone
else made can support the number they reported. It covers how to derive what must not cross a boundary from
the structure of the data in front of you, how to make held-back parts physically unreachable, and what a
proof of either has to print. It prescribes no partitioning strategy, because a strategy is an *output* of
the derivation below, never an input to it.

## 1. What isolation is for

A measured number is a claim about performance on units the solution did not influence. The claim survives
only if nothing about a held-back unit could reach the fitting process — not the unit itself, not a near-copy
of it, not another unit that the same underlying thing produced, not a quantity computed across it together
with others, not its place in an ordering, not its own truth arriving inside the inputs you hand over, and
not knowledge that sits in your stored data but will not exist at the moment a prediction is actually needed.
Each of those is a channel. Isolation is the work of finding which channels are open in *this* data and
closing them. Everything after this section is about proving you closed them rather than intending to.

## 2. Deriving the boundary

Do not pick a strategy. Interrogate the data, answer each question below from what you observe, and let the
answers compose into the boundary. Write each answer and the observation that produced it down as you go: an
unrecorded derivation cannot be audited by anyone, including you a day later.

**What produced these units, and did any one producer produce more than one?** Look at how the data was
assembled and what that process reused. Establish whether some attribute identifies a common origin shared
by several units. Such an attribute is not always stored beside the unit; it may be recoverable only from the
unit's own content, so "there is no such attribute" is licensed only after looking inside the units
themselves — state what you extracted and how you extracted it. Where you looked and found none, say that
explicitly, which is a different statement from not having looked.

Finding one makes it a *candidate* key, not a boundary. It becomes a boundary only when the situation you
claim to measure is one where the producer is unseen at the moment a prediction is required: then all units
sharing a value belong to one part, because splitting them lets the solution learn the producer instead of
the phenomenon. When the same producers are present at prediction time, the opposite holds — the key
must then appear in every part, where its job is to keep the parts comparable rather than to divide them,
and forcing it to divide them measures a question nobody asked while starving the fit. Declare which of the
two situations applies and why, in the words the proofs keep apart: a key declared *non-straddling* is one no
value of which may appear in more than one part, while a key declared *spanning* is one whose values are
expected in several parts, so that the other checks can be computed within each of its values rather than only
across the whole. No key is ever both, and a spanning declaration carries its own obligation — record why the
claim needs that key's values present in more than one part. The reading you rejected is a rejected candidate
with a reason, not an unexamined channel.

A key need not take one value per unit. Where one unit carries several values of the same key, "sharing a
value" becomes a transitive requirement: the inseparable sets are the closures of the sharing relation, and
grouping by any single one of a unit's values will look correct while leaving links running across parts.

**Are there units that are not identical but are substantially the same?** Equality checks find only exact
repeats, and exact repeats are the easy case. Choose a notion of closeness that suits the content you
actually have and apply it. How wide to search is a tradeoff: the cost of one comparison times the number of
comparisons an exhaustive search implies, against the confidence a wider search buys. Where exhaustive is
not affordable, restrict the scope deliberately and print what the restriction was and what proportion of
the possible comparisons it covered — an obligation you cannot afford turns into an unrecorded skip, which is
worse than a declared partial search. Whatever the search flags is a relation between units, not yet a
constraint on the partition: close it transitively, so that units linked directly or through a chain of
flagged links form one set that cannot be divided, and print how many such sets formed and how their sizes
are distributed. Whether the notion you chose is the right one is a judgment, and one of the honest limits in
section 6 — but a notion applied and reported beats no search at all.

**Is there an ordering, and will predictions actually be made under it?** Ordering that merely exists in how
the data is stored may be an artifact of collection. The question is whether the situation you claim to
measure is one where only earlier units are known when a later one must be predicted. If it is, ordering is a
boundary: the parts are consecutive blocks with no overlap, and the requirement must hold both across the
collection as a whole and inside every group the other boundary keys define — a check over the whole can pass
while individual groups interleave. Which stretch each part occupies is derived from the claim too, never
adopted by convention: the part the reported number comes from stands where a prediction will actually be
required, and every part the fitting and searching loop consumes lies on the side of it that will already be
known at that moment. Record the resulting sequence with the reasoning that produced it: it asserts something
about the prediction situation, not about the order the data happens to be stored in.

**What has been, or is about to be, computed across more than one unit?** Any value whose computation
depended on more than one unit carries information between them, and computing it before the partition
exists puts held-back units inside what the fit sees. This covers values already in the data and equally
everything you or the engine is about to build from it; the absence of truth does not make computing across
everything safe, because a quantity fitted on all units has still seen the held-back ones. Establish which
such values exist and how each was built. Then, in order of preference: derive the value from the fitting
part alone and apply that result unchanged to every other part, proving the derivation touched no held-back
unit; or, where it cannot be transported that way, rebuild it inside each part from that part alone, stating
that the parts now carry different values and what that costs their comparability; or, where neither is
possible, treat every unit it touched as one inseparable set that cannot be divided.

**Where is a unit's truth carried, and does it travel with the unit's inputs?** Truth is not always a value
stored apart from the unit. Anything handed over with the inputs can carry it — where the unit is stored,
what it is called, where it falls in an order — and that list does not close; the test is whether a
held-back unit's truth is recoverable from what you hand over. Establish which, then keep two removals
apart, because their scopes are opposite. Truth itself is *present* in the part you fit on, because the fit
needs it, and must be absent from the inputs of every held-back part, wherever that part carries it. When
truth rides in a unit's location or name, handing over the inputs hands over the truth, and every check that
looks only for the absence of a separately stored truth value passes while the channel is wide open.

**Does any information you hold fail to exist at prediction time?** Anything recorded after the fact,
back-filled, corrected later, or whose value could only be known once the outcome was known is information
the deployed situation will not have. This creates no partition boundary; it must be removed from what the
solution can see in *every* part, train included. Removal means the information cannot be reconstructed from
what remains, not that its name is gone: the order of records, of files, or of names can carry the same
quantity after the attribute itself is deleted. So prove the absence by measuring what can still be
recovered, and either destroy the residual encoding before handover or report it as an open channel.

**The mistake that matters most.** The presence of structure in the inputs does not by itself dictate how to
partition. What dictates it is the prediction situation you are claiming to measure: what will be known and
what will be unknown at the moment a prediction is required. Identical structure demands different boundaries
when the claims differ, and one collection demands a different boundary when the claim changes. So derive the
boundary from the claim first, then check that the data can support it. If you cannot tell which claim is
intended, ask rather than choosing silently, and record the answer as part of the derivation.

**When the answers conflict.** The questions above do not always compose. Two keys can be jointly
unsatisfiable — one requiring a set of units to stay together while another requires a cut that runs through
it — and forcing both leaves parts too small to measure anything, or empty. That is a finding, not something
to resolve quietly. Print the size of the conflict: how many groups and how many units straddle each cut.
Then either decide which key the claim you are measuring actually requires and demote the others, recording
each demoted key as UNMET rather than dropping it silently and naming the claim the number therefore does not
support; or discard the units in the conflict, which removes exactly the most linked, or most recent, units
and so changes the population — count them and state what the surviving population is. Either way the number
describes the population that survived rather than the one the request implied, and the report says so.

**Declare the boundary, negatives included.** The derivation's output is: each boundary key, computable for
every unit; the rule each key imposes on the partition; and the candidates you considered and *rejected*,
each with its reason. The negative declarations are what let a later reader disagree with you specifically
instead of vaguely, and they are the only record that a channel was examined rather than missed.

## 3. How many parts, and why

Four roles have to exist and stay distinct for a reported number to mean anything. These labels are used
throughout this skill:

- **train** — what the solution is fitted on. Read without limit.
- **engine_val** — what the search's own internal reward is computed on, so it can tell its candidates
  apart. Read without limit by the search.
- **select** — what you rank finished candidates on, to choose one. Read at most once per candidate.
- **holdout** — spent once, on the single winner, to produce the number you report.

Collapsing any pair inflates the final number, each in its own way:

- **train doing engine_val's job.** The reward then measures recall of what was fitted, and the search climbs
  that reward, so the search is steered toward memorization by construction.
- **engine_val doing select's job.** The search has already optimized against that data, so ranking on it
  picks whichever candidate exploited it hardest. There is a second, sharper reason: the isolation of
  engine_val is *requested* of the search, not enforced on it — its inputs are handed over and only its truth
  is withheld, which withholds nothing at all when the truth is carried by the inputs themselves, and the
  instruction not to look is an instruction given to a code-writing model (`references/engine.md` section 5
  for the exact mechanism). A requested property is not a verified one, so nothing you report may rest on it.
- **select doing holdout's job.** This is the failure that makes a suite feel most trustworthy while lying,
  and no line of the code looks wrong. Choosing the best of many candidates by a number computed on one body
  of data makes the reported number the maximum of a set of noisy estimates; its expected value sits above
  true performance, and the gap widens as you compare more candidates. The only remedy is a body of data that
  took part in no choice at all.
- **holdout spent more than once.** The moment you read it and then change anything, it has become select
  data, and the report must describe it that way.

A role that has no work to do in your procedure need not exist — but then the report cannot make the claim
that role would have supported, and must not imply it.

**When the boundary is an ordering, the roles are positions in a sequence, not merely disjoint bodies.** Each
held-back part is predicted from what lies strictly before it, so the distance between the end of the part
the solution was fitted on and the start of the part a number comes from is part of what that number
measures: a number produced across a larger distance than the situation being claimed answers a different and
harder question, so presenting it as the answer to the question asked misstates what was measured. State that
distance, in the ordering's own units, beside every number you report. If you close
the distance instead by refitting the chosen candidate on everything strictly earlier than the part you are
about to measure, that refit is a new fit and not the candidate that was selected: record the relationship
between the two, and note that it cannot be done inside the search, because advancing the search's own origin
means putting earlier held-back truth on a path the search can reach.

**Sizing.** No fraction is correct, and any fraction stated as correct is wrong somewhere. The tradeoff:
held-back parts must be large enough that the difference you care about is distinguishable from sampling
variation, because a small ranking part turns candidate selection into a coin flip and the final number into
a lottery; train must be large enough that the solution is not crippled, because a starved fit measures data
scarcity rather than the method. Where a boundary key constrains the partition, the count that sets a
held-back part's resolution is the number of distinct key values it holds, not the number of units it holds —
many units from few origins carry the resolution of few — so state both counts wherever you state a size.
Reason about the resolution you need given how many candidates you intend to compare and how variable your
number of record is on this data, decide, and record the sizes with the reasoning. Learning that variability
may require reading truth; do it before the roles exist, look only at the aggregate shape of the truth and
never at per-unit outcomes joined to any candidate's behaviour, and log the read with its purpose so the
accounting of who read what stays complete. Every part must be non-empty and must hold enough units for the
number of record to be defined at all. When the data cannot fill the roles you need without one of them
becoming too small to distinguish anything, that is a finding about the data rather than a problem to divide
your way out of; `SKILL.md` says what to deliver instead of a number.

## 4. Sealing: unreachable, not merely unpassed

Not passing a path is not isolation. Generated code runs as an ordinary process with the same filesystem
access you have: it can list directories, walk upward, follow links, search for names, and open anything the
account can open — and it is written by a model asked to make a number go up. Assume that any reachable truth
will eventually be reached, not out of malice but because "reachable" and "used" are separated only by luck.

Reachable means anything a process starting from a given directory can arrive at:

- relative references inside any file in that directory;
- locations named inside listing, index, or metadata files that live in it;
- symbolic links, including ones reached through other links, and links whose target is a parent;
- absolute locations written into any file in the tree;
- whatever an archive in the tree expands into — the engine can copy an entire input tree elsewhere and
  expand archives in place (`references/engine.md`), so sealing must hold for the copy as well as the
  original.

So the sealed parts live outside every directory path handed to the engine, at a location not derivable from
anything inside those paths. The truth for the part the search computes its reward on lives in a different
directory from the inputs the search is given; the truth for the parts you rank and report on lives under a
root that is never any path the engine receives (`references/engine.md` for which flag takes which path).

Each part's own root must also be self-contained: every reference emitted into a part names something inside
that part, so the part can be moved or handed to someone else without reaching back to a shared origin.
Self-contained is not the same as relative. A relative reference resolves against whatever directory the
consuming process is standing in, and the code the engine runs does not stand in your data
(`references/engine.md` section 2). So either write references that need no root, or have something tell the
code which root to join them to — and prove them the way the consumer resolves them, never from where you
happen to be standing.

Sealing and self-containment govern held-back units and their truth. Material that is available at prediction
time and withheld from no part is not sealed material: every part may reference it from one shared location
instead of carrying its own copy, provided you declare that location with the reason it is legitimate and the
reachability proof accounts for it as a resolution you expected rather than as a violation it found.

Membership is defined on unit identifiers, so every artifact that records membership carries those
identifiers in a form that survives being written and read back. That artifact is itself sealed material: a
record naming which part a unit went to hands a reading process the held-back units by name, even where their
truth is nowhere in reach. Nothing in the partition may depend on position or order of appearance
(`references/scoring.md` for the same discipline applied to joining).

## 5. Proving it

You write these checks yourself, against the data as it actually is, in whatever form fits it. One rule
governs all of them: **a check that prints only a verdict proves nothing**, because a verdict is a restatement
of the claim while the numbers are the evidence. Each check prints the quantities it compared — including the
zeros, next to the totals they were compared against — and only then whether the requirement held. The report
quotes those quantities, not the verdict.

What must be proven, and what its output must contain:

- **Membership is disjoint and accounted for.** Per-part unit counts, their sum, the number of distinct
  identifiers across all parts, the size of every pairwise intersection, and how many units of the source
  landed in no part.
- **No declared boundary value straddles.** Per key declared non-straddling: how many distinct values exist,
  how many of them occur in more than one part, and the offending values themselves when there are any —
  naming them is what makes them fixable. A key declared spanning is not tested here; it supplies the groups
  the other checks are computed within, and section 2 says what declaring one obliges you to record.
- **Near-duplication does not straddle.** The notion of closeness used and its parameters, the scope searched
  and what proportion of the possible comparisons that scope covered, how many cross-part pairs it flagged,
  the identifiers of the closest ones, and how many inseparable sets the flagged relation closed into.
- **A declared ordering is monotone.** For each adjacent pair of parts, the two boundary values being
  compared and the distance between them, across the collection as a whole and then per group: how many
  groups were checked, how many violated, and which.
- **References resolve.** How many references were checked, how many opened when resolved as the consuming
  process will resolve them rather than from inside the part, the directory the check stood in, and the first
  failures by name. Any location deliberately shared by every part is listed here by name, so it is never
  mistaken for a leak and never silently multiplied into copies.
- **Nothing sealed is reachable.** How many locations were resolved by walking outward from the directory the
  engine is given, how many of those landed outside it, and the size of the intersection between sealed
  identifiers and identifiers recoverable from inside it — then, for that intersection, two further counts
  kept apart: how many of those units have their truth recoverable there, and how many have the part they were
  assigned to recoverable there. Those two are the counts that must be zero. The bare intersection may be
  non-zero because you decided to hand a held-back part's inputs over, which is a weaker claim than sealing
  and a decision recorded where `SKILL.md` settles the partition; resolve it by naming that decision and what
  it costs the claim, never by narrowing the search until the count falls.
- **No held-back unit's truth travels with its inputs.** Per held-back part separately, not once for the
  whole tree the engine receives: how many of that part's units have their truth recoverable from what was
  actually handed over, searched in every carrier section 2 names and in whatever else this data affords,
  against the number searched; and whether a unit's identifier and its truth can be recovered together from
  any one place a process could reach.
- **The truth's values are spread across the parts.** Where the truth takes a discrete set of values, per
  part: how many units carry each value; which values appear in a part you will report a number from but not
  in the part the solution is fitted on; and any part where the number of record is undefined for something
  it contains. This is evidence about what the number can mean, not a licence to reshape the partition until
  the table looks tidy.
- **Information that must not be visible is absent.** The names of the attributes you decided to remove, the
  count of their occurrences in what the engine receives against the total searched, and how well the removed
  quantity can still be recovered from the artifact as handed over — printed next to the same measurement on
  that artifact with its incidental orderings destroyed, since a name can be gone while the quantity survives
  in the order of what remains. Occurrences may be legitimate where the fitting part's own truth lives, so a
  non-zero count is resolved by naming the specific locations it came from, never by loosening the test until
  it prints zero.

Run every check on the artifacts you actually hand over, after the last change to them. A partition that was
proven and then regenerated, reordered, moved, or repaired is unproven again. Most of these checks cost
minutes, and skipping them costs a whole run whose number means nothing; the closeness search is the one that
can be genuinely expensive, which is why its scope is a declared decision rather than a silent omission. Keep
the printed output where a reader of the report can find it without rerunning anything (`SKILL.md` on the run
log).

## 6. Honest limits

Each of these goes into the report as a stated limit. Assuming them away silently is precisely how a
rigorous-looking result misleads.

- Reachability checking sees only what exists while it runs. It cannot see a location a running program
  constructs for itself by walking, assembling a string, or searching. Sealing outside the tree is a strong
  barrier; the proof of it is not exhaustive.
- Duplicate detection is exactly as good as the notion of similarity chosen and the scope it was applied
  over, and both are judgments. A notion that finds nothing may mean the data is clean or that the notion is
  blind to what is there. State the notion and the scope alongside the result, so the result can be read for
  what it is.

The limits on read-count discipline and on whether the boundary you declared was the right one are stated
under *What cannot be proven* in `SKILL.md`; both apply to everything above, and the report states them in
that form rather than in a second version of the same sentence.
