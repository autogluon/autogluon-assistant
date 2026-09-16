# Engine reference: driving mlzero

Read this when you are about to invoke the engine, or when you are about to read a result out of a finished
run. Everything here is a fact about a real program, verified in
`/opt/dlami/nvme/autogluon-assistant/src/autogluon/assistant/`; source paths are cited where the behaviour is
subtle or surprising. That is why this page is concrete while the rest of the skill is principled. If the
repository has moved on since this was written, re-verify the claim you are relying on at the cited path
rather than trusting this page — and prefer omitting a claim to guessing one.

Nothing here tells you what kind of problem you have or which library suits it. Those are yours to derive.

## Contents

1. What the engine is, and what that means for what you can promise
2. Invocation
3. Which channels actually reach the generated code
4. The external scorer hook: the wire contract
5. The hardcoded prediction shape you must design around, and getting a prediction out of a finished node
6. Reading the output honestly
7. Environment reality
8. Libraries the engine may choose

## 1. What the engine is, and what that means for what you can promise

The engine runs a tree search. Each iteration it writes a candidate solution, executes it, judges the
execution, and expands the tree. You do not author the solution code and you cannot inspect it before it
runs. Your influence arrives through four channels only: the directory you point it at, the instruction
string, the configuration (including the prompt templates themselves), and the external scorer hook, which
replaces the search's reward with a number you compute. Everything else you do is reading what it produced.

The consequence governs every claim you will later make: anything you asked the engine to do is a *request*.
It becomes a property of the run only when you check the artifact afterwards and can quote what the check
printed. Keep those two words apart in your notes from the start, because the report has to keep them apart.

## 2. Invocation

Flag spellings are exact; several are counter-intuitive. Source: `cli/app.py`, `coding_agent.py`.

```
mlzero -i <dir> -o <dir> [-c <yaml>] [-n <int>] [-t <str>] [-e <dir>]
       [--continuous_improvement] [--conda-env <abs-env-prefix>]
       [--eval-script <path> --eval-ground-truth <path> --eval-validation-features <path>]
       [--eval-metric <str>] [-v 0..4] [--remove-iteration-folders]
       [--enable-per-iteration-instruction] [-m] [--provider <name>]
```

- `-i/--input <dir>` — required for a coding run. This is the directory the engine perceives, summarizes and
  points the code writer at. It bounds what the engine *describes*, not what the generated code can open: the
  generated code is launched as an ordinary process with your own filesystem access and with no working
  directory set (`agents/executer_agent.py` runs it through `Popen` with no `cwd`), so it inherits the
  directory `mlzero` was launched from, and a relative reference inside anything you hand over resolves
  against that directory rather than against the data. Make such references absolute, or have your
  instruction name the location the code should resolve them from. What belongs inside this directory, and
  what actually makes the rest unreachable, is `references/isolation.md`.
- `-o/--output <dir>` — with no `-o`, output goes to `runs/mlzero-mcts-<YYYYMMDD_HHMMSS>-<uuid4>/` relative
  to the repository. An existing `-o` directory is reused silently and node ids restart at 0 on every
  invocation, so pointing two runs at one directory interleaves their nodes and destroys your ability to
  attribute a node to a run. Use a fresh directory per run when you intend to attribute anything, and give it
  as an absolute path: only the copy used to create the directory and set up logging is resolved, while the
  string every later path is built from is the one you typed (`coding_agent.py`), so a relative `-o` becomes a
  relative literal inside generated code (section 5) that resolves against the launch directory rather than
  against the run.
- `-c/--config <yaml>` — `/opt/dlami/nvme/autogluon-assistant/src/autogluon/assistant/configs/default.yaml` is
  always loaded first and this file is merged on top, so a config may be a small overlay. This is also the
  channel that carries a prompt template, the only wording that reaches a coder verbatim and unsummarized
  (section 3).
- `--provider {bedrock,openai,anthropic,sagemaker}` — it is never passed into the run. It only selects which
  configuration file is loaded, and only while `-c` is still at its default; with an explicit `-c` this flag
  has no effect at all and the provider comes from the merged configuration. Do not report a provider because
  you passed this flag; report it from the configuration that was actually merged.
- `-n/--max-iterations <int>` — default 5. Each iteration is one node: one generate-and-execute cycle. The
  tradeoff that sets it: more iterations buy more candidates to choose between and more chances to recover
  from a failed node, at wall-clock and token cost, and — when candidates are ranked on a number computed
  from one fixed set — more opportunity to select for that set's noise rather than for the ability you care
  about. Let the cost of a wrong choice and the size of the evidence you can afford decide, and say which.
- `--continuous_improvement` — spelled with underscores. Without it the loop stops at the first node whose
  verdict is SUCCESS (`coding_agent.py`). If your plan is to rank several candidates against each other, a
  run without this flag may hand you exactly one; set it deliberately rather than discovering this after. It
  also changes what the engine asks the code to validate on when the scorer hook is off (section 4). A run
  ends early for a second reason too: the loop stops when the search has nowhere left to expand, every branch
  having gone terminal, which is reached routinely once `max_debug_depth` failed attempts accumulate on a
  branch. So the node count is a result of the run, never `-n`; section 6 says which stop occurred.
- `-t/--initial-instruction <str>` — see section 3 for where it lands and how much survives.
- `-e/--extract-to <dir>` — copies the input tree into that directory, extracts archives there, and then uses
  it as the input folder. It creates a second copy of everything, so any reasoning you did about what is
  reachable from `-i` must be redone for this directory.
- `--conda-env <path>` — see section 7. It must be an absolute environment prefix, for the reason in section 4.
- `--eval-script` / `--eval-ground-truth` / `--eval-validation-features` / `--eval-metric` — section 4.
- `-v/--verbosity 0..4` — default 1.
- `--remove-iteration-folders` — deletes per-iteration folders as the run proceeds. It deletes the evidence
  you would need to audit a node after the fact; it buys disk. Choose accordingly.
- `--enable-per-iteration-instruction` — suspends the run waiting for input at each iteration. An unattended
  run with this flag set hangs rather than progresses.
- `-m/--enable-meta-prompting` — lets the system rewrite its own prompt templates. If you are relying on
  wording you placed in a template, this flag can rewrite it away; the two features interact.

The flags that decide whether a run is trustworthy are `-i`, `-o`, the scorer trio, and
`--continuous_improvement`. The rest are conveniences.

## 3. Which channels actually reach the generated code

Spend your wording where it survives. Sources: `prompts/python_coder_prompt.py`,
`prompts/variable_provider.py`, `prompts/task_descriptor_prompt.py`, `prompts/tool_selector_prompt.py`.

- **The instruction string reaches the code writer verbatim**, as `{user_input_truncate_end_2048}`, which
  keeps the *first* 2048 characters and discards everything after them. Check the direction yourself before
  relying on it — the name is misleading: in `prompts/variable_provider.py` the `end` mode returns
  `value[:max_length]` and only the `start` mode keeps a tail. The task-describing step gets the same string
  through `{user_input_truncate_end_16384}`, again the leading characters. So the sentence that must be
  honoured belongs at the *start* of the instruction, and anything past 2048 characters reaches no code writer
  at all. Both windows are fixed by the placeholder names inside the templates; no configuration key widens
  them, only a template override (section 5). After the first node, confirm what arrived rather than assuming:
  read that node's `states/python_coder_prompt.txt` and print whether your sentence is in it.
- **A description file inside the input folder reaches only the task-describing step**, as
  `{description_file_contents_truncate_end_16384}`, and that step is instructed to *condense* what it reads.
  Nothing written in a description file reaches any coder verbatim. A constraint stated only there is a
  suggestion that an LLM has paraphrased before any code writer sees it.
- **A prompt template set through configuration reaches the coder verbatim and unsummarized.** This is the
  highest-fidelity channel and the most expensive one, because you take ownership of what you replaced
  (section 5).
- **The same wording also steers library choice.** The tool selector ranks libraries from the condensed task
  description plus a data summary plus the catalog's descriptions, so how you describe the data and the
  objective changes which library is attempted (section 8).

The coder's default template carries these placeholders: `{selected_tool}`,
`{per_iteration_output_folder}`, `{validation_prompt}`, `{tool_prompt}`, `{code_improvement_prompt}`,
`{task_description}`, `{data_prompt}`, `{user_input_truncate_end_2048}`, `{all_previous_error_analyses}`,
`{tutorial_prompt}`. It does not contain description-file contents. The assembled prompt is truncated to
80000 characters if longer, and the tutorial block sits last in the default template, so the tail is what
truncation eats.

Independently of anything you pass, the default coder template hardcodes: save files only to the
per-iteration output folder; save any trained model in a timestamped folder inside it; drop training samples
that have no valid label, from the training data only, "unless explicitly instructed otherwise"; remove an
index column it judges unnecessary; write test predictions to a file named `results` whose extension matches
the test data; preserve the test data's original indices and never drop a test row; and name output columns
exactly as they are named in the training or sample-submission files. These are asserted to the coder whether
or not they fit what your data needs. Two consequences worth deriving before you run: the drop-unlabelled
clause deletes exactly those units whose truth is legitimately empty, so if an absent value is itself a valid
answer for your units, say so — that clause is the one carrying its own escape, and the instruction channel
can countermand it; and the column-naming clause presumes your truth is a named column somewhere the coder
can find it. Where the clause has no escape and does not fit, the only lever is the template-override
decision in section 5, not an instruction string.

## 4. The external scorer hook: the wire contract

This hook is how the number *you* define becomes the number the search optimizes. Source:
`managers/node_manager.py` (`use_external_scorer`, `eval_python`, `eval_predictions_path`,
`_run_external_scorer`, and the override inside `simulate`).

- **Activation** requires all three of `--eval-script`, `--eval-ground-truth`, `--eval-validation-features`;
  `--eval-metric` is optional. With any one missing the hook is silently inactive, and what the engine asks the
  code to validate then has three distinct states (`prompts/python_coder_prompt.py`,
  `_generate_validation_prompt`). Hook active: the block in section 5. Hook inactive with
  `continuous_improvement` set: the coder is told to hold out ten percent of the data it was handed, train on
  the remainder, and compute and print its own metric — a partition each node makes for itself and a quantity
  each node chooses for itself, read back out of the log by a model, so those numbers are neither yours nor
  comparable between nodes. Hook inactive with that flag unset, which is the shipped default: the block
  renders empty, no node is asked for any number, and the no-peek prohibition is absent along with it — the
  search is then steered only by success, failure and the failure penalty. In neither inactive state does the
  engine touch a body you set aside for it, so there is nothing there for an `engine_val` role to be wired to.
- **Which body belongs on which path.** `--eval-validation-features` carries the inputs of the body you are
  content to let the search consume — the `engine_val` role in `references/isolation.md` section 3 — and
  `--eval-ground-truth` carries the truth for exactly those units and no others. Anything held back to rank
  candidates or to be spent once at the end must appear in none of the engine's path arguments: a body named
  on one of them has been read by the search and can no longer play a role that requires the engine never to
  have seen it. Because the shape in section 5 makes predictions arrive in the features file's own row order,
  that order is the only alignment on offer, so the truth must be materialized to match it — if it does not,
  every gate still passes and the reward means nothing.
- **Invocation form**: `<eval_python> <eval_script> --ground-truth <GT> --predictions <PRED>`, plus
  `--metric <M>` only when `--eval-metric` was given. Nothing else is ever passed: not the optional
  `--target` that the implementation's own docstring advertises, and no training-scale or auxiliary-data flag
  either. Anything else your scorer needs must be baked into the scorer or reachable from the ground-truth
  path you hand it.
- **No working directory is set**, so a relative script path resolves against wherever `mlzero` was launched.
  Pass absolute paths for all three.
- **Interpreter selection**: the configuration key `eval_python` if set, else `<provided_conda_env>/bin/python`
  formed by string join, else the interpreter running `mlzero`. Because it is a plain join, a bare environment
  *name* produces the relative path `<name>/bin/python` and the scorer will not start; `--conda-env` must be
  an absolute environment prefix.
- **Where predictions must appear**: `<node_dir>/output/<eval_predictions_name>`, where
  `eval_predictions_name` is a configuration key defaulting to `validation_predictions.csv`. The engine's
  only test on that path is an existence check — and a directory satisfies an existence check. Existence is
  therefore not evidence that an artifact was produced.
- **How the number is read**: from the scorer's stdout, by the regular expression
  `MLZERO_SCORE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)`. Higher is always better on this wire, whatever the
  quantity means; orienting your number to that convention is part of defining it (`references/scoring.md`).
  The scorer is killed after `per_scorer_timeout` seconds, shipped at 600.
- **All three paths are opaque to the engine.** It never opens or parses the ground truth or the predictions
  and imposes no format on either; the validation-features path it does not open, existence-check or
  format-check at all — `managers/node_manager.py` reads it only as a value for the three-flag activation
  test, and `prompts/python_coder_prompt.py` only interpolates it as text into the coder's prompt. So none of
  the three is constrained to any shape by the engine: every format agreement is between your scorer and the
  code the engine writes, which is exactly why the shape in section 5 matters and why `references/scoring.md`
  is where the artifact and identifier contract gets settled. Two consequences follow from the features path
  never being opened. It may be a file or a directory, or resolve to nothing at all: the hook activates on the
  three strings being present, so the block in section 5 renders either way and the generated code is the
  first thing to discover the path is empty. And nothing anywhere checks that this path holds what section 5's
  block asserts it holds — that assertion is made to a model, about a body you built, and what enumerates the
  units on it is yours to build (`references/scoring.md` section 2).
- **Two of the three paths are disclosed verbatim, and the third is not hidden.** The features path and the
  predictions path are interpolated into the coder's prompt as literal absolute strings (section 5), so the
  model that writes the code is told exactly where the body you set aside for the search lives. The
  ground-truth path reaches no prompt — it appears only in the scorer's argument list — but the scorer's
  combined output is written into that node's `states/` directory, one level up from the `output/` folder whose
  absolute path every code writer is given, and a scorer that fails on a file it could not read names that
  file there. Treat all three as disclosed: nothing is protected by going unmentioned, so what makes a body
  safe is what a walk outward from these locations actually reaches, which is why `SKILL.md`'s sealing
  evidence asks for that walk rather than for the list of paths you passed (`references/isolation.md`
  section 5 is how the walk is done).
- **Failure semantics**: a missing predictions path, a non-zero exit, a timeout, or stdout with no matching
  line each produce no score, log only a warning, and let the run continue. The scorer's combined output is
  written to `<node_dir>/states/external_scorer_output` — no extension, so a glob for `*.txt` misses it — as
  stdout, then the literal separator line `---stderr---`, then stderr. Read it when a node has no number.
- **The external score replaces the engine's own scraped score only when that node's verdict is SUCCESS.** A
  SUCCESS node with no score stays SUCCESS-but-unvalidated: it can still be evolved, but it can never be
  crowned best. The first node to receive any score is crowned unconditionally — the crowning test in
  `managers/node_manager.py` short-circuits when there is no best node yet — and only after that does a node
  need a score *strictly exceeding* the best so far. Nothing screens the value on the way in. The expression
  that extracts the external score matches only digits, so a non-finite spelling cannot arrive that way,
  though an overflowing exponent can; on the log-scraped path the executer parses the model's line with a bare
  `float()`, which accepts a non-finite spelling outright. Such a value crowns whatever node produced it, and
  where nothing can exceed it — every comparison against a not-a-number incumbent is false — it is never
  displaced, while also corrupting the search's own score normalisation. So a missing or non-finite score does
  not cost a node the crown; it costs the recorded ranking its meaning. Read the value the engine recorded
  before you quote it, and see section 6 for whose rule ineligibility actually is.

Honest consequence: the hook makes the reward yours, but nothing compels a node to emit a scoreable
artifact. Before saying the search was guided by your number, count how many nodes actually received one, and
quote the count.

## 5. The hardcoded prediction shape you must design around, and getting a prediction out of a finished node

When the hook is active, the block rendered into `{validation_prompt}`
(`prompts/python_coder_prompt.py`, `_generate_validation_prompt`) tells the code writer, in one block, to:
train only on the training data and neither train on nor peek at the validation-features file; load that file
from a given absolute path, of which it asserts, verbatim, "(the target column has already been removed)";
predict exactly one value per row, in the same row order, dropping and reordering nothing; write the
predictions to the predictions path as a single-column CSV named `prediction` with no index; and not compute or
print any score of its own — closing with "If you cannot produce this file the run is treated as unvalidated."

Every clause of that is asserted whether or not it fits what a prediction for one of your units actually is,
and the parenthetical is asserted about a file you built, whatever that file's parts are. If your units are
not rows of a table from which one target column was removed, the code writer has been told something untrue
about the file it must read, and nothing in the engine will correct it. So make the following decision
consciously, and write down which way you went and why.

**Can the number of record be computed from an artifact of that shape, with every row attributable to the
right unit?** The block fixes the column count, the row count and the row order; it says nothing about what a
cell may contain. Left as it is, therefore, the only join the wire offers is row position — precisely the join
`references/scoring.md` forbids. An identifier travelling on the ground-truth side does not repair that: it
fixes which unit row *i* of the truth belongs to, but it cannot establish that the coder's row *i* is that
unit, so the strongest check available becomes a comparison of the two counts, which no count-preserving
reordering or off-by-one error can fail. Under the unmodified block, matching by identifier is therefore not
merely unproven but unavailable: record it as not held, name position as the join you actually used, print
both counts and the count you compared, and never present it as verified.

Three routes get you a join you can check. Each has a real cost.

- **Ask for the identifier inside the artifact.** Because the single permitted column's contents are
  unconstrained, one value per row can be a value that carries its own unit identifier, and a scorer that
  refuses any cell without a recognisable identifier fails closed — a node that ignored the request returns
  unvalidated instead of silently mis-joined. The cost: this is a request in the instruction channel standing
  against a template clause that implies otherwise, so it holds only for the nodes where it landed. Place it
  inside the window that survives (section 3), and count the nodes whose artifact actually carries identifiers
  before claiming the join for any of them.
- **Override the template through configuration.** Set `template:` under the relevant agent's own section in a
  `-c` config — for the code writer, `python_coder.template` — either as a literal string or as a path ending
  in `.txt`. `/opt/dlami/nvme/autogluon-assistant/src/autogluon/assistant/configs/data_visualizer.yaml` is a
  shipped precedent that overrides several templates. Template validation rejects nothing: it logs a warning
  for a placeholder name it does not recognise, requires no placeholder to be present, and renders an
  unrecognised name into the prompt as the literal text `{UNKNOWN_VARIABLE:<name>}`
  (`prompts/base_prompt.py`, `prompts/variable_provider.py`), so a custom coder template may omit
  `{validation_prompt}` entirely and still load, and a mistyped placeholder travels silently to the model. The
  cost: the no-peek prohibition lives in the *same* block as the shape requirement, so dropping the block
  drops the leak prohibition too — restate whatever you still need in your own words, read a node's saved
  `states/python_coder_prompt.txt` to see what the model was actually sent, and verify the resulting artifact
  rather than assuming your wording landed. Meta-prompting, if enabled, may also rewrite what you wrote.
- **Leave the hook off and rank candidates yourself after the run**, from whatever each node left behind. The
  cost: the search was then steered by one of the two inactive-hook states in section 4 — a number each node
  chose and printed for itself, or nothing but success and failure — so the tree explored a different
  objective from the one you report, and no body you set aside is wired to the run at all. You also have to
  produce a comparable artifact for every node you want to rank, which is the next subsection.

The tradeoff is reward fidelity during the search against how much of the engine's prompt you take
responsibility for. Neither answer is default-correct; the one thing that is wrong is choosing by accident.

### Getting a prediction out of a finished node

A run takes one validation-features path, so every node predicted on that one body and nothing else. Ranking
candidates on a body the engine never saw, and spending the final hold-out on the winner, therefore always
means producing a prediction no node produced. What each node leaves you is the recipe, not the answer:
`generated_code.py` and `execution_script.sh`, which are the candidate as text; `output/`, holding what it
wrote, including a timestamped model folder *if* the code chose to save one, which the template asked for and
nothing enforced; `conda_env` when the engine built one there; and `states/python_coder_prompt.txt`, which is
the prompt that produced the code and therefore the record of where its literals came from.

None of it is parameterised. Both the body it reads and the folder it writes to were interpolated into the
prompt and reach the code as literal strings fixed inside `generated_code.py`, and the bash script names that
file by path too, so a copy of the script run from elsewhere still executes the original node's code. No
engine flag redirects either literal, so pointing a candidate at another body is an edit to the candidate's
text: copy the code aside, change what it reads, and record what you changed, because what you then run is no
longer the thing the search executed.

Change what it writes as well, or you destroy your own evidence. The predictions file has a fixed name inside
that one folder, so a re-run overwrites the artifact whose recorded score you are trying to preserve, while a
saved model lands in a *new* timestamped folder beside the old one, after which nothing distinguishes the fit
that was scored from the fit you just made. Copy a node's `output/` aside before re-executing it: for the
crowned node the engine also copied those contents up into the run root (section 6), but for every other node
that copy is yours to make or the evidence is gone. And do not assume a node's `output/` is the product of
that node alone — a node that evolves or debugs another is handed that node's code verbatim, its own output
folder's path included, so one node's code can legitimately read from or write into another's folder.

Two ways to get the missing prediction, and they support different claims:

- **Re-execute the node** with the read path resolving to the other body. Nothing the engine keeps separates a
  node's fitted state from whatever its code happened to save, and the script has no mode that skips fitting,
  so re-execution refits from scratch — unless the code itself reloads a model it finds, which you learn by
  reading it rather than by assuming either way. With a `--conda-env` the environment is re-resolved first
  (section 7), and it may have moved since the original run. What you then measure is a refit of that recipe,
  not the fitted candidate the search actually scored; say so, and print the node's recorded score beside the
  refit's number on the engine's own body, so a reader can see how much of your ranking is the recipe and how
  much is one particular fit.
- **Load the saved model and predict yourself.** This route exists only where the code saved something
  loadable, which was asked of it and never enforced, so look before you plan on it. Nothing is retrained, but
  the inference path is now yours rather than the one the node's code used, and any preparation of inputs that
  lived only inside that script is yours to reconstruct. Establish the reconstruction before trusting it:
  predict the engine's own features body with your loader and show that you reproduce that node's recorded
  number.

What is not available is the shortcut of adding the ranking body or the hold-out body to the features file
with their truth withheld, to harvest predictions for them cheaply. Withholding truth does not make a body
unseen: the search read those inputs and chose among nodes with them present. Once a body has appeared on any
engine path argument, it can no longer play a role that requires the engine never to have seen it.

Asking the code, through the instruction channel of section 3, to write a *second* artifact for a body named
on none of the engine's arguments is a different thing, and it is the only way a node produces the coverage
`SKILL.md` asks its brief to request. It carries two prices. It is wording, so it holds only for the nodes
where it landed and only where a check on the artifact says so; and the code still reads that body's inputs,
which buys the weaker form of independence that `SKILL.md` prices at the point it settles the partition.

## 6. Reading the output honestly

Per node the engine writes `<out>/node_<id>/` containing `generated_code.py`, `execution_script.sh`,
`output/`, `states/`, and `conda_env` when it builds one. A `node_init/` directory also exists.

`<out>/node_<id>/states/` holds uuid-suffixed files including `decision_<hex>.txt`,
`validation_score_<hex>.txt`, `error_summary_<hex>.txt`, `stdout_<hex>.txt`, `stderr_<hex>.txt`, and
non-suffixed `python_code.py` and `python_coder_prompt.txt`. **The verdict is the newest `decision_<hex>.txt`
by modification time**, and its content is `SUCCESS` or `FIX`. More than one of these can exist in a node,
which is why newest matters rather than any.

Know what that verdict is worth before you lean on it. It requires a zero exit status — a non-zero exit is
overridden to `FIX` whatever the judging model concluded — and beyond that it is a model's judgement of the
last 8192 characters of each stream (`agents/executer_agent.py`, `prompts/executer_prompt.py`). The saved
`stdout_` and `stderr_` files are lossy in the same direction: a line identical to any of the previous hundred
on that stream is dropped before saving, so they are usable as narrative and never as a count of how often
something happened. This is why candidacy below is decided by the artifact rather than by the verdict.

**A trap worth knowing before you quote a per-node number.** `validation_score_<hex>.txt` is written where
the executer parses the LLM's reading of the log (`prompts/executer_prompt.py`), which happens *before* the
external score replaces the node's score (`managers/node_manager.py`, inside `simulate`). When the hook is
active that file can therefore disagree with the number the search actually used. The authoritative records
are the scorer's own output file for that node and the winner's score summary below; the only way to
reproduce a per-node number is to run your scorer again on that node's artifact.

`<out>/states/` is run level. Its `selected_tool.txt` holds the comma-joined **ranked list for the whole
run**, not the library any individual node used — reading it as the winner's library is simply wrong. It also
contains `task_description.txt`, `tool_selector_explanation.txt` and `description_files.txt`.

`<out>/best_run` is a **symbolic link** to the winning `node_<id>` directory, so resolve it with a real-path
call to learn the node id. When a node is crowned, the contents of its `output/` directory are additionally
copied up into `<out>/` — so a file at the run root may be a copy; the node directory is the provenance.

The winner's `states/best_run_summary.txt` contains `Linked to: node_<id>`, then
`Reason: best validation score (<value>)` or `Reason: last successful execution`, then `Tool used: <tool>`,
then a `Validation Score Summary:` block with one `Node <id> (<tool>): <score>` line per *scored* node and
` (BEST)` marking the winner — or the single literal line `No validation scores available.` — then
`Available tools:`, `Tools used:` and `Tools not used:` lines.

**Attributing a library to a node** has three records, with different coverage. This summary names the library
for each *scored* node, so a SUCCESS-but-unvalidated node has no line in it at all.
`node_visualization.pdf` tabulates node id, stage, status and tool for *every* node, including unscored ones.
And each node's own `states/python_coder_prompt.txt` contains the library name rendered into the prompt that
node was given. What is never per-node attribution is the run-level `selected_tool.txt` above.

`<out>/logs.txt` is the console-level digest and carries the completion markers
`Task completed successfully! Best node: <id> with validation score <value>`,
`MCTS search completed in <N> seconds`, `Total nodes explored: <N>`, `Best validation score: <value>`,
`Tools used: ...` and `Output saved in <path>`. It also carries the reason the search stopped —
`Stopping search - solution found and continuous improvement is disabled` or
`Stopping search - all nodes are terminal.` — which is how you explain a run that produced fewer nodes than
`-n` instead of reporting a full budget's worth of search. `info_logs.txt`, `detail_logs.txt` and `debugging_logs.txt`
are also written, as are `token_usage.json`, `node_visualization.pdf` and `node_tree_iteration_<k>.pdf`.

**Deciding which nodes are genuine candidates.** A node is a candidate when its newest decision file says
SUCCESS *and* its declared prediction artifact exists *and* opening that artifact yields something your
scorer can consume. Presence alone is not enough: the engine's own check is satisfied by a directory, and a
file can exist while being empty, partial, or malformed. Ineligibility for a missing or non-finite number is
*your* rule, not something the engine enforces — as section 4 says, it will crown a first-scored node whatever
the value was. Apply the rule anyway in every ranking you build: such a candidate is not a zero, and quietly
treating it as one manufactures a ranking out of a failure.

**When no node was ever scored**, `best_run` is linked to the last successful node and the recorded reason is
the literal text `last successful execution`. That is an ordering by time, not by quality. Either state that
the run produced no scoreable candidate, or score the surviving artifacts yourself and say plainly that you
did the scoring after the fact. When no node succeeded at all, or when the node that would have been linked
has no `output/` directory, there is no `best_run` link to resolve; its absence is a fact about the run and
not a filesystem accident.

## 7. Environment reality

`--conda-env` makes the bash coder activate that environment and run
`uv pip install -r <selected tool requirements> --prerelease=allow -r <common requirements>` at **every
node** (`prompts/bash_coder_prompt.py`). Separately, `configure_env` is forced true when the selected tool is
`machine learning`, `huggingface` or `fairseq`, which additionally instructs the coder to install any other
packages it decides it needs.

So the environment you hand over is a shared cache the engine mutates as it runs, not a pin. A later node can
execute against different resolved versions than an earlier one. The honest implication: passing an
environment does not make a run reproducible, and you may not describe it as such. If a version matters to
your conclusion, record what is installed at the moment you read the result, and say when you recorded it.
`autogluon` is a namespace package with no top-level `__version__`, so read its version through installed
metadata (`importlib.metadata`) rather than an attribute.

## 8. Libraries the engine may choose

This section describes the engine's coverage. It is not a classification for you to perform on your data, and
you should not decide the library for the engine.

As registered in this repository, the tools are exactly `autogluon.tabular`, `autogluon.multimodal`,
`autogluon.timeseries`, `FlagEmbedding`, `wav2vec2`, `qwen3`, and `machine learning` — whose directory name
contains a space. `machine learning` is the engine's default library and the fallback used when a ranking
cannot be parsed.

Each catalog entry carries its own description of what that library covers, and several state explicitly what
they do **not** cover. Read those entries at
`/opt/dlami/nvme/autogluon-assistant/src/autogluon/assistant/tools_registry/_common/catalog.json` rather than
any paraphrase, including this page's, because the exclusions are the part that costs you a run: the selector
ranks from your condensed description plus a data summary plus these descriptions, and it can rank a library
whose own text excludes what your data requires. The parser also replaces a name it does not recognise with
the closest available name measured by character-set symmetric difference, and falls back to
`machine learning` when it can parse no ranking at all — so the library that runs is not always the library
that was named.

Reading those entries can also tell you, before you spend an iteration, that no registered entry's own
description covers what your data requires. That is a finding, not a dead end: expect the general fallback,
which forces `configure_env` true and has the code writer install what it decides it needs (section 7), size
the iteration count for a library with no task-specific tutorials behind it, say so in the plan, and carry the
coverage gap into the report as a stated limit rather than presenting a weak number as the ceiling of the
approach.

The search deliberately spreads libraries across nodes: the root expands to at most `initial_root_children`
children (shipped 3), and an evolve node may be assigned the first still-unused library, chosen
deterministically as the first element of the unused list. One run can therefore legitimately mix libraries,
and a small breadth setting means only the top-ranked few are ever attempted. Because of this, "which library
produced this result" is a per-node question, answered from the per-node records in section 6 and never from
the run-level ranked list.

Your leverage here is describing the data and the objective accurately in the channels of section 3, and then
checking afterwards which library the winning node actually used.

Relevant shipped configuration values, for reference when you read a run:
`eval_predictions_name: validation_predictions.csv`, `eval_python: null`, `per_scorer_timeout: 600`,
`initial_root_children: 3`, `max_debug_depth: 3`, `max_debug_children: 2`, `max_evolve_children: 2`,
`configure_env: false`, `continuous_improvement: False`, `per_execution_timeout: 28800`. The shipped
`max_user_input_length: 2048` is read by no code; do not raise it expecting the verbatim window of section 3
to widen.
