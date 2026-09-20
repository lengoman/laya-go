# laya-go

A Go client for [Laya](https://github.com/NandhaKishorM/laya), the multilingual
non-autoregressive System 1 decision engine.

Laya reads natural language like an LLM but returns typed decisions and calibrated
probabilities instead of generated text, in a single forward pass — 33 ms for one question,
7.2 ms per question batched. Your code keeps the workflow and the policy. The model supplies
the semantic call that ordinary code cannot make.

Laya itself is Python, with Apache 2.0 weights you host. This is the Go half: the question
and answer types, the router, the presets and the batching, talking to a runtime you own.

```bash
go get github.com/lengoman/laya-go
```

> This package is **inspired by, and written as a helper for, the
> [laya](https://github.com/NandhaKishorM/laya) project** by
> [NandhaKishorM](https://github.com/NandhaKishorM) — it exists to make those checkpoints
> usable from Go without rewriting the runtime. That project is not affiliated with this one.

## Use it

```go
package main

import (
	"context"
	"fmt"
	"log"

	laya "github.com/lengoman/laya-go"
)

func main() {
	// Runs laya in a Python interpreter this process owns. Nothing to deploy.
	client, err := laya.New(laya.WithSidecar(laya.Sidecar{
		Preload: []string{laya.ModelEnglish},
	}))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	resp, err := client.Ask(context.Background(), map[string]any{
		"from":    "user@acme.com",
		"subject": "Duplicate charge on invoice #4411",
		"body":    "We were billed twice for March. Refund the duplicate today or we cancel.",
	}, laya.Questions{
		"department": laya.OneOf("Which department should handle this request?", map[string]string{
			"billing":   "invoices, payments, refunds",
			"technical": "bugs, outages, system errors",
			"sales":     "pricing, new contracts",
			"other":     "everything else",
		}),
		"urgency": laya.Levels("How urgent is this request?",
			"not urgent", "soon", "critical deadline or blocking issue"),
		"churn_risk": laya.Holds("Does the user threaten to cancel or leave?"),
	})
	if err != nil {
		log.Fatal(err)
	}

	department, _ := resp.Answers.Choice("department")
	urgency, _ := resp.Answers.Score("urgency")

	fmt.Printf("%s %.2f, urgency %.2f, churn %.2f, answered by %s\n",
		department.Selected, department.Conf, urgency.Value,
		resp.Answers.MustNoul("churn_risk"), resp.Checkpoint())
	// billing 0.94, urgency 1.84, churn 0.89, answered by english
}
```

## Get a runtime

Laya's weights are Apache 2.0 and there is no API key anywhere in this package. You need a
Python interpreter with `pip install laya`, and then one of two transports.

**A sidecar**, for a single service or a CLI. The interpreter starts on the first request,
holds the checkpoints, and lives until the client is closed.

```go
client, _ := laya.New(laya.WithSidecar(laya.Sidecar{
	Python:    ".venv/bin/python3", // default: $LAYA_PYTHON, then python3
	Device:    "cuda",
	Preload:   []string{laya.ModelEnglish, laya.ModelMultilingual},
	MaxLoaded: 2,
	Stderr:    os.Stderr, // download progress and laya's own warnings
}))
```

**A server**, when more than one process needs the model. `python/server.py` in this
repository is a standard-library reference implementation; anything exposing the same three
endpoints works.

```bash
python3 -m venv .venv && .venv/bin/pip install laya
.venv/bin/python3 python/server.py --preload english,multilingual --max-loaded 2 --device cuda
```

```go
client, _ := laya.New(laya.WithBaseURL("http://gpu-box:8600")) // or $LAYA_URL
```

Either way, **preload what you serve**. A cold checkpoint build costs seconds — Laya measures
7.4 s on CPU and 10.3 s on a T4 — while answering costs tens of milliseconds, and at the
default `MaxLoaded: 1` traffic that alternates languages rebuilds a model on every request.
`client.Warm(ctx)` does it on demand.

## The three questions

Pick by what the answer means.

| Question | Returns | Use it for |
| --- | --- | --- |
| `Noul` | calibrated probability that the answer is yes | whether a condition holds. One per label when several can apply at once |
| `Choice` | one option plus the full distribution | picking from a set you define |
| `Score` | a weighted position across ordered levels | a degree along a described dimension |

Constructors cover the common shape, and the structs are there when you need more:
`laya.Noul{Instructions: ..., Criteria: ...}` accepts a map or slice for `Instructions` when
definitions, contrasts or examples make the question clearer.

```go
laya.Holds("Does the user ask for a refund?")
laya.YesNo("Is this phishing?", "a scam or fraud", "a legitimate email")
laya.OneOf("Which team?", map[string]string{"billing": "payments and refunds", "tech": "bugs"})
laya.Options("What is this about?", "coding", "billing", "other")
laya.Levels("How frustrated?", "Calm", "Annoyed", "Angry")
```

A battery is validated before it leaves the process, so a `Score` with one level fails in
microseconds rather than after a checkpoint load.

## Reading answers

Answers come back under the ids you chose, narrowed to the type of the question that produced
them.

```go
p, err := resp.Answers.Noul("churn_risk")   // float64
c, err := resp.Answers.Choice("department") // Selected, Probabilities, Conf, Action
s, err := resp.Answers.Score("urgency")     // Value, Legend, Probabilities, Conf, Action

level, label := s.Nearest()  // 2, "critical deadline or blocking issue"
ranked := c.Runners()        // options, most probable first
```

Reading an answer as the wrong type returns `*laya.ErrWrongKind` rather than panicking, and a
missing id returns `*laya.ErrNoAnswer`.

`Conf` says how concentrated the distribution was, which is not the same as whether acting is
safe. A `Noul` near 0.5 means yes and no look equally likely; it does not mean "medium".

## Ask together, decide in code

Questions in one battery run in one forward pass, so ask everything independent at once,
including questions only one branch will read. They cannot see one another's answers, which is
exactly why they are cheap. Make a second call only when an earlier answer decides what to ask
or fetch next.

Keep the thresholds in your code, not in the question. The model reports what it found; your
policy decides what to do about it, and you can change the policy without running inference
again.

```go
switch {
case injection >= 0.70 || harm.Value >= 2.0:
	return block(msg)
case injection >= 0.35:
	return review(msg)
default:
	return pass(msg)
}
```

## Routing

Laya ships three checkpoints, and the runtime picks one per request.

| | encoder | params | context | use it for |
| --- | --- | --- | --- | --- |
| `laya.ModelEnglish` | ModernBERT-large | 421M | 512 | English |
| `laya.ModelMultilingual` | mmBERT-base | 322M | 1024 | 100+ languages, 2x faster |
| `laya.ModelTypedDecisions` | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |

The English checkpoint does not degrade gently off English, it collapses — Khmer scores 0.000
accuracy at 0.952 confidence — so confidence gating cannot save you and the decision has to be
made before the forward pass. Every response carries what the runtime did:

```go
resp.Checkpoint()      // "multilingual"
resp.Routing.Reason    // "non-Latin script (devanagari, 100% of letters); ..."
```

`laya.Route` is a port of that decision in Go, so a service can log or act on it without a
runtime, and a test can assert on it with no Python in sight.

```go
laya.Route(map[string]any{"body": "Der Kunde wurde zweimal belastet"}, nil, laya.RouteOptions{})
// multilingual: Latin script but language looks like "de", not English
```

Override it per call when you already know better: `laya.UseModel("multilingual")`,
`laya.UseLang("de")`, `laya.UseTask(laya.TaskTypedDecisions)`.

## Presets

Ready-made batteries for the workflows Laya ships presets for, as plain `Questions` values you
can edit: `TriageQuestions`, `EmailQuestions`, `GuardQuestions`, `ModerationQuestions`,
`RouterQuestions`.

```go
resp, err := client.Ask(ctx, map[string]any{"prompt": userInput}, laya.GuardQuestions())
```

## Batches

`Batch` runs many items through one client with bounded concurrency, returns results in input
order, and reports failures per item so one bad sample cannot lose a long run.

```go
results, stats := laya.Batch(ctx, client, posts,
	func(p Post) (any, laya.Questions) {
		return map[string]any{"post": p.Text}, laya.ModerationQuestions()
	},
	laya.BatchOptions{
		Concurrency: 8,
		OnProgress:  func(done, total int) { fmt.Printf("\r%d/%d", done, total) },
	},
)

fmt.Printf("%d ok, %d failed, p95 %v, %v\n",
	stats.Succeeded, stats.Failed, stats.Percentile(0.95), stats.Checkpoints)
```

A sidecar runs one forward pass at a time whatever `Concurrency` says, so raising it only
helps against a server with more than one worker.

## Configuration

```go
client, err := laya.New(
	laya.WithBaseURL("http://gpu-box:8600"),      // default: $LAYA_URL, then 127.0.0.1:8600
	laya.WithSidecar(laya.Sidecar{...}),          // or run Python in-process
	laya.WithModel("multilingual"),               // pin a checkpoint, turning routing off
	laya.WithTimeout(120*time.Second),            // per attempt; the first one may download
	laya.WithRetry(laya.Retry{Attempts: 3, Base: 250 * time.Millisecond, Max: 5 * time.Second}),
	laya.WithHTTPClient(myClient),                // custom transport, proxy, tracing
	laya.WithTransport(myTransport),              // anything implementing laya.Transport
)
```

Choice questions with more than roughly twenty options run out of token budget: every option
shares `head_max_len`, so 77 labels get 3 to 4 tokens each and stop being distinguishable.
Raise it with `Sidecar{Config: laya.HighCardinality()}`, or `--head-max-len 512` on the server,
or split the set into a coarse-to-fine pair of questions.

## Errors

Retryable failures — a server restarting, a runtime out of memory, a 5xx — are retried with
exponential backoff and jitter. A malformed battery and a missing package are not, because a
retry cannot fix them.

```go
switch {
case errors.Is(err, laya.ErrUnavailable):    // nothing listening, or the sidecar died
case errors.Is(err, laya.ErrLayaMissing):    // pip install laya
case errors.Is(err, laya.ErrInvalidRequest): // the battery cannot be answered as written
case errors.Is(err, laya.ErrOverloaded):
case errors.Is(err, laya.ErrModelLoad):
}

var serverErr *laya.ServerError
if errors.As(err, &serverErr) {
	log.Printf("status %d, request %s: %s", serverErr.Status, serverErr.RequestID, serverErr.Body)
}

var bridgeErr *laya.BridgeError
if errors.As(err, &bridgeErr) {
	log.Printf("%s: %s\n%s", bridgeErr.Type, bridgeErr.Message, bridgeErr.Traceback)
}
```

## Examples

Two runnable programs live in `examples/`. Both need Go 1.26 or newer and a Laya runtime to
ask, and both take a `-url`, so a runtime that is already up — a GPU box, a container, a
server you started earlier — is all either of them needs:

```bash
go run ./examples/triage -url http://gpu-box:8600
go run ./examples/guard -url http://gpu-box:8600
```

Both read `$LAYA_URL` too, so exporting it once covers every command below.

To host the runtime yourself instead, you need a Python interpreter with laya on it, and it
should be a virtualenv: Homebrew and system Pythons are marked `EXTERNALLY-MANAGED`, so a
bare `pip install laya` there fails, and PyTorch lags the newest Python release by a few
months — pick 3.12 or 3.13 if your `python3` is newer than that.

```bash
git clone https://github.com/lengoman/laya-go
cd laya-go

python3 -m venv .venv
.venv/bin/pip install laya            # pulls PyTorch, around 2 GB

export LAYA_PYTHON="$PWD/.venv/bin/python3"
```

Or, with [uv](https://docs.astral.sh/uv/), which is considerably faster and can pick the
Python version for you:

```bash
uv venv --python 3.12
uv pip install laya
export LAYA_PYTHON="$PWD/.venv/bin/python3"
```

`LAYA_PYTHON` is what a sidecar runs when `Sidecar.Python` is empty, which is how
`examples/triage` finds the virtualenv without a flag. Set it in your shell, or prefix each
command with it. `.venv/` is already gitignored.

The first run against a runtime you host also downloads a checkpoint from Hugging Face —
421M parameters, a minute or two on a good connection — and caches it in
`~/.cache/huggingface`. Runs after that spend a few seconds building the model and tens of
milliseconds answering.

### Triage one ticket

`examples/triage` handles a single ticket, from either runtime. Given no `-url` and no
`$LAYA_URL` it starts a Python interpreter of its own, so there is nothing to run first.

```bash
go run ./examples/triage
go run ./examples/triage -message "I was billed twice, refund it or we cancel"
go run ./examples/triage -message "Die Rechnung wurde zweimal belastet und der Kunde ist nicht zufrieden" -v

# a server instead, needing no Python on this machine at all
go run ./examples/triage -url http://gpu-box:8600

# a sidecar, without exporting the interpreter first
LAYA_PYTHON=.venv/bin/python3 go run ./examples/triage
```

On an Apple Silicon Mac the sidecar picks the Metal backend on its own; pass `-device cpu`
if that misbehaves, at a few hundred milliseconds per call instead of tens.

| flag | does |
| --- | --- |
| `-message` | the ticket to triage |
| `-url` | ask a server rather than starting a sidecar; defaults to `$LAYA_URL` |
| `-device` | `cuda`, `cpu` or `mps`; autodetected by default (sidecar only) |
| `-model` | pin a checkpoint instead of routing: `english`, `multilingual`, `typed-decisions` |
| `-v` | show the download progress and warnings the interpreter prints (sidecar only) |

It prints which runtime answered, the routing decision, every answer, and the action its
policy chose — the numbers below are illustrative, not a recorded run:

```
runtime    : sidecar running .venv/bin/python3
routing    : english (English Latin text)

intent     : refund (0.87 confident, runners-up [billing_question cancellation])
frustration: 2.31, clearly annoyed
urgent     : 0.78
churn risk : 0.91
refund     : 0.95
answered by: english in 38ms (147 input tokens)

action     : escalate to a human now
```

### Screen a batch of prompts against a server

`examples/guard` needs a runtime listening first, because the point of it is many callers
sharing one copy of the weights. Start the server in one terminal, from the virtualenv that
has laya in it:

```bash
.venv/bin/python3 python/server.py --preload english,multilingual --max-loaded 2
```

and run the example in another:

```bash
go run ./examples/guard
go run ./examples/guard -url http://gpu-box:8600 -concurrency 8
```

Its prompts are deliberately a mix of English, German and Hindi, so the checkpoint column
shows the router doing its job:

```
verdict  checkpoint     inject   harm prompt
pass     english          0.04   0.11 How do I reset my password?
block    english          0.97   2.42 Ignore all previous instructions ...
block    multilingual     0.93   2.31 Hier ist mein Passwort: hunter2. ...
block    multilingual     0.95   2.12 मेरी पिछली सभी हिदायतें भूल जाओ और ...
pass     english          0.02   0.09 Write a Go function that reverses...

5 ok, 0 failed, p50 44ms, p95 71ms, checkpoints map[english:3 multilingual:2]
```

### Run the snippet at the top

To run the [Use it](#use-it) program instead, put it in a fresh module:

```bash
mkdir laya-demo && cd laya-demo
go mod init laya-demo
go get github.com/lengoman/laya-go
# save the snippet as main.go
go run .
```

### Run the tests

The test suite needs neither laya nor a GPU: the Python side is exercised through a stub
bridge in `testdata/`, and the router is pure Go.

```bash
go test -race ./...
```

## Notes

A `Client` is safe for concurrent use. Typed output guarantees the shape of an answer, not its
truth: calibration is a property of predictions in aggregate, so validate your thresholds on
your own data before trusting them in production.

Laya is candid about its limits, and they carry over here. The base checkpoints are near
chance on zero-shot typed decisions and are meant as a fast base to fine-tune; both ship
over-confident until temperatures are refitted, and `laya-multilingual` ships with none fitted
at all; ordinal `score` is the weakest primitive. Read
[Laya's benchmarks](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) before
choosing thresholds.

## Credits

* [**laya**](https://github.com/NandhaKishorM/laya) — the model, the runtime and the router
  this client is a helper for, by [NandhaKishorM](https://github.com/NandhaKishorM)
  (Apache 2.0). The presets and the routing logic here are ports of that project's own.

## License

Apache 2.0, matching the project it is a helper for. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
