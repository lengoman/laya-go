// Package laya is a Go client for Laya, the multilingual non-autoregressive
// System 1 decision engine.
//
// Laya evaluates typed questions over a state in a single forward pass and
// returns calibrated probabilities rather than generated text: nothing to
// parse, nothing to hallucinate, and about 33 ms for one question or 7 ms each
// when they are batched. Your code keeps the workflow and the policy; the model
// supplies the semantic call that ordinary code cannot make.
//
// Laya itself is a Python package with Apache 2.0 weights that you host. This
// client is the Go half: the question and answer types, the router, the
// presets and the batching, talking to a Laya runtime over one of two
// transports.
//
// # The three decisions
//
// A request sends one state and a battery of questions about it. Each question
// is one of three types, chosen by what its answer means:
//
//   - [Noul] asks whether a condition holds, and returns the probability of
//     yes. Use one per label when several labels can apply at once.
//   - [Choice] picks one option from a set, and returns the whole distribution.
//   - [Score] rates the state against ordered levels, and can land between them.
//
// # A first request
//
//	client, err := laya.New(laya.WithSidecar(laya.Sidecar{Preload: []string{laya.ModelEnglish}}))
//	if err != nil {
//		return err
//	}
//	defer client.Close()
//
//	resp, err := client.Ask(ctx, map[string]any{
//		"subject": "Duplicate charge on invoice #4411",
//		"body":    "We were billed twice for March. Refund it today or we cancel.",
//	}, laya.Questions{
//		"department": laya.OneOf("Which department should handle this?", map[string]string{
//			"billing":   "invoices, payments, refunds",
//			"technical": "bugs, outages, system errors",
//			"other":     "everything else",
//		}),
//		"churn_risk": laya.Holds("Does the user threaten to cancel or leave?"),
//	})
//	if err != nil {
//		return err
//	}
//
//	department, _ := resp.Answers.Choice("department")
//	fmt.Println(department.Selected, department.Conf, resp.Checkpoint())
//
// # Two transports
//
// [HTTPTransport], the default, talks to a Laya server: one process holds the
// weights and every caller shares them. [Sidecar] runs Laya in a Python
// interpreter this process starts and owns, which needs no service to deploy.
// Both are reference implementations in python/ in this repository, and any
// service exposing the same endpoints works.
//
// Either way, loading is what costs seconds and answering does not, so preload
// the checkpoints you serve and keep them resident; see [Client.Warm].
//
// # Routing
//
// Laya ships three checkpoints. The English one collapses outside English
// while staying confident about it, so the choice has to be made before the
// forward pass rather than from the model's own confidence. The runtime routes
// each request by script and language, and reports what it did in
// [Response.Routing].
//
// [Route] is a port of that decision in Go, so a service can log or act on it
// without a runtime, and a test can assert on it with no Python in sight.
//
// # Ask together, decide in code
//
// Questions in one battery run in one forward pass, so ask everything
// independent at once, including questions only one branch will read. They
// cannot see one another's answers, which is exactly why they are cheap. Make a
// second call only when an answer decides what to ask or fetch next.
//
// Keep the thresholds in your own code. The model reports what it found; your
// policy decides what to do about it, and you can change that policy without
// running inference again.
//
// # Errors
//
// Failed calls return a [ServerError] or a [BridgeError], which [errors.Is]
// matches against [ErrInvalidRequest], [ErrUnavailable], [ErrOverloaded],
// [ErrModelLoad] and [ErrLayaMissing]. Retryable failures are retried with
// exponential backoff; see [WithRetry].
//
// # Honest limits
//
// Typed output guarantees the shape of an answer, not its truth. Laya's base
// checkpoints are near chance on zero-shot typed decisions and are meant as a
// fast base to fine-tune, both checkpoints ship over-confident until
// temperatures are refitted, and choice questions with more than roughly twenty
// options run out of token budget unless you raise it; see [HighCardinality].
// Validate your thresholds on your own data before trusting them in production.
package laya
