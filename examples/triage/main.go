// Command triage rates one support ticket and decides what to do with it.
//
// It takes its answers from either runtime. Against a server there is nothing
// to install here, not even Python:
//
//	go run ./examples/triage -url http://gpu-box:8600
//
// With no -url and no $LAYA_URL it starts a Python interpreter of its own
// instead, so there is no server to deploy. Give that one a virtualenv with
// laya in it, because Homebrew and system Pythons refuse to be installed into:
//
//	python3 -m venv .venv && .venv/bin/pip install laya
//	export LAYA_PYTHON="$PWD/.venv/bin/python3"
//
//	go run ./examples/triage -message "I was billed twice, refund it or we cancel"
//
// The first run against a cold runtime downloads a checkpoint, which takes
// minutes. Every run after that is a few seconds, almost all of it building
// the model; the decision itself is tens of milliseconds.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	laya "github.com/lengoman/laya-go"
)

type config struct {
	message string
	url     string
	device  string
	model   string
	minConf float64
	verbose bool
}

func main() {
	var cfg config
	flag.StringVar(&cfg.message, "message",
		"I have been charged twice for March. Refund it today or we cancel.",
		"the ticket to triage")
	flag.StringVar(&cfg.url, "url", "",
		"a Laya server to ask; default is $LAYA_URL, then a Python sidecar this process owns")
	flag.StringVar(&cfg.device, "device", "", "cuda, cpu or mps; autodetected by default (sidecar only)")
	flag.StringVar(&cfg.model, "model", "", "pin a checkpoint instead of routing: english, multilingual, typed-decisions")
	flag.Float64Var(&cfg.minConf, "min-confidence", 0, "minimum calibrated confidence threshold for gating / abstention")
	flag.BoolVar(&cfg.verbose, "v", false, "show what the interpreter prints while it loads (sidecar only)")
	flag.Parse()

	if err := run(cfg); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		hint(cfg, err)
		os.Exit(1)
	}
}

func run(cfg config) error {
	options, runtime := runtimeFor(cfg)
	if cfg.model != "" {
		options = append(options, laya.WithModel(cfg.model))
	}

	client, err := laya.New(options...)
	if err != nil {
		return err
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	state := map[string]any{"message": cfg.message}

	// Where the answer will come from, and which checkpoint will give it,
	// before anything is loaded.
	fmt.Printf("runtime    : %s\n", runtime)
	fmt.Printf("routing    : %s\n\n", laya.Route(state, nil, laya.RouteOptions{Model: cfg.model}))

	var callOpts []laya.CallOption
	if cfg.minConf > 0 {
		callOpts = append(callOpts, laya.UseMinConfidence(cfg.minConf))
	}

	resp, err := client.Ask(ctx, state, laya.TriageQuestions(), callOpts...)
	if err != nil {
		return err
	}

	intent, err := resp.Answers.Choice("intent")
	if err != nil {
		return err
	}
	frustration, err := resp.Answers.Score("frustration")
	if err != nil {
		return err
	}
	churn := resp.Answers.MustNoul("churn_risk")
	urgent := resp.Answers.MustNoul("is_urgent")

	_, level := frustration.Nearest()
	fmt.Printf("intent     : %s (%.2f calibrated conf, %.2f entropy, runners-up %v)\n",
		intent.Selected, intent.AnswerConfidence(), intent.Conf, intent.Runners()[1:3])
	if intent.LowConfidence {
		fmt.Printf("             [ABSTAINED: low confidence, threshold=%.2f]\n", cfg.minConf)
	}
	fmt.Printf("frustration: %.2f, %s (conf: %.2f)\n", frustration.Value, level, frustration.AnswerConfidence())
	fmt.Printf("urgent     : %.2f (conf: %.2f)\n", urgent, resp.Answers["is_urgent"].AnswerConfidence())
	fmt.Printf("churn risk : %.2f (conf: %.2f)\n", churn, resp.Answers["churn_risk"].AnswerConfidence())
	fmt.Printf("refund     : %.2f (conf: %.2f)\n", resp.Answers.MustNoul("refund_requested"), resp.Answers["refund_requested"].AnswerConfidence())
	fmt.Printf("answered by: %s in %v (%d input tokens)\n\n",
		resp.Checkpoint(), resp.Latency.Round(time.Millisecond), resp.Usage.InputTokens)

	// The thresholds live here, not in the questions, so the policy can change
	// without running inference again.
	switch {
	case intent.LowConfidence:
		fmt.Println("action     : route to human queue due to low model confidence")
	case churn >= 0.7 || frustration.Value >= 2.5:
		fmt.Println("action     : escalate to a human now")
	case intent.Selected == "refund" && intent.AnswerConfidence() >= 0.85:
		fmt.Println("action     : open a refund case automatically")
	case intent.AnswerConfidence() < 0.6:
		fmt.Printf("action     : queue for human triage (only %.2f calibrated confidence)\n", intent.AnswerConfidence())
	default:
		fmt.Printf("action     : route to the %s queue\n", intent.Selected)
	}
	return nil
}

// runtimeFor chooses between the two transports and describes the one it
// picked, because which runtime answered explains most of what a run does.
func runtimeFor(cfg config) ([]laya.Option, string) {
	if url := serverURL(cfg); url != "" {
		if cfg.device != "" || cfg.verbose {
			fmt.Fprintln(os.Stderr, "note: -device and -v configure a sidecar, and this run uses a server")
		}
		return []laya.Option{laya.WithBaseURL(url)}, "server at " + url
	}

	sidecar := laya.Sidecar{Device: cfg.device}
	if cfg.verbose {
		sidecar.Stderr = os.Stderr
	}
	python := os.Getenv(laya.PythonEnv)
	if python == "" {
		python = "python3"
	}
	return []laya.Option{laya.WithSidecar(sidecar)}, "sidecar running " + python
}

// serverURL is the server to ask, or empty to run a sidecar instead.
func serverURL(cfg config) string {
	url := cfg.url
	if url == "" {
		url = os.Getenv(laya.ServerURLEnv)
	}
	return strings.TrimRight(url, "/")
}

func hint(cfg config, err error) {
	switch {
	case errors.Is(err, laya.ErrLayaMissing):
		fmt.Fprint(os.Stderr, "\nGive the sidecar an interpreter with laya on it:\n"+
			"    python3 -m venv .venv && .venv/bin/pip install laya\n"+
			"    export LAYA_PYTHON=\"$PWD/.venv/bin/python3\"\n"+
			"\nOr ask a runtime that is already up, and install nothing here:\n"+
			"    go run ./examples/triage -url http://host:8600\n")
	case errors.Is(err, laya.ErrUnavailable) && serverURL(cfg) != "":
		fmt.Fprint(os.Stderr, "\nNothing answered at "+serverURL(cfg)+". Start a server:\n"+
			"    .venv/bin/python3 python/server.py --preload english,multilingual\n"+
			"\nor unset -url and $LAYA_URL to run the model in a sidecar here.\n")
	case errors.Is(err, laya.ErrUnavailable):
		fmt.Fprint(os.Stderr, "\nThe sidecar stopped answering. -v shows what its interpreter printed.\n")
	}
}
