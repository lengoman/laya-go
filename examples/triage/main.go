// Command triage rates one support ticket and decides what to do with it.
//
// It runs Laya in a Python interpreter it owns, so there is no server to
// deploy. Give it a virtualenv with laya in it, because Homebrew and system
// Pythons refuse to be installed into:
//
//	python3 -m venv .venv && .venv/bin/pip install laya
//	export LAYA_PYTHON="$PWD/.venv/bin/python3"
//
//	go run ./examples/triage -message "I was billed twice, refund it or we cancel"
//
// The first run downloads a checkpoint, which takes minutes. Every run after
// that is a few seconds, almost all of it building the model; the decision
// itself is tens of milliseconds.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	laya "github.com/lengoman/laya-go"
)

func main() {
	message := flag.String("message", "I have been charged twice for March. Refund it today or we cancel.",
		"the ticket to triage")
	device := flag.String("device", "", "cuda, cpu or mps; autodetected by default")
	model := flag.String("model", "", "pin a checkpoint instead of routing: english, multilingual, typed-decisions")
	verbose := flag.Bool("v", false, "show what the interpreter prints while it loads")
	flag.Parse()

	if err := run(*message, *device, *model, *verbose); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		if errors.Is(err, laya.ErrLayaMissing) {
			fmt.Fprint(os.Stderr, "\nInstall the runtime first:\n"+
				"    python3 -m venv .venv && .venv/bin/pip install laya\n"+
				"    export LAYA_PYTHON=\"$PWD/.venv/bin/python3\"\n")
		}
		os.Exit(1)
	}
}

func run(message, device, model string, verbose bool) error {
	sidecar := laya.Sidecar{Device: device}
	if verbose {
		sidecar.Stderr = os.Stderr
	}

	options := []laya.Option{laya.WithSidecar(sidecar)}
	if model != "" {
		options = append(options, laya.WithModel(model))
	}

	client, err := laya.New(options...)
	if err != nil {
		return err
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	state := map[string]any{"message": message}

	// Which checkpoint will answer, and why, before anything is loaded.
	fmt.Printf("routing    : %s\n\n", laya.Route(state, nil, laya.RouteOptions{Model: model}))

	resp, err := client.Ask(ctx, state, laya.TriageQuestions())
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
	fmt.Printf("intent     : %s (%.2f confident, runners-up %v)\n", intent.Selected, intent.Conf, intent.Runners()[1:3])
	fmt.Printf("frustration: %.2f, %s\n", frustration.Value, level)
	fmt.Printf("urgent     : %.2f\n", urgent)
	fmt.Printf("churn risk : %.2f\n", churn)
	fmt.Printf("refund     : %.2f\n", resp.Answers.MustNoul("refund_requested"))
	fmt.Printf("answered by: %s in %v (%d input tokens)\n\n",
		resp.Checkpoint(), resp.Latency.Round(time.Millisecond), resp.Usage.InputTokens)

	// The thresholds live here, not in the questions, so the policy can change
	// without running inference again.
	switch {
	case churn >= 0.7 || frustration.Value >= 2.5:
		fmt.Println("action     : escalate to a human now")
	case intent.Selected == "refund" && intent.Conf >= 0.85:
		fmt.Println("action     : open a refund case automatically")
	case intent.Conf < 0.6:
		fmt.Printf("action     : queue for human triage (only %.2f confident)\n", intent.Conf)
	default:
		fmt.Printf("action     : route to the %s queue\n", intent.Selected)
	}
	return nil
}
