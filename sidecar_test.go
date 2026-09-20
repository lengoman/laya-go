package laya

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// python3 skips the test when there is no interpreter to run a bridge in.
func python3(t *testing.T) string {
	t.Helper()
	path, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not installed")
	}
	return path
}

// hasLaya reports whether the interpreter can import laya.
func hasLaya(python string) bool {
	return exec.Command(python, "-c", "import laya").Run() == nil
}

// stubSidecar is a client wired to the stub bridge in testdata.
func stubSidecar(t *testing.T, opts Sidecar) (*Client, *SidecarTransport) {
	t.Helper()
	opts.Python = python3(t)
	opts.ScriptPath = "testdata/stub_bridge.py"
	if opts.StartupTimeout == 0 {
		opts.StartupTimeout = 30 * time.Second
	}

	transport := NewSidecar(opts)
	client, err := New(WithTransport(transport), WithRetry(Retry{Attempts: 2, Base: time.Millisecond}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client, transport
}

func TestSidecarRoundTrip(t *testing.T) {
	client, _ := stubSidecar(t, Sidecar{})

	resp, err := client.Ask(context.Background(), "I was billed twice", battery())
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}

	if resp.Model != "laya-stub" {
		t.Errorf("model = %q", resp.Model)
	}
	if resp.Checkpoint() != ModelEnglish {
		t.Errorf("checkpoint = %q, want english", resp.Checkpoint())
	}
	department, err := resp.Answers.Choice("department")
	if err != nil {
		t.Fatalf("Choice: %v", err)
	}
	if department.Selected == "" {
		t.Error("no option was selected")
	}
	if resp.Usage.InputTokens != 42 {
		t.Errorf("input tokens = %d, want 42", resp.Usage.InputTokens)
	}
}

func TestSidecarReusesOneInterpreter(t *testing.T) {
	client, transport := stubSidecar(t, Sidecar{})

	for i := range 3 {
		if _, err := client.Ask(context.Background(), "hello", battery()); err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
	}

	transport.mu.Lock()
	proc := transport.proc
	transport.mu.Unlock()
	if proc == nil || !proc.alive() {
		t.Fatal("the interpreter did not survive three calls")
	}
}

func TestSidecarPassesItsOptionsToPython(t *testing.T) {
	_, transport := stubSidecar(t, Sidecar{
		Device:    "cpu",
		Preload:   []string{ModelEnglish},
		MaxLoaded: 2,
		Config:    HighCardinality(),
	})

	// The stub echoes what it was started with, which is the only way to see
	// that the Go options reached Python in the shape it expects.
	raw, err := transport.roundTrip(context.Background(), sidecarRequest{
		Op:      "predict",
		Request: &Request{State: "hello", Questions: battery(), Lang: "de"},
	})
	if err != nil {
		t.Fatalf("roundTrip: %v", err)
	}

	var echoed struct {
		Echo struct {
			Options struct {
				Device    string         `json:"device"`
				Preload   []string       `json:"preload"`
				MaxLoaded int            `json:"max_loaded"`
				Config    map[string]any `json:"config"`
			} `json:"options"`
			Lang string `json:"lang"`
		} `json:"echo"`
	}
	if err := json.Unmarshal(raw, &echoed); err != nil {
		t.Fatalf("decoding echo: %v", err)
	}

	options := echoed.Echo.Options
	if options.Device != "cpu" || options.MaxLoaded != 2 {
		t.Errorf("options = %#v", options)
	}
	if len(options.Preload) != 1 || options.Preload[0] != ModelEnglish {
		t.Errorf("preload = %v, want [english]", options.Preload)
	}
	if options.Config["head_max_len"] != float64(512) {
		t.Errorf("config = %#v, want head_max_len 512", options.Config)
	}
	if echoed.Echo.Lang != "de" {
		t.Errorf("lang = %q, want de", echoed.Echo.Lang)
	}
}

func TestSidecarReportsPythonExceptions(t *testing.T) {
	client, _ := stubSidecar(t, Sidecar{})

	_, err := client.Ask(context.Background(), "boom", battery())

	var bridgeErr *BridgeError
	if !errors.As(err, &bridgeErr) {
		t.Fatalf("err = %v, want *BridgeError", err)
	}
	if bridgeErr.Type != "ValueError" || bridgeErr.Message != "stub failure" {
		t.Errorf("error = %#v", bridgeErr)
	}
	if !strings.Contains(bridgeErr.Traceback, "ValueError") {
		t.Errorf("traceback = %q, want the Python traceback kept", bridgeErr.Traceback)
	}
	// A malformed request is not something a second try can fix.
	if !errors.Is(err, ErrInvalidRequest) {
		t.Errorf("err = %v, want it to match ErrInvalidRequest", err)
	}
}

func TestSidecarTreatsMemoryFailuresAsRetryable(t *testing.T) {
	client, _ := stubSidecar(t, Sidecar{})

	_, err := client.Ask(context.Background(), "oom", battery())

	if !errors.Is(err, ErrOverloaded) {
		t.Errorf("err = %v, want it to match ErrOverloaded", err)
	}
	var bridgeErr *BridgeError
	if errors.As(err, &bridgeErr) && !bridgeErr.Retryable() {
		t.Error("an out-of-memory failure was marked unretryable, though Laya falls back to CPU")
	}
}

func TestSidecarRestartsAfterTheInterpreterDies(t *testing.T) {
	client, _ := stubSidecar(t, Sidecar{})

	if _, err := client.Ask(context.Background(), "hello", battery()); err != nil {
		t.Fatalf("first call: %v", err)
	}
	if _, err := client.Ask(context.Background(), "crash", battery()); err == nil {
		t.Fatal("the crash was not reported")
	}

	// The next call must bring the interpreter back rather than fail forever.
	if _, err := client.Ask(context.Background(), "hello", battery()); err != nil {
		t.Fatalf("call after a crash: %v", err)
	}
}

func TestSidecarRouteAndWarm(t *testing.T) {
	client, transport := stubSidecar(t, Sidecar{})

	decision, err := client.Route(context.Background(), "hello", battery())
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if decision.Reason != "stub" {
		t.Errorf("reason = %q, want the runtime's own answer", decision.Reason)
	}

	if err := client.Warm(context.Background(), "multi"); err != nil {
		t.Fatalf("Warm: %v", err)
	}
	loaded, err := transport.Loaded(context.Background())
	if err != nil {
		t.Fatalf("Loaded: %v", err)
	}
	if len(loaded) != 1 || loaded[0] != ModelMultilingual {
		t.Errorf("loaded = %v, want [multilingual]", loaded)
	}
}

func TestSidecarForwardsStderr(t *testing.T) {
	var log bytes.Buffer
	client, _ := stubSidecar(t, Sidecar{Stderr: &log})

	// The stub rebinds print() to stderr, as the real bridge does, so anything
	// laya prints lands here instead of corrupting the protocol.
	if _, err := client.Ask(context.Background(), "hello", battery()); err != nil {
		t.Fatalf("Ask: %v", err)
	}
}

func TestSidecarCloseStopsTheInterpreter(t *testing.T) {
	client, transport := stubSidecar(t, Sidecar{})

	if err := transport.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := client.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if _, err := client.Ask(context.Background(), "hello", battery()); !errors.Is(err, ErrClosed) {
		t.Errorf("err = %v, want ErrClosed after Close", err)
	}
}

func TestSidecarWithoutAnInterpreter(t *testing.T) {
	client, err := New(
		WithSidecar(Sidecar{Python: "python-that-does-not-exist"}),
		WithRetry(Retry{Attempts: 1}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = client.Close() }()

	_, err = client.Ask(context.Background(), "hello", battery())

	if !errors.Is(err, ErrPythonMissing) {
		t.Errorf("err = %v, want ErrPythonMissing", err)
	}
}

func TestEmbeddedBridgeReportsAMissingLayaPackage(t *testing.T) {
	python := python3(t)
	if hasLaya(python) {
		t.Skip("laya is installed, so the missing-package path cannot be exercised here")
	}

	client, err := New(
		WithSidecar(Sidecar{Python: python, StartupTimeout: 30 * time.Second}),
		WithRetry(Retry{Attempts: 1}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = client.Close() }()

	_, err = client.Ask(context.Background(), "hello", battery())

	if !errors.Is(err, ErrLayaMissing) {
		t.Fatalf("err = %v, want ErrLayaMissing", err)
	}
	if !strings.Contains(err.Error(), "pip install laya") {
		t.Errorf("err = %v, want it to say how to fix the problem", err)
	}
}

func TestEmbeddedBridgeIsValidPython(t *testing.T) {
	python := python3(t)

	cmd := exec.Command(python, "-c", "import ast, sys; ast.parse(sys.stdin.read())")
	cmd.Stdin = strings.NewReader(BridgeSource())
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("the embedded bridge does not parse: %v\n%s", err, out)
	}
}
