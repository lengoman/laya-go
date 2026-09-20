package laya

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

//go:embed python/bridge.py
var bridgeSource string

// BridgeSource returns the Python bridge this package embeds, so you can write
// it somewhere and run it yourself.
func BridgeSource() string { return bridgeSource }

// PythonEnv is the environment variable naming the interpreter a [Sidecar]
// runs.
const PythonEnv = "LAYA_PYTHON"

// Sidecar configures a Laya runtime inside a Python interpreter this process
// starts and owns, for when a shared server is more infrastructure than the
// job needs. The interpreter is launched on the first request and lives until
// the client is closed, so the weights are built once.
//
// It needs `pip install laya` on the interpreter it runs, and enough memory to
// hold the checkpoints it loads: the three together are about 1.16B parameters.
type Sidecar struct {
	// Python is the interpreter to run. Empty means $LAYA_PYTHON, then python3.
	Python string
	// ScriptPath runs a bridge from disk instead of the embedded one, which is
	// useful while changing it. Empty means the embedded [BridgeSource].
	ScriptPath string
	// Device is "cuda", "cpu" or "mps". Empty lets Laya choose, preferring the
	// accelerator it finds.
	Device string
	// Preload names the checkpoints to build at startup. Empty builds nothing
	// until a request needs it, which makes that request pay seconds for the
	// build.
	Preload []string
	// MaxLoaded caps how many checkpoints stay resident, least recently used
	// evicted first. Zero means 1, which is also Laya's default; raise it, or
	// use Preload, when traffic alternates languages.
	MaxLoaded int
	// Default is the checkpoint used for a state with no letters in it. Empty
	// means [ModelEnglish].
	Default string
	// AutoTaskDetection opts in to routing a battery whose ids match one of the
	// four typed-decisions workflows to that checkpoint.
	AutoTaskDetection bool
	// StandaloneRepos downloads each checkpoint from its own repo rather than
	// the bundle.
	StandaloneRepos bool
	// Config overrides entries of each checkpoint's config once it is loaded;
	// see [HighCardinality].
	Config Config
	// Token is a Hugging Face token. Empty means $HF_TOKEN.
	Token string
	// Env replaces the interpreter's environment. Empty inherits this process's.
	Env []string
	// Dir is the interpreter's working directory. Empty inherits this process's.
	Dir string
	// StartupTimeout bounds the wait for the bridge to report itself ready,
	// which on a cold machine includes downloading a checkpoint. Zero means
	// ten minutes.
	StartupTimeout time.Duration
	// Stderr receives the interpreter's stderr: progress bars, and Laya's own
	// warnings about falling back to CPU. Nil discards it, though the last
	// lines are still quoted back in any error.
	Stderr io.Writer
}

// SidecarTransport runs requests through a Python interpreter this process
// owns. Build one with [NewSidecar], or reach it through [WithSidecar].
//
// It is safe for concurrent use, and serialises requests: one interpreter runs
// one forward pass at a time whatever the caller does, so queueing here rather
// than in Python keeps the error reporting honest.
type SidecarTransport struct {
	opts Sidecar

	mu     sync.Mutex
	proc   *bridgeProcess
	closed bool
}

var (
	_ Transport    = (*SidecarTransport)(nil)
	_ RouteQuerier = (*SidecarTransport)(nil)
	_ Warmer       = (*SidecarTransport)(nil)
	_ io.Closer    = (*SidecarTransport)(nil)
)

// NewSidecar builds a transport that runs Laya in its own Python interpreter.
// Nothing starts until the first request, or an explicit [SidecarTransport.Start].
func NewSidecar(opts Sidecar) *SidecarTransport { return &SidecarTransport{opts: opts} }

// Predict implements [Transport].
func (s *SidecarTransport) Predict(ctx context.Context, req *Request) (*Response, error) {
	raw, err := s.roundTrip(ctx, sidecarRequest{Op: "predict", Request: req})
	if err != nil {
		return nil, err
	}
	var out Response
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("laya: decoding bridge result: %w", err)
	}
	return &out, nil
}

// Route implements [RouteQuerier], reporting the runtime's own decision.
func (s *SidecarTransport) Route(ctx context.Context, req *Request) (*RouteDecision, error) {
	raw, err := s.roundTrip(ctx, sidecarRequest{Op: "route", Request: req})
	if err != nil {
		return nil, err
	}
	var out RouteDecision
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("laya: decoding bridge result: %w", err)
	}
	return &out, nil
}

// Warm implements [Warmer], building checkpoints before a request needs them.
func (s *SidecarTransport) Warm(ctx context.Context, models ...string) error {
	_, err := s.roundTrip(ctx, sidecarRequest{Op: "warm", Models: models})
	return err
}

// Start launches the interpreter and waits for it to report itself ready,
// which is where a missing package or a cold download surfaces. Calling it is
// optional: the first request does the same thing.
func (s *SidecarTransport) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.ensure(ctx)
	return err
}

// Loaded reports which checkpoints are resident in the interpreter.
func (s *SidecarTransport) Loaded(ctx context.Context) ([]string, error) {
	raw, err := s.roundTrip(ctx, sidecarRequest{Op: "ping"})
	if err != nil {
		return nil, err
	}
	var out struct {
		Loaded []string `json:"loaded"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("laya: decoding bridge result: %w", err)
	}
	return out.Loaded, nil
}

// Close stops the interpreter. The transport cannot be used afterwards.
func (s *SidecarTransport) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	if s.proc == nil {
		return nil
	}
	err := s.proc.stop()
	s.proc = nil
	return err
}

// sidecarRequest is one line of the bridge protocol.
type sidecarRequest struct {
	Op string `json:"op"`
	*Request
	Models []string `json:"models,omitempty"`
}

// sidecarReply is one line back.
type sidecarReply struct {
	OK     bool            `json:"ok"`
	Event  string          `json:"event"`
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Type      string `json:"type"`
		Message   string `json:"message"`
		Traceback string `json:"traceback"`
	} `json:"error"`
}

// roundTrip sends one request and waits for its reply, restarting an
// interpreter that has died since the last call.
func (s *SidecarTransport) roundTrip(ctx context.Context, req sidecarRequest) (json.RawMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	proc, err := s.ensure(ctx)
	if err != nil {
		return nil, err
	}

	line, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("laya: encoding request: %w", err)
	}
	if _, err := proc.stdin.Write(append(line, '\n')); err != nil {
		s.discard()
		return nil, fmt.Errorf("%w: writing to bridge: %w", ErrUnavailable, err)
	}

	reply, err := proc.read(ctx)
	if err != nil {
		// A half-read reply leaves the stream out of step with the requests, so
		// the interpreter cannot be reused. The next call starts a fresh one.
		s.discard()
		return nil, err
	}
	if !reply.OK {
		return nil, reply.err()
	}
	return reply.Result, nil
}

// ensure returns a running interpreter, starting one when there is none.
func (s *SidecarTransport) ensure(ctx context.Context) (*bridgeProcess, error) {
	if s.closed {
		return nil, ErrClosed
	}
	if s.proc != nil && s.proc.alive() {
		return s.proc, nil
	}
	s.proc = nil

	proc, err := s.spawn(ctx)
	if err != nil {
		return nil, err
	}
	s.proc = proc
	return proc, nil
}

// discard drops a broken interpreter so the next call starts a new one.
func (s *SidecarTransport) discard() {
	if s.proc != nil {
		_ = s.proc.stop()
		s.proc = nil
	}
}

// spawn starts the interpreter and waits for its ready line.
func (s *SidecarTransport) spawn(ctx context.Context) (*bridgeProcess, error) {
	options, err := json.Marshal(map[string]any{
		"device":              s.opts.Device,
		"preload":             s.opts.Preload,
		"max_loaded":          max(s.opts.MaxLoaded, 1),
		"default":             s.opts.Default,
		"auto_task_detection": s.opts.AutoTaskDetection,
		"standalone_repos":    s.opts.StandaloneRepos,
		"config":              s.opts.Config,
		"token":               s.opts.Token,
	})
	if err != nil {
		return nil, fmt.Errorf("laya: encoding sidecar options: %w", err)
	}

	python := s.opts.Python
	if python == "" {
		python = envOr(PythonEnv, "python3")
	}
	args := []string{"-u", "-c", bridgeSource, string(options)}
	if s.opts.ScriptPath != "" {
		args = []string{"-u", s.opts.ScriptPath, string(options)}
	}

	// The interpreter outlives the call that started it, so it must not be tied
	// to that call's context.
	cmd := exec.Command(python, args...) // #nosec G204 -- the interpreter is the caller's choice
	cmd.Env = s.opts.Env
	cmd.Dir = s.opts.Dir

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("laya: sidecar stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("laya: sidecar stdout: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("laya: sidecar stderr: %w", err)
	}

	if err := cmd.Start(); err != nil {
		if errors.Is(err, exec.ErrNotFound) || os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: %s: %w", ErrPythonMissing, python, err)
		}
		return nil, fmt.Errorf("%w: starting %s: %w", ErrUnavailable, python, err)
	}

	proc := &bridgeProcess{
		cmd:    cmd,
		stdin:  stdin,
		lines:  make(chan []byte, 1),
		readAt: make(chan error, 1),
		exited: make(chan struct{}),
	}
	go proc.pump(stdout)
	go proc.drainStderr(stderr, s.opts.Stderr)
	go func() {
		proc.waitErr = cmd.Wait()
		close(proc.exited)
	}()

	timeout := s.opts.StartupTimeout
	if timeout == 0 {
		timeout = 10 * time.Minute
	}
	readyCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ready, err := proc.read(readyCtx)
	if err != nil {
		_ = proc.stop()
		return nil, err
	}
	if !ready.OK {
		_ = proc.stop()
		err := ready.err()
		var bridgeErr *BridgeError
		if errors.As(err, &bridgeErr) &&
			(bridgeErr.Type == "ModuleNotFoundError" || bridgeErr.Type == "ImportError") &&
			strings.Contains(bridgeErr.Message, "laya") {
			return nil, fmt.Errorf("%w (%s)", ErrLayaMissing, python)
		}
		return nil, err
	}
	return proc, nil
}

// err turns a failed reply into the matching Go error.
func (r sidecarReply) err() error {
	if r.Error == nil {
		return fmt.Errorf("laya: bridge reported failure with no error")
	}
	return &BridgeError{Type: r.Error.Type, Message: r.Error.Message, Traceback: r.Error.Traceback}
}

// bridgeProcess is one running interpreter.
type bridgeProcess struct {
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	lines   chan []byte
	readAt  chan error
	exited  chan struct{}
	waitErr error

	stderrMu sync.Mutex
	stderr   []string
}

// pump reads whole protocol lines off stdout until the stream ends.
func (p *bridgeProcess) pump(stdout io.Reader) {
	defer close(p.lines)
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64<<10), 32<<20)
	for scanner.Scan() {
		line := make([]byte, len(scanner.Bytes()))
		copy(line, scanner.Bytes())
		p.lines <- line
	}
	if err := scanner.Err(); err != nil {
		select {
		case p.readAt <- err:
		default:
		}
	}
}

// drainStderr forwards the interpreter's stderr and keeps the tail for errors.
func (p *bridgeProcess) drainStderr(stderr io.Reader, sink io.Writer) {
	scanner := bufio.NewScanner(stderr)
	scanner.Buffer(make([]byte, 0, 4<<10), 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if sink != nil {
			_, _ = io.WriteString(sink, line+"\n")
		}
		p.stderrMu.Lock()
		p.stderr = append(p.stderr, line)
		if len(p.stderr) > 40 {
			p.stderr = p.stderr[len(p.stderr)-40:]
		}
		p.stderrMu.Unlock()
	}
}

// tail is the last of the interpreter's stderr, for quoting back in an error.
func (p *bridgeProcess) tail() string {
	p.stderrMu.Lock()
	defer p.stderrMu.Unlock()
	if len(p.stderr) == 0 {
		return "no output on stderr"
	}
	from := max(len(p.stderr)-8, 0)
	return strings.Join(p.stderr[from:], "\n")
}

// read waits for one reply line, the interpreter exiting, or ctx.
func (p *bridgeProcess) read(ctx context.Context) (sidecarReply, error) {
	var reply sidecarReply
	select {
	case line, ok := <-p.lines:
		if !ok {
			<-p.exited
			return reply, fmt.Errorf("%w: bridge exited (%v): %s", ErrUnavailable, p.waitErr, p.tail())
		}
		if err := json.Unmarshal(line, &reply); err != nil {
			return reply, fmt.Errorf("laya: unreadable bridge reply %q: %w", truncate(string(line), 200), err)
		}
		return reply, nil
	case err := <-p.readAt:
		return reply, fmt.Errorf("%w: reading from bridge: %w", ErrUnavailable, err)
	case <-ctx.Done():
		return reply, fmt.Errorf("laya: %w (bridge stderr: %s)", ctx.Err(), p.tail())
	}
}

// alive reports whether the interpreter is still running.
func (p *bridgeProcess) alive() bool {
	select {
	case <-p.exited:
		return false
	default:
		return true
	}
}

// stop shuts the interpreter down, killing one that will not leave.
func (p *bridgeProcess) stop() error {
	_, _ = io.WriteString(p.stdin, `{"op":"shutdown"}`+"\n")
	_ = p.stdin.Close()

	select {
	case <-p.exited:
		return nil
	case <-time.After(2 * time.Second):
	}
	if p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
	<-p.exited
	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
