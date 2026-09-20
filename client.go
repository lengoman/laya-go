package laya

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"strings"
	"time"
)

// Defaults used when an option is not supplied.
const (
	// DefaultBaseURL is where a laya-serve runtime listens.
	DefaultBaseURL = "http://127.0.0.1:8600"
	// DefaultTimeout bounds one request. A cold checkpoint build costs seconds,
	// so the first call needs far more headroom than the 33 ms a warm one takes.
	DefaultTimeout = 120 * time.Second

	// ServerURLEnv is the environment variable read when no base URL is given.
	ServerURLEnv = "LAYA_URL"

	maxErrorBody = 1 << 12
)

// Transport carries one request to a Laya runtime and brings back its answer.
// The two implementations are [HTTPTransport], for a shared server, and
// [Sidecar], for a Python interpreter this process owns.
type Transport interface {
	Predict(ctx context.Context, req *Request) (*Response, error)
}

// A RouteQuerier is a transport that can report the runtime's own routing
// decision without running a forward pass. Both built-in transports implement
// it; [Client.Route] falls back to the pure-Go [Route] when one does not.
type RouteQuerier interface {
	Route(ctx context.Context, req *Request) (*RouteDecision, error)
}

// A Warmer is a transport that can build checkpoints ahead of the first
// request. Loading is what costs seconds; answering does not.
type Warmer interface {
	Warm(ctx context.Context, models ...string) error
}

// Config overrides entries in a checkpoint's rl_agent_config.json once it is
// loaded. The two that matter are head_max_len, the token budget every option
// of a question shares, and max_len, the whole sequence.
type Config map[string]any

// HighCardinality raises the option budget for choice questions with more than
// roughly twenty options, which otherwise get too few tokens each to stay
// distinguishable. It costs memory and latency, so it is not the default.
func HighCardinality() Config {
	return Config{"head_max_len": 512, "max_len": 1024}
}

// Retry controls how a request is retried after a retryable failure.
type Retry struct {
	// Attempts is the total number of tries, including the first. Below 1 means 1.
	Attempts int
	// Base is the first backoff delay. Each further delay doubles it.
	Base time.Duration
	// Max caps a single backoff delay.
	Max time.Duration
	// Jitter, from 0 to 1, is the fraction of each delay chosen at random. It
	// stops many parallel callers from retrying in lockstep.
	Jitter float64
}

// DefaultRetry is a reasonable policy for a runtime you host yourself: enough
// to ride out a restart, not enough to hide one.
var DefaultRetry = Retry{Attempts: 3, Base: 250 * time.Millisecond, Max: 5 * time.Second, Jitter: 0.3}

// delay returns how long to wait before attempt number try, counting from 1.
func (r Retry) delay(try int) time.Duration {
	backoff := float64(r.Base) * math.Pow(2, float64(try-1))
	backoff = math.Min(backoff, float64(r.Max))
	if r.Jitter > 0 {
		backoff *= 1 - r.Jitter + rand.Float64()*r.Jitter // #nosec G404 -- jitter, not a secret
	}
	return time.Duration(backoff)
}

// Client asks a Laya runtime typed questions. It is safe for concurrent use.
//
// No API key exists to set: Laya's weights are Apache 2.0 and the runtime is
// one you host, either a shared server or a [Sidecar] this process owns.
type Client struct {
	transport Transport
	model     string
	retry     Retry

	// httpDefaults are collected from the options so that New can build the
	// default HTTP transport after every option has been applied.
	baseURL    string
	httpClient *http.Client
	userAgent  string
	custom     bool
}

// Option configures a [Client].
type Option func(*Client)

// WithBaseURL points the client at a Laya server. Defaults to $LAYA_URL, then
// to [DefaultBaseURL].
func WithBaseURL(url string) Option {
	return func(c *Client) { c.baseURL = strings.TrimRight(url, "/") }
}

// WithHTTPClient supplies your own [http.Client], for custom transport,
// proxies or tracing. Its Timeout, if set, bounds each attempt.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) { c.httpClient = hc }
}

// WithTimeout bounds each individual attempt. Leave room for the first call,
// which may download and build a checkpoint.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) { c.httpClient.Timeout = d }
}

// WithTransport replaces the transport entirely, which is how a [Sidecar] or a
// stub in a test is installed.
func WithTransport(t Transport) Option {
	return func(c *Client) {
		c.transport = t
		c.custom = true
	}
}

// WithSidecar runs Laya in a Python interpreter this process owns, instead of
// talking to a server. The interpreter starts on the first request.
func WithSidecar(s Sidecar) Option {
	return WithTransport(NewSidecar(s))
}

// WithModel pins every request to one checkpoint, turning the router off.
// Names and their aliases are listed in [Models]. The default is empty, which
// lets the runtime route per request.
func WithModel(model string) Option {
	return func(c *Client) { c.model = model }
}

// WithRetry replaces the retry policy. Use Retry{Attempts: 1} to disable retries.
func WithRetry(r Retry) Option {
	return func(c *Client) { c.retry = r }
}

// WithUserAgent appends a product token to the User-Agent header.
func WithUserAgent(ua string) Option {
	return func(c *Client) { c.userAgent = ua }
}

// New builds a client. With no transport option it talks to a Laya server over
// HTTP, at $LAYA_URL or [DefaultBaseURL]; see [WithSidecar] to run the model
// in a Python interpreter this process owns instead.
//
// It returns [ErrUnknownModel] when [WithModel] names a checkpoint Laya does
// not ship. Nothing is dialled and no checkpoint is built here, so New is
// cheap and cannot fail for being offline.
func New(opts ...Option) (*Client, error) {
	c := &Client{
		baseURL:    envOr(ServerURLEnv, DefaultBaseURL),
		httpClient: &http.Client{Timeout: DefaultTimeout},
		userAgent:  "laya-go",
		retry:      DefaultRetry,
	}
	for _, opt := range opts {
		opt(c)
	}
	if c.model != "" {
		name, err := NormaliseModel(c.model)
		if err != nil {
			return nil, err
		}
		c.model = name
	}
	if !c.custom {
		c.transport = &HTTPTransport{
			BaseURL:   strings.TrimRight(c.baseURL, "/"),
			Client:    c.httpClient,
			UserAgent: c.userAgent,
		}
	}
	return c, nil
}

// Model reports the checkpoint the client pins every request to, or "" when
// the runtime routes per request.
func (c *Client) Model() string { return c.model }

// Transport reports the transport in use.
func (c *Client) Transport() Transport { return c.transport }

// Close releases the transport, stopping a sidecar interpreter. It is safe to
// call on a client whose transport holds nothing.
func (c *Client) Close() error {
	if closer, ok := c.transport.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

// Request is one evaluation: a state, and the battery of questions to ask about it.
type Request struct {
	// State is the content to evaluate: a string, or any value that marshals to
	// a JSON object or array. Give each question enough state to answer it.
	State any `json:"state"`
	// Questions is the battery, keyed by ids you choose. Independent questions
	// belong in one request: they run in one forward pass.
	Questions Questions `json:"questions"`
	// Model pins this request to one checkpoint, skipping the router. Usually
	// empty.
	Model string `json:"model,omitempty"`
	// Task names a typed-decisions workflow, which selects that checkpoint.
	Task string `json:"task,omitempty"`
	// Lang skips language detection when you already know the language. Any
	// code but "en" routes to the multilingual checkpoint.
	Lang string `json:"lang,omitempty"`
}

// Response is the result of one evaluation.
type Response struct {
	// Model is the runtime's name for the architecture that answered.
	Model string `json:"model"`
	// Answers holds one answer per question id.
	Answers Answers `json:"answers"`
	// Usage counts the tokens the forward pass consumed. Laya generates no
	// text, so OutputTokens is always zero.
	Usage Usage `json:"usage"`
	// Routing records which checkpoint answered and why, when the runtime
	// routed the request.
	Routing *RouteDecision `json:"routing,omitempty"`
	// Latency is the wall time of the successful attempt, measured by the
	// client. It includes a checkpoint build on the first call.
	Latency time.Duration `json:"-"`
}

// Checkpoint reports which checkpoint answered: "english", "multilingual" or
// "typed-decisions". It is empty when the runtime reported no routing.
func (r *Response) Checkpoint() string {
	if r == nil || r.Routing == nil {
		return ""
	}
	return r.Routing.Model
}

// Usage counts tokens for one request.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// CallOption adjusts a single call.
type CallOption func(*Request)

// UseModel pins one call to a checkpoint, skipping the router.
func UseModel(model string) CallOption {
	return func(r *Request) { r.Model = model }
}

// UseTask selects the checkpoint fine-tuned for a task. The one Laya ships is
// [TaskTypedDecisions].
func UseTask(task string) CallOption {
	return func(r *Request) { r.Task = task }
}

// UseLang states the language of the state, so the router does not have to
// guess. Any code but "en" routes to the multilingual checkpoint.
func UseLang(lang string) CallOption {
	return func(r *Request) { r.Lang = lang }
}

// Ask evaluates state against a battery of questions in one forward pass.
//
// Independent questions should go in one Ask: they cost one pass together, and
// a batched question is measured at about 7 ms against 33 ms alone. They
// cannot see one another's answers, which is what lets them run together. Make
// a second call only when an answer decides what to ask or fetch next.
func (c *Client) Ask(ctx context.Context, state any, questions Questions, opts ...CallOption) (*Response, error) {
	req := Request{State: state, Questions: questions, Model: c.model}
	for _, opt := range opts {
		opt(&req)
	}
	return c.Do(ctx, req)
}

// Do sends a fully built [Request]. [Client.Ask] is the convenient form of this.
func (c *Client) Do(ctx context.Context, req Request) (*Response, error) {
	if err := req.Questions.Validate(); err != nil {
		return nil, err
	}
	if req.Model == "" {
		req.Model = c.model
	}
	if req.Model != "" {
		name, err := NormaliseModel(req.Model)
		if err != nil {
			return nil, err
		}
		req.Model = name
	}

	attempts := max(c.retry.Attempts, 1)
	var lastErr error
	for try := 1; try <= attempts; try++ {
		if try > 1 {
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("laya: %w (after %d attempts, last: %v)", ctx.Err(), try-1, lastErr)
			case <-time.After(c.retry.delay(try - 1)):
			}
		}

		started := time.Now()
		resp, err := c.transport.Predict(ctx, &req)
		if err == nil {
			resp.Latency = time.Since(started)
			return resp, nil
		}
		lastErr = err

		if !retryable(err) {
			return nil, err
		}
		if ctx.Err() != nil {
			return nil, fmt.Errorf("laya: %w", ctx.Err())
		}
	}
	return nil, fmt.Errorf("laya: giving up after %d attempts: %w", attempts, lastErr)
}

// Route reports which checkpoint would answer this state, and why, without
// running a forward pass.
//
// It asks the runtime when the transport can answer, so the reason matches
// what a real call would do. Otherwise it falls back to [Route], this
// package's port of the same detection, which needs no runtime at all.
func (c *Client) Route(ctx context.Context, state any, questions Questions, opts ...CallOption) (*RouteDecision, error) {
	req := Request{State: state, Questions: questions, Model: c.model}
	for _, opt := range opts {
		opt(&req)
	}
	if querier, ok := c.transport.(RouteQuerier); ok {
		decision, err := querier.Route(ctx, &req)
		if err == nil {
			return decision, nil
		}
		if !errors.Is(err, ErrUnavailable) {
			return nil, err
		}
	}
	decision := Route(req.State, req.Questions, RouteOptions{
		Model: req.Model, Task: req.Task, Lang: req.Lang,
	})
	return &decision, nil
}

// Warm builds checkpoints before the first request, so no caller pays for the
// load. With no names it warms every checkpoint the runtime serves.
//
// Laya measures a cold build at 7.4 s on CPU and 10.3 s on a T4, against 33 ms
// to answer, so a server that routes between languages should warm at startup.
// It is a no-op on a transport that cannot warm.
func (c *Client) Warm(ctx context.Context, models ...string) error {
	warmer, ok := c.transport.(Warmer)
	if !ok {
		return nil
	}
	names := make([]string, len(models))
	for i, model := range models {
		name, err := NormaliseModel(model)
		if err != nil {
			return err
		}
		names[i] = name
	}
	return warmer.Warm(ctx, names...)
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
