package laya

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// HTTPTransport talks to a Laya runtime over HTTP: one server, many callers,
// one copy of the weights in memory. The reference server is python/server.py
// in this repository, and any service exposing the same three endpoints works.
//
//	POST /v1/predict  {"state": ..., "questions": {...}, "model": "", "task": "", "lang": ""}
//	POST /v1/route    the same body, answered without a forward pass
//	POST /v1/warm     {"models": ["english", "multilingual"]}
type HTTPTransport struct {
	// BaseURL is the root the endpoints hang off. Empty means [DefaultBaseURL].
	BaseURL string
	// Client sends the requests. Its Timeout bounds each attempt. Empty means
	// a client with [DefaultTimeout].
	Client *http.Client
	// UserAgent is sent with every request.
	UserAgent string
}

var (
	_ Transport    = (*HTTPTransport)(nil)
	_ RouteQuerier = (*HTTPTransport)(nil)
	_ Warmer       = (*HTTPTransport)(nil)
)

// Predict implements [Transport].
func (t *HTTPTransport) Predict(ctx context.Context, req *Request) (*Response, error) {
	var out Response
	if err := t.post(ctx, "/v1/predict", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Route implements [RouteQuerier], reporting the server's own decision.
func (t *HTTPTransport) Route(ctx context.Context, req *Request) (*RouteDecision, error) {
	var out RouteDecision
	if err := t.post(ctx, "/v1/route", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Warm implements [Warmer], building checkpoints on the server before the
// first request needs them.
func (t *HTTPTransport) Warm(ctx context.Context, models ...string) error {
	return t.post(ctx, "/v1/warm", map[string]any{"models": models}, nil)
}

func (t *HTTPTransport) baseURL() string {
	if t.BaseURL == "" {
		return DefaultBaseURL
	}
	return strings.TrimRight(t.BaseURL, "/")
}

func (t *HTTPTransport) client() *http.Client {
	if t.Client == nil {
		return &http.Client{Timeout: DefaultTimeout}
	}
	return t.Client
}

// post sends one JSON request and decodes a JSON reply into out, which may be nil.
func (t *HTTPTransport) post(ctx context.Context, path string, payload, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("laya: encoding request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, t.baseURL()+path, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("laya: building request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if t.UserAgent != "" {
		httpReq.Header.Set("User-Agent", t.UserAgent)
	}

	httpResp, err := t.client().Do(httpReq)
	if err != nil {
		// Nothing listening, DNS failure, connection reset: the runtime is not
		// there, which is a different problem from one that answered badly.
		return fmt.Errorf("%w: %s: %w", ErrUnavailable, t.baseURL(), err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, httpResp.Body)
		_ = httpResp.Body.Close()
	}()

	if httpResp.StatusCode/100 != 2 {
		snippet, _ := io.ReadAll(io.LimitReader(httpResp.Body, maxErrorBody))
		return &ServerError{
			Status:    httpResp.StatusCode,
			Body:      strings.TrimSpace(string(snippet)),
			RequestID: httpResp.Header.Get("X-Request-Id"),
		}
	}

	if out == nil {
		return nil
	}
	if err := json.NewDecoder(httpResp.Body).Decode(out); err != nil {
		return fmt.Errorf("laya: decoding response: %w", err)
	}
	return nil
}

// Ping reports whether a server is listening and able to answer, without
// running a forward pass.
func (t *HTTPTransport) Ping(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return t.post(ctx, "/v1/route", &Request{State: "ping", Questions: Questions{
		"ping": Holds("Is this a ping?"),
	}}, &RouteDecision{})
}
