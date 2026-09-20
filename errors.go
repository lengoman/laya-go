package laya

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// Sentinel errors for the conditions worth branching on. Compare with
// [errors.Is], which also matches a [ServerError] or [BridgeError] carrying
// the matching cause.
var (
	// ErrInvalidRequest means the battery failed validation, for example a
	// Score with one level or a question with no instructions. Retrying will
	// not help.
	ErrInvalidRequest = errors.New("laya: invalid request")
	// ErrUnavailable means the runtime could not be reached: no server
	// listening, or a sidecar that died. Starting one fixes it.
	ErrUnavailable = errors.New("laya: runtime unavailable")
	// ErrOverloaded means the runtime is busy or out of memory right now.
	ErrOverloaded = errors.New("laya: runtime overloaded")
	// ErrModelLoad means a checkpoint could not be downloaded or built.
	ErrModelLoad = errors.New("laya: could not load checkpoint")
	// ErrPythonMissing means the sidecar could not start its interpreter.
	ErrPythonMissing = errors.New("laya: python interpreter not found")
	// ErrLayaMissing means the interpreter started but has no laya package.
	ErrLayaMissing = errors.New("laya: the laya package is not installed (pip install laya)")
	// ErrClosed means the client or transport has been closed.
	ErrClosed = errors.New("laya: client is closed")
	// ErrUnknownModel means a checkpoint name is not one Laya ships.
	ErrUnknownModel = errors.New("laya: unknown model")
)

// StatusOverloaded is the non-standard status a server may use when every
// worker is busy.
const StatusOverloaded = 529

// ServerError is a non-2xx response from a Laya HTTP server.
type ServerError struct {
	// Status is the HTTP status code.
	Status int
	// Body is the response body, truncated to a readable length.
	Body string
	// RequestID identifies the request in the server's logs, when it sent one.
	RequestID string
}

func (e *ServerError) Error() string {
	id := ""
	if e.RequestID != "" {
		id = fmt.Sprintf(" (request %s)", e.RequestID)
	}
	return fmt.Sprintf("laya: server error %d %s%s: %s",
		e.Status, http.StatusText(e.Status), id, e.Body)
}

// Is lets [errors.Is] match a ServerError against the sentinel for its status.
func (e *ServerError) Is(target error) bool {
	switch e.Status {
	case http.StatusUnprocessableEntity, http.StatusBadRequest:
		return target == ErrInvalidRequest
	case http.StatusTooManyRequests, StatusOverloaded, http.StatusServiceUnavailable:
		return target == ErrOverloaded
	case http.StatusNotFound:
		return target == ErrUnavailable
	}
	return false
}

// Retryable reports whether sending the same request again could succeed.
func (e *ServerError) Retryable() bool {
	switch e.Status {
	case http.StatusTooManyRequests,
		StatusOverloaded,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	}
	return false
}

// BridgeError is an exception raised by the Python side of a sidecar.
type BridgeError struct {
	// Type is the Python exception class, for example "ValueError".
	Type string
	// Message is the exception message.
	Message string
	// Traceback is the Python traceback, when the bridge sent one.
	Traceback string
}

func (e *BridgeError) Error() string {
	return fmt.Sprintf("laya: bridge raised %s: %s", e.Type, e.Message)
}

// Is maps the common Python failures onto this package's sentinels, so calling
// code can branch without matching on exception names.
func (e *BridgeError) Is(target error) bool {
	switch target {
	case ErrInvalidRequest:
		return e.Type == "ValueError" || e.Type == "KeyError" || e.Type == "TypeError"
	case ErrModelLoad:
		return e.Type == "FileNotFoundError" || strings.Contains(e.Message, "snapshot_download") ||
			strings.Contains(e.Message, "checkpoint")
	case ErrOverloaded:
		return strings.Contains(strings.ToLower(e.Message), "out of memory")
	}
	return false
}

// Retryable reports whether the same request could succeed on a second try. A
// memory failure can, because Laya falls back to CPU; a malformed battery
// cannot.
func (e *BridgeError) Retryable() bool {
	return strings.Contains(strings.ToLower(e.Message), "out of memory")
}

// retryable reports whether err is worth another attempt.
func retryable(err error) bool {
	var (
		server *ServerError
		bridge *BridgeError
	)
	switch {
	case errors.As(err, &server):
		return server.Retryable()
	case errors.As(err, &bridge):
		return bridge.Retryable()
	case errors.Is(err, ErrInvalidRequest), errors.Is(err, ErrUnknownModel),
		errors.Is(err, ErrLayaMissing), errors.Is(err, ErrPythonMissing),
		errors.Is(err, ErrClosed):
		return false
	}
	// A transport that simply could not connect is worth retrying: a server
	// restarting, or a sidecar this client will respawn.
	return errors.Is(err, ErrUnavailable) || errors.Is(err, ErrOverloaded)
}
