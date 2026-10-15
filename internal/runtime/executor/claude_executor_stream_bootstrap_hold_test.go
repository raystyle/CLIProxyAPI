package executor

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
)

const claudeBootstrapHoldHealthyStream = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"glm-5.3","usage":{"input_tokens":10,"output_tokens":1}}}` + "\n\n" +
	"event: content_block_start\n" +
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}` + "\n\n" +
	"event: content_block_stop\n" +
	`data: {"type":"content_block_stop","index":0}` + "\n\n" +
	"event: message_delta\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}` + "\n\n" +
	"event: message_stop\n" +
	`data: {"type":"message_stop"}` + "\n\n"

// zhipu in-body limit form: HTTP 200 stream whose second event is the error.
const claudeBootstrapHoldInBodyErrorStream = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"glm-5.3","usage":{"input_tokens":10,"output_tokens":1}}}` + "\n\n" +
	"event: error\n" +
	`data: {"type":"error","error":{"type":"rate_limit_error","code":"1308","message":"[1308][已达到 5 小时的使用上限。]"}}` + "\n\n"

func startClaudeBootstrapHoldUpstream(t *testing.T, streamData string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if flusher, ok := w.(http.Flusher); ok {
			_, _ = w.Write([]byte(streamData))
			flusher.Flush()
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func executeClaudeBootstrapHoldStream(t *testing.T, serverURL string) *cliproxyexecutor.StreamResult {
	t.Helper()
	exec := NewClaudeExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{
		Attributes: map[string]string{
			"api_key":  "key-123",
			"base_url": serverURL,
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	result, err := exec.ExecuteStream(ctx, auth, cliproxyexecutor.Request{
		Model:   "glm-5.3",
		Payload: []byte(`{"model":"glm-5.3","messages":[{"role":"user","content":"hi"}],"stream":true}`),
	}, cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	})
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}
	return result
}

func TestClaudeExecutor_ExecuteStream_Passthrough_InBodyRateLimitIsStreamErrorBeforeAnyPayload(t *testing.T) {
	server := startClaudeBootstrapHoldUpstream(t, claudeBootstrapHoldInBodyErrorStream)
	result := executeClaudeBootstrapHoldStream(t, server.URL)

	var sawErr error
	payloads := 0
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			sawErr = chunk.Err
			break
		}
		payloads++
	}
	if payloads != 0 {
		t.Fatalf("expected zero payload chunks before the error, got %d", payloads)
	}
	if sawErr == nil {
		t.Fatal("expected a chunk error for the in-body rate limit event")
	}
	var rateLimited claudeRateLimitError
	if !errors.As(sawErr, &rateLimited) {
		t.Fatalf("expected claudeRateLimitError, got %T: %v", sawErr, sawErr)
	}
	if rateLimited.code != http.StatusTooManyRequests {
		t.Fatalf("expected mapped status 429, got %d", rateLimited.code)
	}
	if !strings.Contains(rateLimited.msg, "1308") {
		t.Fatalf("expected upstream body preserved in error, got: %s", rateLimited.msg)
	}
}

func TestClaudeExecutor_ExecuteStream_Passthrough_HealthyStreamDeliversHeldEvents(t *testing.T) {
	server := startClaudeBootstrapHoldUpstream(t, claudeBootstrapHoldHealthyStream)
	result := executeClaudeBootstrapHoldStream(t, server.URL)

	var collected strings.Builder
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("unexpected chunk error: %v", chunk.Err)
		}
		collected.Write(chunk.Payload)
	}
	if !strings.Contains(collected.String(), "message_start") ||
		!strings.Contains(collected.String(), "content_block_delta") ||
		!strings.Contains(collected.String(), "message_stop") {
		t.Fatalf("held events were not delivered in the healthy stream: %q", collected.String())
	}
}

func TestClaudeExecutor_ExecuteStream_Passthrough_NonRateLimitErrorMapsToBadGateway(t *testing.T) {
	stream := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"glm-5.3","usage":{"input_tokens":10,"output_tokens":1}}}` + "\n\n" +
		"event: error\n" +
		`data: {"type":"error","error":{"type":"api_error","message":"upstream exploded"}}` + "\n\n"
	server := startClaudeBootstrapHoldUpstream(t, stream)
	result := executeClaudeBootstrapHoldStream(t, server.URL)

	var sawErr error
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			sawErr = chunk.Err
			break
		}
	}
	if sawErr == nil {
		t.Fatal("expected a chunk error for the in-body api_error event")
	}
	var status statusErr
	if !errors.As(sawErr, &status) {
		t.Fatalf("expected statusErr, got %T: %v", sawErr, sawErr)
	}
	if status.code != http.StatusBadGateway {
		t.Fatalf("expected 502 mapping, got %d", status.code)
	}
}

func TestClaudeStreamEventClassifiers(t *testing.T) {
	errorFrame := []byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"code\":\"1308\"}}\n\n")
	if !claudeStreamEventIsError(errorFrame) {
		t.Fatal("error frame not detected")
	}
	if !claudeStreamEventIsRateLimit(errorFrame) {
		t.Fatal("rate limit frame not detected")
	}
	startFrame := []byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
	if claudeStreamEventIsError(startFrame) {
		t.Fatal("content frame misclassified as error")
	}
	if !claudeStreamEventStartsContent(startFrame) {
		t.Fatal("content_block_start not detected as content")
	}
	pingFrame := []byte("event: ping\ndata: {\"type\":\"ping\"}\n\n")
	if claudeStreamEventIsError(pingFrame) || claudeStreamEventStartsContent(pingFrame) {
		t.Fatal("ping frame misclassified")
	}
}
