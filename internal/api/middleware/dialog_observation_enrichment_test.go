package middleware

import "testing"

func TestScanUsageDataAnthropicFrames(t *testing.T) {
	var u usageSnapshot
	scanUsageData([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":4}}}"), &u)
	if u.input != 10 || u.cacheRead != 4 {
		t.Fatalf("message_start: got %+v", u)
	}
	scanUsageData([]byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":429387,\"output_tokens\":58,\"cache_read_input_tokens\":640,\"cache_creation_input_tokens\":12,\"output_tokens_details\":{\"thinking_tokens\":9}}}"), &u)
	if u.input != 429387 || u.output != 58 || u.cacheRead != 640 || u.cacheCreation != 12 || u.thinking != 9 {
		t.Fatalf("message_delta: got %+v", u)
	}
}

func TestScanUsageDataChatAndResponses(t *testing.T) {
	var u usageSnapshot
	scanUsageData([]byte("data: {\"object\":\"chat.completion.chunk\",\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":2},\"completion_tokens_details\":{\"reasoning_tokens\":5}}}"), &u)
	if u.input != 7 || u.output != 3 || u.cacheRead != 2 || u.thinking != 5 {
		t.Fatalf("chat: got %+v", u)
	}
	var r usageSnapshot
	scanUsageData([]byte("data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":11,\"output_tokens\":22,\"output_tokens_details\":{\"reasoning_tokens\":6}}}"), &r)
	if r.input != 11 || r.output != 22 || r.thinking != 6 {
		t.Fatalf("responses: got %+v", r)
	}
}

func TestScanUsageDataNonStreamBody(t *testing.T) {
	var u usageSnapshot
	scanUsageData([]byte("{\"object\":\"chat.completion\",\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4}}"), &u)
	if u.input != 9 || u.output != 4 {
		t.Fatalf("non-stream: got %+v", u)
	}
}

func TestScanUsageDataSkipsFramesWithoutUsage(t *testing.T) {
	var u usageSnapshot
	scanUsageData([]byte("data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}"), &u)
	if u != (usageSnapshot{}) {
		t.Fatalf("expected zero snapshot, got %+v", u)
	}
}

func TestSessionFromBodyClaudeCode(t *testing.T) {
	body := []byte(`{"model":"zp-cc-glm53","metadata":{"user_id":"user_ab12_account__session_0e4b0e17-1f2a-4c3d-8e5f-6a7b8c9d0e1f"}}`)
	want := "0e4b0e17-1f2a-4c3d-8e5f-6a7b8c9d0e1f"
	if got := sessionFromBody(body); got != want {
		t.Fatalf("claude code: got %q want %q", got, want)
	}
}

func TestSessionFromBodyCodexTopLevel(t *testing.T) {
	body := []byte(`{"model":"zp-cx-glm53","session_id":"sess-123","input":"ping"}`)
	if got := sessionFromBody(body); got != "sess-123" {
		t.Fatalf("codex: got %q", got)
	}
}

func TestSessionFromBodyAbsent(t *testing.T) {
	if got := sessionFromBody([]byte(`{"model":"zp-cx-chat","messages":[]}`)); got != "" {
		t.Fatalf("absent: got %q", got)
	}
}

func TestAgentFromUserAgent(t *testing.T) {
	cases := map[string]string{
		"claude-cli/2.1.5 (external, cli)":                   "claude",
		"codex_cli_rs/0.48.0 (Ubuntu 24.04; x86_64) unknown": "codex",
		"grok-cli/1.0": "grok",
		"kimi-cli/0.9": "kimi",
		"curl/8.7.1":   "curl",
		"":             "",
	}
	for ua, want := range cases {
		if got := agentFromUserAgent(ua); got != want {
			t.Fatalf("ua %q: got %q want %q", ua, got, want)
		}
	}
}
