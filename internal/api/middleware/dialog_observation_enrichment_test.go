package middleware

import (
	"strings"
	"testing"
)

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

func TestSessionAndDeviceClaudeCode(t *testing.T) {
	body := []byte(`{"model":"zp-cc-glm53","metadata":{"user_id":"user_ab12_account__session_0e4b0e17-1f2a-4c3d-8e5f-6a7b8c9d0e1f"}}`)
	want := "0e4b0e17-1f2a-4c3d-8e5f-6a7b8c9d0e1f"
	session, device := sessionAndDeviceFromBody(body)
	if session != want || device != "" {
		t.Fatalf("claude code: got session=%q device=%q want session=%q", session, device, want)
	}
}

func TestSessionAndDeviceCodexTopLevel(t *testing.T) {
	session, device := sessionAndDeviceFromBody([]byte(`{"model":"zp-cx-glm53","session_id":"sess-123","input":"ping"}`))
	if session != "sess-123" || device != "" {
		t.Fatalf("codex: got session=%q device=%q", session, device)
	}
}

func TestSessionAndDeviceIdentityBlob(t *testing.T) {
	// GLM Claude Code build: user_id is a JSON blob string carrying both the
	// device fingerprint and the session uuid.
	body := []byte(`{"model":"zp-cc-glm53","messages":[],"metadata":{"user_id":"{\"device_id\":\"7fcd982ec8f5b87cc6392872a6f5d95e79376b025af192981abad573c9af5c5b\",\"account_uuid\":\"\",\"session_id\":\"2c73588e-29f5-498e-81fe-31da3d5dc160\"}"}}`)
	session, device := sessionAndDeviceFromBody(body)
	if session != "2c73588e-29f5-498e-81fe-31da3d5dc160" {
		t.Fatalf("blob session: got %q", session)
	}
	if device != "7fcd982ec8f5b87cc6392872a6f5d95e79376b025af192981abad573c9af5c5b" {
		t.Fatalf("blob device: got %q", device)
	}
}

func TestSessionAndDeviceIdentityBlobObjectWireForm(t *testing.T) {
	// Same blob sent as a real JSON object instead of an escaped string.
	body := []byte(`{"model":"zp-cc-glm53","metadata":{"user_id":{"device_id":"48b310c3e0f5","session_id":"4a6c3b58-0000-0000-0000-000000000001"}}}`)
	session, device := sessionAndDeviceFromBody(body)
	if session != "4a6c3b58-0000-0000-0000-000000000001" || device != "48b310c3e0f5" {
		t.Fatalf("blob object form: got session=%q device=%q", session, device)
	}
}

func TestSessionAndDeviceLargeBody(t *testing.T) {
	// Multi-turn Claude Code bodies dwarf the old 256KiB budget; metadata sits
	// after the messages array and must still be reached.
	filler := strings.Repeat("a", 300*1024)
	body := []byte(`{"model":"zp-cc-glm53","messages":[{"role":"user","content":"` + filler + `"}],"metadata":{"user_id":"user_ab12_account__session_0e4b0e17-1f2a-4c3d-8e5f-6a7b8c9d0e1f"}}`)
	session, device := sessionAndDeviceFromBody(body)
	if session != "0e4b0e17-1f2a-4c3d-8e5f-6a7b8c9d0e1f" || device != "" {
		t.Fatalf("large body: got session=%q device=%q", session, device)
	}
	if got := modelFromBody(body); got != "zp-cc-glm53" {
		t.Fatalf("large body model: got %q", got)
	}
}

func TestSessionAndDeviceAbsent(t *testing.T) {
	session, device := sessionAndDeviceFromBody([]byte(`{"model":"zp-cx-chat","messages":[]}`))
	if session != "" || device != "" {
		t.Fatalf("absent: got session=%q device=%q", session, device)
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

func TestAgentIdentity(t *testing.T) {
	cases := []struct {
		name, header, ua, device, want string
	}{
		{"header wins", "  lan-win-cc1  ", "claude-cli/2.1.5", "7fcd982eaaaabbbb", "lan-win-cc1"},
		{"device annotates family", "", "claude-cli/2.1.5", "7fcd982eaaaabbbb", "claude@7fcd982e"},
		{"device truncated to eight", "", "claude-cli/2.1.5", "48b310c3e0f5", "claude@48b310c3"},
		{"no device keeps bare family", "", "claude-cli/2.1.5", "", "claude"},
		{"unknown ua still device-tagged", "", "", "b716e77a0043", "client@b716e77a"},
	}
	for _, tc := range cases {
		if got := agentIdentity(tc.header, tc.ua, tc.device); got != tc.want {
			t.Fatalf("%s: got %q want %q", tc.name, got, tc.want)
		}
	}
}
