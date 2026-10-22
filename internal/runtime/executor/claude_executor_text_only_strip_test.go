package executor

import (
	"encoding/json"
	"testing"

	"github.com/tidwall/gjson"
)

func TestSanitizeClaudeBlocksForTextOnlyUpstreamReplacesTopLevelImage(t *testing.T) {
	body := []byte(`{"model":"glm-5.3","messages":[
		{"role":"user","content":[
			{"type":"text","text":"look"},
			{"type":"image","source":{"type":"base64","data":"iVBORw0KGgo"}}
		]}
	]}`)
	out := sanitizeClaudeBlocksForTextOnlyUpstream(body)
	blocks := gjson.GetBytes(out, "messages.0.content").Array()
	if len(blocks) != 2 {
		t.Fatalf("block count = %d, want 2", len(blocks))
	}
	if got := blocks[1].Get("type").String(); got != "text" {
		t.Fatalf("replaced block type = %q, want text", got)
	}
	if got := blocks[1].Get("text").String(); got != claudeTextOnlyImagePlaceholder {
		t.Fatalf("placeholder = %q, want %q", got, claudeTextOnlyImagePlaceholder)
	}
	if got := blocks[0].Get("text").String(); got != "look" {
		t.Fatalf("text block mutated: %q", got)
	}
}

func TestSanitizeClaudeBlocksForTextOnlyUpstreamReplacesNestedToolResultImage(t *testing.T) {
	body := []byte(`{"model":"glm-5.3","messages":[
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"call_1","content":[
				{"type":"text","text":"file meta"},
				{"type":"image","source":{"type":"base64","data":"iVBORw0KGgo"}}
			]}
		]},
		{"role":"user","content":"plain string stays"}
	]}`)
	out := sanitizeClaudeBlocksForTextOnlyUpstream(body)
	nested := gjson.GetBytes(out, "messages.0.content.0.content").Array()
	if len(nested) != 2 {
		t.Fatalf("nested block count = %d, want 2", len(nested))
	}
	if got := nested[1].Get("type").String(); got != "text" {
		t.Fatalf("nested replaced type = %q, want text", got)
	}
	if got := nested[0].Get("text").String(); got != "file meta" {
		t.Fatalf("nested text block mutated: %q", got)
	}
	if got := gjson.GetBytes(out, "messages.1.content").String(); got != "plain string stays" {
		t.Fatalf("string content mutated: %q", got)
	}
}

func TestSanitizeClaudeBlocksForTextOnlyUpstreamLeavesOtherKindsAndCleanBodiesUntouched(t *testing.T) {
	cases := []string{
		`{"model":"glm-5.3","messages":[{"role":"user","content":[{"type":"tool_use","id":"c","name":"Read","input":{}}]}]}`,
		`{"model":"glm-5.3","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"c","content":"string result"}]}]}`,
		`{"model":"glm-5.3","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"sig"}]}]}`,
		`{"model":"glm-5.3","messages":[{"role":"user","content":"just text"}]}`,
		`{"model":"glm-5.3"}`,
	}
	for _, raw := range cases {
		body := []byte(raw)
		out := sanitizeClaudeBlocksForTextOnlyUpstream(body)
		if string(out) != raw {
			t.Fatalf("body changed for input without image blocks:\n in: %s\nout: %s", raw, out)
		}
	}
}

func TestSanitizeClaudeBlocksForTextOnlyUpstreamFlattensUnknownBlockTypes(t *testing.T) {
	body := []byte(`{"messages":[
		{"role":"user","content":[
			{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0"}},
			{"type":"text","text":"pdf attached"},
			{"type":"redacted_thinking","data":"gAAAAAB"}
		]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"c","content":[
				{"type":"search_result","title":"r","source":{"type":"url","url":"https://x"}},
				{"type":"server_tool_use","id":"su","name":"web_search","input":{}},
				{"type":"text","text":"ok"}
			]}
		]}
	]}`)
	out := sanitizeClaudeBlocksForTextOnlyUpstream(body)
	if got := gjson.GetBytes(out, "messages.0.content.0.type").String(); got != "text" {
		t.Fatalf("document block type = %q, want text", got)
	}
	if got := gjson.GetBytes(out, "messages.0.content.0.text").String(); got != "[omitted document block: text-only upstream]" {
		t.Fatalf("document placeholder = %q", got)
	}
	if got := gjson.GetBytes(out, "messages.0.content.2.text").String(); got != "[omitted redacted_thinking block: text-only upstream]" {
		t.Fatalf("redacted_thinking placeholder = %q", got)
	}
	if got := gjson.GetBytes(out, "messages.1.content.0.content.0.type").String(); got != "text" {
		t.Fatalf("nested search_result type = %q, want text", got)
	}
	if got := gjson.GetBytes(out, "messages.1.content.0.content.1.type").String(); got != "text" {
		t.Fatalf("nested server_tool_use type = %q, want text", got)
	}
	if got := gjson.GetBytes(out, "messages.1.content.0.content.2.text").String(); got != "ok" {
		t.Fatalf("nested text block mutated: %q", got)
	}
	if !json.Valid(out) {
		t.Fatalf("output is not valid JSON")
	}
}

func TestSanitizeClaudeBlocksForTextOnlyUpstreamSkipsBlocksWithoutType(t *testing.T) {
	raw := `{"messages":[{"role":"user","content":[{"text":"no type field"}]}]}`
	out := sanitizeClaudeBlocksForTextOnlyUpstream([]byte(raw))
	if string(out) != raw {
		t.Fatalf("untyped block mutated:\n in: %s\nout: %s", raw, out)
	}
}

func TestSanitizeClaudeBlocksForTextOnlyUpstreamMultipleImagesAndRoundTrip(t *testing.T) {
	body := []byte(`{"messages":[
		{"role":"user","content":[{"type":"image","source":{"type":"base64","data":"AAA"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"c","content":[
			{"type":"image","source":{"type":"base64","data":"BBB"}},
			{"type":"image","source":{"type":"base64","data":"CCC"}}
		]}]}
	]}`)
	out := sanitizeClaudeBlocksForTextOnlyUpstream(body)
	count := 0
	for _, b := range gjson.GetBytes(out, "messages.0.content").Array() {
		if b.Get("type").String() == "text" {
			count++
		}
	}
	for _, b := range gjson.GetBytes(out, "messages.1.content.0.content").Array() {
		if b.Get("type").String() == "text" {
			count++
		}
	}
	if count != 3 {
		t.Fatalf("replaced block count = %d, want 3", count)
	}
	if !json.Valid(out) {
		t.Fatalf("output is not valid JSON")
	}
}
