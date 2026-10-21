package executor

import (
	"encoding/json"
	"testing"

	"github.com/tidwall/gjson"
)

func TestSanitizeClaudeImageBlocksReplacesTopLevelImage(t *testing.T) {
	body := []byte(`{"model":"glm-5.3","messages":[
		{"role":"user","content":[
			{"type":"text","text":"look"},
			{"type":"image","source":{"type":"base64","data":"iVBORw0KGgo"}}
		]}
	]}`)
	out := sanitizeClaudeImageBlocks(body)
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

func TestSanitizeClaudeImageBlocksReplacesNestedToolResultImage(t *testing.T) {
	body := []byte(`{"model":"glm-5.3","messages":[
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"call_1","content":[
				{"type":"text","text":"file meta"},
				{"type":"image","source":{"type":"base64","data":"iVBORw0KGgo"}}
			]}
		]},
		{"role":"user","content":"plain string stays"}
	]}`)
	out := sanitizeClaudeImageBlocks(body)
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

func TestSanitizeClaudeImageBlocksLeavesOtherKindsAndCleanBodiesUntouched(t *testing.T) {
	cases := []string{
		`{"model":"glm-5.3","messages":[{"role":"user","content":[{"type":"tool_use","id":"c","name":"Read","input":{}}]}]}`,
		`{"model":"glm-5.3","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"c","content":"string result"}]}]}`,
		`{"model":"glm-5.3","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"sig"}]}]}`,
		`{"model":"glm-5.3","messages":[{"role":"user","content":"just text"}]}`,
		`{"model":"glm-5.3"}`,
	}
	for _, raw := range cases {
		body := []byte(raw)
		out := sanitizeClaudeImageBlocks(body)
		if string(out) != raw {
			t.Fatalf("body changed for input without image blocks:\n in: %s\nout: %s", raw, out)
		}
	}
}

func TestSanitizeClaudeImageBlocksMultipleImagesAndRoundTrip(t *testing.T) {
	body := []byte(`{"messages":[
		{"role":"user","content":[{"type":"image","source":{"type":"base64","data":"AAA"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"c","content":[
			{"type":"image","source":{"type":"base64","data":"BBB"}},
			{"type":"image","source":{"type":"base64","data":"CCC"}}
		]}]}
	]}`)
	out := sanitizeClaudeImageBlocks(body)
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
