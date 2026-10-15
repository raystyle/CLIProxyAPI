package middleware

import (
	"bytes"
	"context"
	"io"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/dialogobs"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/logging"
	"github.com/tidwall/gjson"
)

// Scan gates: byte markers that must appear in a body before the gjson
// lookups run, skipping bodies that cannot carry the field in one
// memchr-class pass. Unlike the old size budget they never drop a field
// that is actually present, whatever the body size.
var (
	userIDMarker  = []byte(`"user_id"`)
	sessionMarker = []byte(`"session_id"`)
	modelMarker   = []byte(`"model"`)
)

const (
	maxClientNameLen = 64
	agentDeviceChars = 8
)

// DialogObservationMiddleware emits the full conversation stream (client
// request body, response bytes as written to the client, final status) to the
// dialog observation JSONL channel. Every emit is non-blocking: when the
// channel is disabled or its queue is full, the request path is unaffected.
// It must be registered after RequestLoggingMiddleware so it observes the
// final bytes handed to the client.
func DialogObservationMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !dialogobs.Enabled() ||
			shouldSkipMethodForRequestLogging(c.Request) ||
			!shouldLogRequest(c.Request.URL.Path) {
			c.Next()
			return
		}

		ctx := dialogobs.WithSeq(c.Request.Context())
		c.Request = c.Request.WithContext(ctx)

		var body []byte
		if c.Request.Body != nil {
			if raw, errRead := io.ReadAll(c.Request.Body); errRead == nil {
				c.Request.Body = io.NopCloser(bytes.NewReader(raw))
				body = decodeCapturedRequestBodyForLog(raw, c.Request.Header.Get("Content-Encoding"))
			}
		}

		requestID := logging.GetGinRequestID(c)
		model := modelFromBody(body)
		session, device := sessionAndDeviceFromBody(body)
		dialogobs.Emit(ctx, dialogobs.Event{
			RequestID: requestID,
			Kind:      dialogobs.KindClientRequest,
			Model:     model,
			Payload:   body,
		})

		start := time.Now()
		tee := &dialogTeeWriter{ResponseWriter: c.Writer, ctx: ctx, requestID: requestID}
		c.Writer = tee

		c.Next()

		dialogobs.Emit(ctx, dialogobs.Event{
			RequestID:      requestID,
			Kind:           dialogobs.KindRequestEnd,
			Status:         tee.finalStatus(),
			DurationMs:     time.Since(start).Milliseconds(),
			Agent:          agentIdentity(c.Request.Header.Get("X-Client-Name"), c.Request.Header.Get("User-Agent"), device),
			SessionID:      session,
			Alias:          model,
			InputTokens:    tee.usage.input,
			OutputTokens:   tee.usage.output,
			CacheRead:      tee.usage.cacheRead,
			CacheCreation:  tee.usage.cacheCreation,
			ThinkingTokens: tee.usage.thinking,
		})
	}
}

// dialogTeeWriter forwards every write to the client first, then mirrors the
// bytes into the observation channel. Mirroring never blocks the response.
type dialogTeeWriter struct {
	gin.ResponseWriter
	ctx       context.Context
	requestID string
	status    int
	usage     usageSnapshot
}

// usageSnapshot accumulates the cumulative token tally as response frames
// pass through the tee writer. Writes on the response path are sequential,
// so no locking is required.
type usageSnapshot struct {
	input         int64
	output        int64
	cacheRead     int64
	cacheCreation int64
	thinking      int64
}

func (w *dialogTeeWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *dialogTeeWriter) Write(data []byte) (int, error) {
	n, err := w.ResponseWriter.Write(data)
	scanUsageData(data, &w.usage)
	dialogobs.Emit(w.ctx, dialogobs.Event{
		RequestID: w.requestID,
		Kind:      dialogobs.KindSSE,
		Payload:   data,
	})
	return n, err
}

func (w *dialogTeeWriter) WriteString(data string) (int, error) {
	n, err := w.ResponseWriter.WriteString(data)
	scanUsageData([]byte(data), &w.usage)
	dialogobs.Emit(w.ctx, dialogobs.Event{
		RequestID: w.requestID,
		Kind:      dialogobs.KindSSE,
		Payload:   []byte(data),
	})
	return n, err
}

func (w *dialogTeeWriter) finalStatus() int {
	if w.status != 0 {
		return w.status
	}
	return w.ResponseWriter.Status()
}

// modelFromBody extracts the "model" field for indexing without retaining the
// body. The gjson lookup only walks top-level keys, so the cost stays bounded
// by the distance to the field regardless of body size.
func modelFromBody(body []byte) string {
	if len(body) == 0 || body[0] != '{' || !bytes.Contains(body, modelMarker) {
		return ""
	}
	if m := gjson.GetBytes(body, "model"); m.Type == gjson.String {
		return m.String()
	}
	return ""
}

// scanUsageData inspects response bytes for usage-bearing frames: SSE
// "data:" lines for streaming responses, or the whole write when it is a
// single JSON body (non-streaming responses). Cheap byte gates keep the hot
// path untouched for frames that carry no usage.
func scanUsageData(data []byte, u *usageSnapshot) {
	if !bytes.Contains(data, []byte("usage")) {
		return
	}
	if data[0] == '{' && !bytes.Contains(data, []byte("\ndata:")) {
		scanUsageJSON(data, u)
		return
	}
	for _, line := range bytes.Split(data, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		scanUsageJSON(bytes.TrimSpace(line[len("data:"):]), u)
	}
}

// scanUsageJSON merges one usage object into the snapshot, tolerating the
// anthropic (message.usage / usage), chat (prompt_tokens family), and
// responses (response.usage) shapes. Later frames overwrite earlier values,
// which matches upstreams finalising the tally in their last event.
func scanUsageJSON(payload []byte, u *usageSnapshot) {
	for _, v := range []gjson.Result{
		gjson.GetBytes(payload, "message.usage"),  // anthropic message_start
		gjson.GetBytes(payload, "response.usage"), // responses completed / non-stream
		gjson.GetBytes(payload, "usage"),          // anthropic delta / chat / responses non-stream
	} {
		if !v.IsObject() {
			continue
		}
		if n := v.Get("input_tokens"); n.Exists() {
			u.input = n.Int()
		}
		if n := v.Get("prompt_tokens"); n.Exists() {
			u.input = n.Int()
		}
		if n := v.Get("output_tokens"); n.Exists() {
			u.output = n.Int()
		}
		if n := v.Get("completion_tokens"); n.Exists() {
			u.output = n.Int()
		}
		if n := v.Get("cache_read_input_tokens"); n.Exists() {
			u.cacheRead = n.Int()
		}
		if n := v.Get("cache_creation_input_tokens"); n.Exists() {
			u.cacheCreation = n.Int()
		}
		if n := v.Get("prompt_tokens_details.cached_tokens"); n.Exists() {
			u.cacheRead = n.Int()
		}
		if n := v.Get("output_tokens_details.thinking_tokens"); n.Exists() {
			u.thinking = n.Int()
		}
		if n := v.Get("output_tokens_details.reasoning_tokens"); n.Exists() {
			u.thinking = n.Int()
		}
		if n := v.Get("completion_tokens_details.reasoning_tokens"); n.Exists() {
			u.thinking = n.Int()
		}
	}
}

// sessionAndDeviceFromBody extracts the client session and device identity
// from the request body. Codex-style clients send a top-level session_id;
// Claude Code embeds a session uuid after "session_" in metadata.user_id;
// the GLM Claude Code builds send metadata.user_id as a JSON blob string
// ({"device_id":...,"session_id":...}) — or object, which gjson renders back
// as JSON text, so one parse covers both wire forms. No size budget: the
// marker gates keep absent-field bodies at one pass, present fields are found
// whatever the body size.
func sessionAndDeviceFromBody(body []byte) (session, device string) {
	if len(body) == 0 {
		return "", ""
	}
	if sid := gjson.GetBytes(body, "session_id"); sid.Type == gjson.String && sid.String() != "" {
		return sid.String(), ""
	}
	if !bytes.Contains(body, userIDMarker) {
		return "", ""
	}
	uid := gjson.GetBytes(body, "metadata.user_id").String()
	if uid == "" {
		return "", ""
	}
	if uid[0] == '{' {
		blob := gjson.Parse(uid)
		if blob.IsObject() {
			if s := blob.Get("session_id"); s.Type == gjson.String {
				session = s.String()
			}
			if d := blob.Get("device_id"); d.Type == gjson.String {
				device = d.String()
			}
			return session, device
		}
	}
	if idx := strings.LastIndex(uid, "session_"); idx >= 0 {
		cand := strings.TrimSpace(uid[idx+len("session_"):])
		if len(cand) >= 32 && len(cand) <= 36 {
			return cand, ""
		}
	}
	return "", ""
}

// agentIdentity picks the per-client value for the agent column: an explicit
// X-Client-Name header (operator-set per workstation, e.g. via the Claude
// Code custom-headers env) wins; otherwise the UA CLI family is annotated
// with the device fingerprint from the identity blob ("claude@7fcd982e");
// the bare UA mapping stays the fallback so the column never regresses.
func agentIdentity(xClientName, ua, device string) string {
	if name := strings.TrimSpace(xClientName); name != "" {
		if len(name) > maxClientNameLen {
			name = name[:maxClientNameLen]
		}
		return name
	}
	family := agentFromUserAgent(ua)
	if family == "" {
		family = "client"
	}
	if device != "" {
		prefix := device
		if len(prefix) > agentDeviceChars {
			prefix = prefix[:agentDeviceChars]
		}
		return family + "@" + prefix
	}
	return family
}

// agentFromUserAgent maps the caller CLI from its User-Agent header
// (claude-cli / codex / grok / kimi). Unmapped clients fall back to the first
// UA token so the column stays attributable.
func agentFromUserAgent(ua string) string {
	lower := strings.ToLower(ua)
	switch {
	case strings.Contains(lower, "claude-cli"), strings.Contains(lower, "claude-code"):
		return "claude"
	case strings.Contains(lower, "codex"):
		return "codex"
	case strings.Contains(lower, "grok"):
		return "grok"
	case strings.Contains(lower, "kimi"):
		return "kimi"
	}
	if fields := strings.Fields(ua); len(fields) > 0 {
		if name := strings.SplitN(fields[0], "/", 2)[0]; name != "" {
			return strings.ToLower(name)
		}
	}
	return ""
}
