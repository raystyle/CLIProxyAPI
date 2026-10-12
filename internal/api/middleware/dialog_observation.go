package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/dialogobs"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/logging"
)

// maxModelScanBytes bounds the request body size for cheap "model" field
// extraction; larger bodies skip the field rather than paying a full parse.
const maxModelScanBytes = 256 * 1024

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
		dialogobs.Emit(ctx, dialogobs.Event{
			RequestID: requestID,
			Kind:      dialogobs.KindClientRequest,
			Model:     modelFromBody(body),
			Payload:   body,
		})

		start := time.Now()
		tee := &dialogTeeWriter{ResponseWriter: c.Writer, ctx: ctx, requestID: requestID}
		c.Writer = tee

		c.Next()

		dialogobs.Emit(ctx, dialogobs.Event{
			RequestID:  requestID,
			Kind:       dialogobs.KindRequestEnd,
			Status:     tee.finalStatus(),
			DurationMs: time.Since(start).Milliseconds(),
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
}

func (w *dialogTeeWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *dialogTeeWriter) Write(data []byte) (int, error) {
	n, err := w.ResponseWriter.Write(data)
	dialogobs.Emit(w.ctx, dialogobs.Event{
		RequestID: w.requestID,
		Kind:      dialogobs.KindSSE,
		Payload:   data,
	})
	return n, err
}

func (w *dialogTeeWriter) WriteString(data string) (int, error) {
	n, err := w.ResponseWriter.WriteString(data)
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
// body. Bodies beyond the scan budget are skipped.
func modelFromBody(body []byte) string {
	if len(body) == 0 || len(body) > maxModelScanBytes {
		return ""
	}
	var probe struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		return ""
	}
	return probe.Model
}
