// Package dialogobs provides an asynchronous, append-only JSONL observation
// channel for full conversation replay. Producers never block: events are
// enqueued with a non-blocking send on a bounded queue, and when the queue is
// full or the log volume is not writable, events are dropped and counted while
// request handling continues unaffected.
//
// Output format: one JSON object per line in hourly files named
// dialog-YYYYMMDD-HH.jsonl under the configured directory. Payloads are
// chunked so no line exceeds maxLineBytes, which keeps vector file sources
// (default 102400-byte line cap) from dropping records. Payloads beyond
// maxEmitPayloadBytes are emitted as a prefix flagged with
// payload_truncated, so one oversized body cannot flood the queue. Payload
// bytes are stored as UTF-8 strings; invalid UTF-8 sequences are replaced
// per encoding/json semantics.
package dialogobs

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	log "github.com/sirupsen/logrus"
)

// Event kinds emitted to the JSONL stream.
const (
	KindClientRequest   = "client_request"
	KindSSE             = "sse"
	KindRequestEnd      = "request_end"
	KindUpstreamAttempt = "upstream_attempt"
)

const (
	// maxLineBytes bounds each JSONL line. Vector file sources commonly drop
	// lines beyond 102400 bytes, so payloads are chunked below that budget.
	maxLineBytes = 64 * 1024
	// defaultChunkBytes is the initial raw payload slice size. JSON escaping
	// can inflate a chunk, so chunks start well below maxLineBytes and are
	// bisected when a marshalled line still exceeds the budget.
	defaultChunkBytes = 32 * 1024
	// defaultQueueSize bounds the asynchronous event queue.
	defaultQueueSize = 8192
	// maxEmitPayloadBytes caps how much payload a single event may enqueue.
	// Beyond the cap the payload is emitted as a prefix flagged with
	// payload_truncated; this keeps one oversized body from flooding the
	// bounded queue and starving other requests' events.
	maxEmitPayloadBytes = 8 << 20
)

// Event is a single observation record. Payload may exceed the line budget;
// it is transparently chunked into sequential lines on emit.
type Event struct {
	Time       time.Time
	RequestID  string
	Kind       string
	Status     int
	Attempt    int
	DurationMs int64
	AuthHint   string
	Provider   string
	Model      string
	Payload    []byte
	// PayloadTruncated is set by Emit when the payload exceeded
	// maxEmitPayloadBytes and was cut to its prefix. Readers reassembling
	// payloads must not treat the result as the full original body.
	PayloadTruncated bool
	// Request-end enrichment: caller attribution and the final usage tally,
	// filled by the observation middleware when emitting request_end.
	Agent          string
	SessionID      string
	Alias          string
	InputTokens    int64
	OutputTokens   int64
	CacheRead      int64
	CacheCreation  int64
	ThinkingTokens int64
}

type jsonLine struct {
	TS               string `json:"ts"`
	RequestID        string `json:"request_id,omitempty"`
	Seq              uint32 `json:"seq"`
	Kind             string `json:"kind"`
	Status           int    `json:"status,omitempty"`
	Attempt          int    `json:"attempt,omitempty"`
	DurationMs       int64  `json:"duration_ms,omitempty"`
	AuthHint         string `json:"auth_hint,omitempty"`
	Provider         string `json:"provider,omitempty"`
	Model            string `json:"model,omitempty"`
	Agent            string `json:"agent,omitempty"`
	SessionID        string `json:"session_id,omitempty"`
	Alias            string `json:"alias,omitempty"`
	InputTokens      int64  `json:"input_tokens,omitempty"`
	OutputTokens     int64  `json:"output_tokens,omitempty"`
	CacheRead        int64  `json:"cache_read,omitempty"`
	CacheCreation    int64  `json:"cache_creation,omitempty"`
	ThinkingTokens   int64  `json:"thinking_tokens,omitempty"`
	Payload          string `json:"payload,omitempty"`
	PayloadTruncated bool   `json:"payload_truncated,omitempty"`
}

// seqKey carries the per-request sequence counter through the request context
// so producers in different layers (middleware, conductor) share one ordering
// per request id.
type seqKey struct{}

type seqCounter struct {
	n uint32
}

// WithSeq attaches a fresh per-request sequence counter to ctx.
func WithSeq(ctx context.Context) context.Context {
	return context.WithValue(ctx, seqKey{}, &seqCounter{})
}

func nextSeq(ctx context.Context) uint32 {
	if ctx == nil {
		return 0
	}
	if s, ok := ctx.Value(seqKey{}).(*seqCounter); ok && s != nil {
		return atomic.AddUint32(&s.n, 1)
	}
	return 0
}

type observer struct {
	queue    chan []byte
	dir      string
	stop     chan struct{}
	done     chan struct{}
	maxBytes int64
	minAge   time.Duration
	dropped  atomic.Uint64
	written  atomic.Uint64
}

var defaultObserver atomic.Pointer[observer]

// Enabled reports whether the dialog observation channel is active.
func Enabled() bool {
	return defaultObserver.Load() != nil
}

// Start initialises the observation channel writing hourly JSONL files under
// dir. When maxTotalSizeMB is greater than zero, a rotation sweep runs every
// minute and deletes the oldest dialog JSONL files (never the active one) to
// keep the volume within the budget, mirroring the logs-max-total-size-mb
// behaviour of the application log directory. When minAgeMinutes is greater
// than zero, the sweep additionally never deletes a file whose last
// modification is more recent than that age, so a retention window survives
// even when a single heavy hour alone exceeds the whole budget. Because files
// rotate hourly, the young-file overshoot is bounded to about one extra file.
// Start is idempotent.
func Start(dir string, maxTotalSizeMB int, minAgeMinutes int) error {
	if Enabled() {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	o := &observer{
		queue:    make(chan []byte, defaultQueueSize),
		dir:      dir,
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
		maxBytes: int64(maxTotalSizeMB) * 1024 * 1024,
		minAge:   time.Duration(minAgeMinutes) * time.Minute,
	}
	if !defaultObserver.CompareAndSwap(nil, o) {
		return nil
	}
	go o.run()
	log.WithField("dir", dir).WithField("max_total_size_mb", maxTotalSizeMB).WithField("min_age_minutes", minAgeMinutes).Info("dialog observation channel started")
	return nil
}

// Stop drains the queue and closes the active file. Safe to call repeatedly.
func Stop() {
	o := defaultObserver.Swap(nil)
	if o == nil {
		return
	}
	close(o.stop)
	<-o.done
}

// Dropped returns the number of events dropped due to queue overflow.
func Dropped() uint64 {
	if o := defaultObserver.Load(); o != nil {
		return o.dropped.Load()
	}
	return 0
}

// Emit enqueues an event without blocking the caller. Payloads beyond
// maxEmitPayloadBytes are emitted as a prefix flagged with
// payload_truncated. Chunks whose marshalled line exceeds the line budget
// are bisected with both halves emitted in order, so reassembly by seq
// reproduces the emitted payload byte-for-byte. Events are dropped (and
// counted) when the queue is full; this never affects request handling.
func Emit(ctx context.Context, ev Event) {
	o := defaultObserver.Load()
	if o == nil {
		return
	}
	if ev.Time.IsZero() {
		ev.Time = time.Now()
	}
	if len(ev.Payload) > maxEmitPayloadBytes {
		ev.Payload = ev.Payload[:maxEmitPayloadBytes]
		ev.PayloadTruncated = true
	}
	var emitChunk func(payload []byte)
	emitChunk = func(payload []byte) {
		seq := nextSeq(ctx)
		line, errMarshal := json.Marshal(jsonLine{
			TS: ev.Time.UTC().Format(time.RFC3339Nano), RequestID: ev.RequestID,
			Seq: seq, Kind: ev.Kind, Status: ev.Status, Attempt: ev.Attempt,
			DurationMs: ev.DurationMs, AuthHint: ev.AuthHint,
			Provider: ev.Provider, Model: ev.Model,
			Agent: ev.Agent, SessionID: ev.SessionID, Alias: ev.Alias,
			InputTokens: ev.InputTokens, OutputTokens: ev.OutputTokens,
			CacheRead: ev.CacheRead, CacheCreation: ev.CacheCreation,
			ThinkingTokens:   ev.ThinkingTokens,
			PayloadTruncated: ev.PayloadTruncated,
			Payload:          string(payload),
		})
		if errMarshal == nil && len(line) <= maxLineBytes {
			o.enqueueLine(line)
			return
		}
		half := len(payload) / 2
		if half < 1 {
			// A single byte still exceeds the budget (only possible for
			// pathological escape amplification); drop rather than block.
			o.countDrop()
			return
		}
		emitChunk(payload[:half])
		emitChunk(payload[half:])
	}
	if len(ev.Payload) == 0 {
		emitChunk(nil)
		return
	}
	rest := ev.Payload
	for len(rest) > 0 {
		chunkSize := defaultChunkBytes
		if chunkSize > len(rest) {
			chunkSize = len(rest)
		}
		emitChunk(rest[:chunkSize])
		rest = rest[chunkSize:]
	}
}

// enqueueLine hands one marshalled line to the writer goroutine without
// blocking. A full queue drops the line and bumps the drop counter.
func (o *observer) enqueueLine(line []byte) {
	select {
	case o.queue <- line:
	default:
		o.countDrop()
	}
}

func (o *observer) countDrop() {
	n := o.dropped.Add(1)
	if n%1000 == 1 {
		log.WithField("dropped", n).Warn("dialog observation queue full, dropping events")
	}
}

func (o *observer) run() {
	defer close(o.done)
	var file *os.File
	var hour int64 = -1
	writeErrCount := uint64(0)
	ensureFile := func(now time.Time) *os.File {
		currentHour := now.Unix() / 3600
		if file != nil && currentHour == hour {
			return file
		}
		path := filepath.Join(o.dir, "dialog-"+now.Format("20060102-15")+".jsonl")
		opened, errOpen := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if errOpen != nil {
			writeErrCount++
			if writeErrCount%100 == 1 {
				log.WithError(errOpen).WithField("path", path).Warn("dialog observation file open failed")
			}
			return nil
		}
		if file != nil {
			_ = file.Close()
		}
		file, hour = opened, currentHour
		return file
	}
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for {
		select {
		case line := <-o.queue:
			if f := ensureFile(time.Now()); f != nil {
				if _, errWrite := f.Write(append(line, '\n')); errWrite != nil {
					writeErrCount++
					o.countDrop()
					if writeErrCount%100 == 1 {
						log.WithError(errWrite).Warn("dialog observation write failed")
					}
				} else {
					o.written.Add(1)
				}
			} else {
				o.countDrop()
			}
		case <-ticker.C:
			if o.maxBytes > 0 && file != nil {
				if _, errSweep := enforceDialogDirSizeLimit(o.dir, o.maxBytes, filepath.Join(o.dir, "dialog-"+time.Now().Format("20060102-15")+".jsonl"), o.minAge); errSweep != nil {
					log.WithError(errSweep).Warn("dialog observation size sweep failed")
				}
			}
		case <-o.stop:
			for {
				select {
				case line := <-o.queue:
					if f := ensureFile(time.Now()); f != nil {
						_, _ = f.Write(append(line, '\n'))
					}
				default:
					if file != nil {
						_ = file.Close()
					}
					if n := o.dropped.Load(); n > 0 {
						log.WithField("dropped", n).Warn("dialog observation dropped events at shutdown")
					}
					return
				}
			}
		}
	}
}

// enforceDialogDirSizeLimit deletes the oldest dialog JSONL files until the
// total size is within maxBytes. The active (protected) file is never removed,
// mirroring the application log directory cleaner. When minAge is greater
// than zero, files modified within the last minAge are also kept, trading a
// bounded budget overshoot (about one extra hourly file) for a guaranteed
// retention window.
func enforceDialogDirSizeLimit(dir string, maxBytes int64, protectedPath string, minAge time.Duration) (int, error) {
	if maxBytes <= 0 {
		return 0, nil
	}
	entries, errRead := os.ReadDir(dir)
	if errRead != nil {
		if os.IsNotExist(errRead) {
			return 0, nil
		}
		return 0, errRead
	}
	protected := ""
	if protectedPath != "" {
		protected = filepath.Clean(protectedPath)
	}
	cutoff := time.Now().Add(-minAge)
	type logFile struct {
		path    string
		size    int64
		modTime time.Time
	}
	var files []logFile
	var total int64
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, "dialog-") || !strings.HasSuffix(name, ".jsonl") {
			continue
		}
		info, errInfo := entry.Info()
		if errInfo != nil || !info.Mode().IsRegular() {
			continue
		}
		files = append(files, logFile{path: filepath.Join(dir, name), size: info.Size(), modTime: info.ModTime()})
		total += info.Size()
	}
	if total <= maxBytes {
		return 0, nil
	}
	sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })
	deleted := 0
	for _, file := range files {
		if total <= maxBytes {
			break
		}
		if protected != "" && filepath.Clean(file.path) == protected {
			continue
		}
		if minAge > 0 && file.modTime.After(cutoff) {
			continue
		}
		if errRemove := os.Remove(file.path); errRemove != nil {
			continue
		}
		total -= file.size
		deleted++
	}
	return deleted, nil
}
