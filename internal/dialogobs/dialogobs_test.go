package dialogobs

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func readAllLines(t *testing.T, dir string) []jsonLine {
	t.Helper()
	var lines []jsonLine
	entries, errList := os.ReadDir(dir)
	if errList != nil {
		t.Fatalf("read dir: %v", errList)
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".jsonl") {
			continue
		}
		file, errOpen := os.Open(filepath.Join(dir, entry.Name()))
		if errOpen != nil {
			t.Fatalf("open: %v", errOpen)
		}
		scanner := bufio.NewScanner(file)
		scanner.Buffer(make([]byte, 0, 128*1024), 128*1024)
		for scanner.Scan() {
			var line jsonLine
			if errUnmarshal := json.Unmarshal(scanner.Bytes(), &line); errUnmarshal != nil {
				t.Fatalf("unmarshal %q: %v", scanner.Text(), errUnmarshal)
			}
			lines = append(lines, line)
		}
		if errScan := scanner.Err(); errScan != nil {
			t.Fatalf("scan: %v", errScan)
		}
		_ = file.Close()
	}
	return lines
}

func TestStartEmitStopRoundtrip(t *testing.T) {
	dir := t.TempDir()
	if errStart := Start(dir, 0); errStart != nil {
		t.Fatalf("start: %v", errStart)
	}
	if !Enabled() {
		t.Fatal("expected enabled after start")
	}
	ctx := WithSeq(context.Background())
	Emit(ctx, Event{RequestID: "rid-1", Kind: KindClientRequest, Model: "m", Payload: []byte(`{"ping":1}`)})
	Emit(ctx, Event{RequestID: "rid-1", Kind: KindSSE, Payload: []byte("data: hello\n\n")})
	Emit(ctx, Event{RequestID: "rid-1", Kind: KindRequestEnd, Status: 200, DurationMs: 42})
	Stop()
	if Enabled() {
		t.Fatal("expected disabled after stop")
	}
	lines := readAllLines(t, dir)
	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %d", len(lines))
	}
	if lines[0].Kind != KindClientRequest || lines[0].RequestID != "rid-1" || lines[0].Model != "m" {
		t.Fatalf("unexpected first line: %+v", lines[0])
	}
	if lines[0].Payload != `{"ping":1}` {
		t.Fatalf("unexpected payload: %q", lines[0].Payload)
	}
	if lines[2].Kind != KindRequestEnd || lines[2].Status != 200 || lines[2].DurationMs != 42 {
		t.Fatalf("unexpected end line: %+v", lines[2])
	}
	if lines[0].Seq != 1 || lines[1].Seq != 2 || lines[2].Seq != 3 {
		t.Fatalf("expected sequential seq, got %d %d %d", lines[0].Seq, lines[1].Seq, lines[2].Seq)
	}
}

func TestPayloadChunkingKeepsLineBudget(t *testing.T) {
	dir := t.TempDir()
	if errStart := Start(dir, 0); errStart != nil {
		t.Fatalf("start: %v", errStart)
	}
	ctx := WithSeq(context.Background())
	payload := strings.Repeat("a", defaultChunkBytes*3+7)
	Emit(ctx, Event{RequestID: "rid-2", Kind: KindSSE, Payload: []byte(payload)})
	Stop()
	var joined strings.Builder
	for _, line := range readAllLines(t, dir) {
		if line.Kind != KindSSE {
			t.Fatalf("unexpected kind: %+v", line)
		}
		joined.WriteString(line.Payload)
	}
	if joined.String() != payload {
		t.Fatalf("chunked payload roundtrip mismatch: got %d bytes, want %d", joined.Len(), len(payload))
	}
}

func TestDisabledEmitIsNoop(t *testing.T) {
	ctx := WithSeq(context.Background())
	Emit(ctx, Event{RequestID: "rid-3", Kind: KindSSE, Payload: []byte("x")})
	if Dropped() != 0 {
		t.Fatalf("expected zero drops, got %d", Dropped())
	}
}

func TestConcurrentEmitDoesNotBlock(t *testing.T) {
	dir := t.TempDir()
	if errStart := Start(dir, 0); errStart != nil {
		t.Fatalf("start: %v", errStart)
	}
	ctx := WithSeq(context.Background())
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				Emit(ctx, Event{RequestID: "rid-4", Kind: KindSSE, Payload: []byte("chunk")})
			}
		}()
	}
	wg.Wait()
	Stop()
	if got := len(readAllLines(t, dir)) + int(Dropped()); got != 16*200 {
		t.Fatalf("expected written+dropped == %d, got written lines + %d drops", 16*200, Dropped())
	}
}

func TestEmitBisectPreservesAllBytes(t *testing.T) {
	dir := t.TempDir()
	if errStart := Start(dir, 0); errStart != nil {
		t.Fatalf("start: %v", errStart)
	}
	ctx := WithSeq(context.Background())
	// Control bytes (0x00-0x08, excluding \t \n \r) marshal to six-byte
	// \u00XX escapes, forcing every default chunk past the line budget and
	// down the bisection path.
	payload := make([]byte, defaultChunkBytes*3+7)
	for i := range payload {
		payload[i] = byte(i % 9)
	}
	Emit(ctx, Event{RequestID: "rid-bisect", Kind: KindSSE, Payload: payload})
	Stop()
	var joined []byte
	for _, line := range readAllLines(t, dir) {
		if line.Kind != KindSSE {
			t.Fatalf("unexpected kind: %+v", line)
		}
		if line.PayloadTruncated {
			t.Fatal("unexpected truncation flag on in-budget payload")
		}
		joined = append(joined, line.Payload...)
	}
	if !bytes.Equal(joined, payload) {
		t.Fatalf("bisected payload roundtrip mismatch: got %d bytes, want %d", len(joined), len(payload))
	}
}

func TestEmitPayloadCapFlagsTruncation(t *testing.T) {
	dir := t.TempDir()
	if errStart := Start(dir, 0); errStart != nil {
		t.Fatalf("start: %v", errStart)
	}
	ctx := WithSeq(context.Background())
	payload := bytes.Repeat([]byte("x"), maxEmitPayloadBytes+13)
	Emit(ctx, Event{RequestID: "rid-cap", Kind: KindClientRequest, Payload: payload})
	Stop()
	emitted := 0
	for _, line := range readAllLines(t, dir) {
		if !line.PayloadTruncated {
			t.Fatal("expected payload_truncated on every line of the capped event")
		}
		emitted += len(line.Payload)
	}
	if emitted != maxEmitPayloadBytes {
		t.Fatalf("expected %d emitted bytes, got %d", maxEmitPayloadBytes, emitted)
	}
}

func TestStartIdempotent(t *testing.T) {
	dir := t.TempDir()
	if errStart := Start(dir, 0); errStart != nil {
		t.Fatalf("start: %v", errStart)
	}
	if errSecond := Start(dir, 0); errSecond != nil {
		t.Fatalf("second start: %v", errSecond)
	}
	Stop()
	Stop()
}

func TestSizeLimitSweepDeletesOldestKeepsActive(t *testing.T) {
	dir := t.TempDir()
	write := func(name, content string) {
		if errWrite := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); errWrite != nil {
			t.Fatalf("write %s: %v", name, errWrite)
		}
	}
	write("dialog-20260101-00.jsonl", strings.Repeat("a", 4000))
	write("dialog-20260101-01.jsonl", strings.Repeat("b", 4000))
	write("dialog-20260101-02.jsonl", strings.Repeat("c", 4000))
	// Budget fits one old file plus the protected active file.
	deleted, errSweep := enforceDialogDirSizeLimit(dir, 8000, filepath.Join(dir, "dialog-20260101-02.jsonl"))
	if errSweep != nil {
		t.Fatalf("sweep: %v", errSweep)
	}
	if deleted != 1 {
		t.Fatalf("expected 1 deletion, got %d", deleted)
	}
	if _, errStat := os.Stat(filepath.Join(dir, "dialog-20260101-00.jsonl")); !os.IsNotExist(errStat) {
		t.Fatal("expected oldest file removed")
	}
	if _, errStat := os.Stat(filepath.Join(dir, "dialog-20260101-01.jsonl")); errStat != nil {
		t.Fatal("expected second file kept")
	}
	if _, errStat := os.Stat(filepath.Join(dir, "dialog-20260101-02.jsonl")); errStat != nil {
		t.Fatal("expected active file kept")
	}
}
