package executor

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor/helps"
	sigcompat "github.com/router-for-me/CLIProxyAPI/v8/internal/signature"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ClaudeExecutor handles Anthropic Claude over the messages API and keeps bounded
// in-memory thread alias state for OAuth continuations.
// If api_key is unavailable on auth, it falls back to legacy via ClientAdapter.
type ClaudeExecutor struct {
	cfg                     *config.Config
	requestLogProvider      string
	upstreamModelNormalizer func(string) string
	oauthProfileFetcher     claudeOAuthProfileFetcher
	// oauthToolAliases is shared by every ForAPIKey copy of this executor;
	// constructors allocate it so copies made per request reuse one store.
	oauthToolAliases *claudeOAuthToolAliasStore
}

type claudeOAuthCancellationError struct {
	cause error
}

func (e *claudeOAuthCancellationError) Error() string {
	if e == nil || e.cause == nil {
		return ""
	}
	return e.cause.Error()
}

func (e *claudeOAuthCancellationError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

func (e *claudeOAuthCancellationError) IsRequestScoped() bool {
	return e != nil
}

func newClaudeOAuthCancellationError(ctx context.Context, oauth bool, err error) error {
	if !oauth {
		return nil
	}
	cause := err
	if ctx != nil && ctx.Err() != nil {
		cause = ctx.Err()
	}
	if !errors.Is(cause, context.Canceled) {
		return nil
	}
	return &claudeOAuthCancellationError{cause: cause}
}

func shouldSanitizeClaudeMessagesForUpstream(baseModel string) bool {
	return sigcompat.SignatureProviderFromModelName(baseModel) == sigcompat.SignatureProviderClaude
}

func sanitizeClaudeMessagesForClaudeUpstreamWithDebug(ctx context.Context, body []byte, baseModel string, preserveEmptyThinkingBlocks ...bool) []byte {
	sanitized := body
	preserveEmpty := len(preserveEmptyThinkingBlocks) > 0 && preserveEmptyThinkingBlocks[0]
	if shouldSanitizeClaudeMessagesForUpstream(baseModel) || preserveEmpty {
		var report sigcompat.SignatureSanitizeReport
		sanitized, report = sigcompat.SanitizeClaudeMessagesForClaudeUpstream(body, baseModel, preserveEmptyThinkingBlocks...)
		logClaudeSignatureSanitizeReport(ctx, baseModel, report)
	}
	return helps.NormalizeClaudeToolCallIDs(sanitizeClaudeBlocksForTextOnlyUpstream(sanitizeClaudeWebSearchDomains(sanitized)))
}

// claudeTextOnlyImagePlaceholder replaces image content blocks on their way to
// text-only Anthropic-compatible upstreams. GLM's /api/anthropic face rejects
// content.type outside a narrow allowed set with a hard 400 ("messages.content.type
// 参数非法"), which bricks any client session once the offending block lands in
// history (2026-10-20 lan-win incident: tool_result carrying base64 PNG, 40
// consecutive 400s; 2026-10-22 recurrence: non-image blocks after images were
// patched). The placeholder keeps the request legal and the omission visible
// to the model. Fleet fork note: applied unconditionally on this executor
// because every claude-api-key upstream here is a text-only compat face; if a
// genuine Anthropic (vision-capable) upstream is ever added, gate this on
// isAnthropicUpstreamURL before forwarding.
const claudeTextOnlyImagePlaceholder = "[image omitted: text-only upstream]"

// claudeTextOnlyAllowedBlockTypes is the whitelist of content block types
// forwarded as-is. text/tool_use/tool_result/thinking are empirically accepted
// by GLM's face; every other type seen in the wild (image, document from PDF
// reads, redacted_thinking, search_result, server_tool_use,
// web_search_tool_result, ...) is rejected, so anything unknown is flattened
// to a text placeholder rather than enumerated one 400 at a time.
var claudeTextOnlyAllowedBlockTypes = map[string]struct{}{
	"text":        {},
	"tool_use":    {},
	"tool_result": {},
	"thinking":    {},
}

func claudeTextOnlyBlockPlaceholder(blockType string) string {
	if blockType == "image" {
		return claudeTextOnlyImagePlaceholder
	}
	return fmt.Sprintf("[omitted %s block: text-only upstream]", blockType)
}

// sanitizeClaudeBlocksForTextOnlyUpstream walks messages.content and nested
// tool_result content arrays, replacing every block whose type is outside
// claudeTextOnlyAllowedBlockTypes with a text placeholder.
func sanitizeClaudeBlocksForTextOnlyUpstream(body []byte) []byte {
	messages := gjson.GetBytes(body, "messages")
	if !messages.Exists() || !messages.IsArray() {
		return body
	}
	var targets [][2]string // {json path, block type}
	messages.ForEach(func(mi, message gjson.Result) bool {
		content := message.Get("content")
		if !content.Exists() || !content.IsArray() {
			return true
		}
		content.ForEach(func(ci, block gjson.Result) bool {
			blockType := block.Get("type").String()
			if blockType != "" {
				if _, allowed := claudeTextOnlyAllowedBlockTypes[blockType]; !allowed {
					targets = append(targets, [2]string{
						fmt.Sprintf("messages.%d.content.%d", mi.Int(), ci.Int()), blockType,
					})
				}
			}
			if blockType == "tool_result" {
				nested := block.Get("content")
				if nested.Exists() && nested.IsArray() {
					nested.ForEach(func(ni, nestedBlock gjson.Result) bool {
						nestedType := nestedBlock.Get("type").String()
						if nestedType != "" {
							if _, allowed := claudeTextOnlyAllowedBlockTypes[nestedType]; !allowed {
								targets = append(targets, [2]string{
									fmt.Sprintf("messages.%d.content.%d.content.%d", mi.Int(), ci.Int(), ni.Int()), nestedType,
								})
							}
						}
						return true
					})
				}
			}
			return true
		})
		return true
	})
	if len(targets) == 0 {
		return body
	}
	kinds := make([]string, 0, len(targets))
	for _, target := range targets {
		if updated, errSet := sjson.SetBytes(body, target[0], map[string]any{
			"type": "text",
			"text": claudeTextOnlyBlockPlaceholder(target[1]),
		}); errSet == nil {
			body = updated
		}
		kinds = append(kinds, target[1])
	}
	log.WithFields(log.Fields{
		"component":   "text_only_sanitizer",
		"executor":    "claude",
		"action":      "strip_non_text_blocks",
		"replaced":    len(targets),
		"block_types": kinds,
	}).Debug("claude executor: replaced non-text blocks with text placeholders for text-only upstream")
	return body
}

// sanitizeClaudeWebSearchDomains removes empty allowed_domains/blocked_domains
// arrays from built-in web_search tools. Some clients (e.g. litellm) emit an
// empty array instead of omitting the field, and Anthropic rejects it with
// "Empty list of domains is ambiguous. Provide at least one domain or null.".
// Deleting the key is equivalent to leaving it unset.
func sanitizeClaudeWebSearchDomains(body []byte) []byte {
	tools := gjson.GetBytes(body, "tools")
	if !tools.Exists() || !tools.IsArray() {
		return body
	}
	tools.ForEach(func(index, tool gjson.Result) bool {
		if !strings.HasPrefix(tool.Get("type").String(), "web_search_") {
			return true
		}
		for _, field := range []string{"allowed_domains", "blocked_domains"} {
			value := tool.Get(field)
			if value.Exists() && value.IsArray() && len(value.Array()) == 0 {
				path := fmt.Sprintf("tools.%d.%s", index.Int(), field)
				if updated, errDelete := sjson.DeleteBytes(body, path); errDelete == nil {
					body = updated
				}
			}
		}
		return true
	})
	return body
}

func logClaudeSignatureSanitizeReport(ctx context.Context, baseModel string, report sigcompat.SignatureSanitizeReport) {
	if report.DroppedBlocks == 0 && report.DroppedSignatures == 0 && report.ReplacedSignatures == 0 {
		return
	}

	fields := log.Fields{
		"component":           "signature_sanitizer",
		"executor":            "claude",
		"action":              "sanitize_claude_messages",
		"target_provider":     string(report.TargetProvider),
		"target_model":        baseModel,
		"preserved":           report.Preserved,
		"dropped_blocks":      report.DroppedBlocks,
		"dropped_signatures":  report.DroppedSignatures,
		"replaced_signatures": report.ReplacedSignatures,
	}
	if len(report.Decisions) > 0 {
		decision := report.Decisions[0]
		fields["first_block_kind"] = string(decision.BlockKind)
		fields["first_detected_provider"] = string(decision.DetectedProvider)
		fields["first_reason"] = decision.Reason
	}

	helps.LogWithRequestID(ctx).WithFields(fields).Debug("claude executor: sanitized signature history before upstream")
}

// Anthropic-compatible upstreams may reject or even crash when Claude models
// omit max_tokens. Prefer registered model metadata before using a fallback.
const defaultModelMaxTokens = 1024

func NewClaudeExecutor(cfg *config.Config) *ClaudeExecutor {
	return &ClaudeExecutor{cfg: cfg, oauthToolAliases: &claudeOAuthToolAliasStore{}}
}

func (e *ClaudeExecutor) Identifier() string { return "claude" }

func (e *ClaudeExecutor) modelLevelCooling() bool {
	return e != nil && e.cfg != nil && e.cfg.Claude.ModelLevelCooling
}

func (e *ClaudeExecutor) upstreamRequestLogProvider() string {
	if provider := strings.TrimSpace(e.requestLogProvider); provider != "" {
		return provider
	}
	return e.Identifier()
}

func (e *ClaudeExecutor) upstreamModel(baseModel string) string {
	if e.upstreamModelNormalizer != nil {
		return e.upstreamModelNormalizer(baseModel)
	}
	return baseModel
}

func (e *ClaudeExecutor) restoreResponseModel(payload []byte, model string) []byte {
	if e.upstreamModelNormalizer == nil || strings.TrimSpace(model) == "" {
		return payload
	}
	return restoreClaudeResponseModel(payload, model)
}

func restoreClaudeResponseModel(payload []byte, model string) []byte {
	if updated, changed := setClaudeResponseModel(payload, model); changed {
		return updated
	}

	trimmed := bytes.TrimSpace(payload)
	if !bytes.HasPrefix(trimmed, []byte("data:")) {
		return payload
	}
	dataIndex := bytes.Index(payload, []byte("data:"))
	if dataIndex < 0 {
		return payload
	}
	rawJSON := bytes.TrimSpace(payload[dataIndex+len("data:"):])
	updated, changed := setClaudeResponseModel(rawJSON, model)
	if !changed {
		return payload
	}
	rebuilt := make([]byte, 0, dataIndex+len("data: ")+len(updated))
	rebuilt = append(rebuilt, payload[:dataIndex]...)
	rebuilt = append(rebuilt, []byte("data: ")...)
	rebuilt = append(rebuilt, updated...)
	return rebuilt
}

func setClaudeResponseModel(payload []byte, model string) ([]byte, bool) {
	if !gjson.ValidBytes(payload) {
		return payload, false
	}
	updated := payload
	changed := false
	for _, path := range []string{"model", "message.model"} {
		if !gjson.GetBytes(updated, path).Exists() {
			continue
		}
		next, errSet := sjson.SetBytes(updated, path, model)
		if errSet != nil {
			continue
		}
		updated = next
		changed = true
	}
	return updated, changed
}

// PrepareRequest injects Claude credentials into the outgoing HTTP request.
func (e *ClaudeExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	apiKey, _ := claudeCreds(auth)
	useAPIKey := auth != nil && (auth.AuthKind() == cliproxyauth.AuthKindAPIKey || (auth.Attributes != nil && strings.TrimSpace(auth.Attributes["api_key"]) != ""))
	isAnthropicBase := isAnthropicUpstreamURL(req.URL)
	if strings.TrimSpace(apiKey) != "" {
		if isAnthropicBase && useAPIKey {
			req.Header.Del("Authorization")
			req.Header.Set("x-api-key", apiKey)
		} else {
			req.Header.Del("x-api-key")
			req.Header.Set("Authorization", "Bearer "+apiKey)
		}
	} else {
		req.Header.Del("Authorization")
		req.Header.Del("x-api-key")
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

// HttpRequest injects Claude credentials into the request and executes it.
func (e *ClaudeExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("claude executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := helps.NewUtlsHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

// SupportsApplyPatch reports the actual executor contract, independent of its provider name.
func (e *ClaudeExecutor) SupportsApplyPatch() bool { return e != nil }
