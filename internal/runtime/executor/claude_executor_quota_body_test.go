package executor

import (
	"net/http"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor/helps"
)

// Anthropic-compatible relays (e.g. bigmodel) answer a quota-exhausted
// credential with 429 and a refusal that lives only in the JSON body: no
// Anthropic rate-limit headers, no Retry-After. Without a body parse the
// conductor treats it as an ordinary model-level 429 whose sub-second backoff
// expires immediately, so the dead credential keeps winning selection and each
// request pays a wasted upstream roundtrip before the error surfaces.
func TestClassifyClaudeUpstreamError_VendorQuotaBodyGetsBoundedCooldown(t *testing.T) {
	weeklyBody := []byte(`{"type":"error","error":{"type":"rate_limit_error","code":"1310","message":"[1310][已达到每周/每月使用上限。您的限额将在 2026-10-24 14:43:04 重置。][2026102115214823deca91d57a419e]"},"request_id":"2026102115214823deca91d57a419e"}`)
	err := classifyClaudeUpstreamError(http.StatusTooManyRequests, nil, weeklyBody)

	rateLimit, ok := err.(claudeRateLimitError)
	if !ok {
		t.Fatalf("vendor weekly quota refusal = %T, want claudeRateLimitError", err)
	}
	if !rateLimit.credentialScoped {
		t.Fatal("vendor quota refusal must be credential-scoped")
	}
	if rateLimit.StatusCode() != http.StatusTooManyRequests {
		t.Fatalf("status was not preserved for the caller: %d", rateLimit.StatusCode())
	}
	if err.Error() != string(weeklyBody) {
		t.Fatal("body was rewritten on the way back to the caller")
	}
	if ra := rateLimit.RetryAfter(); ra == nil || *ra > 2*time.Hour+time.Minute {
		t.Fatalf("weekly reset 3 days out must cap the cooldown at 2h, got %v", ra)
	}
}

// A 5h window refusal with a near-future reset deadline should cool the
// credential down until that deadline (plus grace), not the 15m fallback.
func TestClassifyClaudeUpstreamError_VendorQuotaBodyHonoursResetDeadline(t *testing.T) {
	deadline := time.Now().UTC().Add(31 * time.Minute).Format("2006-01-02 15:04:05")
	body := []byte(`{"type":"error","error":{"type":"rate_limit_error","code":"1308","message":"[1308][已达到 5 小时的使用上限。您的限额将在 ` + deadline + ` 重置。][202610211439224952b77da0c44e1e]"}`)
	err := classifyClaudeUpstreamError(http.StatusTooManyRequests, nil, body)

	rateLimit, ok := err.(claudeRateLimitError)
	if !ok {
		t.Fatalf("vendor 5h quota refusal = %T, want claudeRateLimitError", err)
	}
	if ra := rateLimit.RetryAfter(); ra == nil || *ra < 30*time.Minute || *ra > 35*time.Minute {
		t.Fatalf("5h window refusal should cool until the parsed deadline (~31m), got %v", ra)
	}
}

// Balance depletion has no reset deadline; a manual recharge is the only way
// out, so a moderate probe interval is right.
func TestClassifyClaudeUpstreamError_VendorBalanceRefusalGetsFallbackCooldown(t *testing.T) {
	body := []byte(`{"error":{"code":"1113","message":"余额不足或无可用资源包,请充值。"}}`)
	err := classifyClaudeUpstreamError(http.StatusTooManyRequests, nil, body)

	rateLimit, ok := err.(claudeRateLimitError)
	if !ok {
		t.Fatalf("balance refusal = %T, want claudeRateLimitError", err)
	}
	if !rateLimit.credentialScoped {
		t.Fatal("balance refusal must be credential-scoped")
	}
	if ra := rateLimit.RetryAfter(); ra == nil || *ra != 15*time.Minute {
		t.Fatalf("balance refusal should use the 15m fallback, got %v", ra)
	}
}

// Ordinary model-level burst errors must keep the plain rotation path: no
// synthesized cooldown, no credential scoping.
func TestClassifyClaudeUpstreamError_BurstRateLimitStaysModelLevel(t *testing.T) {
	body := []byte(`{"type":"error","error":{"type":"rate_limit_error","message":"Number of requests has exceeded your rate limit."}}`)
	err := classifyClaudeUpstreamError(http.StatusTooManyRequests, nil, body)

	rateLimit, ok := err.(claudeRateLimitError)
	if !ok {
		t.Fatalf("burst rate limit = %T, want claudeRateLimitError", err)
	}
	if rateLimit.credentialScoped {
		t.Fatal("burst rate limit must stay model-level")
	}
	if ra := rateLimit.RetryAfter(); ra != nil {
		t.Fatalf("burst rate limit must not synthesize a cooldown, got %v", *ra)
	}
}

// modelLevelCooling keeps per-model cooldown granularity: the vendor quota
// body must not upgrade the failure to whole-credential scope.
func TestClassifyClaudeUpstreamError_ModelLevelCoolingKeepsGranularity(t *testing.T) {
	body := []byte(`{"type":"error","error":{"type":"rate_limit_error","code":"1310","message":"[1310][已达到每周/每月使用上限。您的限额将在 2026-10-24 14:43:04 重置。][x]"}`)
	err := classifyClaudeUpstreamErrorWithCooling(http.StatusTooManyRequests, nil, body, true)

	rateLimit, ok := err.(claudeRateLimitError)
	if !ok {
		t.Fatalf("model-level cooling refusal = %T, want claudeRateLimitError", err)
	}
	if rateLimit.credentialScoped {
		t.Fatal("model-level cooling must not upgrade the vendor quota body to credential scope")
	}
}

func TestParseClaudeQuotaBodyReset_EdgeCases(t *testing.T) {
	now := time.Date(2026, 10, 21, 14, 39, 22, 0, time.UTC)
	cases := []struct {
		name string
		body string
		want time.Duration // 0 means nil
	}{
		{"empty", ``, 0},
		{"unrelated", `{"error":{"message":"model not found"}}`, 0},
		{
			"past deadline falls back",
			`{"error":{"message":"已达到 5 小时的使用上限。您的限额将在 2026-10-21 14:00:00 重置。"}}`,
			15 * time.Minute,
		},
		{
			"non-json body text",
			`余额不足或无可用资源包,请充值。`,
			15 * time.Minute,
		},
		{
			"english usage limit",
			`{"error":{"message":"This organization has exceeded its usage limit."}}`,
			15 * time.Minute,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := helps.ParseClaudeQuotaBodyReset([]byte(tc.body), now)
			if tc.want == 0 {
				if got != nil {
					t.Fatalf("want nil, got %v", *got)
				}
				return
			}
			if got == nil {
				t.Fatal("want a cooldown, got nil")
			}
			if *got != tc.want {
				t.Fatalf("cooldown = %v, want %v", *got, tc.want)
			}
		})
	}
}
