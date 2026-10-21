package helps

import (
	cryptorand "crypto/rand"
	"math"
	"math/big"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
)

const (
	defaultClaudeRateLimitFuzzMinSeconds = 1
	defaultClaudeRateLimitFuzzMaxSeconds = 30
	// maxClaudeRateLimitWindow is the longest Anthropic rate-limit window (7 days)
	// plus a small clock-skew allowance. A later header timestamp is a billing
	// boundary, not a recoverable rate limit.
	maxClaudeRateLimitWindow = 7*24*time.Hour + time.Hour
)

// ClaudeHeadersIndicateUnifiedRateLimitRejection reports whether response headers explicitly
// declare an Anthropic shared 5h or 7d rate-limit rejection. An overage-only or Fable-only rejection
// remains model-scoped when shared subscription windows are not rejected (status "allowed", "allowed_warning",
// or unexhausted when omitted).
func ClaudeHeadersIndicateUnifiedRateLimitRejection(headers http.Header) bool {
	if headers == nil {
		return false
	}
	unifiedStatus := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-Status")))
	status5h := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-5h-Status")))
	if status5h == "rejected" {
		return true
	}
	status7d := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-7d-Status")))
	if status7d == "rejected" {
		return true
	}
	if unifiedStatus != "rejected" {
		return false
	}
	status7dOI := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-7d_oi-Status")))
	return !isOverageOrFableOnlyRejection(headers, unifiedStatus, status5h, status7d, status7dOI)
}

func isClaudeWindowAllowed(status string) bool {
	return status == "allowed" || status == "allowed_warning"
}

func isOverageOrFableOnlyRejection(headers http.Header, unifiedStatus, status5h, status7d, status7dOI string) bool {
	if status5h == "rejected" || status7d == "rejected" {
		return false
	}

	overageStatus := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-Overage-Status")))
	overageDisabledReason := strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-Overage-Disabled-Reason"))
	representativeClaim := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-Representative-Claim")))

	isOverageRejected := status7dOI == "rejected" ||
		overageStatus == "rejected" ||
		overageDisabledReason != "" ||
		strings.Contains(representativeClaim, "overage")

	if !isOverageRejected {
		return false
	}

	shared5hAllowed := isClaudeWindowAllowed(status5h)
	shared7dAllowed := isClaudeWindowAllowed(status7d)

	if shared5hAllowed && shared7dAllowed {
		return true
	}

	// When Anthropic evaluates an overage-only or 7d_oi claim, it often omits
	// the 5h status header while 5h utilization is 0.00 (or vice versa).
	// Require an explicit, valid non-negative utilization below 1.0 before
	// treating the unmentioned window as healthy.
	if shared7dAllowed && status5h == "" {
		if isClaudeUtilizationHealthy(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-5h-Utilization")) {
			return true
		}
	}
	if shared5hAllowed && status7d == "" {
		if isClaudeUtilizationHealthy(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-7d-Utilization")) {
			return true
		}
	}

	// Anthropic omits both shared-window headers when it did not evaluate them.
	// Only a rejected overage claim uses that absence. An allowed or missing
	// unified status must still honor a short Retry-After. Other signals
	// (overage disabled, overage status rejected) do not widen this exception.
	if unifiedStatus == "rejected" && status5h == "" && status7d == "" && strings.Contains(representativeClaim, "overage") {
		util5h := getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-5h-Utilization")
		util7d := getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-7d-Utilization")
		util5hAbsentOrHealthy := strings.TrimSpace(util5h) == "" || isClaudeUtilizationHealthy(util5h)
		util7dAbsentOrHealthy := strings.TrimSpace(util7d) == "" || isClaudeUtilizationHealthy(util7d)
		if util5hAbsentOrHealthy && util7dAbsentOrHealthy {
			return true
		}
	}

	return false
}

func isClaudeUtilizationHealthy(raw string) bool {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return false
	}
	u, errParse := strconv.ParseFloat(raw, 64)
	if errParse != nil || math.IsNaN(u) || math.IsInf(u, 0) {
		return false
	}
	return u >= 0 && u < 1.0
}

// ParseClaudeRateLimitReset inspects Anthropic response headers for shared and Fable-specific
// unified rate-limit and standard Retry-After reset information, returning the conservative cooldown
// duration including a bounded non-negative random grace period.
// If no valid future reset information is present, it returns nil.
func ParseClaudeRateLimitReset(headers http.Header, now time.Time) *time.Duration {
	return parseClaudeRateLimitResetWithFuzz(headers, now, defaultClaudeRateLimitFuzzMinSeconds, defaultClaudeRateLimitFuzzMaxSeconds)
}

func parseClaudeRateLimitResetWithFuzz(headers http.Header, now time.Time, minFuzzSec, maxFuzzSec int) *time.Duration {
	if headers == nil {
		return nil
	}

	unifiedStatus := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-Status")))
	status5h := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-5h-Status")))
	status7d := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-7d-Status")))
	status7dOI := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-7d_oi-Status")))
	overageOnlyRejection := isOverageOrFableOnlyRejection(headers, unifiedStatus, status5h, status7d, status7dOI)

	var candidateDeadlines []time.Time
	var rejectedWindows []string

	if unifiedStatus == "rejected" {
		rejectedWindows = append(rejectedWindows, "unified")
	}
	if status5h == "rejected" {
		rejectedWindows = append(rejectedWindows, "5h")
	}
	if status7d == "rejected" {
		rejectedWindows = append(rejectedWindows, "7d")
	}
	if status7dOI == "rejected" {
		rejectedWindows = append(rejectedWindows, "7d_oi")
	}

	// 1. Retry-After header (skipped for an overage/Fable-only rejection, which does not describe the credential)
	if !overageOnlyRejection {
		if rawRetryAfter := getHeaderCaseInsensitive(headers, "Retry-After"); rawRetryAfter != "" {
			if !containsString(rejectedWindows, "retry-after") {
				rejectedWindows = append(rejectedWindows, "retry-after")
			}
			if t, ok := parseRetryAfterHeader(rawRetryAfter, now); ok && t.After(now) {
				candidateDeadlines = append(candidateDeadlines, t)
			}
		}
	}

	// 2. 5-hour window reset (only when rejected)
	if status5h == "rejected" {
		if raw := getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-5h-Reset"); raw != "" {
			if t, ok := parseUnixOrTimestamp(raw); ok && t.After(now) {
				candidateDeadlines = append(candidateDeadlines, t)
			}
		}
	}

	// 3. 7-day window reset (only when rejected)
	if status7d == "rejected" {
		if raw := getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-7d-Reset"); raw != "" {
			if t, ok := parseUnixOrTimestamp(raw); ok && t.After(now) {
				candidateDeadlines = append(candidateDeadlines, t)
			}
		}
	}

	// 4. Fable-specific 7-day window reset (only when rejected and not an overage/Fable-only rejection)
	if status7dOI == "rejected" && !overageOnlyRejection {
		if raw := getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-7d_oi-Reset"); raw != "" {
			if t, ok := parseUnixOrTimestamp(raw); ok && t.After(now) {
				candidateDeadlines = append(candidateDeadlines, t)
			}
		}
	}

	// 5. Unified reset header:
	unifiedRejected := !overageOnlyRejection && (unifiedStatus == "rejected" || status5h == "rejected" || status7d == "rejected" || status7dOI == "rejected" ||
		(unifiedStatus == "" && !isClaudeWindowAllowed(status5h) && !isClaudeWindowAllowed(status7d)))

	if unifiedRejected {
		if raw := getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-Reset"); raw != "" {
			if !containsString(rejectedWindows, "unified") {
				rejectedWindows = append(rejectedWindows, "unified")
			}
			// When the representative claim is overage and this timestamp matches
			// the overage reset, it is the billing-period boundary, not a rate-limit
			// recovery time.
			if !claudeUnifiedResetIsOverageBillingBoundary(headers, raw) {
				if resetAt, ok := parseUnixOrTimestamp(raw); ok && resetAt.After(now) {
					candidateDeadlines = append(candidateDeadlines, resetAt)
				}
			}
		}
	}

	candidateDeadlines = filterClaudeRateLimitDeadlines(candidateDeadlines, now)
	if len(candidateDeadlines) == 0 {
		if len(rejectedWindows) > 0 {
			log.WithFields(log.Fields{
				"rejected_windows": strings.Join(rejectedWindows, ","),
				"status":           "fallback_exponential_backoff",
			}).Info("Anthropic rate limit window rejected; falling back to generic exponential backoff")
		}
		return nil
	}

	// Pick the latest applicable deadline across rejected windows
	var latestDeadline time.Time
	for _, deadline := range candidateDeadlines {
		if deadline.After(latestDeadline) {
			latestDeadline = deadline
		}
	}

	if latestDeadline.IsZero() || !latestDeadline.After(now) {
		if len(rejectedWindows) > 0 {
			log.WithFields(log.Fields{
				"rejected_windows": strings.Join(rejectedWindows, ","),
				"status":           "fallback_exponential_backoff",
			}).Info("Anthropic rate limit window rejected; falling back to generic exponential backoff")
		}
		return nil
	}

	baseDuration := latestDeadline.Sub(now)
	fuzz := randomClaudeFuzzDuration(minFuzzSec, maxFuzzSec)
	effectiveDuration := baseDuration + fuzz

	log.WithFields(log.Fields{
		"rejected_windows":   strings.Join(rejectedWindows, ","),
		"effective_cooldown": effectiveDuration.String(),
		"base_cooldown":      baseDuration.String(),
		"fuzz":               fuzz.String(),
		"deadline":           latestDeadline.Format(time.RFC3339),
	}).Info("parsed Anthropic rate limit reset headers")

	return &effectiveDuration
}

func claudeUnifiedResetIsOverageBillingBoundary(headers http.Header, unifiedReset string) bool {
	claim := strings.ToLower(strings.TrimSpace(getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-Representative-Claim")))
	if !strings.Contains(claim, "overage") {
		return false
	}
	overageReset := getHeaderCaseInsensitive(headers, "Anthropic-Ratelimit-Unified-Overage-Reset")
	unifiedAt, okUnified := parseUnixOrTimestamp(unifiedReset)
	overageAt, okOverage := parseUnixOrTimestamp(overageReset)
	if !okUnified || !okOverage {
		return false
	}
	return unifiedAt.Equal(overageAt)
}

func filterClaudeRateLimitDeadlines(deadlines []time.Time, now time.Time) []time.Time {
	if len(deadlines) == 0 {
		return deadlines
	}
	kept := deadlines[:0]
	for _, deadline := range deadlines {
		if !deadline.After(now) {
			continue
		}
		if deadline.Sub(now) > maxClaudeRateLimitWindow {
			continue
		}
		kept = append(kept, deadline)
	}
	return kept
}

func containsString(list []string, target string) bool {
	for _, item := range list {
		if item == target {
			return true
		}
	}
	return false
}

func getHeaderCaseInsensitive(h http.Header, target string) string {
	if h == nil {
		return ""
	}
	if val := h.Get(target); val != "" {
		return val
	}
	for k, v := range h {
		if strings.EqualFold(k, target) && len(v) > 0 {
			return v[0]
		}
	}
	return ""
}

func parseUnixOrTimestamp(raw string) (time.Time, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}, false
	}
	if sec, err := strconv.ParseFloat(raw, 64); err == nil && sec > 0 {
		secInt := int64(sec)
		nsec := int64((sec - float64(secInt)) * 1e9)
		return time.Unix(secInt, nsec), true
	}
	if t, err := time.Parse(time.RFC3339, raw); err == nil {
		return t, true
	}
	if t, err := http.ParseTime(raw); err == nil {
		return t, true
	}
	return time.Time{}, false
}

func parseRetryAfterHeader(raw string, now time.Time) (time.Time, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}, false
	}
	if sec, err := strconv.ParseFloat(raw, 64); err == nil && sec > 0 {
		d := time.Duration(sec * float64(time.Second))
		return now.Add(d), true
	}
	if t, err := http.ParseTime(raw); err == nil {
		return t, true
	}
	if t, err := time.Parse(time.RFC3339, raw); err == nil {
		return t, true
	}
	return time.Time{}, false
}

func randomClaudeFuzzDuration(minSec, maxSec int) time.Duration {
	if maxSec <= minSec {
		if minSec < 0 {
			return 0
		}
		return time.Duration(minSec) * time.Second
	}
	nBig, err := cryptorand.Int(cryptorand.Reader, big.NewInt(int64(maxSec-minSec+1)))
	if err != nil {
		return time.Duration(minSec) * time.Second
	}
	return time.Duration(minSec+int(nBig.Int64())) * time.Second
}

const (
	// claudeQuotaBodyCooldownFallback bounds the synthesized cooldown when the
	// body reports account quota exhaustion without a parseable reset deadline
	// (balance/resource-pack refusals that only a manual recharge clears).
	claudeQuotaBodyCooldownFallback = 15 * time.Minute
	// claudeQuotaBodyCooldownMax caps body-derived cooldowns so vendor timezone
	// skew or a far-future reset deadline cannot lock a credential out for days.
	claudeQuotaBodyCooldownMax = 2 * time.Hour
	// claudeQuotaBodyResetGrace pads a parsed reset deadline so the cooldown
	// releases slightly after the quota window actually resets.
	claudeQuotaBodyResetGrace = 30 * time.Second
)

// claudeQuotaBodyResetRe matches a reset deadline embedded in vendor quota
// messages. Chinese phrasing puts the deadline before the verb ("您的限额将在
// 2026-10-24 14:43:04 重置"), English after it ("resets at 2026-10-24
// 14:43:04"), so any full datetime inside an already quota-gated message is
// the deadline.
var claudeQuotaBodyResetRe = regexp.MustCompile(`(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})`)

// ParseClaudeQuotaBodyReset inspects a 429 body for account-level quota
// exhaustion that the response headers do not describe. Anthropic-compatible
// relays (e.g. bigmodel) report window exhaustion or balance depletion only in
// the JSON body, with no Anthropic rate-limit headers; without this parse the
// conductor classifies those as ordinary model-level 429s whose sub-second
// backoff expires immediately, so a dead credential keeps winning selection
// and each request pays a wasted upstream roundtrip. It returns a bounded
// cooldown duration, or nil when the body does not look like an account quota
// refusal.
//
// Reset deadlines are parsed as UTC on purpose: a vendor stamp in local time
// then only extends the cooldown (bounded by claudeQuotaBodyCooldownMax), never
// releases it early.
func ParseClaudeQuotaBodyReset(body []byte, now time.Time) *time.Duration {
	message := strings.TrimSpace(gjson.GetBytes(body, "error.message").String())
	if message == "" {
		message = strings.TrimSpace(gjson.GetBytes(body, "error").String())
	}
	if message == "" {
		message = strings.TrimSpace(string(body))
	}
	if message == "" || !claudeBodyIndicatesAccountQuota(message) {
		return nil
	}
	cooldown := claudeQuotaBodyCooldownFallback
	if m := claudeQuotaBodyResetRe.FindStringSubmatch(message); m != nil {
		if deadline, errParse := time.ParseInLocation("2006-01-02 15:04:05", strings.Replace(m[1], "T", " ", 1), time.UTC); errParse == nil {
			if untilReset := deadline.Sub(now) + claudeQuotaBodyResetGrace; untilReset > cooldown {
				cooldown = untilReset
			}
		}
	}
	if cooldown > claudeQuotaBodyCooldownMax {
		cooldown = claudeQuotaBodyCooldownMax
	}
	d := cooldown
	return &d
}

// claudeBodyIndicatesAccountQuota matches vendor quota refusal wording: Chinese
// window/balance exhaustion (bigmodel style) plus generic English account limit
// phrases. Model-scoped burst errors stay unmatched on purpose so they keep the
// ordinary model-level 429 path.
func claudeBodyIndicatesAccountQuota(message string) bool {
	if strings.Contains(message, "使用上限") || strings.Contains(message, "余额不足") || strings.Contains(message, "配额") {
		return true
	}
	lower := strings.ToLower(message)
	return strings.Contains(lower, "usage limit") ||
		(strings.Contains(lower, "insufficient") && strings.Contains(lower, "balance"))
}
