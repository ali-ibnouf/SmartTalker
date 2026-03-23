/**
 * Maskki — Admin Dashboard Logic
 * Fetches KPIs, drift alerts, guardrail violations, and review queues.
 */

"use strict";

const $ = (sel) => document.querySelector(sel);

const dom = {
    btnRefresh: $("#refresh-btn"),
    inpAvatar: $("#cfg-avatar"),

    // KPIs
    kpiConversations: $("#kpi-conversations"),
    kpiAutonomy: $("#kpi-autonomy"),
    kpiResolution: $("#kpi-resolution"),
    kpiCost: $("#kpi-cost"),

    // Panels
    driftAlerts: $("#drift-alerts"),
    guardrailViolations: $("#guardrail-violations"),
    reviewQueue: $("#review-queue"),
    activityTimeline: $("#activity-timeline"),
};

const API_KEY = "test_key_123"; // Using default test key for now
const HEADERS = {
    "Authorization": `Bearer ${API_KEY}`,
    "Content-Type": "application/json"
};

function getApiBase() {
    if (location.hostname === "localhost" || location.hostname === "127.0.0.1") return "http://localhost:8000";
    return "https://ws.maskki.com";
}

/** Fetch JSON from API */
async function fetchApi(path, options = {}) {
    options.headers = { ...HEADERS, ...options.headers };
    try {
        const url = path.startsWith("http") ? path : getApiBase() + path;
        const res = await fetch(url, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        return await res.json();
    } catch (err) {
        console.error(`Error fetching ${path}:`, err);
        return null;
    }
}

/** Format time */
function formatTime(timestamp) {
    if (!timestamp) return "N/A";
    const d = new Date(timestamp * 1000);
    return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/** Refresh all data */
async function refreshDashboard() {
    const avatarId = dom.inpAvatar.value.trim() || "default";

    // 1. Fetch KPIs
    const kpis = await fetchApi(`/api/v1/analytics/${avatarId}/kpis`);
    if (kpis) {
        dom.kpiConversations.textContent = kpis.total_conversations;
        dom.kpiAutonomy.textContent = `${kpis.autonomy_percent.toFixed(1)}%`;
        dom.kpiResolution.textContent = `${kpis.resolution_time_avg_s.toFixed(1)}s`;
        dom.kpiCost.textContent = `$${kpis.total_cost.toFixed(2)}`;
    }

    // 2. Fetch Drift Alerts
    const drift = await fetchApi(`/api/v1/analytics/${avatarId}/drift`);
    if (drift && drift.count > 0) {
        dom.driftAlerts.innerHTML = drift.alerts.map(a => `
            <div class="alert-item ${a.severity === 'urgent' ? 'danger' : ''}">
                <div class="alert-title">${a.metric} (${a.severity})</div>
                <div>Changed by ${a.change_percent.toFixed(1)}%. Current: ${a.current_value.toFixed(2)}</div>
            </div>
        `).join("");
    } else {
        dom.driftAlerts.innerHTML = `<div class="empty-state">No drift detected</div>`;
    }

    // 3. Fetch Guardrail Violations
    const violations = await fetchApi(`/api/v1/guardrails/${avatarId}/violations?limit=5`);
    if (violations && violations.count > 0) {
        dom.guardrailViolations.innerHTML = violations.violations.map(v => `
            <div class="alert-item ${v.severity === 'high' ? 'danger' : ''}">
                <div class="alert-title">${v.violation_type.toUpperCase()} - ${formatTime(v.created_at)}</div>
                <div style="margin-bottom: 4px"><em>User:</em> "${escapeHtml(v.original_text || '')}"</div>
                <div><em>AI:</em> "${escapeHtml(v.sanitized_text || '')}"</div>
            </div>
        `).join("");
    } else {
        dom.guardrailViolations.innerHTML = `<div class="empty-state">No recent violations</div>`;
    }

    // 4. Fetch Review Queue
    const queue = await fetchApi(`/api/v1/supervisor/review-queue?reviewed=false&limit=10`);
    if (queue && queue.count > 0) {
        dom.reviewQueue.innerHTML = queue.reviews.map(r => `
            <div class="queue-item" id="queue-${r.review_id}">
                <div class="queue-flags">⚠️ ${r.flagged_reason} (Conf: ${(r.confidence * 100).toFixed(1)}%)</div>
                <div class="queue-qa">
                    <strong>Q:</strong> ${escapeHtml(r.question)}<br/>
                    <strong>A:</strong> ${escapeHtml(r.ai_response)}
                </div>
                <div class="queue-actions">
                    <button class="btn-approve" onclick="submitReview('${r.review_id}', 'approved')">✓ Approve</button>
                    <button class="btn-reject" onclick="submitReview('${r.review_id}', 'rejected')">✗ Reject</button>
                </div>
            </div>
        `).join("");
    } else {
        dom.reviewQueue.innerHTML = `<div class="empty-state">Queue is empty! 🎉</div>`;
    }

    // 5. Fetch Activity Timeline
    const timeline = await fetchApi(`/api/v1/supervisor/activity-timeline?limit=10`);
    if (timeline && timeline.count > 0) {
        dom.activityTimeline.innerHTML = timeline.entries.map(e => `
            <div class="timeline-item">
                <div class="tl-time">${formatTime(e.created_at)}</div>
                <div class="tl-content">
                    <span class="tl-type">${e.action_type.toUpperCase()}</span> by ${e.operator_id} 
                    <div style="color: #8b949e; font-size: 0.85rem; margin-top: 4px;">
                        ${typeof e.details === 'object' ? JSON.stringify(e.details) : escapeHtml(e.details || "")}
                    </div>
                </div>
            </div>
        `).join("");
    } else {
        dom.activityTimeline.innerHTML = `<div class="empty-state">No recent activity</div>`;
    }
}

/** Submit a review decision */
async function submitReview(reviewId, verdict) {
    const payload = {
        reviewer_id: "admin",
        verdict: verdict,
        corrected_response: "" // Simplification for now
    };

    const res = await fetchApi(`/api/v1/supervisor/review-queue/${reviewId}`, {
        method: "POST",
        body: JSON.stringify(payload)
    });

    if (res) {
        const item = document.getElementById(`queue-${reviewId}`);
        if (item) {
            item.innerHTML = `<div style="text-align: center; color: var(--success-color); padding: 20px;">Review submitted successfully!</div>`;
            setTimeout(() => {
                if (item.parentNode) item.parentNode.removeChild(item);
            }, 2000);
        }
    }
}

/** Escapes HTML tags to prevent XSS */
function escapeHtml(str) {
    if (!str) return "";
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return str.replace(/[&<>"']/g, (c) => map[c]);
}

// Bind events
dom.btnRefresh.addEventListener("click", refreshDashboard);

// Auto-refresh every 10 seconds
setInterval(refreshDashboard, 10000);

// Initial load
window.addEventListener("DOMContentLoaded", refreshDashboard);
