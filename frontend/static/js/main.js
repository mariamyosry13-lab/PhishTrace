// ── Config ───────────────────────────────────────────────
const API_BASE = 'http://127.0.0.1:5000';

const ENDPOINTS = {
    analyze:   '/analyze',
    dashboard: '/api/dashboard',
    campaigns: '/api/campaigns',
    history:   '/api/history',
    scan:      '/api/scan',
};

// ── Toast ────────────────────────────────────────────────
function showToast(msg, type = 'info') {
    const wrap = document.getElementById('toastWrap');
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    const icons = { error: 'fa-circle-xmark', success: 'fa-circle-check', info: 'fa-circle-info' };
    t.innerHTML = `<i class="fa-solid ${icons[type] || icons.info} mr-2"></i><span>${msg}</span>`;
    wrap.appendChild(t);
    setTimeout(() => t.remove(), 3700);
}

// ── Nav ──────────────────────────────────────────────────
function toggleMobile() { document.getElementById('mobileNav').classList.toggle('hidden'); }
function clearInput()   { const inp = document.getElementById('urlInput'); inp.value = ''; inp.focus(); }

function setConn(ok) {
    document.getElementById('connDot').className = ok
        ? 'w-[7px] h-[7px] rounded-full bg-[#2ec4b6] animate-pulse'
        : 'w-[7px] h-[7px] rounded-full bg-[#e63946]';
    const txt = document.getElementById('connText');
    txt.textContent = ok ? 'Connected' : 'Disconnected';
    txt.className   = ok
        ? 'text-[11px] text-gray-500 hidden sm:inline'
        : 'text-[11px] text-red-400 hidden sm:inline';
}

function navigateTo(page) {
    document.querySelectorAll('[id^="page-"]').forEach(el => el.classList.add('hidden'));
    const target = document.getElementById(`page-${page}`);
    if (target) target.classList.remove('hidden');
    document.querySelectorAll('#desktopNav .tab-btn').forEach(b => b.classList.remove('active'));
    const active = document.querySelector(`#desktopNav .tab-btn[data-page="${page}"]`);
    if (active) active.classList.add('active');
    if (page === 'dashboard') loadDashboard();
    if (page === 'campaigns') loadCampaigns();
    if (page === 'history')   loadHistory();
}

// ── Analyze ──────────────────────────────────────────────
let analyzing = false;

async function analyzeURL() {
    const inp = document.getElementById('urlInput');
    const url = inp.value.trim();
    if (!url) { showToast('Please enter a URL or email address', 'error'); inp.focus(); return; }
    if (analyzing) return;
    analyzing = true;

    const btn     = document.getElementById('analyzeBtn');
    const btnTxt  = document.getElementById('analyzeBtnText');
    const loading = document.getElementById('loadingState');
    const results = document.getElementById('resultsSection');

    btn.disabled = true;
    btnTxt.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i>Analyzing...';
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    results.querySelectorAll('.fade-up').forEach(el => el.classList.remove('visible'));
    document.getElementById('resultCard').classList.remove('danger-pulse');

    const steps = [
        'Extracting features...', 'Analyzing URL structure...',
        'Running classifier...', 'Computing SHAP values...',
        'Generating report...'
    ];
    let si = 0;
    const stepTimer = setInterval(() => {
        si = (si + 1) % steps.length;
        document.getElementById('loadingStep').textContent = steps[si];
    }, 1100);

    const t0 = Date.now();
    try {
        const res = await fetch(`${API_BASE}${ENDPOINTS.analyze}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || `Server error: ${res.status}`);
        }
        const data = await res.json();
        data._elapsed = Date.now() - t0;
        clearInterval(stepTimer);
        loading.classList.add('hidden');
        renderResults(data);
        results.classList.remove('hidden');
        setTimeout(() => {
            results.querySelectorAll('.fade-up').forEach((el, i) => {
                setTimeout(() => el.classList.add('visible'), i * 140);
            });
        }, 40);
        setConn(true);
        showToast('Analysis complete', 'success');
    } catch (err) {
        clearInterval(stepTimer);
        loading.classList.add('hidden');
        if (err.message.includes('fetch') || err.message.includes('NetworkError') || err.message.includes('Failed')) {
            showToast('Cannot reach server — make sure the backend is running', 'error');
            setConn(false);
        } else {
            showToast(err.message, 'error');
        }
    } finally {
        analyzing = false;
        btn.disabled = false;
        btnTxt.innerHTML = '<i class="fa-solid fa-crosshairs mr-2"></i>Analyze';
    }
}

// ── Render Results ───────────────────────────────────────
function renderResults(d) {
    const score = d.score ?? 0;
    const pct   = Math.round(score * 100);
    // API returns: verdict (not label)
    const label = d.verdict ?? 'Unknown';

    // Score ring animation
    const arc  = document.getElementById('scoreArc');
    const circ = 2 * Math.PI * 80;
    let color  = '#2ec4b6';
    if (label === 'Dangerous')  color = '#e63946';
    if (label === 'Suspicious') color = '#e9c46a';
    arc.style.stroke = color;
    arc.style.strokeDasharray  = circ;
    arc.style.strokeDashoffset = circ;
    requestAnimationFrame(() => requestAnimationFrame(() => {
        arc.style.strokeDashoffset = circ - (circ * score);
    }));
    const scoreEl = document.getElementById('scoreValue');
    scoreEl.textContent = pct + '%';
    scoreEl.style.color = color;

    // Verdict badge
    const map = {
        Safe:       { text: 'Likely Safe',  icon: 'fa-circle-check',        cls: 'bg-[rgba(46,196,182,0.1)] text-[#8eddd5] border border-[rgba(46,196,182,0.2)]' },
        Suspicious: { text: 'Suspicious',   icon: 'fa-triangle-exclamation', cls: 'bg-[rgba(233,196,106,0.1)] text-[#e9c46a] border border-[rgba(233,196,106,0.2)]' },
        Dangerous:  { text: 'Dangerous',    icon: 'fa-skull-crossbones',     cls: 'bg-[rgba(230,57,70,0.1)] text-[#f0a0a8] border border-[rgba(230,57,70,0.2)]' },
    };
    const cfg = map[label] || map.Suspicious;
    document.getElementById('resultBadge').className   = `inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold ${cfg.cls}`;
    document.getElementById('resultIcon').className    = `fa-solid ${cfg.icon}`;
    document.getElementById('resultLabel').textContent = cfg.text;
    document.getElementById('resultCard').classList.toggle('danger-pulse', label === 'Dangerous');

    // Summary text — adds campaign info if available
    let summary = `The URL scored ${pct}% risk and was classified as "${cfg.text}".`;
    if (d.campaign_id) {
        summary += ` It has been linked to <strong>${d.campaign_id}</strong>.`;
        document.getElementById('resultSummary').innerHTML = summary;
    } else {
        document.getElementById('resultSummary').textContent = summary;
    }

    document.getElementById('resultModel').innerHTML     = `<i class="fa-solid fa-brain mr-1"></i>Model: Random Forest`;
    document.getElementById('resultTime').innerHTML      = `<i class="fa-solid fa-stopwatch mr-1"></i>Time: ${d._elapsed ?? '—'}ms`;
    document.getElementById('resultThreshold').innerHTML = `<i class="fa-solid fa-sliders mr-1"></i>Threshold: 0.75 / 0.45`;

    // Show scan ID as small badge if available
    if (d.scan_id) {
        document.getElementById('resultModel').innerHTML += `&nbsp;<span class="opacity-40">#${d.scan_id}</span>`;
    }

    renderFeatures(d.features || {});
    renderExplanation(d.reasons || [], score, label);
    renderDetails(d);

    // Store scan data globally for report download
    window._lastScan = d;

    // Show download button — inline with the meta pills
    const existingBtn = document.getElementById('downloadReportBtn');
    if (existingBtn) existingBtn.remove();
    const btn = document.createElement('button');
    btn.id = 'downloadReportBtn';
    btn.className = 'text-[12px] text-gray-400 hover:text-white bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] px-4 py-2 rounded-lg transition-all flex items-center gap-2 flex-shrink-0';
    btn.innerHTML = '<i class="fa-solid fa-file-arrow-down text-[11px]"></i>Download Report';
    btn.onclick = downloadReport;
    // Insert right after resultThreshold — same flex row
    const threshEl = document.getElementById('resultThreshold');
    if (threshEl) threshEl.insertAdjacentElement('afterend', btn);
}


// ── Download Report ──────────────────────────────────────
function downloadReport() {
    const d = window._lastScan;
    if (!d) return;

    const score      = Math.round((d.score || 0) * 100);
    const verdict    = d.verdict || 'Unknown';
    const feats      = d.features || {};
    const reasons    = d.reasons  || [];
    const now        = new Date().toLocaleString();
    const verdictColor = { Dangerous:'#e63946', Suspicious:'#e9c46a', Safe:'#2ec4b6' }[verdict] || '#888';
    const verdictText  = verdict === 'Safe' ? 'Likely Safe' : verdict;

    // ── SHAP rows ─────────────────────────────────────────
    const shapRows = reasons.map(r => {
        const isRisk = r.contribution > 0;
        const barW   = Math.min(Math.abs(r.contribution) * 400, 100);
        const barCol = isRisk ? '#e63946' : '#2ec4b6';
        return `<tr>
            <td style="padding:7px 10px;border-bottom:1px solid #f0f0f0;font-size:12px;font-family:monospace;font-weight:500">${r.feature}</td>
            <td style="padding:7px 10px;border-bottom:1px solid #f0f0f0;font-size:11px;color:#666">${r.text_en || ''}</td>
            <td style="padding:7px 10px;border-bottom:1px solid #f0f0f0;width:140px">
                <div style="background:#f0f0f0;border-radius:4px;height:8px">
                    <div style="background:${barCol};height:100%;border-radius:4px;width:${barW}%"></div>
                </div>
            </td>
            <td style="padding:7px 10px;border-bottom:1px solid #f0f0f0;font-weight:700;font-size:12px;color:${barCol};font-family:monospace;text-align:right">
                ${isRisk ? '+' : ''}${r.contribution.toFixed(4)}
            </td>
        </tr>`;
    }).join('');

    // ── Features grid — all entries as 3-column grid cards ─
    const featEntries = Object.entries(feats).filter(([,v]) => v !== undefined && v !== null);
    const featCards = featEntries.map(([k, v]) => {
        const isBad = (k === 'has_ip' && v) || (k === 'has_suspicious_word' && v) ||
                      (k === 'has_at_in_url' && v) || (k === 'double_slash' && v) ||
                      (k === 'num_subdomains' && v >= 3) || (k === 'num_hyphens' && v >= 2) ||
                      (k === 'url_length' && v > 75) || (k === 'num_suspicious_words' && v >= 2);
        const isGood = k === 'has_https' && v;
        const dot    = isBad ? '#e63946' : isGood ? '#2ec4b6' : '#ccc';
        return `<div style="padding:8px 12px;border:1px solid #f0f0f0;border-radius:8px;display:flex;align-items:center;justify-content:space-between;gap:8px">
            <div style="display:flex;align-items:center;gap:6px">
                <div style="width:7px;height:7px;border-radius:50%;background:${dot};flex-shrink:0"></div>
                <span style="font-size:11px;color:#666">${k}</span>
            </div>
            <span style="font-size:12px;font-weight:600;font-family:monospace;color:#1a1a1a">${v}</span>
        </div>`;
    }).join('');

    const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>PhishTrace Report</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',Arial,sans-serif;color:#1a1a1a;background:#fff;padding:32px;max-width:900px;margin:0 auto}
  .header{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;padding-bottom:16px;border-bottom:2px solid #f0f0f0}
  .logo-row{display:flex;align-items:center;gap:10px}
  .logo-icon{width:38px;height:38px;background:linear-gradient(135deg,#e63946,#a4161a);border-radius:10px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:18px}
  .logo-name{font-size:20px;font-weight:700}
  .logo-sub{font-size:10px;color:#aaa;margin-top:1px}
  .report-meta{font-size:11px;color:#aaa;text-align:right;line-height:1.8}
  .verdict-box{display:flex;align-items:center;gap:20px;padding:20px 24px;background:#fafafa;border:1.5px solid #eee;border-radius:14px;margin-bottom:20px}
  .score-ring{width:76px;height:76px;border-radius:50%;border:5px solid ${verdictColor};display:flex;align-items:center;justify-content:center;flex-shrink:0}
  .score-num{font-size:22px;font-weight:700;color:${verdictColor}}
  .verdict-name{font-size:22px;font-weight:700;color:${verdictColor};margin-bottom:4px}
  .verdict-url{font-size:11px;font-family:monospace;color:#555;word-break:break-all;margin-bottom:8px}
  .pills{display:flex;gap:6px;flex-wrap:wrap}
  .pill{font-size:10px;padding:2px 10px;border-radius:10px;background:#f0f0f0;color:#555}
  .pill-warn{background:rgba(233,196,106,0.12);color:#a07800;border:1px solid rgba(233,196,106,0.3)}
  .section{margin-bottom:22px}
  .sec-title{font-size:13px;font-weight:600;color:#333;padding-bottom:8px;border-bottom:1px solid #f0f0f0;margin-bottom:12px}
  table{width:100%;border-collapse:collapse}
  th{text-align:left;padding:7px 10px;font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#aaa;font-weight:600;border-bottom:1px solid #eee;background:#fafafa}
  .feat-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
  .footer{margin-top:24px;padding-top:14px;border-top:1px solid #f0f0f0;font-size:10px;color:#bbb;text-align:center;line-height:1.8}
  @media print{body{padding:16px}}
</style>
</head>
<body>

<div class="header">
  <div class="logo-row">
    <div class="logo-icon">🛡</div>
    <div>
      <div class="logo-name">PhishTrace</div>
      <div class="logo-sub">AI-Powered Phishing Detection & Campaign Analysis</div>
    </div>
  </div>
  <div class="report-meta">
    <div>Generated: ${now}</div>
    ${d.scan_id ? `<div>Scan ID: #${d.scan_id}</div>` : ''}
    <div>Model: Random Forest &nbsp;·&nbsp; Threshold: 0.75 / 0.45</div>
  </div>
</div>

<div class="verdict-box">
  <div class="score-ring"><div class="score-num">${score}%</div></div>
  <div style="flex:1">
    <div class="verdict-name">${verdictText}</div>
    <div class="verdict-url">${d.url}</div>
    <div class="pills">
      <span class="pill">Risk Score: ${score}%</span>
      <span class="pill">SHAP Explainability</span>
      <span class="pill">Random Forest</span>
      ${d.campaign_id ? `<span class="pill pill-warn">⚑ ${d.campaign_id}</span>` : ''}
    </div>
  </div>
</div>

<div class="section">
  <div class="sec-title">📊 Decision Explanation (SHAP) — Top ${reasons.length} Features</div>
  <table>
    <thead><tr><th style="width:160px">Feature</th><th>Why it matters</th><th style="width:140px">Impact</th><th style="width:90px;text-align:right">SHAP Value</th></tr></thead>
    <tbody>${shapRows}</tbody>
  </table>
</div>

<div class="section">
  <div class="sec-title">🔬 Extracted Features (${featEntries.length} total)</div>
  <div class="feat-grid">${featCards}</div>
</div>

<div class="footer">
  PhishTrace &nbsp;·&nbsp; Graduation Project &nbsp;·&nbsp; Faculty of Engineering<br>
  Random Forest &nbsp;·&nbsp; F1 = 96.86% &nbsp;·&nbsp; AUC = 98.17% &nbsp;·&nbsp; Test Set: 15,877 samples
</div>

</body></html>`;

    // Direct download as HTML file (no print dialog)
    const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    const safeName = (d.url || 'report').replace(/[^a-zA-Z0-9]/g, '_').slice(0, 40);
    a.href     = url;
    a.download = `PhishTrace_${safeName}_${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Report downloaded', 'success');
}


function renderFeatures(f) {
    const grid = document.getElementById('featuresGrid');
    grid.innerHTML = '';
    const defs = [
        { k: 'url_length',           label: 'URL Length',        icon: 'fa-ruler-horizontal',     fmt: v => v + ' chars',  bad: v => v > 75 },
        { k: 'has_ip',               label: 'IP Address',        icon: 'fa-server',               fmt: v => v ? 'Yes':'No', bad: v => !!v },
        { k: 'num_subdomains',       label: 'Subdomains',        icon: 'fa-layer-group',          fmt: v => v,              bad: v => v >= 3 },
        { k: 'has_https',            label: 'HTTPS',             icon: 'fa-lock',                 fmt: v => v ? 'Yes':'No', bad: v => !v },
        { k: 'has_suspicious_word',  label: 'Suspicious Word',   icon: 'fa-key',                  fmt: v => v ? 'Yes':'No', bad: v => !!v },
        { k: 'num_hyphens',          label: 'Hyphens',           icon: 'fa-minus',                fmt: v => v,              bad: v => v >= 2 },
        { k: 'num_dots',             label: 'Dots',              icon: 'fa-ellipsis',             fmt: v => v,              bad: v => v >= 4 },
        { k: 'path_length',          label: 'Path Length',       icon: 'fa-folder-open',          fmt: v => v + ' chars',   bad: v => v > 50 },
        { k: 'num_digits',           label: 'Digits in URL',     icon: 'fa-hashtag',              fmt: v => v,              bad: v => v > 10 },
        { k: 'has_at_in_url',        label: '@ in URL',          icon: 'fa-at',                   fmt: v => v ? 'Yes':'No', bad: v => !!v },
        { k: 'double_slash',         label: 'Double Slash',      icon: 'fa-slash',                fmt: v => v ? 'Yes':'No', bad: v => !!v },
        { k: 'num_suspicious_words', label: 'Suspicious Count',  icon: 'fa-triangle-exclamation', fmt: v => v,              bad: v => v >= 2 },
    ];
    defs.forEach(def => {
        const val = f[def.k];
        if (val === undefined || val === null) return;
        const flagged = def.bad(val);
        const chip    = document.createElement('div');
        chip.className = `feature-chip ${flagged ? 'flagged' : 'clean'}`;
        chip.innerHTML = `
            <i class="fa-solid ${def.icon} text-[11px] ${flagged ? 'text-[#e63946]' : 'text-[#2ec4b6]'}"></i>
            <span class="text-[11px] text-gray-500">${def.label}:</span>
            <span class="text-[12px] font-semibold">${def.fmt(val)}</span>`;
        grid.appendChild(chip);
    });
    if (!grid.children.length) {
        grid.innerHTML = '<p class="col-span-full text-sm text-gray-500 text-center py-4">No features available</p>';
    }
}

function renderExplanation(reasons, score, label) {
    document.getElementById('explainSummary').textContent =
        `Top factors influencing the "${label}" classification (score: ${Math.round(score * 100)}%):`;

    const list = document.getElementById('contributionsList');
    list.innerHTML = '';
    if (!reasons.length) {
        list.innerHTML = '<p class="text-sm text-gray-500 text-center py-4">No explanation data</p>';
        return;
    }
    const maxAbs = Math.max(...reasons.map(r => Math.abs(r.contribution)), 0.001);

    reasons.forEach(item => {
        const abs      = Math.abs(item.contribution);
        const isDanger = item.contribution > 0;
        const barColor = isDanger ? '#e63946' : '#2ec4b6';
        const barW     = Math.min((abs / maxAbs) * 100, 100);

        // Use Arabic text from API if available, fallback to feature name
        const explanationText = item.text_en || item.feature;

        const row = document.createElement('div');
        row.className = 'flex items-center gap-4';
        row.innerHTML = `
            <div class="w-52 sm:w-64 text-left flex-shrink-0">
                <div class="text-sm font-semibold font-mono">${item.feature}</div>
                <div class="text-[11px] text-gray-500 leading-tight mt-0.5">${explanationText}</div>
            </div>
            <div class="flex-1">
                <div class="contrib-bar">
                    <div class="fill" style="background:${barColor};opacity:0.65;" data-w="${barW}"></div>
                </div>
            </div>
            <div class="w-20 text-right flex-shrink-0">
                <span class="text-sm font-mono font-bold" style="color:${barColor}">
                    ${isDanger ? '+' : ''}${item.contribution.toFixed(4)}
                </span>
            </div>`;
        list.appendChild(row);
    });

    setTimeout(() => {
        list.querySelectorAll('.fill').forEach(f => { f.style.width = f.dataset.w + '%'; });
    }, 180);
}

function renderDetails(d) {
    const grid  = document.getElementById('detailsGrid');
    grid.innerHTML = '';
    const feats = d.features || {};
    const items = [
        { label: 'Full URL',       icon: 'fa-link',         val: d.url },
        { label: 'Scan ID',        icon: 'fa-fingerprint',  val: d.scan_id ? '#' + d.scan_id : null },
        { label: 'Campaign',       icon: 'fa-diagram-project', val: d.campaign_id || 'Not linked' },
        { label: 'URL Length',     icon: 'fa-ruler',        val: feats.url_length + ' characters' },
        { label: 'Has HTTPS',      icon: 'fa-lock',         val: feats.has_https ? 'Yes' : 'No' },
        { label: 'Has IP Host',    icon: 'fa-server',       val: feats.has_ip ? 'Yes' : 'No' },
        { label: 'Subdomains',     icon: 'fa-layer-group',  val: feats.num_subdomains },
        { label: 'Path Length',    icon: 'fa-folder',       val: feats.path_length + ' characters' },
        { label: 'Digit Count',    icon: 'fa-hashtag',      val: feats.num_digits },
        { label: 'Hyphen Count',   icon: 'fa-minus',        val: feats.num_hyphens },
    ];
    items.forEach(item => {
        if (item.val === undefined || item.val === null) return;
        const div = document.createElement('div');
        div.className = 'p-4 rounded-xl bg-white/[0.018] border border-white/[0.04]';
        div.innerHTML = `
            <div class="flex items-center gap-2 mb-2">
                <i class="fa-solid ${item.icon} text-[11px] text-gray-500"></i>
                <span class="text-[11px] text-gray-500 font-semibold uppercase tracking-wider">${item.label}</span>
            </div>
            <div class="text-sm font-mono break-all text-gray-300">${item.val}</div>`;
        grid.appendChild(div);
    });
}

// ── Dashboard ────────────────────────────────────────────
async function loadDashboard() {
    try {
        // Fetch dashboard stats + recent scans in parallel
        const [dashRes, histRes] = await Promise.all([
            fetch(`${API_BASE}${ENDPOINTS.dashboard}`),
            fetch(`${API_BASE}${ENDPOINTS.history}?limit=8`),
        ]);
        if (!dashRes.ok) throw new Error();
        const dashData = await dashRes.json();
        const histData = histRes.ok ? await histRes.json() : {};
        dashData.recent_scans = histData.scans || [];
        renderDashboard(dashData);
        setConn(true);
    } catch (e) {
        console.error('Dashboard fetch failed:', e);
        setConn(false);
    }
}

function renderDashboard(d) {
    // ── Scan counts ──────────────────────────────────────
    document.getElementById('statTotal').textContent  = (d.total_scans ?? 0).toLocaleString();
    document.getElementById('statDanger').textContent = (d.dangerous   ?? 0).toLocaleString();
    document.getElementById('statSusp').textContent   = (d.suspicious  ?? 0).toLocaleString();
    document.getElementById('statSafe').textContent   = (d.safe        ?? 0).toLocaleString();

    // ── Model metrics (from evaluation_results.json via DB seed) ─
    const m   = d.model_metrics || {};
    const fmt = v => (v != null && v > 0) ? (v * 100).toFixed(1) + '%' : '—';
    document.getElementById('mAcc').textContent  = fmt(m.accuracy);
    document.getElementById('mPrec').textContent = fmt(m.precision);
    document.getElementById('mRec').textContent  = fmt(m.recall);
    document.getElementById('mF1').textContent   = fmt(m.f1);
    document.getElementById('mFPR').textContent  = fmt(m.fpr);

    // ── Recent scans table (replaces Timeline + False Positives) ─
    renderRecentScans(d.recent_scans || []);
}

function renderRecentScans(scans) {
    const box = document.getElementById('timelineBox');
    if (!box) return;

    if (!scans.length) {
        box.innerHTML = `<div class="empty-state py-8">
            <i class="fa-solid fa-inbox opacity-20 text-3xl mb-3"></i>
            <p class="text-sm text-gray-500">No scans yet — analyze a URL to see activity</p>
        </div>`;
        return;
    }

    const styles = {
        Safe:       'bg-[rgba(46,196,182,0.08)] text-[#8eddd5]',
        Suspicious: 'bg-[rgba(233,196,106,0.08)] text-[#e9c46a]',
        Dangerous:  'bg-[rgba(230,57,70,0.08)] text-[#f0a0a8]',
    };

    box.innerHTML = `
        <div class="overflow-x-auto">
            <table class="data-table">
                <thead><tr><th>URL</th><th>Result</th><th>Score</th><th>Time</th></tr></thead>
                <tbody>
                    ${scans.slice(0, 8).map(s => `<tr>
                        <td class="font-mono text-xs text-gray-400 max-w-xs truncate" title="${s.url}">${s.url}</td>
                        <td><span class="text-xs px-2.5 py-1 rounded-lg font-semibold ${styles[s.verdict] || ''}">${s.verdict ?? '—'}</span></td>
                        <td class="font-mono text-sm">${s.score != null ? Math.round(s.score * 100) + '%' : '—'}</td>
                        <td class="text-xs text-gray-500">${s.timestamp ? s.timestamp.slice(0, 16).replace('T', ' ') : '—'}</td>
                    </tr>`).join('')}
                </tbody>
            </table>
        </div>`;
}

// ── Campaigns ────────────────────────────────────────────
async function loadCampaigns() {
    const list = document.getElementById('campaignsList');
    list.innerHTML = `<div class="empty-state py-10"><i class="fa-solid fa-spinner fa-spin text-2xl mb-3"></i><p class="text-sm">Loading campaigns...</p></div>`;
    try {
        const res = await fetch(`${API_BASE}${ENDPOINTS.campaigns}`);
        if (!res.ok) throw new Error();
        renderCampaigns(await res.json());
    } catch (e) {
        console.error('Campaigns fetch failed:', e);
        list.innerHTML = `<div class="empty-state"><i class="fa-solid fa-triangle-exclamation"></i><p class="text-sm">Failed to load campaigns</p></div>`;
    }
}

function renderCampaigns(d) {
    const list = document.getElementById('campaignsList');

    // Campaigns page shows ONLY campaigns found from the user's own scans
    // (URLs linked to a campaign_id during analysis)
    // We fetch from /api/history and group by campaign_id
    fetch(`${API_BASE}${ENDPOINTS.history}?limit=200`)
        .then(r => r.json())
        .then(histData => renderCampaignsFromHistory(list, histData.scans || []))
        .catch(() => {
            list.innerHTML = `<div class="empty-state">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <p class="text-sm">Failed to load scan data</p>
            </div>`;
        });
}

function renderCampaignsFromHistory(list, scans) {
    // Group scans by campaign_id (only Dangerous/Suspicious with a campaign)
    const grouped = {};
    scans.forEach(s => {
        if (!s.campaign_id) return;
        if (!grouped[s.campaign_id]) grouped[s.campaign_id] = [];
        grouped[s.campaign_id].push(s);
    });

    const campaignNames = Object.keys(grouped);

    if (!campaignNames.length) {
        list.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-shield-halved opacity-20"></i>
                <p class="text-sm text-gray-400 mt-2">No phishing campaigns detected in your scans</p>
                <p class="text-[11px] text-gray-600 mt-1">
                    When you scan phishing URLs, the system will automatically<br>
                    group similar ones into campaigns here.
                </p>
            </div>`;
        return;
    }

    // Sort by count descending
    campaignNames.sort((a, b) => grouped[b].length - grouped[a].length);
    const total = campaignNames.reduce((s, k) => s + grouped[k].length, 0);

    list.innerHTML = `
        <div class="flex items-center gap-6 mb-6 px-1">
            <div class="text-sm text-gray-400">
                <span class="text-white font-bold text-lg">${campaignNames.length}</span> campaigns detected in your scans
            </div>
            <div class="text-sm text-gray-400">
                <span class="text-white font-bold text-lg">${total}</span> phishing URLs linked
            </div>
        </div>
        <div id="campCards"></div>`;

    const container = document.getElementById('campCards');

    campaignNames.forEach(campName => {
        const campScans = grouped[campName];
        const count     = campScans.length;
        const maxCount  = grouped[campaignNames[0]].length || 1;
        const barPct    = Math.round((count / maxCount) * 100);
        const lastSeen  = campScans[0]?.timestamp?.slice(0, 16).replace('T', ' ') || '—';

        // Detect common features across URLs in this campaign
        const tags = [];
        const allHasIp   = campScans.every(s => s.features?.has_ip);
        const allSuspWord = campScans.some(s => s.features?.has_suspicious_word);
        const allSuspTld  = campScans.some(s => s.features?.tld_suspicious);
        if (allHasIp)    tags.push({ label: 'IP-based',        color: '#e63946' });
        if (allSuspWord) tags.push({ label: 'suspicious-words', color: '#e9c46a' });
        if (allSuspTld)  tags.push({ label: 'suspicious-TLD',   color: '#e9c46a' });

        const tagsHtml = tags.map(t =>
            `<span class="text-[10px] px-2 py-0.5 rounded-md border mr-1.5 mt-1" style="color:${t.color};border-color:${t.color}44">${t.label}</span>`
        ).join('');

        const card = document.createElement('div');
        card.className = 'campaign-card mb-4';
        card.innerHTML = `
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-2" onclick="toggleCampDetail(this.closest('.campaign-card'))">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-xl bg-[rgba(233,196,106,0.08)] flex items-center justify-center flex-shrink-0">
                        <i class="fa-solid fa-bullseye text-[#e9c46a] text-sm"></i>
                    </div>
                    <div>
                        <div class="font-bold text-sm font-mono">${campName}</div>
                        <div class="text-[11px] text-gray-500">${count} URL${count > 1 ? 's' : ''} · Last seen ${lastSeen}</div>
                    </div>
                </div>
                <div class="flex items-center gap-3">
                    <span class="text-[#e9c46a] font-bold font-mono text-sm">${count} URL${count > 1 ? 's' : ''}</span>
                    <i class="fa-solid fa-chevron-down text-gray-600 text-[10px] transition-transform duration-300 chev-icon"></i>
                </div>
            </div>
            ${tagsHtml ? `<div class="flex flex-wrap mt-1 mb-2">${tagsHtml}</div>` : ''}
            <div class="h-1 rounded-full bg-white/[0.04] overflow-hidden mb-1">
                <div class="h-full rounded-full bg-[rgba(233,196,106,0.45)] transition-all duration-700" style="width:${barPct}%"></div>
            </div>
            <div class="camp-detail hidden mt-4 pt-4 border-t border-white/[0.04]">
                <div class="text-[11px] text-gray-500 uppercase tracking-wider font-semibold mb-3">Scanned URLs in this campaign</div>
                ${campScans.slice(0, 5).map(s => `
                    <div class="flex items-center justify-between py-2 border-b border-white/[0.03] gap-3">
                        <span class="font-mono text-xs text-gray-400 truncate flex-1" title="${s.url}">${s.url}</span>
                        <span class="text-xs font-mono font-bold ${s.verdict === 'Dangerous' ? 'text-[#f0a0a8]' : 'text-[#e9c46a]'} flex-shrink-0">${Math.round((s.score||0)*100)}%</span>
                    </div>`).join('')}
                ${campScans.length > 5 ? `<p class="text-[11px] text-gray-600 mt-2">+${campScans.length - 5} more URLs</p>` : ''}
            </div>`;
        container.appendChild(card);
    });
}

function toggleCampDetail(card) {
    const det  = card.querySelector('.camp-detail');
    const chev = card.querySelector('.chev-icon');
    if (!det) return;
    const isHidden = det.classList.contains('hidden');
    det.classList.toggle('hidden');
    if (chev) chev.style.transform = isHidden ? 'rotate(180deg)' : '';
}

// ── History ──────────────────────────────────────────────
async function loadHistory() {
    const wrap = document.getElementById('historyTableWrap');
    wrap.innerHTML = `<div class="empty-state py-10"><i class="fa-solid fa-spinner fa-spin text-2xl mb-3"></i><p class="text-sm">Loading history...</p></div>`;
    try {
        const res = await fetch(`${API_BASE}${ENDPOINTS.history}?limit=50`);
        if (!res.ok) throw new Error();
        renderHistory(await res.json());
    } catch (e) {
        console.error('History fetch failed:', e);
        wrap.innerHTML = `<div class="empty-state"><i class="fa-solid fa-triangle-exclamation"></i><p class="text-sm">Failed to load history</p></div>`;
    }
}

function renderHistory(d) {
    const wrap  = document.getElementById('historyTableWrap');
    // API returns: scans[] with fields: id, url, score, verdict, timestamp, campaign_id
    const scans = d.scans || [];

    if (!scans.length) {
        wrap.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-inbox"></i>
                <p class="text-sm">History is empty</p>
                <p class="text-[11px] text-gray-600 mt-1">Scan a URL and it will appear here</p>
            </div>`;
        return;
    }

    const styles = {
        Safe:       'bg-[rgba(46,196,182,0.08)] text-[#8eddd5]',
        Suspicious: 'bg-[rgba(233,196,106,0.08)] text-[#e9c46a]',
        Dangerous:  'bg-[rgba(230,57,70,0.08)] text-[#f0a0a8]',
    };

    wrap.innerHTML = `
        <div class="overflow-x-auto">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>URL</th>
                        <th>Result</th>
                        <th>Score</th>
                        <th>Campaign</th>
                        <th>Time</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    ${scans.map(s => `
                    <tr>
                        <td class="text-[11px] text-gray-600 font-mono">${s.id ?? '—'}</td>
                        <td class="font-mono text-xs text-gray-400 max-w-xs truncate" title="${s.url}">${s.url}</td>
                        <td>
                            <span class="text-xs px-2.5 py-1 rounded-lg font-semibold ${styles[s.verdict] || ''}">
                                ${s.verdict ?? '—'}
                            </span>
                        </td>
                        <td class="font-mono text-sm">${s.score != null ? Math.round(s.score * 100) + '%' : '—'}</td>
                        <td class="text-[11px] font-mono text-gray-500">${s.campaign_id || '—'}</td>
                        <td class="text-xs text-gray-500">${s.timestamp ? s.timestamp.slice(0, 16).replace('T', ' ') : '—'}</td>
                        <td>
                            <button
                                class="text-gray-500 hover:text-white transition-colors px-2 py-1 rounded hover:bg-white/5"
                                onclick="reAnalyze('${s.url.replace(/'/g, "\\'")}')"
                                title="Re-analyze">
                                <i class="fa-solid fa-arrows-rotate text-xs"></i>
                            </button>
                        </td>
                    </tr>`).join('')}
                </tbody>
            </table>
        </div>
        <div class="px-4 py-3 border-t border-white/[0.04] text-[11px] text-gray-600">
            Showing ${scans.length} most recent scans
        </div>`;
}

function reAnalyze(url) {
    document.getElementById('urlInput').value = url;
    navigateTo('scanner');
    analyzeURL();
}

// ── Init ─────────────────────────────────────────────────
async function checkConn() {
    try {
        await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        setConn(true);
    } catch {
        setConn(false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    checkConn();
    document.getElementById('urlInput').addEventListener('keydown', e => {
        if (e.key === 'Enter') analyzeURL();
    });
    document.getElementById('urlInput').focus();
});