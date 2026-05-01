const API_BASE = 'http://127.0.0.1:5000';

const ENDPOINTS = {
    analyze:   '/analyze',
    dashboard: '/api/dashboard',
    campaigns: '/api/campaigns',
    history:   '/api/history',
};

// ── Toast ────────────────────────────────────────────────
function showToast(msg, type = 'info') {
    const wrap = document.getElementById('toastWrap');
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    const icons = { error: 'fa-circle-xmark', success: 'fa-circle-check', info: 'fa-circle-info' };
    t.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}"></i><span>${msg}</span>`;
    wrap.appendChild(t);
    setTimeout(() => t.remove(), 3700);
}

// ── Nav ──────────────────────────────────────────────────
function toggleMobile() { document.getElementById('mobileNav').classList.toggle('hidden'); }
function clearInput() { const inp = document.getElementById('urlInput'); inp.value = ''; inp.focus(); }

function setConn(ok) {
    document.getElementById('connDot').className = ok
        ? 'w-[7px] h-[7px] rounded-full bg-[#2ec4b6] animate-pulse'
        : 'w-[7px] h-[7px] rounded-full bg-[#e63946]';
    const txt = document.getElementById('connText');
    txt.textContent = ok ? 'Connected' : 'Disconnected';
    txt.className = ok
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

    const btn    = document.getElementById('analyzeBtn');
    const btnTxt = document.getElementById('analyzeBtnText');
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
        if (err.message.includes('fetch') || err.message.includes('NetworkError')) {
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
    const label = d.verdict ?? 'Unknown';

    // Score ring
    const arc   = document.getElementById('scoreArc');
    const circ  = 2 * Math.PI * 80;
    let color   = '#2ec4b6';
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

    // Badge
    const map = {
        Safe:       { text: 'Likely Safe',  icon: 'fa-circle-check',        cls: 'bg-[rgba(46,196,182,0.1)] text-[#8eddd5] border border-[rgba(46,196,182,0.2)]' },
        Suspicious: { text: 'Suspicious',   icon: 'fa-triangle-exclamation', cls: 'bg-[rgba(233,196,106,0.1)] text-[#e9c46a] border border-[rgba(233,196,106,0.2)]' },
        Dangerous:  { text: 'Dangerous',    icon: 'fa-skull-crossbones',     cls: 'bg-[rgba(230,57,70,0.1)] text-[#f0a0a8] border border-[rgba(230,57,70,0.2)]' },
    };
    const cfg = map[label] || map.Suspicious;
    document.getElementById('resultBadge').className = `inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold ${cfg.cls}`;
    document.getElementById('resultIcon').className  = `fa-solid ${cfg.icon}`;
    document.getElementById('resultLabel').textContent = cfg.text;
    document.getElementById('resultCard').classList.toggle('danger-pulse', label === 'Dangerous');
    document.getElementById('resultSummary').textContent = `The URL scored ${pct}% risk and was classified as "${cfg.text}".`;
    document.getElementById('resultModel').innerHTML     = `<i class="fa-solid fa-brain mr-1"></i>Model: Random Forest`;
    document.getElementById('resultTime').innerHTML      = `<i class="fa-solid fa-stopwatch mr-1"></i>Time: ${d._elapsed ?? '—'}ms`;
    document.getElementById('resultThreshold').innerHTML = `<i class="fa-solid fa-sliders mr-1"></i>Threshold: 0.75 / 0.45`;

    renderFeatures(d.features || {});
    renderExplanation(d.reasons || [], score, label);
    renderDetails(d);
}

function renderFeatures(f) {
    const grid = document.getElementById('featuresGrid');
    grid.innerHTML = '';
    const defs = [
        { k: 'url_length',          label: 'URL Length',       icon: 'fa-ruler-horizontal', fmt: v => v + ' chars',  bad: v => v > 75 },
        { k: 'has_ip',              label: 'IP Address',       icon: 'fa-server',           fmt: v => v ? 'Yes':'No', bad: v => !!v },
        { k: 'num_subdomains',      label: 'Subdomains',       icon: 'fa-layer-group',      fmt: v => v,              bad: v => v >= 3 },
        { k: 'has_https',           label: 'HTTPS',            icon: 'fa-lock',             fmt: v => v ? 'Yes':'No', bad: v => !v },
        { k: 'has_suspicious_word', label: 'Suspicious Word',  icon: 'fa-key',              fmt: v => v ? 'Yes':'No', bad: v => !!v },
        { k: 'num_hyphens',         label: 'Hyphens',          icon: 'fa-minus',            fmt: v => v,              bad: v => v >= 2 },
        { k: 'num_dots',            label: 'Dots',             icon: 'fa-ellipsis',         fmt: v => v,              bad: v => v >= 4 },
        { k: 'path_length',         label: 'Path Length',      icon: 'fa-folder-open',      fmt: v => v + ' chars',   bad: v => v > 50 },
        { k: 'num_digits',          label: 'Digits in URL',    icon: 'fa-hashtag',          fmt: v => v,              bad: v => v > 10 },
        { k: 'has_at_in_url',       label: '@ in URL',         icon: 'fa-at',               fmt: v => v ? 'Yes':'No', bad: v => !!v },
        { k: 'double_slash',        label: 'Double Slash',     icon: 'fa-slash',            fmt: v => v ? 'Yes':'No', bad: v => !!v },
        { k: 'num_suspicious_words',label: 'Suspicious Count', icon: 'fa-triangle-exclamation', fmt: v => v,          bad: v => v >= 2 },
    ];
    defs.forEach(def => {
        const val = f[def.k];
        if (val === undefined || val === null) return;
        const flagged = def.bad(val);
        const chip = document.createElement('div');
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
    const summaryEl = document.getElementById('explainSummary');
    summaryEl.textContent = `Top factors influencing the "${label}" classification (score: ${Math.round(score*100)}%):`;
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
        const row = document.createElement('div');
        row.className = 'flex items-center gap-4';
        row.innerHTML = `
            <div class="w-44 sm:w-56 text-left flex-shrink-0">
                <div class="text-sm font-semibold">${item.feature}</div>
                <div class="text-[11px] text-gray-500">${item.direction === 'increases' ? '↑ increases risk' : '↓ decreases risk'}</div>
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
    const grid = document.getElementById('detailsGrid');
    grid.innerHTML = '';
    const feats = d.features || {};
    const items = [
        { label: 'Full URL',      icon: 'fa-link',      val: d.url },
        { label: 'URL Length',    icon: 'fa-ruler',     val: feats.url_length + ' characters' },
        { label: 'Has HTTPS',     icon: 'fa-lock',      val: feats.has_https ? 'Yes ✅' : 'No ⚠️' },
        { label: 'Has IP Host',   icon: 'fa-server',    val: feats.has_ip ? 'Yes ⚠️' : 'No ✅' },
        { label: 'Subdomains',    icon: 'fa-layer-group', val: feats.num_subdomains },
        { label: 'Path Length',   icon: 'fa-folder',    val: feats.path_length + ' characters' },
        { label: 'Digit Count',   icon: 'fa-hashtag',   val: feats.num_digits },
        { label: 'Hyphen Count',  icon: 'fa-minus',     val: feats.num_hyphens },
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
        const res = await fetch(`${API_BASE}${ENDPOINTS.dashboard}`);
        if (!res.ok) throw new Error();
        renderDashboard(await res.json());
    } catch (e) { console.error('Dashboard fetch failed:', e); }
}

function renderDashboard(d) {
    document.getElementById('statTotal').textContent  = d.total_scans ?? '—';
    document.getElementById('statDanger').textContent = d.dangerous ?? '—';
    document.getElementById('statSusp').textContent   = d.suspicious ?? '—';
    document.getElementById('statSafe').textContent   = d.safe ?? '—';
    const m = d.model_metrics || {};
    const fmt = v => v != null ? (v * 100).toFixed(1) + '%' : '—';
    document.getElementById('mAcc').textContent  = fmt(m.accuracy);
    document.getElementById('mPrec').textContent = fmt(m.precision);
    document.getElementById('mRec').textContent  = fmt(m.recall);
    document.getElementById('mF1').textContent   = fmt(m.f1);
    document.getElementById('mFPR').textContent  = fmt(m.fpr);
    if (d.timeline?.length) renderTimeline(d.timeline);
}

function renderTimeline(tl) {
    const box = document.getElementById('timelineBox');
    const mx  = Math.max(...tl.map(t => t.scans), 1);
    box.innerHTML = `<div class="flex items-end gap-1.5 h-36">${tl.map(t => {
        const h  = Math.max((t.scans / mx) * 100, 5);
        const dH = t.dangerous ? Math.max((t.dangerous / mx) * 100, 0) : 0;
        return `<div class="flex-1 flex flex-col items-center gap-1 group cursor-pointer" title="${t.date}: ${t.scans} scans">
            <span class="text-[10px] text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity">${t.scans}</span>
            <div class="w-full rounded-t-md bg-[rgba(230,57,70,0.15)] relative" style="height:${h}%">
                <div class="absolute bottom-0 w-full rounded-t-md bg-[rgba(230,57,70,0.6)]" style="height:${(dH/h)*100}%"></div>
            </div>
            <span class="text-[9px] text-gray-600">${t.date.slice(5)}</span>
        </div>`;
    }).join('')}</div>`;
}

// ── Campaigns ────────────────────────────────────────────
async function loadCampaigns() {
    try {
        const res = await fetch(`${API_BASE}${ENDPOINTS.campaigns}`);
        if (!res.ok) throw new Error();
        renderCampaigns(await res.json());
    } catch (e) { console.error('Campaigns fetch failed:', e); }
}

function renderCampaigns(d) {
    const list  = document.getElementById('campaignsList');
    const camps = d.campaigns || [];
    if (!camps.length) return;
    list.innerHTML = camps.map(c => `
        <div class="campaign-card mb-4" onclick="toggleCampDetail(this)">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-xl bg-[rgba(233,196,106,0.08)] flex items-center justify-center flex-shrink-0">
                        <i class="fa-solid fa-bullseye text-[#e9c46a] text-sm"></i>
                    </div>
                    <div>
                        <div class="font-bold text-sm">${c.name}</div>
                        <div class="text-[11px] text-gray-500">${c.urls_count} URLs</div>
                    </div>
                </div>
                <div class="flex items-center gap-4 text-xs">
                    <span class="text-[#e63946] font-bold font-mono">${c.avg_danger_score != null ? Math.round(c.avg_danger_score*100)+'%' : '—'}</span>
                    <i class="fa-solid fa-chevron-down text-gray-600 text-[10px] transition-transform duration-300"></i>
                </div>
            </div>
            <div class="camp-detail hidden mt-4 pt-4 border-t border-white/[0.04]">
                <div class="text-[11px] text-gray-500 mb-2 uppercase tracking-wider font-semibold">Sample URLs</div>
                ${(c.sample_urls || []).map(u => `<div class="text-xs font-mono text-gray-400 py-1.5 border-b border-white/[0.025] truncate">${u}</div>`).join('')}
            </div>
        </div>`).join('');
}

function toggleCampDetail(card) {
    const det  = card.querySelector('.camp-detail');
    const chev = card.querySelector('.fa-chevron-down');
    if (det) { det.classList.toggle('hidden'); chev.style.transform = det.classList.contains('hidden') ? '' : 'rotate(180deg)'; }
}

// ── History ──────────────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch(`${API_BASE}${ENDPOINTS.history}`);
        if (!res.ok) throw new Error();
        renderHistory(await res.json());
    } catch (e) { console.error('History fetch failed:', e); }
}

function renderHistory(d) {
    const wrap  = document.getElementById('historyTableWrap');
    const scans = d.scans || [];
    if (!scans.length) return;
    const styles = {
        Safe:       'bg-[rgba(46,196,182,0.08)] text-[#8eddd5]',
        Suspicious: 'bg-[rgba(233,196,106,0.08)] text-[#e9c46a]',
        Dangerous:  'bg-[rgba(230,57,70,0.08)] text-[#f0a0a8]'
    };
    wrap.innerHTML = `<div class="overflow-x-auto"><table class="data-table">
        <thead><tr><th>URL</th><th>Result</th><th>Score</th><th>Time</th><th></th></tr></thead>
        <tbody>${scans.map(s => `<tr>
            <td class="font-mono text-sm text-gray-400 max-w-xs truncate">${s.url}</td>
            <td><span class="text-xs px-2.5 py-1 rounded-lg font-semibold ${styles[s.label]||''}">${s.label}</span></td>
            <td class="font-mono text-sm">${s.score != null ? Math.round(s.score*100)+'%' : '—'}</td>
            <td class="text-xs text-gray-500">${s.timestamp || '—'}</td>
            <td><button class="text-gray-500 hover:text-white transition-colors" onclick="reAnalyze('${s.url.replace(/'/g,"\\'")}')"><i class="fa-solid fa-arrows-rotate text-xs"></i></button></td>
        </tr>`).join('')}</tbody></table></div>`;
}

function reAnalyze(url) { document.getElementById('urlInput').value = url; navigateTo('scanner'); analyzeURL(); }

// ── Init ─────────────────────────────────────────────────
async function checkConn() {
    try {
        await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        setConn(true);
    } catch { setConn(false); }
}

document.addEventListener('DOMContentLoaded', () => {
    checkConn();
    document.getElementById('urlInput').focus();
});