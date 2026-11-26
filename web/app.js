// Fitting Room front end. Plain ES module, no build step, same-origin API only.
const MOCK = new URLSearchParams(location.search).has('mock');
if (MOCK) await import('./mock-api.js');

/* ------------------------------------------------------------------ helpers */
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : '');
const num = (n) => String(Math.round(n * 10) / 10);
const imgUrl = (id) => (MOCK ? `${window.__MOCK_IMG__}/${Number(id)}.jpg` : `/images/${Number(id)}.jpg`);
const icons = () => window.lucide && window.lucide.createIcons();
const jsonReq = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = '';
    try { detail = (await r.json()).detail; } catch { /* no body */ }
    const err = new Error(`${r.status}`);
    err.status = r.status;
    err.detail = typeof detail === 'string' ? detail : '';
    throw err;
  }
  return r.status === 204 ? null : r.json();
}

// URLs that come from the API are only used when they stay on this origin.
function safeUrl(u) {
  try {
    const url = new URL(String(u), location.href);
    return url.origin === location.origin ? url.href : '';
  } catch { return ''; }
}

// Replace a region's markup, keeping keyboard focus on the same control.
function setHTML(el, html) {
  const a = document.activeElement;
  const fk = a && el.contains(a) ? a.dataset.fk : null;
  el.innerHTML = html;
  icons();
  if (fk) {
    const t = el.querySelector(`[data-fk="${CSS.escape(fk)}"]`);
    if (t) t.focus({ preventScroll: true });
  }
}

const raf = (() => {
  const pending = new Set();
  return (fn) => {
    if (pending.has(fn)) return;
    pending.add(fn);
    requestAnimationFrame(() => { pending.delete(fn); fn(); });
  };
})();

/* ------------------------------------------------------------------ state */
const S = {
  view: 'find',
  convId: null, title: '', messages: [],
  results: null, selected: null, sizes: {}, sizeState: 'idle', // idle | loading | error
  state: { wants: {}, profile: {} },
  busy: false, thinking: false,
  shortlist: [], items: {},
  conversations: [], tryons: [],
  note: '', noteWhere: 'grid', measureOpen: false, sizeMsg: '', status: null,
  photo: null, uploading: false, photoError: '',
  tryItem: null, job: null, viewSize: {},
};
const EXAMPLES = [
  'A shirt for a beach wedding in June',
  'Office shirts under $80',
  'A plain tee for hot weather',
  'Something smart for a job interview',
];
const MAX_SHORTLIST = 3; // table width only; the server owns the real limit

const profile = () => S.state.profile || {};
const canSize = () => !!((profile().height_cm && profile().weight_kg) || profile().chest_cm);
const inList = (id) => S.shortlist.some((s) => s.id === id);
const itemOf = (id) => S.items[id] || (S.results && S.results.items.find((i) => i.id === id)) || null;
// A size is only ever shown when it came from a size recommendation. Shortlist sizes count only
// once the client knows a profile, so a server default can never surface as "your size".
const validSize = (z) => (z && z.size && typeof z.chest_cm === 'number' ? z : null);
const sizeFor = (id) => validSize(S.sizes[id]) || (canSize() ? validSize((S.shortlist.find((s) => s.id === id) || {}).size) : null);
const NEED_SIZE = 'No size yet. ';
const needSizeBtn = '<button class="text" type="button" data-act="need-size" data-fk="need-size">Tell me your height and weight in Find.</button>';
const jobActive = () => S.job && (S.job.status === 'queued' || S.job.status === 'running');
const photoOK = () => !!(S.photo && S.photo.check && S.photo.check.ok);

function remember(items) { (items || []).forEach((it) => { if (it && it.id != null) S.items[it.id] = { ...(S.items[it.id] || {}), ...it }; }); }

/* ------------------------------------------------------------------ routing */
const VIEWS = ['find', 'try', 'list'];
function show(name) {
  if (!VIEWS.includes(name)) name = 'find';
  S.view = name;
  VIEWS.forEach((v) => { $('#v-' + v).hidden = v !== name; });
  document.querySelectorAll('.nav button').forEach((b) => {
    if (b.dataset.view === name) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
  });
  if (name === 'try') { ensureTryItem(); renderTry(); refreshTryons(); loadShortlist(); }
  if (name === 'list') { renderList(); loadShortlist(); }
  if (name === 'find') raf(() => { scrollThread(); if (S.focusMessage) { S.focusMessage = false; $('#message').focus(); } });
}
function go(name) {
  if (location.hash.slice(1) === name) show(name); else location.hash = name;
}
window.addEventListener('hashchange', () => show(location.hash.slice(1)));

/* ------------------------------------------------------------------ copy helpers */
const SIZE_NAME = { XXS: 'extra extra small', XS: 'extra small', S: 'small', M: 'medium', L: 'large', XL: 'extra large', XXL: 'extra extra large', XXXL: 'triple extra large' };
const sizeName = (s) => (SIZE_NAME[s] ? cap(SIZE_NAME[s]) : String(s));
const withA = (name) => (/^[AEIOU]/i.test(name) && name.length > 1 ? 'an ' : 'a ') + name;
const LENGTH_NOTE = { short: 'Sits above the hip', mid_hip: 'Ends mid-hip', long: 'Falls below the hip' };
const ISSUES = {
  no_person: "I can't see a person in this photo.",
  no_person_found: "I can't see a person in this photo.",
  multiple_people: 'There is more than one person in this photo. Use one with just you in it.',
  upper_body_not_visible: "Your upper body isn't fully in view. Try a photo from the waist up.",
  upper_body_not_found: "Your upper body isn't fully in view. Try a photo from the waist up.",
  low_resolution: 'This photo is small. A larger one will give a cleaner result.',
  too_small: 'This photo is small. A larger one will give a cleaner result.',
  blurry: 'This photo is blurry. A sharper one will work better.',
};
function issueSentence(raw) {
  const s = String(raw || '').trim();
  if (!s) return '';
  if (ISSUES[s]) return ISSUES[s];
  const t = /\s/.test(s) ? s : s.replace(/[_-]+/g, ' ');
  return /[.!?]$/.test(t) ? cap(t) : cap(t) + '.';
}
const STAGES = { queued: 'Waiting in line', running: 'Working on it' };
function stageText(job) {
  const raw = String(job.stage || '').trim();
  if (!raw || raw === job.status) return STAGES[job.status] || 'Working on it';
  const t = raw.replace(/[_-]+/g, ' ');
  return cap(t);
}
function rich(text) { return esc(text).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\n/g, '<br>'); }
function paragraphs(text) { return String(text || '').split(/\n{2,}/).map((p) => p.trim()).filter(Boolean); }

function sizeHeading(sz) {
  const n = Math.abs(sz.moved || 0);
  if (!n) return 'The size on the chart';
  const word = ['', 'One', 'Two', 'Three'][n] || String(n);
  return `${word} size${n > 1 ? 's' : ''} ${sz.moved > 0 ? 'up' : 'down'} from the chart`;
}
function sizeBecause(sz) {
  const p = profile();
  const out = [];
  if (sz.chest_estimated) {
    out.push(p.height_cm && p.weight_kg
      ? `At ${num(p.height_cm)} cm and ${num(p.weight_kg)} kg your chest is about ${Math.round(sz.chest_cm)} cm.`
      : `Your chest is about ${Math.round(sz.chest_cm)} cm.`);
  } else {
    out.push(`You gave your chest as ${num(sz.chest_cm)} cm.`);
  }
  const notes = sz.notes || [];
  if (notes.includes('below_chart')) out.push(`That is smaller than this chart goes, so ${sizeName(sz.size)} is the closest.`);
  else if (notes.includes('above_chart')) out.push(`That is bigger than this chart goes, so ${sizeName(sz.size)} is the closest.`);
  else if (typeof sz.chart_room_cm === 'number') out.push(`That is ${withA(sizeName(sz.chart_size))}, with ${num(sz.chart_room_cm)} cm of room.`);
  else out.push(`By the chart that is ${withA(sizeName(sz.chart_size))}.`);
  const room = sz.room_cm;
  const sameAsChart = !sz.moved && typeof sz.chart_room_cm === 'number' && !notes.some((n) => n.endsWith('_chart'));
  if (typeof room === 'number' && !sameAsChart) {
    let tail;
    if (room < 0) tail = `${sizeName(sz.size)} would be about ${num(-room)} cm tight at the chest.`;
    else {
      let why = '.';
      if (sz.moved > 0) why = notes.includes('one_up_for_loose_fit') ? ', which is the looser fit you asked for.' : ', a bit looser than the chart size.';
      else if (sz.moved < 0) why = notes.includes('one_down_for_slim_fit') ? ', which is the slimmer fit you asked for.' : ', a bit closer than the chart size.';
      tail = `${sizeName(sz.size)} gives you ${num(room)} cm${why.startsWith('.') ? ' of room.' : ' of room' + why}`;
    }
    out.push(tail);
  }
  if (notes.includes('near_size_boundary')) out.push("You're close to the edge between two sizes, so check the measurements below.");
  return out.join(' ');
}
function shoulderNote(sz) {
  const d = sz.shoulder_diff_cm;
  if (typeof d !== 'number') return '';
  if (d >= 0.5) return `${num(Math.round(d))} cm wider than yours`;
  if (d <= -0.5) return `${num(Math.round(-d))} cm narrower than yours`;
  return 'About as wide as yours';
}

/* ------------------------------------------------------------------ chat */
const thread = $('#thread');
const scrollThread = () => { thread.scrollTop = thread.scrollHeight; };
const nearBottom = () => thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80;

function threadHTML() {
  const out = [];
  const last = S.messages.length - 1;
  if (!S.messages.length) out.push('<p class="them">Tell me what you need a shirt for, and when you will wear it.</p>');
  S.messages.forEach((m, i) => {
    if (m.role === 'user') { out.push(`<div class="me">${esc(m.text)}</div>`); return; }
    if (m.kind === 'trace') { out.push(traceHTML(m, i)); return; }
    if (m.kind === 'ref') {
      const it = itemOf(m.item_id);
      if (!it) return;
      out.push(`<button class="ref" type="button" data-ref="${Number(it.id)}" data-fk="ref"><div class="th"><img src="${imgUrl(it.id)}" alt=""></div><div><b>See it on you</b><span>Uses one photo of you</span></div></button>`);
      return;
    }
    paragraphs(m.text).forEach((p) => out.push(`<p class="them${m.kind === 'note' ? ' quiet' : ''}">${rich(p)}</p>`));
    if (m.options && m.options.length) {
      const next = S.messages[i + 1];
      const chosen = next && next.role === 'user' ? String(next.text).trim().toLowerCase() : '';
      const live = i === last && !S.busy;
      out.push('<div class="opts">' + m.options.map((o, k) => {
        const on = chosen && String(o).trim().toLowerCase() === chosen;
        return `<button class="opt" type="button" data-opt="${i}:${k}" data-fk="opt-${i}-${k}" aria-pressed="${on ? 'true' : 'false'}"${live ? '' : ' disabled'}>${esc(o)}</button>`;
      }).join('') + '</div>');
    }
  });
  if (S.thinking) out.push('<p class="them quiet thinking">Thinking</p>');
  return out.join('');
}
function secs(ms) { return `${(ms / 1000).toFixed(1)} s`; }
function traceHTML(m, i) {
  const total = m.steps.reduce((t, s) => t + (s.status === 'done' ? Number(s.ms) || 0 : 0), 0);
  const head = m.live
    ? '<span class="trace-toggle live">Working</span>'
    : `<button class="trace-toggle" type="button" data-trace="${i}" data-fk="trace-${i}" aria-expanded="${!!m.open}">Worked for ${total < 100 ? 'under 0.1 s' : secs(total)}</button>`;
  const lines = (m.live || m.open) ? `<ol class="steps">${m.steps.map((s) => {
    const run = s.status !== 'done';
    const ms = !run && Number(s.ms) >= 50 ? `<span class="ms">${secs(Number(s.ms))}</span>` : '<span class="ms"></span>';
    return `<li class="step ${run ? 'run' : 'done'}"${run ? ` style="--d:-${Math.round(performance.now() % 2400)}ms"` : ''}><i data-lucide="${run ? 'loader' : 'check'}"></i><span class="txt"><span class="lab">${esc(s.label)}</span>${!run && s.detail ? `<span class="det">${esc(s.detail)}</span>` : ''}</span>${ms}</li>`;
  }).join('')}</ol>` : '';
  return `<div class="trace">${head}${lines}</div>`;
}
function renderThread(force) {
  const stick = force === true || nearBottom();
  setHTML(thread, threadHTML());
  if (stick) scrollThread();
  $('#send').disabled = S.busy;
}

function pushAssistant(text, extra = {}) { S.messages.push({ role: 'assistant', text, ...extra }); }

async function readSSE(body, onEvent) {
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  const flush = (block) => {
    let ev = 'message';
    const data = [];
    block.split(/\r?\n/).forEach((line) => {
      if (!line || line.startsWith(':')) return;
      const c = line.indexOf(':');
      const field = c < 0 ? line : line.slice(0, c);
      const val = c < 0 ? '' : line.slice(c + 1).replace(/^ /, '');
      if (field === 'event') ev = val; else if (field === 'data') data.push(val);
    });
    if (!data.length) return;
    let parsed = {};
    try { parsed = JSON.parse(data.join('\n')); } catch { return; }
    onEvent(ev, parsed);
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let m;
    while ((m = buf.match(/\r?\n\r?\n/))) {
      flush(buf.slice(0, m.index));
      buf = buf.slice(m.index + m[0].length);
    }
  }
  buf += dec.decode();
  if (buf.trim()) flush(buf);
}

function onChatEvent(ev, d, turn) {
  switch (ev) {
    case 'conversation':
      S.convId = d.id; S.title = d.title || '';
      break;
    case 'question':
      if (turn.silent) break;
      S.thinking = false; turn.text = null; turn.got = true;
      S.messages.push({ role: 'assistant', kind: 'question', text: d.text, options: Array.isArray(d.options) ? d.options : [] });
      raf(renderThread);
      break;
    case 'text':
      if (turn.silent) break;
      S.thinking = false; turn.got = true;
      if (!turn.text) { turn.text = { role: 'assistant', kind: 'text', text: '' }; S.messages.push(turn.text); }
      turn.text.text += d.delta || '';
      raf(renderThread);
      break;
    case 'results':
      S.thinking = false; turn.text = null; turn.got = true;
      d.items = Array.isArray(d.items) ? d.items : [];
      d.filters = Array.isArray(d.filters) ? d.filters : [];
      remember(d.items);
      S.results = d;
      S.selected = (d.items.find((i) => i.best) || d.items[0] || {}).id ?? null;
      S.note = '';
      raf(renderThread); renderRack();
      break;
    case 'size':
      turn.text = null; turn.got = true;
      S.sizes[d.item_id] = d;
      if (S.results && S.results.items.some((i) => i.id === d.item_id)) S.selected = d.item_id;
      turn.ref = d.item_id;
      renderRack();
      break;
    case 'step': {
      if (turn.silent) break;
      S.thinking = false; turn.stepSeen = true; clearTimeout(turn.timer);
      if (!turn.trace) {
        S.messages.forEach((x) => { if (x.kind === 'trace') x.open = false; });
        turn.trace = { role: 'assistant', kind: 'trace', steps: [], open: true, live: true };
        S.messages.push(turn.trace);
      }
      const steps = turn.trace.steps;
      let st = null;
      for (let k = steps.length - 1; k >= 0; k -= 1) {
        if (steps[k].id === d.id && (d.status === 'done' ? steps[k].status !== 'done' : true)) { st = steps[k]; break; }
      }
      if (!st) { st = { id: d.id, label: d.label, detail: null, ms: null, status: 'start' }; steps.push(st); }
      if (d.label) st.label = d.label;
      if (d.status === 'done') { st.status = 'done'; st.detail = d.detail || null; st.ms = d.ms; }
      raf(renderThread);
      break;
    }
    case 'state':
      S.state = { wants: d.wants || {}, profile: d.profile || {} };
      if (S.results) renderSizing();
      break;
    case 'done':
      S.thinking = false;
      break;
    default: break;
  }
}

async function send(raw, opts = {}) {
  const text = String(raw || '').trim();
  if (!text || S.busy) return;
  const silent = !!opts.removeFilter; // a filter removal updates the rack only
  if (!silent) {
    S.messages = S.messages.filter((m) => m.kind !== 'ref');
    S.messages.push({ role: 'user', text });
  }
  S.busy = true; S.thinking = false;
  renderThread(true);
  if (!S.results) renderRack();
  const turn = { text: null, got: false, ref: null, silent, trace: null, stepSeen: false, timer: 0 };
  // "Thinking" is only a fallback when no step has arrived after 600 ms
  if (!silent) turn.timer = setTimeout(() => { if (!turn.stepSeen && !turn.got && S.busy) { S.thinking = true; renderThread(); } }, 600);
  try {
    const body = { conversation_id: S.convId, message: text };
    if (opts.removeFilter) body.remove_filter = opts.removeFilter;
    const r = await fetch('/api/chat', jsonReq('POST', body));
    if (!r.ok) {
      let detail = '';
      try { detail = (await r.json()).detail; } catch { /* no body */ }
      if (r.status === 429 && typeof detail === 'string' && detail) {
        turn.got = true;
        S.messages.push({ role: 'assistant', kind: 'note', text: detail });
        return;
      }
      throw new Error('chat');
    }
    if (!r.body) throw new Error('chat');
    await readSSE(r.body, (ev, d) => onChatEvent(ev, d, turn));
    if (!turn.got) throw new Error('empty');
  } catch {
    if (!turn.got) pushAssistant("I couldn't get an answer just now. Send that again in a moment.");
  } finally {
    clearTimeout(turn.timer);
    S.busy = false; S.thinking = false;
    if (turn.trace) { turn.trace.live = false; turn.trace.steps.forEach((x) => { if (x.status !== 'done') { x.status = 'done'; x.ms = null; } }); }
    if (turn.ref != null) S.messages.push({ role: 'assistant', kind: 'ref', item_id: turn.ref });
    renderThread();
    if (S.selected != null && canSize() && !S.sizes[S.selected]) loadSize(S.selected);
    refreshHistory();
    if (S.view === 'find') $('#message').focus({ preventScroll: true });
  }
}

$('#composer').addEventListener('submit', (e) => {
  e.preventDefault();
  const input = $('#message');
  const v = input.value;
  if (!v.trim() || S.busy) return;
  input.value = '';
  send(v);
});

function newSearch() {
  Object.assign(S, { convId: null, title: '', messages: [], results: null, selected: null, sizes: {}, sizeState: 'idle', state: { wants: {}, profile: {} }, busy: false, thinking: false, note: '', measureOpen: false });
  renderThread(); renderRack(); renderHistory();
  go('find');
  $('#message').focus({ preventScroll: true });
}

async function openConversation(id) {
  try {
    const d = await api('/api/conversations/' + encodeURIComponent(id));
    S.convId = d.id; S.title = d.title || '';
    let skipReply = false; // filter removals are silent live, so hide them (and their reply) on restore too
    S.messages = (d.messages || []).flatMap((m) => {
      const text = m.text ?? m.content ?? '';
      if (m.role === 'user') {
        skipReply = /^Remove the ["“].*["”] filter\.$/.test(String(text).trim());
        return skipReply ? [] : [{ role: 'user', text }];
      }
      if (skipReply) { skipReply = false; return []; }
      const msg = { role: 'assistant', kind: m.options && m.options.length ? 'question' : 'text', text, options: m.options || [] };
      const steps = Array.isArray(m.steps) ? m.steps.filter((x) => x && x.label) : [];
      return steps.length
        ? [{ role: 'assistant', kind: 'trace', open: false, live: false, steps: steps.map((x) => ({ id: x.id, label: x.label, detail: x.detail || null, ms: x.ms, status: 'done' })) }, msg]
        : [msg];
    });
    S.results = d.results && Array.isArray(d.results.items) ? { ...d.results, filters: d.results.filters || [] } : null;
    if (S.results) remember(S.results.items);
    S.state = { wants: (d.state && d.state.wants) || {}, profile: (d.state && d.state.profile) || {} };
    S.sizes = {};
    S.sizeState = 'idle'; S.measureOpen = false; S.note = '';
    S.selected = S.results ? (S.results.items.find((i) => i.best) || S.results.items[0] || {}).id ?? null : null;
    if (d.size && d.size.item_id != null) {
      S.sizes[d.size.item_id] = d.size;
      if (S.results && S.results.items.some((i) => i.id === d.size.item_id)) S.selected = d.size.item_id;
      S.messages.push({ role: 'assistant', kind: 'ref', item_id: d.size.item_id });
    }
    S.busy = false; S.thinking = false;
    renderThread(true); renderRack(); renderHistory();
    go('find');
    if (S.selected != null && canSize() && !S.sizes[S.selected]) loadSize(S.selected);
  } catch {
    S.note = '';
  }
}

/* ------------------------------------------------------------------ rack */
function renderRack() { renderResults(); renderSizing(); }

function renderResults() {
  const el = $('#rack-top');
  const R = S.results;
  if (!R && S.messages.length) {
    setHTML(el, `<div class="empty-rack"><h1 style="font-size:26px;font-weight:600;letter-spacing:-0.035em;line-height:1.15">A couple of questions first</h1>
      <p>Your shirts will appear here once I know enough to choose.</p></div>`);
    return;
  }
  if (!R) {
    setHTML(el, `<div class="empty-rack"><h1 style="font-size:26px;font-weight:600;letter-spacing:-0.035em;line-height:1.15">What are you looking for?</h1>
      <p>Describe the occasion, the weather or a colour. I'll pick shirts from the catalog and work out your size.</p>
      <div class="opts">${EXAMPLES.map((t, i) => `<button class="opt" type="button" data-ex="${i}" data-fk="ex-${i}">${esc(t)}</button>`).join('')}</div></div>`);
    return;
  }
  const crit = R.filters.map((f, i) => `<button type="button" data-crit="${i}" data-fk="crit-${esc(f.key)}-${i}" aria-label="Remove filter: ${esc(f.label)}">${esc(f.label)}<i data-lucide="x"></i></button>`).join('');
  const cells = R.items.map((it) => {
    const on = it.id === S.selected;
    const liked = inList(it.id);
    return `<div class="cell">
      <button class="piece" type="button" data-pick="${Number(it.id)}" data-fk="piece-${Number(it.id)}" aria-pressed="${on}">
        <div class="frame"><img src="${imgUrl(it.id)}" alt="${esc(`${it.name} by ${it.brand}, $${it.price}`)}" loading="lazy">${it.exact === false ? '<span class="note close">Close match</span>' : it.best ? '<span class="note">Best match</span>' : ''}</div>
      </button>
      <button class="like" type="button" data-heart="${Number(it.id)}" data-fk="heart-${Number(it.id)}" aria-pressed="${liked}" aria-label="Shortlist ${esc(it.name)}"${liked ? ' data-on' : ''}><i data-lucide="heart"></i></button>
      <div class="cap"><span class="brand">${esc(it.brand)}</span><h3>${esc(it.name)}</h3><span class="price">$${esc(it.price)}</span>${it.exact === false && it.near ? `<p class="near">${esc(it.near)}</p>` : ''}${it.reason ? `<p class="why">${esc(it.reason)}</p>` : ''}</div>
    </div>`;
  }).join('');
  setHTML(el, `<div class="rack-head"><h1>${esc(R.title)}</h1></div>
    <div class="crit">${crit}</div>
    ${R.items.length ? `<div class="grid">${cells}</div>` : '<p class="quiet hint">Nothing matched. Try taking one of the filters off.</p>'}
    ${S.note && S.noteWhere === 'grid' ? `<p class="quiet note-line" role="status">${esc(S.note)}</p>` : ''}`);
}

function renderSizing() {
  const el = $('#sizing');
  const it = S.selected != null ? itemOf(S.selected) : null;
  if (!S.results || !it) { setHTML(el, ''); return; }
  const sz = S.sizes[it.id];
  if (!sz) {
    let msg;
    if (!canSize()) msg = `Tell me your height and weight in the chat and I'll size the ${esc(it.name.toLowerCase())} for you.`;
    else if (S.sizeState === 'error') msg = esc(S.sizeMsg) || "I couldn't work out a size for this one. Try entering a chest measurement.";
    else msg = 'Working out your size.';
    setHTML(el, `<div class="sizing-empty"><p class="quiet" role="status">${msg}</p></div>`);
    return;
  }
  const ranges = sz.ranges || [];
  const n = ranges.length || 1;
  const seg = 100 / n;
  const selIdx = Math.max(0, ranges.findIndex((r) => r.size === sz.size));
  let you = 0;
  if (ranges.length) {
    const c = sz.chest_cm;
    if (c <= ranges[0].lo) you = 0;
    else if (c >= ranges[n - 1].hi) you = 100;
    else {
      const k = ranges.findIndex((r, i) => c >= r.lo && (c < r.hi || i === n - 1));
      const r = ranges[Math.max(0, k)];
      you = seg * (Math.max(0, k) + (c - r.lo) / (r.hi - r.lo));
    }
  }
  const youAt = Math.min(98, Math.max(2, you));
  const ticks = Array.from({ length: n + 1 }, (_, i) => `<div class="tick" style="left:${num(i * seg)}%"></div>`).join('');
  const labs = ranges.map((r, i) => `<span class="lab${i === selIdx ? ' sel' : ''}" style="left:${num((i + 0.5) * seg)}%"><b>${esc(r.size)}</b>${esc(r.lo)} to ${esc(r.hi)}${i === 0 ? ' cm' : ''}</span>`).join('');
  const g = sz.garment || {};
  const fit = profile().fit || 'regular';
  const liked = inList(it.id);
  const sho = shoulderNote(sz);
  const lenNote = LENGTH_NOTE[sz.length_note] || '';
  let chestLine;
  if (sz.chest_estimated) chestLine = 'Your chest is estimated from height and weight. ';
  else chestLine = 'This uses the chest measurement you gave. ';
  const measure = S.measureOpen
    ? `<form class="measure" id="measure-form"><label class="sr-only" for="chest-input">Chest in centimetres</label><input id="chest-input" type="number" inputmode="decimal" min="60" max="150" step="0.5" required placeholder="Chest, cm" value="${profile().chest_cm ? esc(profile().chest_cm) : ''}"><button class="btn sm solid" type="submit">Update</button><button class="text" type="button" data-act="measure-cancel" data-fk="measure-cancel">Cancel</button></form>`
    : `<button class="text" type="button" data-act="measure" data-fk="measure">${sz.chest_estimated ? 'Enter a measurement' : 'Change it'}</button>`;
  setHTML(el, `<div class="sizing" aria-label="Your size">
    <div>
      <div class="letter">${esc(sz.size)}</div>
      <p class="lede">Your size in the ${esc(it.brand)} ${esc(it.name.toLowerCase())}</p>
    </div>
    <div>
      <h2>${esc(sizeHeading(sz))}</h2>
      <p class="because">${esc(sizeBecause(sz))}</p>
      <div class="scale" aria-hidden="true">
        <div class="axis"></div>${ticks}
        <div class="on" style="left:${num(selIdx * seg)}%;width:${num(seg)}%"></div>
        ${labs}
        <span class="you-lab" style="left:${num(youAt)}%">You</span>
        <div class="you" style="left:${num(youAt)}%"></div>
      </div>
      <dl class="meas">
        <div><dt>Chest</dt><dd>${esc(g.chest)} cm${typeof sz.room_cm === 'number' ? `<small>${esc(num(sz.room_cm))} cm of room</small>` : ''}</dd></div>
        <div><dt>Length</dt><dd>${esc(g.length)} cm${lenNote ? `<small>${esc(lenNote)}</small>` : ''}</dd></div>
        <div><dt>Shoulder</dt><dd>${esc(g.shoulder)} cm${sho ? `<small>${esc(sho)}</small>` : ''}</dd></div>
      </dl>
      <div class="row">
        <button class="btn solid" type="button" data-tryit="${Number(it.id)}" data-fk="tryit">Try it on</button>
        <button class="btn" type="button" data-heart="${Number(it.id)}" data-fk="size-heart" aria-pressed="${liked}"><i data-lucide="heart"></i>${liked ? 'Shortlisted' : 'Shortlist'}</button>
        <div class="switch" role="group" aria-label="Fit" style="margin-left:auto">
          ${['slim', 'regular', 'loose'].map((f) => `<button type="button" data-fit="${f}" data-fk="fit-${f}" aria-pressed="${fit === f}">${cap(f)}</button>`).join('')}
        </div>
      </div>
      ${S.note && S.noteWhere === 'size' ? `<p class="quiet note-line" role="status">${esc(S.note)}</p>` : ''}
      <div class="quiet" style="margin-top:16px">${chestLine}${measure}</div>
    </div>
  </div>`);
  if (S.measureOpen && !$('#chest-input', el).dataset.seen) { const inp = $('#chest-input', el); inp.dataset.seen = '1'; inp.focus(); }
}

let sizeToken = 0;
async function loadSize(id) {
  if (!canSize()) { S.sizeState = 'idle'; renderSizing(); return; }
  const token = ++sizeToken;
  S.sizeState = 'loading';
  if (!S.sizes[id]) renderSizing();
  try {
    const body = { item_id: id, profile: { ...profile(), fit: profile().fit || 'regular' } };
    if (S.convId) body.conversation_id = S.convId;
    const p = await api('/api/size', jsonReq('POST', body));
    if (token !== sizeToken) return;
    S.sizes[id] = p; S.sizeState = 'idle'; S.sizeMsg = '';
  } catch (e) {
    if (token !== sizeToken) return;
    S.sizeState = 'error'; S.sizeMsg = e.status === 429 ? e.detail : '';
  }
  renderSizing();
}

function pick(id) {
  S.selected = id;
  S.measureOpen = false;
  renderRack();
  if (!S.sizes[id] && canSize()) loadSize(id);
}

function changeProfile(patch) {
  S.state = { ...S.state, profile: { ...profile(), ...patch } };
  S.sizes = {};
  renderSizing();
  if (S.selected != null) loadSize(S.selected);
}

/* ------------------------------------------------------------------ shortlist */
async function loadShortlist() {
  try {
    const list = await api('/api/shortlist');
    S.shortlist = Array.isArray(list) ? list : [];
    remember(S.shortlist);
  } catch { /* keep what we have */ }
  syncShortlistUI();
}
function syncShortlistUI() {
  const n = S.shortlist.length;
  $('#list-count').textContent = n ? String(n) : '';
  if (S.results) renderRack();
  if (S.view === 'list') renderList();
  if (S.view === 'try') renderTry();
}
let noteTimer = 0;
function setNote(msg, where = 'grid') {
  S.note = msg; S.noteWhere = where;
  renderResults(); renderSizing();
  clearTimeout(noteTimer);
  noteTimer = setTimeout(() => { S.note = ''; renderResults(); renderSizing(); }, 6000);
}
async function toggleShortlist(id, where) {
  const on = inList(id);
  try {
    if (on) await api('/api/shortlist/' + Number(id), { method: 'DELETE' });
    else await api('/api/shortlist', jsonReq('POST', { item_id: id }));
  } catch (e) {
    setNote(e.detail && (e.status === 409 || e.status === 429) ? e.detail : "That didn't save. Try again.", where);
    return;
  }
  await loadShortlist();
}

function renderList() {
  const el = $('#list-in');
  const L = S.shortlist.slice(0, MAX_SHORTLIST);
  if (!L.length) {
    setHTML(el, `<div class="list-head"><div><h1>Shortlist</h1></div></div>
      <div class="list-empty"><p class="quiet">Heart a shirt in Find and it will wait here, side by side with the others.</p><button class="btn solid" type="button" data-act="to-find" data-fk="to-find">Find shirts</button></div>`);
    return;
  }
  const words = ['', 'One shirt', 'Two shirts', 'Three shirts'];
  const sized = L.map((s) => sizeFor(s.id));
  const allSized = sized.every(Boolean);
  const prices = L.map((s) => Number(s.price));
  const best = L.length > 1 && new Set(prices).size > 1 ? Math.min(...prices) : null;
  const row = (label, cells) => `<div class="r" role="row"><div class="c k" role="rowheader">${label}</div>${cells.map((c) => `<div class="c${c.better ? ' better' : ''}" role="cell">${c.html}</div>`).join('')}</div>`;
  const tried = (id) => S.tryons.some((t) => t.item_id === id);
  const heads = L.map((it) => `<div class="c head" role="columnheader"><div class="frame"><img src="${imgUrl(it.id)}" alt="${esc(it.name)}"></div><h3>${esc(it.name)}</h3><span class="quiet">${esc(it.brand)}</span><button class="text" type="button" data-unlist="${Number(it.id)}" data-fk="unlist-${Number(it.id)}" aria-label="Remove ${esc(it.name)} from shortlist">Remove</button></div>`).join('');
  setHTML(el, `<div class="list-head">
      <div><h1>Shortlist</h1><p class="quiet">${words[L.length]}${allSized ? ', sized for you.' : '.'}</p></div>
      <button class="btn solid" type="button" data-act="try-list" data-fk="try-list">Try them on</button>
    </div>
    <div class="cmp" role="table" aria-label="Shortlist comparison" style="--cols:${L.length}">
      <div class="r" role="row"><div class="c head" role="columnheader"></div>${heads}</div>
      ${row('Price', L.map((s) => ({ html: `$${esc(s.price)}`, better: best !== null && Number(s.price) === best })))}
      ${row('Your size', L.map((s, i) => ({ html: sized[i] ? esc(sized[i].size) : '<span class="quiet">&ndash;</span>' })))}
      ${row('Room at chest', L.map((s, i) => ({ html: sized[i] && typeof sized[i].room_cm === 'number' ? `${esc(num(sized[i].room_cm))} cm` : '<span class="quiet">&ndash;</span>' })))}
      ${row('Fabric', L.map((s) => ({ html: esc(cap(s.fabric || '')) })))}
      ${row('Pattern', L.map((s) => ({ html: esc(cap(s.pattern || '')) })))}
      ${row('Sleeve', L.map((s) => ({ html: esc(s.sleeve ? cap(s.sleeve) + ' sleeves' : '') })))}
      ${row('Tried on', L.map((s) => ({ html: tried(s.id) ? 'Yes' : `<button class="text" type="button" data-tryit="${Number(s.id)}" data-fk="tryrow-${Number(s.id)}">Try it on</button>` })))}
    </div>
    ${allSized ? '' : `<p class="quiet" style="margin-top:20px">${NEED_SIZE}${needSizeBtn}</p>`}`);
}

function madeNote() { return ''; }

/* ------------------------------------------------------------------ history */
function whenOf(u) {
  if (typeof u === 'number') return new Date(u < 1e12 ? u * 1000 : u);
  if (/^\d+(\.\d+)?$/.test(String(u))) return whenOf(Number(u));
  return new Date(u);
}
async function refreshHistory() {
  try { S.conversations = await api('/api/conversations'); } catch { /* keep */ }
  renderHistory();
}
async function refreshTryons() {
  try { S.tryons = await api('/api/tryons'); } catch { return; }
  const missing = [...new Set(S.tryons.map((t) => t.item_id))].filter((id) => !S.items[id]);
  await Promise.all(missing.map((id) => api('/api/items/' + Number(id)).then((it) => remember([it])).catch(() => {})));
  renderHistory();
  if (S.view === 'list') renderList();
}
function renderHistory() {
  const now = new Date();
  const today = [], earlier = [];
  (S.conversations || []).forEach((c) => (whenOf(c.updated).toDateString() === now.toDateString() ? today : earlier).push(c));
  const conv = (c) => `<button type="button" class="hist" data-conv="${esc(c.id)}" data-fk="conv-${esc(c.id)}"${c.id === S.convId ? ' aria-current="true"' : ''}><span>${esc(c.title || 'Untitled search')}</span></button>`;
  let html = '';
  if (today.length) html += `<h2>Today</h2>${today.map(conv).join('')}`;
  if (earlier.length) html += `<h2>Earlier</h2>${earlier.map(conv).join('')}`;
  if (S.tryons.length) {
    html += '<h2>Tried on</h2>' + S.tryons.map((t) => {
      const it = S.items[t.item_id];
      const cur = S.job && S.job.job_id === t.job_id && S.view === 'try';
      return `<button type="button" class="hist" data-tryon="${esc(t.job_id)}" data-fk="tryon-${esc(t.job_id)}"${cur ? ' aria-current="true"' : ''}><span>${esc(it ? it.name : 'Shirt ' + t.item_id)}</span></button>`;
    }).join('');
  }
  setHTML($('#history'), html);
}

/* ------------------------------------------------------------------ try on */
function wearItems() {
  const ids = [];
  S.shortlist.forEach((s) => ids.push(s.id));
  [S.selected, S.tryItem].forEach((id) => { if (id != null && !ids.includes(id) && itemOf(id)) ids.push(id); });
  return ids.map(itemOf).filter(Boolean);
}
function ensureTryItem() {
  const ids = wearItems().map((i) => i.id);
  if (S.tryItem == null || !ids.includes(S.tryItem)) S.tryItem = (S.selected != null && ids.includes(S.selected) ? S.selected : ids[0]) ?? null;
}
function shownSize(id) {
  const sz = sizeFor(id);
  if (!sz) return '';
  return S.viewSize[id] || sz.size;
}
function sizeChoices(it) {
  const list = it && it.size_chart && it.size_chart.sizes ? it.size_chart.sizes.map((s) => s.size) : [];
  return list.length ? list : ['S', 'M', 'L', 'XL'];
}
function currentResult() {
  const j = S.job;
  if (j && j.item_id === S.tryItem) return j;
  const t = S.tryons.find((x) => x.item_id === S.tryItem);
  return t ? { job_id: t.job_id, item_id: t.item_id, status: 'done', url: t.url, seconds: t.seconds } : null;
}

function renderTry() {
  ensureTryItem();
  const it = S.tryItem != null ? itemOf(S.tryItem) : null;
  const res = it ? currentResult() : null;
  const active = res && (res.status === 'queued' || res.status === 'running');
  const size = it ? shownSize(it.id) : '';

  // left: photo
  let left;
  if (S.photo && safeUrl(S.photo.url)) {
    left = `<div class="frame"><img src="${esc(safeUrl(S.photo.url))}" alt="Your photo">${S.uploading ? '<div class="veil" role="status">Checking your photo</div>' : ''}</div>
      <figcaption><b style="font-weight:500">Your photo</b><button class="text" type="button" data-act="pick-photo" data-fk="change-photo">Change photo</button></figcaption>`;
  } else {
    left = `<button class="frame drop" type="button" data-act="pick-photo" data-fk="drop"><span><i data-lucide="image-plus"></i>${S.uploading ? 'Checking your photo' : 'Drop a photo of yourself here, or click to choose one. Face the camera with your upper body in view.'}</span></button>
      <figcaption><b style="font-weight:500">Your photo</b><span></span></figcaption>`;
  }
  const issues = [];
  if (S.photoError) issues.push(S.photoError);
  if (S.photo && S.photo.check) {
    const list = (S.photo.check.issues || []).map(issueSentence).filter(Boolean);
    if (!S.photo.check.ok) issues.push("This photo won't work yet.");
    issues.push(...list);
  }
  const issuesHTML = issues.length ? `<div class="issues" role="status">${issues.map((s) => `<p>${esc(s)}</p>`).join('')}</div>` : '';

  // right: result
  let inner, cap2 = '';
  if (res && res.status === 'done' && safeUrl(res.url)) {
    inner = `<img src="${esc(safeUrl(res.url))}" alt="${esc(`You wearing the ${it.name.toLowerCase()}`)}">`;
    cap2 = typeof res.seconds === 'number' ? `<span>Made in ${Math.max(1, Math.round(res.seconds))} seconds</span>` : '';
  } else if (res && active) {
    inner = `<div class="veil"><div class="veil-in"><b data-stage role="status">${esc(stageText(res))}</b><span class="quiet" data-elapsed aria-hidden="true">${elapsed(res)}</span><span class="quiet">It can take up to a minute.</span></div></div>`;
  } else if (res && res.status === 'error') {
    inner = `<div class="veil" role="alert"><b style="font-weight:500">The try-on didn't finish.</b><span class="quiet">${esc(res.msg || 'Your photo is fine. Give it another go.')}</span><button class="btn sm" type="button" data-act="generate" data-fk="retry">Try again</button></div>`;
  } else {
    let msg = 'Press Generate to see it on you.';
    if (!S.photo || !S.photo.check) msg = 'Your try-on appears here once you add a photo.';
    else if (!S.photo.check.ok) msg = 'Add a photo that works and your try-on appears here.';
    else if (!it) msg = 'Pick a shirt on the right.';
    inner = `<div class="wait">${esc(msg)}</div>`;
  }
  const capName = it ? `${esc(it.name)}${size ? `, size ${esc(size)}` : ''}` : 'Your try-on';
  const right = `<div class="frame" style="position:relative">${inner}</div>
    <figcaption><b style="font-weight:500">${capName}</b>${cap2}</figcaption>`;
  setHTML($('#looks'), `<figure class="look">${left}${issuesHTML}</figure><figure class="look out">${right}</figure>`);

  // panel
  const wear = wearItems();
  const rows = wear.map((w) => {
    const on = w.id === S.tryItem;
    const sz = shownSize(w.id);
    return `<button type="button" data-wear="${Number(w.id)}" data-fk="wear-${Number(w.id)}" aria-pressed="${on}"><div class="th"><img src="${imgUrl(w.id)}" alt=""></div><div><b>${esc(w.name)}</b><span>${esc(w.brand)}${sz ? `, size ${esc(sz)}` : ''}</span></div>${on ? '<i data-lucide="check"></i>' : '<span></span>'}</button>`;
  }).join('');
  const canGo = photoOK() && it && !jobActive() && !S.uploading;
  const again = res && res.status !== 'error';
  const save = res && res.status === 'done' && safeUrl(res.url)
    ? `<a class="btn solid" href="${esc(safeUrl(res.url))}" download="fitting-room-${Number(it.id)}.jpg"><i data-lucide="download"></i>Save image</a>`
    : '<button class="btn solid" type="button" disabled><i data-lucide="download"></i>Save image</button>';
  setHTML($('#panel'), `<div><h2>Wearing</h2>
      ${wear.length ? '' : '<p class="quiet sole">Shortlist a shirt or pick one in Find and it will show up here.</p>'}
      <div class="wear">${rows}<button type="button" class="add" data-act="find-another" data-fk="find-another"><div class="th"><i data-lucide="plus"></i></div><div><b>Find another shirt</b></div></button></div></div>
    ${it && !sizeFor(it.id) ? `<p class="quiet">${NEED_SIZE}${needSizeBtn}</p>` : ''}
    ${it && sizeFor(it.id) ? `<div><h2>Size</h2><div class="switch" role="group" aria-label="Size">${sizeChoices(it).map((s) => `<button type="button" data-size="${esc(s)}" data-fk="size-${esc(s)}" aria-pressed="${s === size}">${esc(s)}</button>`).join('')}</div></div>` : ''}
    <p class="heads-up"><i data-lucide="info"></i>The picture shows how the shirt looks on you. It can't show exactly how a size drapes, so go by the size advice for fit.</p>
    <div class="stack" style="margin-top:auto">
      ${save}
      <button class="btn" type="button" data-act="generate" data-fk="generate"${canGo ? '' : ' disabled'}><i data-lucide="refresh-cw"></i>${again ? 'Generate again' : 'Generate'}</button>
      ${madeNote() ? `<p class="made">${esc(madeNote())}</p>` : ''}
    </div>`);
}

function elapsed(job) { const s = Math.max(0, Math.floor((Date.now() - (job.started || Date.now())) / 1000)); return `${s} ${s === 1 ? 'second' : 'seconds'}`; }
let ticker = 0;
function syncTicker() {
  if (jobActive() && !ticker) {
    ticker = setInterval(() => { const e = $('[data-elapsed]'); if (e && S.job) e.textContent = elapsed(S.job); }, 500);
  } else if (!jobActive() && ticker) { clearInterval(ticker); ticker = 0; }
}

async function startTryon() {
  if (!photoOK() || S.tryItem == null || jobActive()) return;
  const job = { item_id: S.tryItem, status: 'queued', stage: 'queued', started: Date.now(), fails: 0 };
  S.job = job;
  renderTry(); syncTicker();
  try {
    const r = await api('/api/tryon', jsonReq('POST', { photo_id: S.photo.photo_id, item_id: job.item_id }));
    job.job_id = r.job_id;
    setTimeout(() => poll(job), 1500);
  } catch (e) {
    job.status = 'error';
    if (e.status === 429 && e.detail) job.msg = e.detail;
    renderTry(); syncTicker();
  }
}
async function poll(job) {
  if (S.job !== job) return;
  const before = `${job.status}|${job.stage}`;
  try {
    const r = await api('/api/tryon/' + encodeURIComponent(job.job_id));
    Object.assign(job, { status: r.status, stage: r.stage, seconds: r.seconds, url: r.url, error: r.error, fails: 0 });
  } catch {
    if (++job.fails >= 4) job.status = 'error';
  }
  const active = job.status === 'queued' || job.status === 'running';
  if (active) setTimeout(() => poll(job), 1500);
  if (!active) { if (S.view === 'try') renderTry(); }
  else if (before !== `${job.status}|${job.stage}`) {
    const st = $('[data-stage]'); // update the words in place so nothing re-renders or jumps
    if (st) st.textContent = stageText(job);
  }
  syncTicker();
  if (job.status === 'done') refreshTryons();
}

async function uploadPhoto(file) {
  if (!file) return;
  S.photoError = '';
  if (!/^image\//.test(file.type)) { S.photoError = 'That file is not a picture. Choose a JPG or PNG photo.'; renderTry(); return; }
  S.uploading = true; renderTry();
  try {
    const fd = new FormData();
    fd.append('file', file);
    S.photo = await api('/api/photo', { method: 'POST', body: fd });
    S.job = null; syncTicker();
  } catch (e) {
    S.photoError = e.status === 429 && e.detail ? e.detail : "That photo didn't upload. Try again or pick another one.";
  }
  S.uploading = false;
  renderTry();
  if (photoOK() && S.tryItem != null) startTryon();
}

function goTry(id, start) {
  if (id != null) S.tryItem = id;
  go('try');
  if (start && photoOK()) startTryon();
}

/* ------------------------------------------------------------------ events */
const fileInput = $('#photo-file');
fileInput.addEventListener('change', () => { const f = fileInput.files[0]; fileInput.value = ''; uploadPhoto(f); });

['dragover', 'drop'].forEach((t) => window.addEventListener(t, (e) => { if (e.dataTransfer && [...e.dataTransfer.types].includes('Files')) e.preventDefault(); }));
const looks = $('#looks');
looks.addEventListener('dragover', (e) => { const d = $('.frame.drop', looks); if (d) d.classList.add('over'); });
looks.addEventListener('dragleave', (e) => { if (!looks.contains(e.relatedTarget)) { const d = $('.frame.drop', looks); if (d) d.classList.remove('over'); } });
looks.addEventListener('drop', (e) => { const d = $('.frame.drop', looks); if (d) d.classList.remove('over'); const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) uploadPhoto(f); });

document.addEventListener('submit', (e) => {
  if (e.target.id !== 'measure-form') return;
  e.preventDefault();
  const input = $('#chest-input');
  if (!input.reportValidity()) return;
  S.measureOpen = false;
  changeProfile({ chest_cm: Number(input.value) });
});

document.addEventListener('click', (e) => {
  const t = e.target.closest('button, a');
  if (!t) return;
  const d = t.dataset;
  if (d.view) { go(d.view); return; }
  if (t.id === 'new-search') { newSearch(); return; }
  if (d.pick) { pick(Number(d.pick)); return; }
  if (d.heart) { toggleShortlist(Number(d.heart), t.dataset.fk === 'size-heart' ? 'size' : 'grid'); return; }
  if (d.unlist) { toggleShortlist(Number(d.unlist)); return; }
  if (d.crit !== undefined) { const f = S.results && S.results.filters[Number(d.crit)]; if (f) send(`Remove the "${f.label}" filter.`, { removeFilter: f.key }); return; }
  if (d.ex !== undefined) { send(EXAMPLES[Number(d.ex)]); return; }
  if (d.opt) { const [i, k] = d.opt.split(':').map(Number); const m = S.messages[i]; if (m && m.options) send(m.options[k]); return; }
  if (d.trace !== undefined) { const m = S.messages[Number(d.trace)]; if (m) { m.open = !m.open; renderThread(); } return; }
  if (d.fit) { if ((profile().fit || 'regular') !== d.fit) changeProfile({ fit: d.fit }); return; }
  if (d.tryit) { goTry(Number(d.tryit), true); return; }
  if (d.ref) { goTry(Number(d.ref), true); return; }
  if (d.wear) { S.tryItem = Number(d.wear); renderTry(); return; }
  if (d.size) { S.viewSize[S.tryItem] = d.size; renderTry(); return; }
  if (d.conv) { openConversation(d.conv); return; }
  if (d.tryon) {
    const r = S.tryons.find((x) => x.job_id === d.tryon);
    if (r) { S.tryItem = r.item_id; S.job = { job_id: r.job_id, item_id: r.item_id, status: 'done', url: r.url, seconds: r.seconds }; syncTicker(); go('try'); renderTry(); renderHistory(); }
    return;
  }
  switch (d.act) {
    case 'measure': S.measureOpen = true; renderSizing(); break;
    case 'measure-cancel': S.measureOpen = false; renderSizing(); $('[data-act="measure"]')?.focus(); break;
    case 'pick-photo': fileInput.click(); break;
    case 'generate': startTryon(); break;
    case 'find-another': go('find'); break;
    case 'to-find': go('find'); break;
    case 'need-size': S.focusMessage = true; go('find'); break;
    case 'try-list': goTry(S.shortlist[0] ? S.shortlist[0].id : null, true); break;
    default: break;
  }
});

/* ------------------------------------------------------------------ boot */
renderThread();
renderRack();
show(location.hash.slice(1));
loadShortlist();
refreshHistory();
refreshTryons();
