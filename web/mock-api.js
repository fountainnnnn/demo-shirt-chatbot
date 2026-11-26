// Canned backend for the front end, loaded only when the page is opened with ?mock=1.
// Shapes follow docs/contracts.md ("HTTP API"). Extra switches:
//   ?mock=1&fail=tryon   the try-on job ends in the error status
//   ?mock=1&rate=1       the second chat message gets HTTP 429
//   ?mock=1&status=local /api/status reports local mode (default is hosted)
//   ?mock=1&guess=1      shortlist returns a size even before any height/weight (must not be shown)
//   ?mock=1&slow=1       chat waits 2.5 s before its first event (to see the Thinking state)
window.__MOCK_API__ = true;
// Photos: the real catalog images when the server has them, otherwise the design mock's four.
window.__MOCK_IMG__ = await fetch('/images/30802.jpg', { method: 'HEAD' }).then((r) => (r.ok ? '/images' : '/design/img')).catch(() => '/design/img');

const params = new URLSearchParams(location.search);
const realFetch = window.fetch.bind(window);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const enc = new TextEncoder();

/* ------------------------------------------------------------------ catalog */
const chart = (fit) => ({
  fit,
  sizes: [
    { size: 'S', body_chest: [86, 94], garment_chest: 98, length: 70, shoulder: 44 },
    { size: 'M', body_chest: [94, 102], garment_chest: 104, length: 73, shoulder: 46 },
    { size: 'L', body_chest: [102, 110], garment_chest: 110, length: 76, shoulder: 48 },
    { size: 'XL', body_chest: [110, 118], garment_chest: 116, length: 79, shoulder: 50 },
  ],
});
const mk = (o) => ({ gender: 'men', type: 'shirt', season: 'summer', usage: 'casual', synthetic: ['price', 'fabric', 'size_chart'], image: `images/${o.id}.jpg`, ...o });
const CATALOG = [
  mk({ id: 30802, name: 'Blue cotton shirt', brand: 'Fabindia', colour: 'blue', sleeve: 'short', pattern: 'solid', fabric: 'light cotton', price: 42, size_chart: chart('regular'), description: 'A pale blue short-sleeved cotton shirt.', source_name: 'Fabindia Men Blue Shirt' }),
  mk({ id: 13870, name: 'Beige stripe shirt', brand: 'Scullers', colour: 'beige', sleeve: 'short', pattern: 'striped', fabric: 'cotton blend', price: 48, size_chart: chart('relaxed'), description: 'A beige striped short-sleeved shirt.', source_name: 'Scullers Men Beige Shirt' }),
  mk({ id: 56825, name: 'Brown check shirt', brand: 'John Players', colour: 'brown', sleeve: 'short', pattern: 'checked', fabric: 'cotton twill', price: 36, size_chart: chart('regular'), description: 'A brown checked short-sleeved shirt.', source_name: 'John Players Men Brown Shirt' }),
  mk({ id: 14052, name: 'White poplin shirt', brand: 'Wrangler', colour: 'white', sleeve: 'long', pattern: 'solid', fabric: 'cotton poplin', price: 55, usage: 'formal', size_chart: chart('regular'), description: 'A white long-sleeved poplin shirt.', source_name: 'Wrangler Men White Shirt' }),
];
const byId = (id) => CATALOG.find((c) => c.id === Number(id));
const REASONS = {
  30802: 'Light, plain, easy to dress up.',
  13870: 'The most relaxed of the three.',
  56825: 'Darker, better for the evening.',
};

/* ------------------------------------------------------------------ sizing (same rules as app/sizing.py) */
function recommend(item, p) {
  const sizes = item.size_chart.sizes;
  const fit = p.fit || 'regular';
  let chest, estimated, shoulder = null;
  if (p.height_cm && p.weight_kg) {
    chest = 105.9 + 0.55 * (p.weight_kg - 85.5) + 0.05 * (p.height_cm - 175.6);
    shoulder = 46 + 0.1 * (p.weight_kg - 85.5) + 0.1 * (p.height_cm - 175.6);
  }
  if (p.chest_cm) { chest = Number(p.chest_cm); estimated = false; } else { estimated = true; }
  if (chest == null) return null;
  chest = Math.round(chest * 10) / 10;
  let base = sizes.findIndex((s, i) => chest >= s.body_chest[0] && (chest < s.body_chest[1] || i === sizes.length - 1));
  const notes = [];
  if (chest < sizes[0].body_chest[0]) { base = 0; notes.push('below_chart'); }
  if (chest > sizes[sizes.length - 1].body_chest[1]) { base = sizes.length - 1; notes.push('above_chart'); }
  let idx = base;
  if (fit === 'loose' && idx < sizes.length - 1) { idx += 1; notes.push('one_up_for_loose_fit'); }
  else if (fit === 'slim' && idx > 0 && !notes.length) {
    const [lo, hi] = sizes[idx].body_chest;
    if (chest < lo + (hi - lo) / 3) { idx -= 1; notes.push('one_down_for_slim_fit'); }
  }
  const s = sizes[idx];
  const bounds = [...sizes.map((x) => x.body_chest[0]), sizes[sizes.length - 1].body_chest[1]];
  if (!notes.some((n) => n.endsWith('_chart')) && Math.min(...bounds.map((b) => Math.abs(chest - b))) <= 1.5) notes.push('near_size_boundary');
  const ratio = p.height_cm ? s.length / p.height_cm : null;
  return {
    size: s.size, chart_size: sizes[base].size, moved: idx - base,
    chest_cm: chest, chest_estimated: estimated,
    room_cm: Math.round((s.garment_chest - chest) * 10) / 10,
    garment: { chest: s.garment_chest, length: s.length, shoulder: s.shoulder },
    shoulder_diff_cm: shoulder == null ? null : Math.round((s.shoulder - shoulder) * 10) / 10,
    ranges: sizes.map((x) => ({ size: x.size, lo: x.body_chest[0], hi: x.body_chest[1] })),
    chart_room_cm: Math.round((sizes[base].garment_chest - chest) * 10) / 10,
    confidence: estimated ? 0.85 : 0.95, notes,
    length_note: ratio == null ? null : ratio < 0.41 ? 'short' : ratio >= 0.45 ? 'long' : 'mid_hip',
  };
}

/* ------------------------------------------------------------------ in-memory state */
const DB = { convs: new Map(), shortlist: [], tryons: [], jobs: new Map(), photos: new Map(), seq: 1 };
let lastProfile = {};

const json = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
const sseResponse = (events) => {
  const stream = new ReadableStream({
    async start(controller) {
      for (const [name, data, delay] of events) {
        if (delay) await sleep(delay);
        controller.enqueue(enc.encode(`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`));
      }
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
};
const words = (text, per = 70) => text.split(/(?<=\s)/).map((w) => ['text', { delta: w }, per]);

/* ------------------------------------------------------------------ chat */
function chat(body) {
  let conv = body.conversation_id && DB.convs.get(body.conversation_id);
  const events = [];
  const msg = String(body.message || '');
  const low = msg.toLowerCase();
  if (!conv) {
    conv = { id: 'c' + DB.seq++, title: msg.length > 38 ? msg.slice(0, 36).trim() + '...' : msg, messages: [], results: null, size: null, state: { wants: {}, profile: {} }, asked: 0, filters: [], selected: null, updated: new Date().toISOString() };
    DB.convs.set(conv.id, conv);
  }
  conv.updated = new Date().toISOString();
  conv.messages.push({ role: 'user', text: msg });
  events.push(['conversation', { id: conv.id, title: conv.title }, params.has('slow') ? 2500 : 250]);
  const say = (text) => { conv.messages.push({ role: 'assistant', text }); events.push(...words(text)); };
  const firstIdx = conv.messages.length;
  const trace = [];
  let writing = false, wrote = false;
  // a stage that takes `ms` (the mock waits at most 700 ms so tests stay quick)
  const step = (id, label, detail, ms) => {
    events.push(['step', { id, status: 'start', label, detail: null, ms: 0 }, 120]);
    events.push(['step', { id, status: 'done', label, detail, ms }, Math.min(ms, 700)]);
    trace.push({ id, label, detail, ms });
  };
  const writeStart = () => { if (wrote || writing) return; writing = true; events.push(['step', { id: 'write', status: 'start', label: 'Writing the reply', detail: null, ms: 0 }, 80]); };
  const writeEnd = () => { if (!writing) return; writing = false; wrote = true; events.push(['step', { id: 'write', status: 'done', label: 'Writing the reply', detail: null, ms: 1100 }, 40]); trace.push({ id: 'write', label: 'Writing the reply', detail: null, ms: 1100 }); };

  // profile in the message
  const h = msg.match(/(\d{3})\s*cm/i), w = msg.match(/(\d{2,3})\s*kg/i);
  const p = conv.state.profile;
  if (h) p.height_cm = Number(h[1]);
  if (w) p.weight_kg = Number(w[1]);
  if (/loose/i.test(msg)) p.fit = 'loose'; else if (/slim/i.test(msg)) p.fit = 'slim'; else if (/regular/i.test(msg)) p.fit = 'regular';
  if (p.height_cm || p.weight_kg) lastProfile = { ...p };

  const drop = body.remove_filter ? [msg, (conv.filters.find((f) => f.key === body.remove_filter) || {}).label] : null;
  if (body.remove_filter) conv.messages.pop(); // a filter removal is not a chat turn
  const wants = conv.state.wants;
  if (/short/.test(low)) wants.sleeve = 'short'; else if (/long/.test(low)) wants.sleeve = 'long'; else if (/either/.test(low)) wants.sleeve = 'any';

  if (drop) {
    conv.filters = conv.filters.filter((f) => f.label !== drop[1]);
    conv.results = { ...conv.results, filters: conv.filters };
    events.push(['results', conv.results, 300]);
  } else if (!conv.results) {
    step('understand', 'Reading your message', null, 380);
    step('decide', 'Deciding what to do next', null, 120);
    if (!wants.sleeve && conv.asked < 1) {
      conv.asked += 1;
      const q = { text: 'Short sleeves or long? And is there a colour you want to avoid?', options: ['Short sleeves', 'Long, rolled up', 'Either'] };
      conv.messages.push({ role: 'assistant', text: q.text, options: q.options });
      events.push(['question', q, 400]);
    } else {
      const filters = [];
      if (wants.sleeve === 'short') filters.push({ key: 'sleeve', label: 'Short sleeves' });
      if (wants.sleeve === 'long') filters.push({ key: 'sleeve', label: 'Long sleeves' });
      if (/(no|not) white/.test(low)) filters.push({ key: 'avoid_colours', label: 'Not white' });
      const under = low.match(/under \$?(\d+)/);
      if (under) filters.push({ key: 'max_price', label: `Under $${under[1]}` });
      if (/beach|summer|june|hot/.test(low) || conv.messages.some((m) => /beach|summer|june|hot/i.test(m.text))) filters.push({ key: 'fabric', label: 'Light fabric' });
      conv.filters = filters;
      const wedding = conv.messages.some((m) => m.role === 'user' && /wedding/i.test(m.text));
      const widen = !!under; // the budget filter leaves too few exact matches in this scenario
      const items = [30802, 13870, 56825].map((id, i) => ({ ...byId(id), reason: REASONS[id], best: i === 0, exact: !(widen && i === 2), ...(widen && i === 2 ? { near: 'Checked, not plain' } : {}) }));
      step('search', 'Searching 100 shirts', widen ? '2 match every filter' : '14 match every filter', 412);
      if (widen) step('widen', 'Widening the search', 'Dropped "plain" to find one more', 240);
      step('pick', 'Choosing the best three', null, 1800);
      conv.results = { title: wedding ? 'Three shirts for a beach wedding' : 'Three shirts for you', filters, exact_count: widen ? 2 : 3, items };
      conv.selected = 30802;
      events.push(['results', conv.results, 300]);
      writeStart();
      say("Three on the right. The blue Fabindia is the one I'd pick: light cotton, and plain enough for a ceremony. ");
      if (!(p.height_cm && p.weight_kg)) { say("\n\nGive me your height and weight and I'll size it."); writeEnd(); }
    }
  } else if (!h && !w && !/loose|slim|regular|\d+\s*(cm|kg)/i.test(msg)) {
    step('understand', 'Reading your message', null, 300);
    writeStart();
    say('I can change the filters on the right if you tell me what to drop, or size a shirt if you give me your height and weight.');
    writeEnd();
  }

  if (conv.results && (p.height_cm && p.weight_kg) && (h || w || /loose|slim|regular/i.test(msg) || (!drop && conv.results && !conv.size))) {
    const item = byId(conv.selected || conv.results.items[0].id);
    const sz = recommend(item, { ...p, fit: p.fit || 'regular' });
    conv.size = { item_id: item.id, ...sz };
    const nm = { S: 'Small', M: 'Medium', L: 'Large', XL: 'Extra large' };
    step('size', 'Sizing the shirt', `${sz.size}, ${sz.room_cm} cm of room`, 30);
    writeStart();
    let t = `\n\nYou're a **${nm[sz.size]}** in the ${item.brand}.`;
    if (sz.moved) t += ` By the chart you're a ${nm[sz.chart_size]}, so I went one size ${sz.moved > 0 ? 'up' : 'down'} for the ${sz.moved > 0 ? 'looser' : 'slimmer'} fit.`;
    say(t);
    events.push(['size', conv.size, 200]);
    writeEnd();
  }
  writeEnd();
  const firstReply = conv.messages.slice(firstIdx).find((x) => x.role === 'assistant');
  if (firstReply && trace.length) firstReply.steps = trace;
  events.push(['state', { wants: conv.state.wants, profile: conv.state.profile }, 50]);
  events.push(['done', {}, 20]);
  return sseResponse(events);
}

/* ------------------------------------------------------------------ router */
async function route(method, path, init, url) {
  let m;
  if (method === 'POST' && path === '/api/chat') {
    const b = JSON.parse(init.body);
    DB.chats = (DB.chats || 0) + 1;
    if (params.has('rate') && DB.chats === 2) return json({ detail: 'That is a lot of messages in a short time. Give it a minute and try again.' }, 429);
    return chat(b);
  }
  await sleep(80);
  if (method === 'GET' && path === '/api/status') {
    return params.get('status') === 'local'
      ? json({ mode: 'local', privacy: 'Private. Runs on this laptop, offline.', chat_model: 'mock', tryon_provider: 'local', ready: true })
      : json({ mode: 'hosted', privacy: 'Messages and photos go to hosted models through OpenRouter.', chat_model: 'mock', tryon_provider: 'openrouter', ready: true });
  }
  if (method === 'GET' && path === '/api/conversations') {
    return json([...DB.convs.values()].sort((a, b) => b.updated.localeCompare(a.updated)).map((c) => ({ id: c.id, title: c.title, updated: c.updated })));
  }
  if (method === 'GET' && (m = path.match(/^\/api\/conversations\/(.+)$/))) {
    const c = DB.convs.get(decodeURIComponent(m[1]));
    if (!c) return json({ detail: 'not found' }, 404);
    return json({ id: c.id, title: c.title, messages: c.messages, results: c.results, size: c.size, state: c.state });
  }
  if (method === 'GET' && (m = path.match(/^\/api\/items\/(\d+)$/))) {
    const it = byId(m[1]);
    return it ? json(it) : json({ detail: 'not found' }, 404);
  }
  if (method === 'POST' && path === '/api/size') {
    const b = JSON.parse(init.body);
    const it = byId(b.item_id);
    const sz = it && recommend(it, b.profile || {});
    return sz ? json(sz) : json({ detail: 'need chest, or height and weight' }, 422);
  }
  if (path === '/api/shortlist' && method === 'GET') {
    // ?guess=1 imitates a server that returns a size even with no profile (the client must not show it)
    return json(DB.shortlist.map((id) => ({ ...byId(id), size: params.has('guess') && !lastProfile.height_cm ? { size: 'S', chest_cm: 90, room_cm: 8 } : recommend(byId(id), { ...lastProfile, fit: lastProfile.fit || 'regular' }) })));
  }
  if (path === '/api/shortlist' && method === 'POST') {
    const { item_id } = JSON.parse(init.body);
    if (!byId(item_id)) return json({ detail: 'unknown item' }, 404);
    if (!DB.shortlist.includes(item_id) && DB.shortlist.length >= 3) return json({ detail: 'Your shortlist holds three shirts. Take one off to add another.' }, 409);
    if (!DB.shortlist.includes(item_id)) DB.shortlist.push(item_id);
    return route('GET', '/api/shortlist', {}, url);
  }
  if (method === 'DELETE' && (m = path.match(/^\/api\/shortlist\/(\d+)$/))) {
    DB.shortlist = DB.shortlist.filter((id) => id !== Number(m[1]));
    return route('GET', '/api/shortlist', {}, url);
  }
  if (method === 'POST' && path === '/api/photo') {
    const file = init.body.get('file');
    const name = (file && file.name) || '';
    const id = 'p' + DB.seq++;
    const check = /bad/i.test(name) ? { ok: false, issues: ['upper_body_not_visible'] }
      : /two/i.test(name) ? { ok: false, issues: ['There is more than one person in this photo.'] }
        : { ok: true, issues: [] };
    const photo = { photo_id: id, url: URL.createObjectURL(file), check };
    DB.photos.set(id, photo);
    return json(photo);
  }
  if (method === 'POST' && path === '/api/tryon') {
    const b = JSON.parse(init.body);
    if (!DB.photos.has(b.photo_id) || !byId(b.item_id)) return json({ detail: 'bad request' }, 422);
    const job_id = 'j' + DB.seq++;
    DB.jobs.set(job_id, { job_id, item_id: b.item_id, created: Date.now(), saved: false });
    return json({ job_id });
  }
  if (method === 'GET' && (m = path.match(/^\/api\/tryon\/(.+)$/))) {
    const j = DB.jobs.get(decodeURIComponent(m[1]));
    if (!j) return json({ detail: 'not found' }, 404);
    const t = Date.now() - j.created;
    const base = { status: 'queued', stage: 'queued', seconds: null, url: null, error: null };
    if (params.get('fail') === 'tryon' && t > 2200) return json({ ...base, status: 'error', stage: 'failed', error: 'model crashed' });
    if (t < 1000) return json(base);
    if (t < 2500) return json({ ...base, status: 'running', stage: 'Reading your photo' });
    if (t < 4000) return json({ ...base, status: 'running', stage: 'Fitting the shirt' });
    const out = { status: 'done', stage: 'done', seconds: 4.1, url: `${window.__MOCK_IMG__}/${j.item_id}.jpg`, error: null };
    if (!j.saved) { j.saved = true; DB.tryons.unshift({ job_id: j.job_id, item_id: j.item_id, url: out.url, seconds: out.seconds }); }
    return json(out);
  }
  if (method === 'GET' && path === '/api/tryons') return json(DB.tryons);
  return json({ detail: 'not found' }, 404);
}

window.fetch = async (input, init = {}) => {
  const url = new URL(typeof input === 'string' ? input : input.url, location.href);
  if (!url.pathname.startsWith('/api/')) return realFetch(input, init);
  return route((init.method || 'GET').toUpperCase(), url.pathname, init, url);
};
