// Code the queue (index.html) and the map (map.html) both use: where the lists live, reading them,
// and asking Semantic Scholar about papers. It's a plain script, so everything here is a global.

// Where the lists live. Change these if the repo moves.
const REPO = 'cho104/AIResearchers';
const BRANCH = 'main';
const DIR = 'papers';           // next to this page on the site
const REPO_DIR = 'docs/papers'; // the same folder in the repo
const CACHE_KEY = 'air-paper-queue';
const S2_BATCH = 'https://api.semanticscholar.org/graph/v1/paper/batch';

const MD_LINK = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/;
const BARE_LINK = /<?(https?:\/\/[^\s<>]+)>?/;
const TAG = /(^|\s)#([\p{L}\p{N}][\p{L}\p{N}_-]*)/gu;

const $ = id => document.getElementById(id);

// ---------- Finding and fetching the lists ----------

function isListFile(name) {
  return /\.(md|json)$/i.test(name) && !/^readme\.md$/i.test(name) && !/^[_.]/.test(name) && !name.includes('/');
}

// GitHub Pages can't list a folder, so ask the GitHub API what's in papers/.
// A local preview (python3 -m http.server) reads the server's folder listing instead.
async function listFiles() {
  if (!location.hostname.endsWith('github.io')) {
    try {
      const res = await fetch(DIR + '/', { cache: 'no-cache' });
      if (res.ok) {
        const page = new DOMParser().parseFromString(await res.text(), 'text/html');
        return [...page.querySelectorAll('a[href]')]
          .map(a => decodeURIComponent(a.getAttribute('href')))
          .filter(isListFile);
      }
    } catch {}
  }
  const res = await fetch(`https://api.github.com/repos/${REPO}/contents/${REPO_DIR}?ref=${BRANCH}`);
  if (res.status === 404) return [];
  if (!res.ok) {
    const reset = Number(res.headers.get('x-ratelimit-reset'));
    if (res.headers.get('x-ratelimit-remaining') === '0' && reset) {
      const at = new Date(reset * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
      throw new Error(`GitHub's API limit was reached. Try again after ${at}.`);
    }
    throw new Error(`GitHub answered ${res.status}.`);
  }
  return (await res.json()).filter(f => f.type === 'file').map(f => f.name).filter(isListFile);
}

async function loadFiles() {
  const names = await listFiles();
  const files = await Promise.all(names.map(async name => {
    try {
      const res = await fetch(`${DIR}/${encodeURIComponent(name)}`, { cache: 'no-cache' });
      // A file merged a moment ago may not be deployed yet; it'll appear on the next visit.
      return res.ok ? { name, text: await res.text() } : null;
    } catch { return null; }
  }));
  return files.filter(Boolean);
}

// ---------- Parsing ----------

const plain = s => s.replace(/\*\*|`/g, '').replace(/\s+/g, ' ').trim();

// "- [Title](link) — why #tag #reviewed", or just "- link".
function parseLine(text) {
  const md = text.match(MD_LINK), bare = text.match(BARE_LINK);
  let match, url, title = null;
  if (md && md.index < bare.index) { match = md; url = md[2]; title = plain(md[1]); }
  else if (bare) { match = bare; url = bare[1].replace(/[.,;:!?)\]]+$/, ''); }
  else return null;

  let reviewed = false;
  const tags = [];
  const rest = (text.slice(0, match.index) + ' ' + text.slice(match.index + match[0].length))
    .replace(TAG, (_, space, tag) => {
      const status = tag.toLowerCase();
      if (status === 'reviewed') reviewed = true;
      else if (status !== 'tbd') tags.push(tag);
      return space;
    });
  const why = plain(rest).replace(/^[\s:|·•–—-]+|[\s:|·•–—-]+$/g, '');
  return { url, title, why, tags, reviewed };
}

function parseMarkdown(text, fallbackName) {
  let name = null, fenced = false;
  const papers = [];
  text.split(/\r?\n/).forEach((line, i) => {
    if (/^\s*(```|~~~)/.test(line)) { fenced = !fenced; return; }
    if (fenced) return;
    const heading = line.match(/^#\s+(.+?)\s*#*\s*$/);
    if (heading) { name ??= heading[1]; return; }
    // Only top-level list items count; indented lines are the author's own notes.
    const item = line.match(/^(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?(.*)$/);
    const paper = item && parseLine(item[1]);
    if (paper) papers.push({ ...paper, line: i + 1 });
  });
  return { name: name || fallbackName, papers, problems: [] };
}

function parseJSON(text, fallbackName) {
  const data = JSON.parse(text);
  const list = Array.isArray(data) ? data : data && data.papers;
  if (!Array.isArray(list)) throw new Error('expected {"name": "...", "papers": [...]}');
  const papers = [], problems = [];
  list.forEach((entry, i) => {
    const p = typeof entry === 'string' ? { link: entry } : entry || {};
    const url = String(p.link || p.url || '');
    if (!/^https?:\/\/\S+$/i.test(url)) return problems.push(`paper ${i + 1} needs an http(s) "link"`);
    const status = String(p.status || 'tbd').toLowerCase().replace(/^#/, '');
    if (status !== 'reviewed' && status !== 'tbd') problems.push(`paper ${i + 1}: status should be "reviewed" or "tbd"`);
    const tags = (Array.isArray(p.tags) ? p.tags : String(p.tags || '').split(/[\s,]+/))
      .map(t => String(t).replace(/^#/, '')).filter(Boolean);
    papers.push({ url, title: p.title ? plain(String(p.title)) : null, why: p.why ? plain(String(p.why)) : '', tags, reviewed: status === 'reviewed', line: null });
  });
  return { name: (!Array.isArray(data) && data.name && String(data.name)) || fallbackName, papers, problems };
}

// The same paper under different URLs (arXiv abs/pdf/versions, OpenReview forum/pdf) is one paper.
function paperKey(url) {
  try {
    const u = new URL(url);
    const host = u.hostname.toLowerCase().replace(/^www\./, '');
    const path = u.pathname.replace(/\/+$/, '');
    let m;
    if (/(^|\.)arxiv\.org$/.test(host) && (m = path.match(/^\/(?:abs|pdf|html)\/(.+?)(?:v\d+)?(?:\.pdf)?$/))) return 'arxiv:' + m[1];
    if (host === 'huggingface.co' && (m = path.match(/^\/papers\/(\d{4}\.\d{4,5})/))) return 'arxiv:' + m[1];
    if (host === 'openreview.net' && u.searchParams.get('id')) return 'openreview:' + u.searchParams.get('id');
    return host + path + u.search;
  } catch { return url; }
}

function collect(files) {
  const byKey = new Map(), problems = [];
  for (const file of files) {
    const stem = file.name.replace(/\.(md|json)$/i, '');
    let list;
    try {
      list = /\.json$/i.test(file.name) ? parseJSON(file.text, stem) : parseMarkdown(file.text, stem);
    } catch (err) {
      problems.push(`Couldn't read ${REPO_DIR}/${file.name}: ${err.message}`);
      continue;
    }
    list.problems.forEach(p => problems.push(`Skipped in ${REPO_DIR}/${file.name}: ${p}`));
    for (const p of list.papers) {
      const key = paperKey(p.url);
      let paper = byKey.get(key);
      if (!paper) byKey.set(key, paper = { key, order: byKey.size, url: p.url, title: null, reviewed: false, tags: [], notes: [], people: [] });
      paper.title ||= p.title;
      paper.reviewed ||= p.reviewed;
      for (const t of p.tags) if (!paper.tags.some(x => x.toLowerCase() === t.toLowerCase())) paper.tags.push(t);
      if (!paper.people.some(w => w.name === list.name)) paper.people.push({ name: list.name, file: file.name, line: p.line });
      if (p.why) paper.notes.push({ text: p.why, by: list.name });
    }
  }
  return { papers: [...byKey.values()], problems };
}

// ---------- Semantic Scholar ----------

// The ID Semantic Scholar knows a paper by, or null for sites it can't look up (like OpenReview).
function s2Id(paper) {
  if (paper.key.startsWith('arxiv:')) return 'ARXIV:' + paper.key.slice(6);
  try {
    const u = new URL(paper.url);
    const host = u.hostname.toLowerCase().replace(/^www\./, '');
    let m;
    if (host === 'doi.org' || host === 'dx.doi.org') return 'DOI:' + decodeURIComponent(u.pathname.slice(1));
    if (host === 'semanticscholar.org' && (m = u.pathname.match(/([0-9a-f]{40})\/?$/))) return m[1];
    if (host === 'aclanthology.org' && (m = u.pathname.match(/^\/([^/]+?)(?:\.pdf)?\/?$/))) return 'ACL:' + m[1];
    if (/(^|\.)(aclweb\.org|acm\.org|biorxiv\.org)$/.test(host)) return 'URL:' + paper.url;
  } catch {}
  return null;
}

// GET the URL, or POST `body` to it as JSON.
// The free APIs are often busy (429, which the browser only sees as a failed fetch), so try a few times.
async function fetchJSON(url, body) {
  const init = body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
  let error;
  for (const wait of [0, 2000, 5000, 10000]) {
    await new Promise(r => setTimeout(r, wait));
    try {
      const res = await fetch(url, init);
      if (res.ok) return await res.json();
      error = new Error(`Semantic Scholar answered ${res.status}.`);
    } catch (err) { error = err; }
  }
  throw error;
}

// Run `task` on each item, a few at a time, so we don't flood an API.
async function inBatches(items, task, size = 4) {
  for (let i = 0; i < items.length; i += size) await Promise.all(items.slice(i, i + size).map(task));
}

// ---------- Tags ----------

const SHARED_TAGS = ['ssl', 'vlm', 'segmentation', 'diffusion', 'agents']; // each has its own color in site.css
const NOT_TOPICS = ['rejected']; // tags that say something about a paper other than its field
const EXTRA_COLORS = 3; // other tags take --tag-x1 to --tag-x3 in order of use, then gray

const collator = new Intl.Collator(['ko', 'en'], { numeric: true, sensitivity: 'base' });
const fmt = n => n.toLocaleString('en');
const isTopic = t => !NOT_TOPICS.includes(t.toLowerCase());

// How often each name comes up, ignoring case, most first.
function tally(names) {
  const counts = new Map();
  for (const name of names) {
    const c = counts.get(name.toLowerCase()) || { key: name.toLowerCase(), name, n: 0 };
    c.n++;
    counts.set(c.key, c);
  }
  return [...counts.values()].sort((a, b) => b.n - a.n || collator.compare(a.name, b.name));
}

// Tag key -> CSS color, given the tags in tally() order.
function tagColors(tags) {
  const colors = new Map(SHARED_TAGS.map(t => [t, `var(--tag-${t})`]));
  let extra = 0;
  for (const t of tags) {
    if (colors.has(t.key) || !isTopic(t.name)) continue;
    colors.set(t.key, extra < EXTRA_COLORS ? `var(--tag-x${++extra})` : 'var(--tag-none)');
  }
  return colors;
}

// ---------- Small DOM helpers ----------

const ICONS = {
  close: '<path d="m4.5 4.5 7 7m0-7-7 7"/>',
  chevron: '<path d="m4.5 6.5 3.5 3.5 3.5-3.5"/>',
  out: '<path d="M5.5 10.5 10.5 5.5M6.5 5.5h4v4"/>',
};

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('class', 'ico');
  svg.setAttribute('aria-hidden', 'true');
  svg.innerHTML = ICONS[name];
  return svg;
}

// One group in the sidebar: a heading, then items with their counts. Long groups show the first few
// until "more". Each item is { q, label, n, color }, and `active` is the q of the chosen one.
function sideGroup(box, title, items, onPick, active, limit = 6) {
  const open = box.dataset.open === 'true';
  const list = el('ul', 'side-list');
  items.forEach((item, i) => {
    const b = button('side-item', null, () => onPick(item.q));
    b.dataset.q = item.q.toLowerCase();
    b.setAttribute('aria-pressed', b.dataset.q === active);
    if (item.color) {
      const swatch = el('span', 'swatch');
      swatch.style.setProperty('--c', item.color);
      b.append(swatch);
    }
    b.append(item.label);
    if (item.n != null) b.append(el('span', 'n', item.n));
    const li = el('li');
    li.hidden = !open && i >= limit && b.dataset.q !== active;
    li.append(b);
    list.append(li);
  });
  const parts = [el('p', 'side-h', title), list];
  if (items.length > limit) {
    const more = button('side-more', open ? 'less' : 'more', () => {
      box.dataset.open = open ? '' : 'true';
      sideGroup(box, title, items, onPick, box.querySelector('[aria-pressed="true"]')?.dataset.q ?? active, limit);
    });
    more.setAttribute('aria-expanded', open);
    more.append(icon('chevron'));
    parts.push(more);
  }
  box.replaceChildren(...parts);
}

// Mark the sidebar item whose q is `q` as chosen.
function pressSide(q) {
  for (const b of document.querySelectorAll('.side-item[data-q]')) b.setAttribute('aria-pressed', b.dataset.q === q);
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function link(href, text, cls) {
  const a = el('a', cls, text);
  a.href = href;
  a.target = '_blank';
  a.rel = 'noopener';
  return a;
}

function button(cls, text, onClick) {
  const b = el('button', cls, text);
  b.type = 'button';
  b.addEventListener('click', onClick);
  return b;
}

const host = url => { try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return ''; } };
const shortUrl = url => url.replace(/^https?:\/\/(www\.)?/, '').replace(/\/$/, '');
