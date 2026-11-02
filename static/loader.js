// Dead Money loader — behavior for templates/_loader.html (one per page).
//
// Every séance row is a real request: in 'run' mode the loader fetches each
// source's url itself; in 'nav' mode the one row is the page navigation
// that's still pending. Nothing here runs on a fake timer — the bar is the
// share of rows that are done, and the slow/failed states key off how long
// the real requests have actually taken.
//
//   < 400 ms   nothing shown, so a fast load never flashes
//   8 s        "The crypt is busy": background jobs from /api/loader/jobs,
//              plus "Show last saved" when the service worker has a copy
//   30 s       a row times out → "It didn't rise": Try again, or open the
//              page without that source
(function () {
    var SHOW_AFTER = 400, SLOW_AFTER = 8000, FAIL_AFTER = 30000;
    var MAX_PARALLEL = 4;          // leaves the browser's per-host slots for the page itself
    var PAGE_CACHE = 'dm-pages-v1'; // see service-worker.js
    var SEEN_KEY = 'dm.lastSeen';  // written by today.html's open-bets rail

    var el = document.getElementById('dmLoader');
    if (!el) return;
    function q(name) { return el.querySelector('[data-dm="' + name + '"]'); }

    var mode = el.getAttribute('data-mode');
    var rows = [], t0 = 0, tick = null, showTimer = null, jobsTimer = null;
    var slowDismissed = false, target = '', running = false, savedTs = null;

    function fmtTime(ms) {
        return new Date(ms).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
    }
    function secs(ms) { return ms < 100 ? '<0.1s' : ms < 10000 ? (ms / 1000).toFixed(1) + 's' : Math.round(ms / 1000) + 's'; }
    function names(list) {
        var n = list.map(function (r) { return r.name; });
        return n.length < 3 ? n.join(' and ') : n.slice(0, -1).join(', ') + ' and ' + n[n.length - 1];
    }

    // ---- requests -------------------------------------------------------
    function startRow(r) {
        r.state = 'active';
        r.started = Date.now();
        var ctl = window.AbortController ? new AbortController() : null;
        r.abort = function () { if (ctl) ctl.abort(); };
        r.timer = setTimeout(function () { finish(r, 'fail', 'Timed out · 30s'); r.abort(); }, FAIL_AFTER);
        fetch(r.url, {credentials: 'same-origin', cache: 'no-store', signal: ctl ? ctl.signal : undefined})
            .then(function (resp) {
                if (!resp.ok) throw new Error('Error · ' + resp.status);
                return resp.text();
            })
            .then(function () { finish(r, 'done', secs(Date.now() - r.started)); },
                  function (e) { finish(r, 'fail', e && /^Error/.test(e.message) ? e.message : 'No answer'); });
    }

    function finish(r, state, status) {
        if (r.state !== 'active') return;
        clearTimeout(r.timer);
        r.state = state;
        r.status = status;
        pump();
        render();
    }

    function pump() {
        var active = rows.filter(function (r) { return r.state === 'active'; }).length;
        rows.forEach(function (r) {
            if (r.state === 'wait' && r.url && active < MAX_PARALLEL) { startRow(r); active++; }
        });
        if (mode === 'run' && rows.length && rows.every(function (r) { return r.state === 'done'; })) {
            go(target);
        }
    }

    function go(url) {
        clearTimeout(showTimer);
        clearInterval(tick);
        clearInterval(jobsTimer);
        running = false;
        window.location.replace(url);
    }

    // ---- state + render -------------------------------------------------
    function stage() {
        if (rows.some(function (r) { return r.state === 'fail'; })) return 'error';
        if (Date.now() - t0 >= SLOW_AFTER) return 'slow';
        return 'loading';
    }

    var COPY = {
        loading: {eyebrow: 'GRABBING TONIGHT’S DATA', title: 'Raising the dead'},
        slow:    {eyebrow: 'STILL DIGGING', title: 'The crypt is busy',
                  sub: 'The server is mid-job. You can wait it out or open what we already have.'},
        error:   {eyebrow: 'SOMETHING STAYED BURIED', title: 'It didn’t rise'}
    };

    function rowHTML(r) {
        var icon = {
            done: '<svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="#1E2A16"></circle><path d="M7.5 12.5l3 3 6-6.5" stroke="#A6E35A" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
            active: '<span class="dm-pulse dm-loader-dot"></span>',
            wait: '<span class="dm-loader-ring"></span>',
            fail: '<svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="#3A1A1E"></circle><path d="M8.5 8.5l7 7M15.5 8.5l-7 7" stroke="#FF8A8A" stroke-width="2.4" stroke-linecap="round"></path></svg>'
        }[r.state];
        var status = r.state === 'active' ? 'Summoning…' : r.state === 'wait' ? 'In line' : r.status;
        var li = document.createElement('li');
        li.className = 'dm-loader-row is-' + r.state;
        li.innerHTML = '<span class="dm-loader-row-icon">' + icon + '</span>' +
            '<span class="dm-loader-row-text"><span class="dm-loader-row-name"></span><span class="dm-loader-row-detail"></span></span>' +
            '<span class="dm-loader-row-status"></span>';
        li.querySelector('.dm-loader-row-name').textContent = r.label;
        li.querySelector('.dm-loader-row-detail').textContent = r.detail || '';
        li.querySelector('.dm-loader-row-status').textContent = status;
        return li;
    }

    function render() {
        if (!running) return;
        var st = stage(), copy = COPY[st], elapsed = Date.now() - t0;
        var done = rows.filter(function (r) { return r.state === 'done'; }).length;
        var failed = rows.filter(function (r) { return r.state === 'fail'; });
        var pending = rows.some(function (r) { return r.state === 'active' || r.state === 'wait'; });
        var pct = rows.length ? Math.round(done / rows.length * 100) : 0;

        el.setAttribute('data-stage', st);
        q('eyebrow').textContent = st === 'slow' ? copy.eyebrow + ' · ' + Math.round(elapsed / 1000) + 'S' : copy.eyebrow;
        q('title').textContent = copy.title;
        q('quips').hidden = st !== 'loading';
        var sub = st === 'slow' ? copy.sub
            : st === 'error' ? names(failed) + ' timed out. ' +
                               (pending ? 'The rest are still coming.' : rows.length > failed.length ? 'Everything else is ready.' : '')
            : '';
        q('subtitle').textContent = sub;
        q('subtitle').hidden = !sub;
        q('count').textContent = done + ' of ' + rows.length + ' risen';
        q('elapsed').textContent = Math.floor(elapsed / 1000) + 's';
        q('bar').setAttribute('aria-valuenow', pct);
        q('fill').style.width = pct + '%';
        el.querySelector('.dm-loader-main').setAttribute('aria-label',
            copy.title + '. ' + done + ' of ' + rows.length + ' sources loaded.');

        var list = q('rows');
        list.innerHTML = '';
        rows.forEach(function (r) { list.appendChild(rowHTML(r)); });

        q('slow').hidden = st !== 'slow' || slowDismissed;
        if (st === 'slow' && !jobsTimer) { loadJobs(); jobsTimer = setInterval(loadJobs, 4000); }
        if (st !== 'slow' && jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; }

        // The failure card waits for everything else to settle, so "Open
        // without" really does open with everything that did come back.
        var fail = q('fail');
        fail.hidden = st !== 'error' || pending;
        if (!fail.hidden) {
            q('fail-title').textContent = names(failed) + ' didn’t come back';
            if (mode === 'run') {
                q('fail-text').textContent = done
                    ? 'Everything else loaded. You can open the page without ' + names(failed) + ' and it’ll fill in on a later refresh.'
                    : 'Nothing came back. The server may be restarting.';
                q('without').textContent = 'Open without ' + names(failed);
                q('without').hidden = !done;
            } else {
                q('fail-text').textContent = 'The page took more than 30 seconds, so we stopped waiting.' +
                    (savedTs ? ' You can open the copy saved at ' + fmtTime(savedTs) + ' instead.' : '');
                q('without').textContent = savedTs ? 'Show last saved · ' + fmtTime(savedTs) : 'Stay here';
                q('without').hidden = false;
            }
        }
    }

    function loadJobs() {
        fetch('/api/loader/jobs', {cache: 'no-store'}).then(function (r) { return r.json(); }).then(function (d) {
            var jobs = (d && d.jobs) || [];
            if (jobs.length) {
                q('slow-title').textContent = 'Background jobs are hogging the shovel';
                q('slow-text').textContent = 'Running on the server: ' + jobs.map(function (j) {
                    return j.label + ' (' + (j.running_s < 60 ? j.running_s + 's' : Math.round(j.running_s / 60) + 'm') + ' in)';
                }).join(', ') + '.';
            } else {
                q('slow-title').textContent = 'Nothing else is running';
                q('slow-text').textContent = 'No background jobs are holding things up, so the sources themselves are slow to answer.';
            }
        }).catch(function () {
            q('slow-title').textContent = 'The server is slow to answer';
            q('slow-text').textContent = 'We couldn’t reach the job queue either.';
        });
    }

    // ---- "Show last saved" (service worker page cache) -----------------
    function savedCopy(url) {
        if (!window.caches) return Promise.resolve(null);
        return caches.open(PAGE_CACHE).then(function (c) { return c.match(url); }).then(function (resp) {
            return resp ? Number(resp.headers.get('x-dm-saved-at')) || null : null;
        }).catch(function () { return null; });
    }

    function savedUrl(url) {
        var u = new URL(url, location.href);
        u.searchParams.set('saved', '1');
        return u.pathname + u.search;
    }

    function offerSaved(url) {
        var btn = q('saved');
        btn.hidden = true;
        savedTs = null;
        savedCopy(url).then(function (ts) {
            if (!ts || url !== target) return;
            savedTs = ts;
            render();
            btn.textContent = 'Show last saved · ' + fmtTime(ts);
            btn.hidden = false;
            btn.onclick = function () { go(savedUrl(url)); };
        });
    }

    // ---- "Last seen alive" ---------------------------------------------
    function renderCached() {
        var seen = null;
        try { seen = JSON.parse(localStorage.getItem(SEEN_KEY) || 'null'); } catch (e) {}
        var box = q('cached');
        if (!seen || !seen.bets || !seen.bets.length) { box.hidden = true; return; }
        q('cached-when').textContent = 'LAST SEEN ALIVE · ' + fmtTime(seen.ts);
        var ul = q('cached-rows');
        ul.innerHTML = '';
        seen.bets.slice(0, 3).forEach(function (b) {
            var li = document.createElement('li');
            li.innerHTML = '<span class="dm-loader-cached-bet"></span><span class="dm-loader-cached-state"></span>';
            li.firstChild.textContent = b.bet;
            li.lastChild.textContent = b.state;
            if (b.tone) li.lastChild.classList.add(b.tone);
            ul.appendChild(li);
        });
        box.hidden = false;
    }

    // ---- lifecycle ------------------------------------------------------
    function begin(sourceRows, url) {
        rows = sourceRows;
        target = url;
        t0 = Date.now();
        running = true;
        slowDismissed = false;
        offerSaved(url);
        renderCached();
        render();
        clearTimeout(showTimer);
        showTimer = setTimeout(function () {
            if (!running) return;
            el.hidden = false;
            document.documentElement.classList.add('dm-loader-open');
        }, SHOW_AFTER);
        clearInterval(tick);
        tick = setInterval(render, 1000);
    }

    function reset() {
        running = false;
        clearTimeout(showTimer);
        clearInterval(tick);
        clearInterval(jobsTimer);
        jobsTimer = null;
        rows.forEach(function (r) { clearTimeout(r.timer); });
        el.hidden = true;
        document.documentElement.classList.remove('dm-loader-open');
    }

    q('wait').addEventListener('click', function () { slowDismissed = true; render(); });
    q('retry').addEventListener('click', function () {
        if (mode === 'run') {
            rows.forEach(function (r) { if (r.state === 'fail') { r.state = 'wait'; r.status = ''; } });
            t0 = Date.now();
            pump();
            render();
        } else {
            var url = target, label = rows[0].label;
            reset();
            watch(url, label);
            window.location.assign(url);
        }
    });
    q('without').addEventListener('click', function () {
        if (mode !== 'run') {
            if (savedTs) go(savedUrl(target)); else reset();
            return;
        }
        var skip = rows.filter(function (r) { return r.state === 'fail'; }).map(function (r) { return r.key; });
        var u = new URL(target, location.href);
        u.searchParams.set('without', skip.join(','));
        go(u.pathname + u.search);
    });

    function sourceRows(list) {
        return list.map(function (s) {
            return {key: s.key, name: s.label.replace(/^\W+\s*/, '') || s.label, label: s.label, detail: s.detail,
                    url: s.url, state: 'wait', status: ''};
        });
    }

    if (mode === 'run') {
        var sources = [];
        try { sources = JSON.parse(el.getAttribute('data-sources') || '[]'); } catch (e) {}
        begin(sourceRows(sources), el.getAttribute('data-then') || '/');
        pump();
        render();
        return;
    }

    // ---- 'nav' mode: same-site page loads -------------------------------
    // The browser does the request; the loader just watches it. At 30 s it
    // calls window.stop() so "Try again" is a real retry, not a second
    // request racing the first.
    function skipLink(a, e) {
        if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return true;
        if (a.target && a.target !== '_self') return true;
        if (a.hasAttribute('download') || a.hasAttribute('data-no-loader')) return true;
        if (a.origin !== location.origin) return true;
        if (/^\/(api|static)\//.test(a.pathname)) return true;
        // Same page, different hash: no request.
        if (a.pathname === location.pathname && a.search === location.search && a.hash) return true;
        return false;
    }

    if (document.body.classList.contains('embed-mode')) return;

    document.addEventListener('click', function (e) {
        var a = e.target.closest && e.target.closest('a[href]');
        if (!a || skipLink(a, e)) return;
        var label = (a.getAttribute('aria-label') || a.textContent || '').replace(/\s+/g, ' ').trim();
        // The browser navigates on its own; this only starts the watch.
        var url = a.href;
        setTimeout(function () { if (!e.defaultPrevented) watch(url, label); }, 0);
    });

    document.addEventListener('submit', function (e) {
        var f = e.target;
        if (e.defaultPrevented || (f.method || 'get').toLowerCase() !== 'get' || f.hasAttribute('data-no-loader')) return;
        if (f.target && f.target !== '_self') return;
        setTimeout(function () {
            if (e.defaultPrevented) return;
            var u = new URL(f.action, location.href);
            u.search = new URLSearchParams(new FormData(f)).toString();
            watch(u.href, 'Search');
        }, 0);
    });

    function watch(url, label) {
        label = label || 'This page';
        var row = {key: 'page', name: label.replace(/^\W+\s*/, '') || label, label: label,
                   detail: new URL(url, location.href).pathname, state: 'active', status: ''};
        row.timer = setTimeout(function () {
            if (!running) return;
            try { window.stop(); } catch (e) {}
            row.state = 'fail';
            row.status = 'Timed out · 30s';
            render();
        }, FAIL_AFTER);
        begin([row], url);
    }

    // Back/forward restores this page from bfcache with the loader still up.
    window.addEventListener('pageshow', function (e) { if (e.persisted) reset(); });

    // Arrived on a "Show last saved" copy: say so, and drop the flag so a
    // reload goes back to the live page.
    var here = new URL(location.href);
    if (here.searchParams.get('saved') === '1') {
        here.searchParams.delete('saved');
        history.replaceState(null, '', here.pathname + here.search + here.hash);
        savedCopy(here.pathname + here.search).then(function (ts) {
            if (ts && window.showToast) showToast('Showing last saved · ' + fmtTime(ts));
        });
    }
})();
