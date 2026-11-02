"""Shared state behind the Dead Money loader (templates/_loader.html,
static/loader.js), kept in small JSON files instead of module globals
because gunicorn runs 3 workers (see Dockerfile): the scheduler jobs live in
whichever worker won the leader flock, and a loader's polling requests can
land on any of the three.

Two things live here:

- Running background jobs. `job(label)` wraps a scheduler/startup function
  and drops a file for as long as it runs, so the loader's "The crypt is
  busy" state can name what the server is actually doing.
- Today-hub slices. Each sport's slice of the hub (_hub_sport_data's
  summary + picks) is written here when it's built, so a cold worker can
  assemble the hub from slices another worker (or the loader's per-sport
  requests) already built instead of blocking on a 6-sport rebuild.
"""
import functools
import json
import os
import time
import uuid

_dir = None


def init(directory):
    global _dir
    _dir = directory
    os.makedirs(_dir, exist_ok=True)


def _path(name):
    return os.path.join(_dir, name)


def _write(name, data):
    # Write-then-rename so a reader in another worker never sees half a file.
    tmp = _path(f'.{name}.{os.getpid()}.tmp')
    with open(tmp, 'w') as fh:
        json.dump(data, fh, default=str)
    os.replace(tmp, _path(name))


def _read(name):
    try:
        with open(_path(name)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def job(label):
    """Decorator: record `label` as a running background job while the
    wrapped function runs. Never lets bookkeeping break the job itself."""
    def wrap(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            name = None
            if _dir:
                name = f'job-{os.getpid()}-{uuid.uuid4().hex[:8]}.json'
                try:
                    _write(name, {'label': label, 'started': time.time(), 'pid': os.getpid()})
                except OSError:
                    name = None
            try:
                return fn(*args, **kwargs)
            finally:
                if name:
                    try:
                        os.remove(_path(name))
                    except OSError:
                        pass
        return run
    return wrap


def running_jobs():
    """[{label, started}] oldest first. Files left behind by a worker that
    died mid-job are dropped (and cleaned up) by the pid check."""
    if not _dir:
        return []
    out = []
    for name in os.listdir(_dir):
        if not (name.startswith('job-') and name.endswith('.json')):
            continue
        data = _read(name)
        if not data:
            continue
        if not _pid_alive(data.get('pid', 0)):
            try:
                os.remove(_path(name))
            except OSError:
                pass
            continue
        out.append({'label': data['label'], 'started': data['started']})
    out.sort(key=lambda j: j['started'])
    return out


def put_slice(key, data):
    if _dir:
        try:
            _write(f'slice-{key}.json', {'ts': time.time(), 'data': data})
        except (OSError, TypeError, ValueError):
            pass


def get_slice(key, max_age):
    """(ts, data) for a slice no older than max_age seconds, else None."""
    if not _dir:
        return None
    hit = _read(f'slice-{key}.json')
    if not hit or time.time() - hit['ts'] > max_age:
        return None
    return hit['ts'], hit['data']
