"""Pull ESPN pickcenter closing lines for completed CBB games still inside
ESPN's ~300-day retention window. Cached to instance/cbb_market_lines.json
({event_id: {spread_close, home_moneyline, ...} | null}); re-runs only fetch
events not yet attempted."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import cbb_rating
import espn_pickcenter

PATH = os.path.join(cbb_rating._DATA_DIR, 'cbb_market_lines.json')


def main(seasons=(2026,), floor_days=295):
    cache = json.load(open(PATH)) if os.path.exists(PATH) else {}
    floor = (datetime.now() - timedelta(days=floor_days)).strftime('%Y-%m-%d')
    todo = [g['id'] for s in seasons for g in cbb_rating.load_season(s)
            if g['date'] >= floor and g['id'] not in cache]
    print(f'{len(todo)} events to fetch (floor {floor})', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (eid, res) in enumerate(zip(todo, pool.map(lambda e: espn_pickcenter.fetch_pickcenter('CBB', e), todo))):
            cache[eid] = res
            if i % 500 == 0:
                print(i, flush=True)
                json.dump(cache, open(PATH, 'w'))
    json.dump(cache, open(PATH, 'w'))
    print('have lines for', sum(1 for v in cache.values() if v and v.get('spread_close') is not None))


if __name__ == '__main__':
    main()
