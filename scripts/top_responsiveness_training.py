"""
Top 100 players by Responsiveness from the full training dataset (raw ggData).
Responsiveness = avg(Acceleration, SprintSpeed, Agility, Balance, Reactions)
Uses base attributes (0-chem) to avoid chem-boosted duplicates.
"""
import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RAW_DIR = Path(r'd:\Dev\fut.gg\data\raw\ggData')

players = []
files = list(RAW_DIR.glob('*_ggData.json'))
print(f"Scanning {len(files)} player files...")

for f in files:
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        p = data.get('data', {})
        acc = p.get('attributeAcceleration')
        spd = p.get('attributeSprintSpeed')
        agi = p.get('attributeAgility')
        bal = p.get('attributeBalance')
        rea = p.get('attributeReactions')
        if None in (acc, spd, agi, bal, rea):
            continue
        resp = (acc + spd + agi + bal + rea) / 5.0
        players.append({
            'name': p.get('commonName') or p.get('nickname') or f"{p.get('firstName','')} {p.get('lastName','')}".strip(),
            'overall': p.get('overall', '?'),
            'acc': acc, 'spd': spd, 'agi': agi, 'bal': bal, 'rea': rea,
            'resp': resp
        })
    except Exception:
        continue

players.sort(key=lambda x: x['resp'], reverse=True)

print(f"Total unique player cards: {len(players)}\n")
header = f"{'#':<5}{'Name':<35}{'OVR':<6}{'Acc':<6}{'Spd':<6}{'Agi':<6}{'Bal':<6}{'Rea':<6}{'Resp':<7}"
print(header)
print('-' * len(header))
for i, p in enumerate(players[:100], 1):
    print(f"{i:<5}{p['name']:<35}{p['overall']:<6}{p['acc']:<6}{p['spd']:<6}{p['agi']:<6}{p['bal']:<6}{p['rea']:<6}{p['resp']:<7.1f}")
