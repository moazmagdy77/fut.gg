import json, sys, io
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

data_dir = Path(__file__).resolve().parents[1] / "data"
data = json.load(open(data_dir / "club_final.json", encoding='utf-8'))

rows = []
for p in data:
    vals = [p.get(k) for k in ('attributeAcceleration', 'attributeSprintSpeed', 'attributeAgility', 'attributeBalance', 'attributeReactions')]
    if not all(isinstance(v, (int, float)) for v in vals):
        continue
    p['responsiveness'] = sum(vals) / 5.0
    rows.append(p)
data = rows
data.sort(key=lambda x: x['responsiveness'], reverse=True)

top100 = data[:100]
print(f'Total club players: {len(data)}')
print()
header = f"{'#':<4} {'Name':<30} {'OVR':<5} {'Acc':<5} {'Spd':<5} {'Agi':<5} {'Bal':<5} {'Rea':<5} {'Resp':<6}"
print(header)
print('-' * len(header))
for i, p in enumerate(top100, 1):
    name = p.get('commonName') or 'Unknown'
    ovr = p.get('overall', '?')
    acc = p['attributeAcceleration']
    spd = p['attributeSprintSpeed']
    agi = p['attributeAgility']
    bal = p['attributeBalance']
    rea = p['attributeReactions']
    resp = p['responsiveness']
    print(f"{i:<4} {name:<30} {ovr:<5} {acc:<5} {spd:<5} {agi:<5} {bal:<5} {rea:<5} {resp:<6.1f}")
