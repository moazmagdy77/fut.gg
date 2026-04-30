import json, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data = json.load(open(r'd:\Dev\fut.gg\data\club_final.json', encoding='utf-8'))

for p in data:
    p['responsiveness'] = (
        p['attributeAcceleration'] +
        p['attributeSprintSpeed'] +
        p['attributeAgility'] +
        p['attributeBalance'] +
        p['attributeReactions']
    ) / 5.0

data.sort(key=lambda x: x['responsiveness'], reverse=True)

top100 = data[:100]
print(f'Total club players: {len(data)}')
print()
header = f"{'#':<4} {'Name':<30} {'OVR':<5} {'Acc':<5} {'Spd':<5} {'Agi':<5} {'Bal':<5} {'Rea':<5} {'Resp':<6}"
print(header)
print('-' * len(header))
for i, p in enumerate(top100, 1):
    name = p['commonName']
    ovr = p['overall']
    acc = p['attributeAcceleration']
    spd = p['attributeSprintSpeed']
    agi = p['attributeAgility']
    bal = p['attributeBalance']
    rea = p['attributeReactions']
    resp = p['responsiveness']
    print(f"{i:<4} {name:<30} {ovr:<5} {acc:<5} {spd:<5} {agi:<5} {bal:<5} {rea:<5} {resp:<6.1f}")
