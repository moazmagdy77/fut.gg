import json
import shutil
import sys
import io
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

def main():

    print("🚀 Starting Raw Data Migration...")
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw"
    
    gg_data_dir = raw_dir / "ggData"
    gg_meta_dir = raw_dir / "ggMeta"
    es_meta_dir = raw_dir / "esMeta"
    
    if not gg_data_dir.exists():
        print("❌ No existing raw/ggData folder found. Nothing to migrate.")
        return
        
    # Load player IDs
    player_ids_path = data_dir / "player_ids.json"
    club_ids_path = data_dir / "all_club_ids.json"
    
    training_ids = set()
    if player_ids_path.exists():
        try:
            training_ids = set(str(x) for x in json.loads(player_ids_path.read_text(encoding='utf-8')))
        except Exception as e:
            print(f"⚠️ Error reading player_ids.json: {e}")
            
    club_ids = set()
    if club_ids_path.exists():
        try:
            club_ids = set(str(x) for x in json.loads(club_ids_path.read_text(encoding='utf-8')))
        except Exception as e:
            print(f"⚠️ Error reading all_club_ids.json: {e}")

    # Create new subfolders
    categories = ['club - main', 'club - rest', 'training']
    subdirs = ['ggData', 'ggMeta', 'esMeta']
    
    for cat in categories:
        for s in subdirs:
            (raw_dir / cat / s).mkdir(parents=True, exist_ok=True)
            
    gg_files = list(gg_data_dir.glob("*_ggData.json"))
    print(f"Found {len(gg_files)} files in raw/ggData to migrate.")
    
    migrated_counts = {'club - main': 0, 'club - rest': 0, 'training': 0}
    
    for i, f in enumerate(gg_files):
        ea_id = f.name.split('_')[0]
        
        # Read rating
        overall = 0
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            overall = int(data.get('data', {}).get('overall', 0))
        except Exception:
            pass
            
        # Determine targets
        targets = []
        
        # 1. Check if in training set
        if ea_id in training_ids:
            targets.append('training')
            
        # 2. Check if in club set
        if ea_id in club_ids:
            if overall >= 75:
                targets.append('club - main')
            else:
                targets.append('club - rest')
                
        # 3. Fallback if not explicitly in either (determine by rating)
        if not targets:
            if overall >= 75:
                targets.append('club - main')
            else:
                targets.append('club - rest')
                
        # Copy files to targets
        for target in targets:
            migrated_counts[target] += 1
            # Copy ggData
            shutil.copy2(f, raw_dir / target / "ggData" / f.name)
            
            # Copy ggMeta if exists
            gg_meta_f = gg_meta_dir / f"{ea_id}_ggMeta.json"
            if gg_meta_f.exists():
                shutil.copy2(gg_meta_f, raw_dir / target / "ggMeta" / gg_meta_f.name)
                
            # Copy esMeta if exists
            es_meta_f = es_meta_dir / f"{ea_id}_esMeta.json"
            if es_meta_f.exists():
                shutil.copy2(es_meta_f, raw_dir / target / "esMeta" / es_meta_f.name)
                
        if (i + 1) % 1000 == 0 or (i + 1) == len(gg_files):
            print(f"Processed {i + 1}/{len(gg_files)} players...")
            
    print("\n🎉 Migration finished!")
    print(f"  - club - main: {migrated_counts['club - main']} players")
    print(f"  - club - rest: {migrated_counts['club - rest']} players")
    print(f"  - training: {migrated_counts['training']} players")
    
    # We leave the original directories intact for now to ensure safety.
    # The user or verification steps can remove them later.

if __name__ == "__main__":
    main()
