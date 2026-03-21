import pandas as pd
import json
from pathlib import Path

tradeable_file = Path("data") / "tradeable_details.json"
try:
    with open(tradeable_file, "r", encoding="utf-8") as f:
        tradeable_details = json.load(f)
except Exception as e:
    print("Error loading details:", e)
    tradeable_details = []

if tradeable_details:
    trad_df = pd.DataFrame(tradeable_details)
    prices_dir = Path("data") / "raw" / "prices"
    
    def load_price(ea_id):
        clean_id = str(ea_id).split(".")[0]
        pfile = prices_dir / f"{clean_id}.json"
        
        if not pfile.exists():
            print(f"File missing: {pfile}")
            return 0
            
        try:
            with open(pfile, "r", encoding="utf-8") as pf:
                data = json.load(pf)
                return data.get("price", 0)
        except Exception as e:
            print("Error loading json", pfile, e)
            return 0
            
    trad_df["price"] = trad_df["__true_player_id"].apply(load_price)
    
    print("Zero prices count:", (trad_df["price"] == 0).sum())
    print("Total prices sum:", trad_df["price"].sum())
    
    trad_df["price"] = pd.to_numeric(trad_df["price"], errors="coerce").fillna(0).astype("Int64")
    trad_df["discardValue"] = pd.to_numeric(trad_df.get("discardValue", 0), errors="coerce").fillna(0).astype("Int64")
    
    print("After numeric:")
    print("Zero prices count:", (trad_df["price"] == 0).sum())
    print("Total prices sum:", trad_df["price"].sum())
else:
    print("No tradeable details")
