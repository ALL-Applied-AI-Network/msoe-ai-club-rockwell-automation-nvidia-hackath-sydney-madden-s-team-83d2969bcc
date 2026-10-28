"""Run AutoTagger on all 30 vials of Video 5 and report results for vials 11 to 30."""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from tagger.auto_tagger import AutoTagger

def main():
    tagger = AutoTagger()
    print("Running AutoTagger on Video 5 (all 30 vials)...")
    res = tagger.process_video(video_id=5, max_vials=30)
    
    if "recipes" in res and res["recipes"]:
        recipes_df = pd.DataFrame(res["recipes"])
        pours_df = pd.DataFrame(res["pours"])
    else:
        recipes_df = pd.read_csv(res["recipes_csv"])
        pours_df = pd.read_csv(res["pours_csv"])
        
    gt_df = pd.read_csv("data/given_tags/video5_vial_recipes.csv")
    
    print("\n" + "=" * 90)
    print(f"{'Vial':<5} | {'Type':<15} | {'Est Total mL':<14} | {'GT Total mL':<13} | {'Pours (Dispensed mL)':<25} | Status")
    print("-" * 90)
    
    matched = 0
    total = len(recipes_df)
    
    for _, v_rec in recipes_df.iterrows():
        v = int(v_rec["vial"])
        # Find matching GT
        gt_matches = gt_df[gt_df["vial"] == v]
        if gt_matches.empty:
            continue
        v_gt = gt_matches.iloc[0]
        v_pours = pours_df[pours_df["vial"] == v]
        
        est_tot = int(v_rec["total_ml"])
        gt_tot = int(v_gt["total_ml"])
        pours_desc = ", ".join([f"{r['tank_id']}:{int(r['dispensed_ml'])}mL" for _, r in v_pours.iterrows()])
        
        is_match = (est_tot == gt_tot) or (gt_tot % 10 != 0 and abs(est_tot - gt_tot) <= 5)
        if is_match:
            matched += 1
        status = "OK" if is_match else f"MISMATCH ({est_tot - gt_tot:+d}mL)"
        
        # Highlight vials 11-30 specifically
        marker = " [V11-30]" if v >= 11 else ""
        print(f"{v:<5} | {v_rec['type']:<15} | {est_tot:<14} | {gt_tot:<13} | {pours_desc:<25} | {status}{marker}")
        
    print("=" * 90)
    
    # Also report specifically for 11 to 30
    sub_rec = recipes_df[recipes_df["vial"] >= 11]
    sub_matched = 0
    for _, r in sub_rec.iterrows():
        e_tot = int(r["total_ml"])
        g_tot = int(gt_df[gt_df["vial"] == r["vial"]].iloc[0]["total_ml"])
        if e_tot == g_tot or (g_tot % 10 != 0 and abs(e_tot - g_tot) <= 5):
            sub_matched += 1
    print(f"Overall Vials 1-30: {matched} / {total} exact matches ({matched/total*100:.1f}%)")
    print(f"Remaining Vials 11-30: {sub_matched} / {len(sub_rec)} exact matches ({sub_matched/len(sub_rec)*100:.1f}%)")

if __name__ == "__main__":
    main()
