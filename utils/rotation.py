"""
AgriSense India — Crop Rotation Utility Module
File: utils/rotation.py

Import: from utils.rotation import get_rotation_plan, get_rotation_benefits
"""

import json, os

_RULES_PATH = "assets/rotation_rules.json"
_rules_cache = None


def _load_rules():
    global _rules_cache
    if _rules_cache is None:
        if not os.path.exists(_RULES_PATH):
            raise FileNotFoundError(f"Rotation rules not found: {_RULES_PATH}")
        with open(_RULES_PATH, encoding="utf-8") as f:
            _rules_cache = json.load(f)
    return _rules_cache


def get_rotation_plan(current_crop, n_seasons=3):
    rules = _load_rules()
    plan  = [current_crop.lower()]
    current = current_crop.lower()
    seasons = ["kharif","rabi","zaid","kharif","rabi"]
    season_idx = 1

    for _ in range(n_seasons - 1):
        rule   = rules.get(current, {})
        avoid  = rule.get("avoid_after", [])
        prefer = rule.get("prefer_after", [])
        cur_fam = rule.get("disease_family", "")

        scored = []
        for crop, r in rules.items():
            if crop == current or crop in avoid:
                continue
            score = 0
            if crop in prefer: score += 10
            if r.get("disease_family","") != cur_fam: score += 5
            if r.get("nitrogen_impact", 0) > 0: score += 3
            if seasons[season_idx % len(seasons)] in r.get("seasons",[]): score += 4
            scored.append((crop, score))

        scored.sort(key=lambda x: -x[1])
        next_crop = scored[0][0] if scored else list(rules.keys())[0]
        plan.append(next_crop)
        current = next_crop
        season_idx += 1

    return plan


def get_rotation_benefits(plan):
    rules = _load_rules()
    benefits = []
    total_n  = 0
    for i, crop in enumerate(plan):
        r = rules.get(crop, {})
        n_imp = r.get("nitrogen_impact", 0)
        total_n += n_imp
        benefits.append({
            "season": i + 1,
            "crop": crop,
            "nitrogen_impact": n_imp,
            "rationale": r.get("rationale", ""),
            "seasons": r.get("seasons", []),
        })
    return {"plan": benefits, "total_n_balance": total_n}
