"""features.py - turns the raw UCI 697 columns into model features.

v2 adds 4 personal features + 18 parent-occupation features to the original 19.
Occupation is reduced to the 9 broad ISCO-08 major groups. India's NCO-2015 divisions 1-9 follow the
same 9 groups, so an Indian college can map a parent's job to a group 1-9 directly.
Group 0 / missing / unknown -> all occupation columns are 0 for that parent.
"""
import re
import numpy as np
import pandas as pd

# ---- original 19 features (unchanged) --------------------------------------------------------
OLD_A = ["prev_qual_pct", "admission_pct", "admission_missing", "hostel", "first_gen", "scholarship"]
OLD_B = OLD_A + ["fee_dues", "s1_registered", "s1_passed", "s1_pass_rate", "s1_sgpa", "s1_no_reg"]
OLD_C = OLD_B + ["s2_registered", "s2_passed", "s2_pass_rate", "s2_sgpa", "s2_no_reg",
                 "delta_pass_rate", "delta_sgpa"]
# ---- new features ------------------------------------------------------------------------------
PERSONAL = ["age", "gender_male", "special_needs", "international"]
OCC = [f"{who}_occ_{g}" for who in ("mother", "father") for g in range(1, 10)]
NEW = PERSONAL + OCC

NEW_A = OLD_A + NEW
NEW_B = OLD_B + NEW
NEW_C = OLD_C + NEW
CHECKPOINT_C = NEW_C          # everything build_features() returns, in a fixed order


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def occupation_group(code):
    """UCI occupation code -> broad group 1-9, or 0 when unknown / armed forces / 'student' / blank.
    UCI codes: 1-9 are the major groups themselves; 10 and 101-103 armed forces; 90 other; 99 blank;
    11x..19x are detailed codes whose middle digit is the major group (e.g. 123 -> group 2)."""
    if pd.isna(code):
        return 0
    c = int(code)
    if 1 <= c <= 9:
        return c
    if c >= 110:
        g = c // 10 - 10
        return g if 1 <= g <= 9 else 0
    return 0


def build_features(data):
    column_map = {normalize_name(col): col for col in data.columns}

    def get_column(name):
        return column_map.get(normalize_name(name))

    f = pd.DataFrame(index=data.index)

    prev_grade = get_column("Previous qualification (grade)")
    admission_grade = get_column("Admission grade")
    if prev_grade:
        f["prev_qual_pct"] = data[prev_grade] / 2
    if admission_grade:
        f["admission_missing"] = data[admission_grade].isna().astype(float)
        f["admission_pct"] = data[admission_grade] / 2

    displaced, scholarship = get_column("Displaced"), get_column("Scholarship holder")
    if displaced:
        f["hostel"] = data[displaced]
    if scholarship:
        f["scholarship"] = data[scholarship]

    mother, father = get_column("Mother's qualification"), get_column("Father's qualification")
    higher_education = {2, 3, 4, 5, 6, 40, 41, 42, 43, 44}
    if mother and father:
        f["first_gen"] = (~data[mother].isin(higher_education) & ~data[father].isin(higher_education)).astype(float)

    debtor, fees = get_column("Debtor"), get_column("Tuition fees up to date")
    if debtor and fees:
        f["fee_dues"] = ((data[debtor] == 1) | (data[fees] == 0)).astype(float)

    for semester, prefix in [("1st sem", "s1"), ("2nd sem", "s2")]:
        enrolled_col = get_column(f"Curricular units {semester} (enrolled)")
        approved_col = get_column(f"Curricular units {semester} (approved)")
        grade_col = get_column(f"Curricular units {semester} (grade)")
        if enrolled_col and approved_col and grade_col:
            enrolled, approved = data[enrolled_col], data[approved_col]
            f[f"{prefix}_registered"] = enrolled.astype(float)
            f[f"{prefix}_passed"] = approved.astype(float)
            f[f"{prefix}_pass_rate"] = np.where(enrolled > 0, approved / enrolled, 0.0)
            f[f"{prefix}_sgpa"] = data[grade_col] / 2
            f[f"{prefix}_no_reg"] = (enrolled == 0).astype(float)
    if "s1_pass_rate" in f.columns and "s2_pass_rate" in f.columns:
        f["delta_pass_rate"] = f["s2_pass_rate"] - f["s1_pass_rate"]
        f["delta_sgpa"] = f["s2_sgpa"] - f["s1_sgpa"]

    # ---- new personal features ----
    age, gender = get_column("Age at enrollment"), get_column("Gender")
    needs, nat = get_column("Educational special needs"), get_column("Nacionality")
    if age:
        f["age"] = data[age].astype(float)
    if gender:
        f["gender_male"] = (data[gender] == 1).astype(float)          # UCI: 1 = male, 0 = female
    if needs:
        f["special_needs"] = data[needs].astype(float)
    if nat:
        f["international"] = (data[nat] != 1).astype(float)           # UCI: 1 = Portuguese (home country)
    # ---- new occupation features (9 broad groups, one 0/1 column per group) ----
    for who, name in (("mother", "Mother's occupation"), ("father", "Father's occupation")):
        col = get_column(name)
        if col:
            grp = data[col].map(occupation_group)
            for g in range(1, 10):
                f[f"{who}_occ_{g}"] = (grp == g).astype(float)

    missing = set(CHECKPOINT_C) - set(f.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    return f[CHECKPOINT_C].astype(float)
