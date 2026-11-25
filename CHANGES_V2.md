# v2 - what changed

PUSH THESE 3 FILES to GitHub (replace the old ones, one commit):
  main.py   engine.py   model_bundle.joblib
(requirements.txt is unchanged. features.py and train_model.py are only for retraining on your laptop.)

New OPTIONAL fields in POST /predict (old requests still work):
  age_at_enrollment (14-80), gender ("male"/"female"), special_needs (0/1), international (0/1),
  mother_occupation_group (1-9), father_occupation_group (1-9)   # 0 or omitted = unknown
Occupation groups = NCO-2015 divisions 1-9 (same 9 groups as ISCO-08):
  1 Managers/senior officials  2 Professionals  3 Technicians & associate professionals
  4 Clerical support  5 Service & sales  6 Skilled agriculture  7 Craft & trades
  8 Plant & machine operators  9 Elementary occupations

Response adds: model_version ("v2"), optional_fields_not_provided.
Gender, special needs, nationality and occupation are used by the model but never listed in top_factors.

Training-row cross-validation (macro-F1, old -> new): A 0.429 -> 0.479 | B 0.656 -> 0.658 | C 0.699 -> 0.700
(B and C changes are inside the noise; only checkpoint A improves clearly.)
