# Limitations

- The main setting is retrospective discharge-summary risk modeling, not pre-discharge decision support.
- Readmissions are same-system readmissions observed within MIMIC-IV/BIDMC data.
- Public clinical baselines may have MIMIC-derived pretraining overlap.
- Raw MIMIC data and raw note text cannot be released.
- Note timing depends on available `storetime` and `charttime` fields.
- Rule-based section parsing may fail on unusual note templates.
- The paper runs used one NVIDIA H200 GPU for clean domain adaptation and representation extraction and one NVIDIA L40S GPU for final downstream training. End-to-end efficiency was not compared across systems under matched hardware and software conditions.
- The resulting model is not a clinical deployment system.
