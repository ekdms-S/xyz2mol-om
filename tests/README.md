# tests/

`pytest` — no workspace and no network. Every test is a small hard-coded structure with its Mayer
bond orders pinned as constants, so the suite runs on a bare install.

```bash
pip install -e ".[dev]" && pytest -q
```

| file | what it pins |
|---|---|
| `test_api_smoke.py` | one `[Mo(≡N)(OH)Cl₃]⁻` through every stage — T1 · T4 · T8 `Mo≡N` · `q_L` · `OS(Mo)=+6` · ⑥ · ligand SMILES round trip |
| `test_molecules.py` | a disconnected input splits into molecules, and `OS` is solved **inside** one |
| `test_radical.py` | `n_unpaired=1` — the electron on a ligand, on the metal, and the refusals |
| `test_ml_budget_and_eta.py` | agostic and haptic spend nothing in the ④ budget · η^k is per ligand |
| `test_bridging_carbonyl.py` | μ-CO comes out `3c2e` with the `C≡O` intact |
| `test_complex_smiles_and_bridge.py` | the complex SMILES, the bridge tags, the 3c2e budget exclusion |
| `test_r7_haptic_ring.py` | R7 fires — the S of an η⁵-thienyl turns haptic |
| `test_eta2_carries_the_pi.py` | an η² is written across a `Double`, never a `Single` |
| `test_pi_shift.py` | the post-⑥ repairs — the π moves where it cancels two charges, and the dianionic-chelate and peroxide exclusions hold |
| `test_agostic_carbon.py` | the carbon side of an agostic `C–H···M` is dropped too, unless that orphans the fragment |
| `test_boron_sextet.py` | trivalent boron is neutral (sextet), four-coordinate boron unchanged |
| `test_boron_ligand_atom.py` | boron is a ligand atom everywhere; `B₂H₆` is `B(+1)` + bridging `[H-]`, and the `B–B` the two bridges already pay for is dropped |
| `test_rule_a_headroom.py` | rule A's π headroom counts the σ M–L bonds, and NHC · σ-aryl · haptic ring atoms are untouched |
| `test_pi_suppressed.py` | the `pi_suppressed` report: when it fires, when it must stay silent, that it reads the raw likelihood margin, and that it survives the JSON round trip as tuples |
| `test_invariants.py` | geometry-free unit tests on the decision functions themselves |
| `test_assemble.py` · `test_serialize.py` · `test_draw.py` | reassembly, `save_json`/`load_json` bond keys, and that `draw()` renders every bond kind it claims to |
