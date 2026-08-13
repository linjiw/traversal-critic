# Stage A Public Release — Frozen Protocols and Audit Methodology

This directory is **Stage A** of the project's staged artifact release: the
frozen preregistrations, outcome-blind interpretation notes, and audit
methodology, published **before the policy training matrix closes**, so that
their outcome-blindness is publicly checkable rather than merely asserted. A
pre-commitment is only demonstrably outcome-blind while the outcomes do not
yet exist; publishing now timestamps that property. Nothing in this directory
reports or depends on any policy-matrix outcome.

Every file is a sanitized copy of a private original; the only sanitization is
the replacement of absolute private path prefixes by the placeholders
`<REPO>` (home-directory prefix) and `<DATA>` (data-volume prefix). All
SHA-256 values, protocol rules, numbers, and frozen sentences are verbatim.
Full per-file provenance — source path, pre- and post-sanitization SHA-256,
exact change, and the verification greps run — is in
[`RELEASE_MANIFEST.md`](RELEASE_MANIFEST.md).

## Contents

### `protocols/`

| Document | Role | SHA-256 (released copy) |
|---|---|---|
| `shortcut_battery_extension_protocol_2026-08-11.md` | Frozen 17-row shortcut-battery protocol (frozen 2026-08-11, executed 2026-08-12) | `4a34c6c9e3ce38f6808fa5dc77d582b3bb35ab8f9dc2f15310c4c90d9de873dd` |
| `constrained_decoding_interface_comparator_protocol_2026-08-11.md` | E1 constrained-decoding interface-comparator factorial, preregistered before any cell existed | `0d621c48984f5d28017a76cad640e6d99d305b4df296f9de25484b50a9fa49d6` |
| `e1_amendment_expected_score_readout_2026-08-11.md` | E1 amendment: expected-score readout definition | `50aaf1db92932173888a19ab360fef15ee977f603ee67a01bbe36b341f66c6f1` |
| `e5_temporal_interface_amendment_2026-08-11.md` | E5 temporal-interface amendment (three-interface audit design) | `06d78a3e808d9da6a6cc0162196188cf129329a6e084bda210909f19bfdf444c` |
| `matrix_interpretation_note_outcome_blind_2026-08-11.md` | Outcome-blind interpretation note pre-committing the branch texts for the policy matrix, written before any held-out endpoint existed | `8709e330ac9372ca81e92433b6b2f9b32d90cdb389d96e3db5db59a9a41064a3` |
| `temporal_matched_v1_evaluation_protocol_2026-08-11.md` | Temporal-matched v1 critic evaluation protocol (corrected-generation design) | `6cd97611febf1a63ba1fbaed5f1646612653f020ee6b4e00fed43f63023224c2` |
| `temporal_matched_v1_prelaunch_seal_2026-08-11.md` | Prelaunch seal binding the temporal-matched v1 configuration by hash before launch | `d02807371256f58fe28213fcc6eac517285d3e57151301f3a941f065ba2319c2` |
| `heldout_test_set_protocol_2026-08-12.md` | Held-out fresh-scene confirmatory test-set protocol, frozen before any test scene exists | `64937f0d920713acd786ac677c9eccab76f3fe8ce05914ec056e1ecb36c91e81` |
| `external_baseline_protocol_2026-08-12.md` | External-baseline (B1–B3) preregistration, frozen before any baseline cell exists | `998a5e4cf6f02847a08a18830ca99ecacdf857ae41b4cd0b7536d702f3bd4a44` |

### `reports/run-260812-0034/`

| Document | Role | SHA-256 (released copy) |
|---|---|---|
| `run_report.md` | Shortcut-battery run of record (fingerprint replay, sanity gate, 17-row table, artifact hashes) | `9571d09dae2987d66ca3e5188131a4938e283c098b1d16f703e491adb94d3b30` |
| `shortcut_battery_results.json` | Full machine-readable battery results bound to the report | `0575a9c108c73a6435528c4f526dd82098b6226610f2ea47b74bb04a4cac55ea` |

The two `.npz` feature/seed artifacts named in the report are withheld until
Stage C and remain bound by the SHA-256s printed in the report.

### `scripts/`

Portable audit instruments (with their test suites, which pass against these
copies): `temporal_route_materializer.py` (shared temporal-route authority),
`run_shortcut_battery_extension.py`, `evaluate_temporal_shortcuts.py`,
`derive_e1_expected_score.py`, `audit_critic_temporal_preprocessing.py`.
Per-file hashes are in the manifest. One test file
(`temporal_route_materializer_test.py`) ships at Stage B with its run-record
fixtures; see the manifest's exclusion note.

## Later stages

Per the staging plan: **Stage B** (at matrix closure/adjudication) releases the
matrix results package, matrix analysis/audit code, E1 and temporal-matched
results, and remaining run-audit records; **Stage C** (at camera-ready)
releases data, checkpoints (license permitting), the hash-bound `.npz`
artifacts, and the full code; **Stage D** enumerates what is never released.
