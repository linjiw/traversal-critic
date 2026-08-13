# Stage A Release Manifest

Stage A of the public artifact release, executed per the staging plan
`artifact_release_staging_2026-08-12.md` (research repo, commit `cf62cbd`).
Every file below is a *copy* of a private original; nothing is symlinked and
the private repository is unmodified.

## Placeholder scheme

Two placeholders sanitize absolute private paths. The substitution is a pure
prefix replacement — nothing else in any file was altered by sanitization:

- `<REPO>` replaces the private home-directory prefix. The research repository
  checkout is therefore `<REPO>/traversal-critic-research` and the Cosmos
  framework checkout is `<REPO>/cosmos-framework`.
- `<DATA>` replaces the private data-volume prefix; the project data root is
  `<DATA>/traversal-critic/data`.

Every SHA-256, protocol rule, number, and frozen sentence is verbatim from the
original. Where a released copy is sanitized, any SHA-256 of that file printed
in *another* document binds the **private original**; the "pre" hash column
below is that original's hash, so the binding remains checkable.

## Files

Source paths are relative to `<REPO>/traversal-critic-research/`.
"pre" = SHA-256 of the private original; "post" = SHA-256 of the released copy.
"verbatim" means pre == post (byte-identical copy).

### `protocols/` (staging-plan items A1–A7)

| Released file | Source | pre SHA-256 | post SHA-256 | Change |
|---|---|---|---|---|
| `shortcut_battery_extension_protocol_2026-08-11.md` | `docs/reviews/` (same name) | `4a34c6c9e3ce38f6808fa5dc77d582b3bb35ab8f9dc2f15310c4c90d9de873dd` | same (verbatim) | none |
| `constrained_decoding_interface_comparator_protocol_2026-08-11.md` | `docs/reviews/` | `0d621c48984f5d28017a76cad640e6d99d305b4df296f9de25484b50a9fa49d6` | same (verbatim) | none |
| `e1_amendment_expected_score_readout_2026-08-11.md` | `docs/reviews/` | `50aaf1db92932173888a19ab360fef15ee977f603ee67a01bbe36b341f66c6f1` | same (verbatim) | none |
| `e5_temporal_interface_amendment_2026-08-11.md` | `docs/reviews/` | `06d78a3e808d9da6a6cc0162196188cf129329a6e084bda210909f19bfdf444c` | same (verbatim) | none |
| `matrix_interpretation_note_outcome_blind_2026-08-11.md` | `docs/reviews/` | `8709e330ac9372ca81e92433b6b2f9b32d90cdb389d96e3db5db59a9a41064a3` | same (verbatim) | none |
| `temporal_matched_v1_evaluation_protocol_2026-08-11.md` | `docs/reviews/` | `6cd97611febf1a63ba1fbaed5f1646612653f020ee6b4e00fed43f63023224c2` | same (verbatim) | none |
| `temporal_matched_v1_prelaunch_seal_2026-08-11.md` | `docs/reviews/` | `2f8405fc1a7e97409de66f9a1e8fdcf0dc1e6889abb4420a44a1fcd49342064d` | `d02807371256f58fe28213fcc6eac517285d3e57151301f3a941f065ba2319c2` | 7 path-prefix substitutions (2 home, 5 data); the seal's 1 SHA-256 verbatim |
| `heldout_test_set_protocol_2026-08-12.md` | `docs/reviews/` | `58703143216459f7cceecc6a3e97909e295cf4978195594fd8660ad94d5583e7` | `64937f0d920713acd786ac677c9eccab76f3fe8ce05914ec056e1ecb36c91e81` | 3 path-prefix substitutions (data); 2 SHA-256s verbatim |
| `external_baseline_protocol_2026-08-12.md` | `docs/reviews/` | `33b32dd774452106c0142dc044268d5d49cb86aec55a68845157af827775bb9a` | `998a5e4cf6f02847a08a18830ca99ecacdf857ae41b4cd0b7536d702f3bd4a44` | 5 path-prefix substitutions (data); 18 SHA-256s (incl. prompt/anchor hashes) verbatim |

### `reports/run-260812-0034/` (item A8 — shortcut-battery run of record)

| Released file | Source | pre SHA-256 | post SHA-256 | Change |
|---|---|---|---|---|
| `run_report.md` | `autoresearch/run-260812-0034/run_report.md` | `b78b248fb83ca13d5881c910c3b247942c06cea67dbc774c0a2be1183ad6213a` | `9571d09dae2987d66ca3e5188131a4938e283c098b1d16f703e491adb94d3b30` | 2 path-prefix substitutions (home) + a clearly-marked release annotation block after the title; 7 printed SHA-256s verbatim |
| `shortcut_battery_results.json` | `autoresearch/run-260812-0034/shortcut_battery_results.json` | `c61cb964d080c3409f91a0b765b1c06c51c9589efcdac39213dd08d9d6237588` | `0575a9c108c73a6435528c4f526dd82098b6226610f2ea47b74bb04a4cac55ea` | 10 path-prefix substitutions (7 home, 3 data) in meta/config string fields; all numbers and 11 embedded SHA-256s verbatim; JSON validity re-checked |

The `run_report.md` prints the SHA-256 of `shortcut_battery_results.json` as
`c61cb96…` — that is the **private original's** hash (the "pre" value above);
the private original differs from the released copy only in the 10 path
prefixes. The two `.npz` artifacts named in the report
(`shortcut_battery_features.npz`, `shortcut_battery_seed_index_artifact.npz`)
are withheld until Stage C and are bound by the SHA-256s printed in the report.

### `scripts/` (item A9 — audit-methodology instruments)

Copies are verbatim except (a) private-path strings in docstring usage
examples / default constants replaced by the placeholders, and (b) a one-line
header comment added after the SPDX line of each modified script:
`# Public release copy (stage A); private-path defaults replaced by <REPO>/<DATA> placeholders — see RELEASE_MANIFEST.md`.
The working private originals differ from these copies **only** in those
default/example path strings and the absence of that header line.

| Released file | Source | pre SHA-256 | post SHA-256 | Change |
|---|---|---|---|---|
| `temporal_route_materializer.py` | `scripts/` (same name) | `7efd595b8ef82e15c9958eb94a8f7c9d313bde72c242898460a0d97a986229bf` | same (verbatim) | none (0 private paths in original) |
| `run_shortcut_battery_extension.py` | `scripts/` | `f552faea2d8eef9100f07852b1470766c5c3eb5f6e69c54202afba1720004950` | `7a4e0f422f84632bfc93efebfc01b206ed5335b28fe149ec226c2cd942eecf2b` | 4 path substitutions (1 home, 3 data; docstring usage example) + header line |
| `evaluate_temporal_shortcuts.py` | `scripts/` | `0e7f422c5a3829fd543c2ffe2a3a8a92a1c161df7df2456361e319060957a79e` | `1f01208bb67e6798747e49c7bbec4e21e5e400d1de546685e6df413ca798dbeb` | 3 path substitutions (data; docstring usage example) + header line |
| `derive_e1_expected_score.py` | `scripts/` | `9a7fa9913908d90f70c51ce59dd3c6460feca3584487873bdb3ab788343b7c70` | `7c032727fccb3b166dcbc9a096d8d895783307984f35798b39d504c4a3463091` | 1 path substitution (data; `DEFAULT_ROOT` constant) + header line |
| `audit_critic_temporal_preprocessing.py` | `scripts/` | `6c6e39d229bcc66208f16af0d5e98214c9e3d332dab8d6a6fb34acbd7aa6a7f2` | `886eb4cee8c7090d8c9fbf6abcbaa5197e294c31ddb59144fe8748d7f5e9777a` | 3 path substitutions (data; docstring usage example) + header line |
| `run_shortcut_battery_extension_test.py` | `scripts/` | `c8c982c163dc7b3cddf00aca8aaa8f962ca62250b26743db800ef8b339bbed6d` | same (verbatim) | none |
| `evaluate_temporal_shortcuts_test.py` | `scripts/` | `3278781271e6f36900676782759baa1b406571d75b175d1ea4082a5e28267bc1` | same (verbatim) | none |
| `derive_e1_expected_score_test.py` | `scripts/` | `da8430314305a6a693a23baa0dab3330c09d948ee88a305cfcbd90e46725634c` | same (verbatim) | none |
| `audit_critic_temporal_preprocessing_test.py` | `scripts/` | `adb2ae7280b4eb45764a452d58e1d353c3340479c89938bc03d9b8fc3a15c77d` | same (verbatim) | none |

**Exclusion:** `temporal_route_materializer_test.py` (original SHA-256
`647ae4254846221d4ce5540cf44ef70c76a46cf7ed339f2a17cdf50349411bec`) is **not**
released in Stage A. At import time it loads repo fixtures that are staged
later or cannot ship: `autoresearch/run-260811-1753/temporal_preprocessing_audit.json`
(a run-260811 audit JSON — Stage B per the staging plan), the
`cosmos_overlay/.../traversal_critic_temporal.py` processor tree, and a local
processor checkpoint referenced by absolute path inside that audit JSON. These
fixture references cannot be sanitized meaningfully. The test was nevertheless
executed against the released (byte-identical) `temporal_route_materializer.py`
copy in a private harness supplying those fixtures: 49 passed. The test file
itself will ship with its fixtures at Stage B.

### Site `paper/` refresh (item A10)

| File | Source | SHA-256 | Change |
|---|---|---|---|
| `paper/draft.md` (site top level, not under `release/`) | `paper/draft.md` | `10b3fe944d13066ccac656defd004484a58bc3ba88571f8037699460dfb89043` | verbatim refresh from the research repo's current version (commit `6c66353`) |
| `paper/figures/*` (10 files: 6 PNG, 4 SVG) | `paper/figures/` | byte-identical copies | verbatim refresh |

Grep for the private path prefixes and username over `paper/draft.md` and
`paper/figures/*.svg`: 0 hits.

## Verification record

Grep battery run over every file above individually and over the whole
`release/` tree (case-insensitive, extended regex), covering: the private
home-directory path prefix, the private data-volume path prefix, the private
username (bare, and in `user@host` form), the gitignored private tool-state
directory name, credential vocabulary (key, provider token-prefix, and
passphrase patterns), live-matrix state names (the
scoring-queue directory prefix, socket-file suffix), and the matrix's total
step count — **0 hits for every pattern**, with two reviewed false positives
on looser variants of the patterns:

1. Bare `hf_` (without the token-length constraint) matches once:
   `scripts/audit_critic_temporal_preprocessing.py` docstring example path
   `/path/to/hf_exports/iter_000000100` — a directory name ("Hugging Face
   exports"), not a credential. Left intact.
2. The scoring-queue directory-name pattern matches once: the frozen prelaunch
   seal pre-registers a queue **name** in its scoring-binding list as part of
   its design. This is frozen protocol text (a design reference, not live state
   and not an outcome) and is preserved verbatim.

Item-specific verifications:

- A2: `grep -inE 'branch [A-F]|success rate|fall rate'` over both E1
  documents — 0 hits (no outcome text).
- A5–A7 + A8: reverse substitution (`<REPO>` → private home prefix, `<DATA>` →
  private data prefix) reproduces each private original byte-for-byte (for
  `run_report.md`, modulo the marked annotation block), proving sanitization
  touched only path prefixes. The full set of 64-hex SHA-256 strings in each
  sanitized copy equals the original's set.
- A8: `python -m json.tool` parses the released results JSON; `grep -iE
  'ppo|matrix|branch'` over it matches only the bootstrap "draw matrix" of the
  seed/index artifact (battery-internal resampling design, unrelated to the
  policy matrix); no `.npz` staged.
- A9: `pytest` over the four released test suites, run against the released
  script copies — 47 passed. Excluded materializer test: 49 passed against the
  released module copy in the private fixture harness (see exclusion note).
- Outcome-blindness: 0 occurrences anywhere in `release/` of any policy-matrix
  outcome, score, per-arm number, or the matrix step count; the protocols
  reference only the matrix's design, as they are entitled to.
