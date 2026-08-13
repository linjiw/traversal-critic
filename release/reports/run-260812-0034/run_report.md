# Shortcut-Battery Extension Run Report

> **Stage-A release annotation (added in this published copy; not part of the
> frozen report):** the two `.npz` artifacts listed below
> (`shortcut_battery_features.npz`, `shortcut_battery_seed_index_artifact.npz`)
> are withheld until Stage C of the release staging plan and remain bound to
> this report by the SHA-256 values printed below. Absolute private path
> prefixes in this copy were replaced by the placeholders `<REPO>` (private
> home-directory prefix) and `<DATA>` (private data-volume prefix); the
> published copy of `shortcut_battery_results.json` is sanitized the same way,
> so its SHA-256 printed below binds the private original, which differs from
> the published copy only in those path prefixes (see `RELEASE_MANIFEST.md`).

- Executed: 2026-08-12 (protocol frozen 2026-08-11)
- Authority: <REPO>/traversal-critic-research/docs/reviews/shortcut_battery_extension_protocol_2026-08-11.md
- Script: <REPO>/traversal-critic-research/scripts/run_shortcut_battery_extension.py
- Fingerprint replay: MATCH (train and val)
- Sanity gate: PASS (row 5 recomputed 0.6827 vs pilot 0.6827; shuffled-label control 0.0098)

## Artifact SHA-256

- `shortcut_battery_results.json`: `c61cb964d080c3409f91a0b765b1c06c51c9589efcdac39213dd08d9d6237588`
- `shortcut_battery_seed_index_artifact.npz`: `1bc39839f8514d0817487ae9b589a0debf879150f4c18f7daf5a163eff416f43`
- `shortcut_battery_features.npz`: `ab62f3e19ab1b8078b0b8d1f3a0fa11d49f9b8daa97139a6133214e611191afe`
- script `run_shortcut_battery_extension.py`: `f552faea2d8eef9100f07852b1470766c5c3eb5f6e69c54202afba1720004950`
- pilot results (input authority): `02aa81fd26857744b47e49c769e256354909d27e8d380f1ab29b34073f893ca2`
- pilot feature cache (input authority): `f8a9b5f81c94bca5bd240e0af8a31cf6692aab8808fcd36bb83f7636144fb852`
- protocol document: `4a34c6c9e3ce38f6808fa5dc77d582b3bb35ab8f9dc2f15310c4c90d9de873dd`

## Row summary (validation Pearson [95% CI descriptive], Spearman, rounded Pearson, lambda)

| Row | Key | Pearson | 95% CI | Spearman | Rounded Pearson | Lambda | OOF sel. Pearson |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | duration_only | 0.2687 | [0.1776, 0.3562] | 0.4293 | 0.2758 | 0.01 | 0.2439 |
| 2 | first_frame_only | 0.1480 | [0.0374, 0.2566] | 0.1342 | 0.1024 | 1 | 0.1357 |
| 3 | terminal_frame_only | 0.6661 | [0.6049, 0.7223] | 0.7003 | 0.6522 | 100 | 0.6358 |
| 4 | first_plus_terminal | 0.6496 | [0.5785, 0.7133] | 0.6665 | 0.6308 | 100 | 0.6106 |
| 5 | duration_plus_terminal | 0.6827 | [0.6349, 0.7300] | 0.7820 | 0.6665 | 100 | 0.6662 |
| 6 | duration_first_terminal | 0.6737 | [0.6159, 0.7265] | 0.7540 | 0.6625 | 100 | 0.6449 |
| 7 | random_single_frame | 0.3719 | [0.2719, 0.4613] | 0.3776 | 0.3220 | 100 | 0.3596 |
| 8 | onset_prefix_25 | 0.3259 | [0.2240, 0.4244] | 0.3253 | 0.2575 | 100 | 0.2921 |
| 9 | onset_prefix_50 | 0.4401 | [0.3348, 0.5350] | 0.4494 | 0.4120 | 100 | 0.3487 |
| 10 | endpoint_masked_last_10 | 0.5737 | [0.4954, 0.6441] | 0.5954 | 0.5009 | 100 | 0.5520 |
| 11 | endpoint_masked_last_25 | 0.4915 | [0.4212, 0.5597] | 0.5511 | 0.4500 | 100 | 0.4949 |
| 12 | shuffled_order_first_plus_terminal | 0.6415 | [0.5831, 0.6941] | 0.6751 | 0.5956 | 100 | 0.5805 |
| 13 | shuffled_order_duration_first_terminal | 0.6575 | [0.6136, 0.6999] | 0.7513 | 0.6101 | 100 | 0.6109 |
| 14 | shuffled_order_onset_prefix_25 | 0.2374 | [0.1419, 0.3281] | 0.2324 | 0.2203 | 100 | 0.2585 |
| 15 | shuffled_order_onset_prefix_50 | 0.3217 | [0.2164, 0.4211] | 0.3444 | 0.3206 | 100 | 0.3175 |
| 16 | shuffled_label_control | 0.0098 | [-0.0990, 0.1210] | 0.0100 | -0.0045 | 0.0001 | 0.0368 |
| 17 | post_hoc_pilot (duration+terminal) | 0.6827 | n/a (reprint) | 0.7820 | 0.6665 | 100 | 0.6674 |

Row 17 reprints the post-hoc pilot; its other reprinted rows (duration-only,
terminal-only, permuted-labels) are in `post_hoc_pilot` inside the results JSON.
All intervals are descriptive. Ratios vs critic 0.5646 (mismatched-interface
historical record) and probe 0.7053 are per row in the results JSON.
