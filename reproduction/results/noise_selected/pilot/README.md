# Excluded exploratory runs

These runs preceded the final replicated validation protocol. None contributes
to `selection.json`, timing points, fitted slopes or accepted accuracy results.

`initial-mean-validation.json` was a single mean-fit smoke check.
`two_seed_selection/` used one noisy realization per geometry rather than the
final eight. Its parent was stopped before the final protocol began. The last
already-running child, `n32768-s0.05-ell0.12-seed3102`, finished while the helper
source was being revised: its recorded self-source hash may identify the new
file rather than the code loaded by that child. That record has ambiguous
self-source provenance and is excluded. Other pilot source hashes refer to
exploratory versions, not the final committed helper.

The final `validation/` directory is separate, source-verified, and uses two
geometry seeds times eight noise realizations per candidate.
