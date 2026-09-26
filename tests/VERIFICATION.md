# Repository reorganization verification

Local verification on 2026-09-27, Windows, existing Python 3.11.9 environment.
Direct dependencies: NumPy 2.4.6, Matplotlib 3.11.2, Pillow 12.3.0.

## Preservation and execution

- The shared core, seven experimental scripts, eight numerical diagnostic scripts
  and original VPython script were preserved byte-for-byte. All historical
  `posiciones*.txt` files and the old README were also compared against a backup.
- Only result-directory resolution changed in the three final scripts. No physics,
  integrator, mission parameter or diagnostic threshold was changed.
- All Python files passed syntax parsing and shared-core import resolution.
- Final simulation completed with 14,957 stored points. Regenerated CSV SHA-256:
  `9ad8806d344d06f74a09b46bb303cb00186f6ef2f89f97dae17fee4f87c77686`.
  This exactly matches the CSV present before reorganization.
- Final flyby altitude: 300.0061032088047 km; closest approach:
  243.87774603044983 days; relative speed: 5.5432624994224655 km/s.
- The plotting script regenerated its five PNG outputs. The animation completed
  all 700 update calls, rendered frames 0/350/699 and exported a three-frame GIF
  using Pillow. The GIF frame count was checked. Plotting and animation were run
  from outside the repository to verify path independence.
- The heliocentric plot and a mid-mission animation frame were visually inspected.
- `python -m experiments.main` and `python -m experiments.mars_transfer` completed.
  Long optimization sweeps were not rerun; their source is unchanged.

## Completed diagnostics

All eight commands exited with code 0:

```text
python -m tests.test_earth_orbit
python -m tests.test_time_step_convergence
python -m tests.test_barycentric_convergence
python -m tests.test_escape_convergence
python -m tests.test_hybrid_timestep
python -m tests.test_transfer_timestep
python -m tests.test_mars_encounter_convergence
python -m tests.final_phase_validation
```

## Numerical interpretation

The existing diagnostics print measurements rather than assert pass/fail bounds.
Their successful execution checks relocation and runtime compatibility, not a
universal claim of scientific accuracy.

- Earth orbit: maximum relative energy error approximately 2.71e-10.
- Barycentric convergence: observed orders 1.99838–1.99998.
- Escape refinement: last position difference approximately 77.61 km and velocity
  difference 0.000190 km/s. This is not evidence of metre-level departure accuracy.
- Hybrid cruise refinement: closest-approach distance difference 0.0394 km.
- Mars encounter refinement: last altitude difference 0.00219 km; observed order
  for that sampled periapsis quantity is approximately 0.842, not 2. The diagnostic
  uses an earlier phase candidate and returns approximately 316.4 km altitude,
  so it must not be mistaken for the final 300 km trajectory.
- The historical fixed-step transfer test has a 127,719.48 km closest-approach
  difference between its last two resolutions and a 3.54548-day timing difference.
  This configuration is not numerically converged; it is preserved as a historical
  diagnostic and must not be cited as validation of the final hybrid trajectory.
- Final phase validation prints `FINAL PHASE ACCEPTED`, with a 0.00610 km error
  against its 300 km target and 10 km acceptance tolerance. Agreement with this
  numerical target does not establish absolute physical accuracy.

## Scope and limitations

Matplotlib used the noninteractive Agg backend. Warnings that windows cannot be
shown are expected; a font-cache write warning did not prevent rendering.
Interactive desktop playback, the full 700-frame GIF export and the legacy
VPython/Tkinter UI were not exercised. No fresh environment installation was
attempted; dependency pins reflect the working local environment rather than a
fully resolved transitive lockfile.

Detailed local numerical logs and animation smoke artifacts are retained under
`results/verification/` (ignored by Git). Existing extra PNGs were retained.
The pre-change backup is outside the repository in the task workspace as
`ejercicio_final31-before-cleanup.zip`. No commit, push or history rewrite was made.

## Final Git review

`git diff --check` passed. The index was left unchanged. `git status` shows the
original tracked paths as deleted and their new `legacy/` paths as untracked
until staging; the files themselves are preserved. The new core, results,
experiments, tests, requirements and ignore rules are ready to stage together.
The environment, bytecode and verification artifacts are correctly ignored.
