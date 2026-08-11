# Timestep validation

## Long-run correction (2026-08-11)

The historical 200 ps NVE recommendation below is superseded. No Cranberry NVE
timestep has yet passed a 1 microsecond 1l2x validation. Full-model runs failed
at 5 fs near 135.75 ns, 3 fs near 294 ns with initial COM velocity removed,
and 2.75 fs near 421 ns. A 2.5 fs run reached only 60 ns before it was stopped.
Use 1 fs only as a provisional diagnostic timestep with explicit energy-drift
monitoring; do not describe it as 1 microsecond validated.

The 8 fs Langevin recommendation remains supported at 298 K by a completed
1 microsecond 1l2x survival run, local-coordinate comparisons against 1 fs,
and constrained velocity/kinetic-energy distribution checks. Those checks are
necessary but not sufficient: configurational reference comparisons, not a
Maxwell velocity histogram alone, are the primary finite-timestep criterion.

The 3 fs failure is progressive numerical heating rather than an isolated bond
rupture. Total energy rose by 8812 kJ/mol and the instantaneous kinetic
temperature rose from about 307 K to 2630 K before the first recorded physical
bond excursion above 0.1 nm (residue 10 P-S3 at 293.068 ns). Force ablations
localize the dominant sensitivity to the C3 sugar bonded geometry, especially
the 27000 kJ/mol/nm^2 S2-B1 coordinate, with additional spline, pairing, and
WCA contributions. C3-only compound and split ordinary bond/angle plus CBT
torsion implementations both drifted strongly, so CustomCompoundBondForce
itself is not the identified cause.

---

The historical short-screen analysis below had proposed:

- ~~`dt_nve = 5 fs` for energy-conserving dynamics~~ (withdrawn by the long-run correction above).
- `dt_nvt = 8 fs` for production Langevin dynamics and REMD.

The existing 5 fs runtime default is a legacy operational value, not a validated NVE setting, for commands
that do not distinguish ensembles. An NVT/REMD workflow may explicitly select
8 fs. These values are model-specific recommendations, not universal stability
limits.

There is no separate `dt_nvt_dynamics`. The recommendations apply to the
current Cranberry masses, constraints, forces, and OpenMM integrators. Changes
to any of those require rerunning `benchmarks/timestep_study.py`.

## Methods

All reported production measurements used OpenMM 8.5.2 on an RTX 2060 with
CUDA mixed precision. Systems represented a two-nucleotide fragment
(`2ntCG`), folded RNAs (`1zih`, `157d`, and `1l2x`), and a long
single strand (`rU40`). Tests covered 298 K and 420 K.

Each condition started from a 1 fs Langevin-equilibrated configuration.
Independent seeds received independently drawn velocities. NVE removed the
initial center-of-mass velocity and used `VerletIntegrator`; NVT used
`LangevinMiddleIntegrator` with Cranberry's production friction. Raw samples
were written after every condition, so interrupted matrices can resume without
repeating completed work.

The short screens covered 1--10 fs in NVE and 1--15 fs in NVT for all five
systems. Confirmation used 200 ps trajectories:

- NVE: `2ntCG`, `1l2x`, and `rU40`; 298 and 420 K; three seeds; 1, 3,
  5, and 8 fs. There were 18 trajectories per timestep.
- NVT: `1l2x`; 298 and 420 K; three seeds; a 0.5 fs reference and 1, 5, 8,
  12, and 15 fs candidates. Each candidate had 20 ps burn-in followed by
  200 ps production.
- GHMC diagnostic: `2ntCG`, `1l2x`, and `rU40`; 298 and 420 K; 1--15
  fs. One-step Metropolization measures local Verlet proposal error without
  requiring independently diverged trajectories to visit the same global RNA
  conformation.
- NVT geometry: `2ntCG` and `1zih`; 298 and 420 K; three seeds; a 1 fs
  reference and 5, 8, 10, and 12 fs candidates. Each condition used 100 ps
  candidate burn-in and 300 ps production, sampled every 0.2 ps.

There is no universal pass percentage for either ensemble. NVE selection used
bounded total-energy oscillation and convergence as the timestep was reduced;
the fitted drift was retained as a secondary diagnostic because CUDA mixed
precision itself contributes drift. NVT selection compared local coordinate
distributions with 1 fs. Wasserstein-1 distances were normalized by the 1 fs
standard deviation and compared with leave-one-seed-out disagreement among the
three 1 fs replicas.

## NVE result

Drift was obtained by least-squares regression of total energy against time
and normalized by `DOF R T`. Oscillatory energy error was reported as RMS
total-energy fluctuation on the same scale.

| Timestep (fs) | Median absolute drift (kBT/DOF/ns) | 90th percentile | Maximum | Median RMS fluctuation (kBT/DOF) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00256 | 0.01057 | 0.01169 | 0.001132 |
| 3 | 0.01173 | 0.03205 | 0.05366 | 0.001477 |
| 5 | 0.01612 | 0.03568 | 0.05361 | 0.002084 |
| 8 | 0.04508 | 0.12397 | 0.22737 | 0.004036 |

The earlier draft imposed an ad hoc acceptance region, then selected one grid
point below the largest passing value. That rule was not taken from a published
validation protocol and was too conservative. The data support **5 fs as the
current NVE recommendation**: its median RMS error is 0.00208 kBT/DOF, while 8 fs
roughly doubles that error and has a much larger tail of fitted short-trajectory
drift.

The 5 fs recommendation is therefore a tolerance choice at the observed
convergence knee, not a mathematical stability boundary. A 1 ns replicated
double-precision matrix was attempted, but an RTX 2060 is too slow in CUDA
double precision and the CPU Reference platform required about 40 seconds for
only 100 ps of 2ntCG at 1 fs. The mixed-precision 18-replica comparison is the
reviewed production-platform result.

## NVT result

Survival was not used as the decision rule: every 20 ps screening trajectory
survived even at 15 fs. Direct 0.5 fs comparisons showed no monotonic bias in
bond energy, dihedral energy, or kinetic temperature through 5 fs. Global
pairing, stacking, and radius-of-gyration means varied non-monotonically across
all timesteps, consistent with ordinary metastable trajectory divergence over
200 ps rather than a timestep trend.

GHMC acceptance was also measured as a local-error diagnostic:

| Timestep (fs) | Minimum acceptance | Median acceptance |
| ---: | ---: | ---: |
| 1 | 0.99588 | 0.99800 |
| 3 | 0.98428 | 0.99118 |
| 5 | 0.95960 | 0.97880 |
| 8 | 0.89088 | 0.93072 |
| 10 | 0.86440 | 0.87620 |
| 12 | 0.78119 | 0.79990 |
| 15 | 0.61285 | 0.64076 |

The earlier 95% cutoff was an arbitrary project-local screen, not a standard
GHMC criterion. GHMC rejection is not a calibrated measure of configurational
bias in the corresponding un-Metropolized Langevin integrator, so it cannot
set the NVT boundary. A follow-up 2ntCG screen (three seeds, 500 ps production,
298 and 420 K) found no monotonic bond or pucker energy bias through 15 fs. At
8 fs the mean bond energy differed from 1 fs by -0.5% and -0.7%, respectively.
Energy means can hide geometric redistribution, so the follow-up compared every
stiff bond, P--S3 bond, S2--B1 pucker bond, and inferred sugar phase. Raw phase
occupancy differed strongly and non-monotonically among replicas because 300 ps
does not converge C2-prime-endo/C3-prime-endo transitions. Comparisons were
therefore also conditioned on pucker state. At 8 fs, the 95th-percentile
normalized Wasserstein distances across the four system/temperature combinations
were 0.05--0.12 for P--S3, 0.13--0.49 for S2--B1, and 0.13--0.26 for phase. The
corresponding 1 fs leave-one-seed-out ranges were 0.08--0.14, 0.22--0.36, and
0.14--0.24. Occasional excesses were small, below half a reference standard
deviation, and did not increase monotonically at 10 or 12 fs. Thus no local
configurational bias distinguishes 8 fs from the 1 fs sampling uncertainty.

**8 fs is the NVT/REMD recommendation.** It is deliberately below the region
where the fastest local mode has fewer than about ten integration points per
period. This is not an assertion that 10 or 12 fs is unstable; those values
need more sampling before they could be recommended.

## Cross-model and fast-mode audit

3SPN.2 reports 20 fs Langevin production but no timestep-convergence criterion.
Its 83--150 Da sites and soft-quadratic/quartic bonds are dynamically unlike
Cranberry. Released 1CPN inputs use 60 fs for Langevin production and 20 fs for
three NVE integration tests; those tests only enforce a 10% total-energy band.
iConRNA uses 8 fs Langevin after 1 fs equilibration, with no published timestep
convergence found. The original RACER paper used 4 fs, not 20 fs.

A constrained, mass-weighted Hessian of minimized 2ntCG gives a 115 fs fastest
period. About 63% of its curvature is ordinary bond and 37% pucker, concentrated
in coupled pyrimidine-base/sugar motion. Pairing, stacking, spline, and
electrostatics contribute negligibly. A force-group NVE ablation confirmed the
same limiter: at 20 fs, removing both bond and pucker reduced oscillatory energy
error by roughly 200-fold. Coarse graining permits a larger timestep only when
it removes, softens, constrains, or mass-loads the fastest degrees of freedom.

The 115 fs period gives about 23, 14, and 5.8 steps per cycle at 5, 8, and
20 fs. That explains why Cranberry cannot inherit 3SPN.2's 20 fs solely by
being coarse-grained: Cranberry retains light 26 Da sites, 27,000--32,000
kJ/mol/nm^2 unconstrained bonds, and a stiff state-dependent S2--B1 pucker
coordinate. Released 3SPN.2 sites are 83--150 Da and its local bond curvature
is much softer.

## PBC and OpenMMTools failure

The historical sugar-pucker failure was not caused by replica swaps. OpenMMTools requests
getState(enforcePeriodicBox=True) after each MCMC move, stores the
resulting coordinates, and reapplies them to the next context.

The old System partitioned periodic 1l2x into 28 imaging components: the
81-particle backbone was one component, while each base-site group was separate.
The input Topology did contain the sugar/base edges, but Cranberry omitted them
from every System bond or constraint because their energy was evaluated by
custom forces. Context.getMolecules() therefore could not infer the intended
molecule graph.

When an intact RNA was translated across a box face, the old nonperiodic pucker
energy changed from 854.1646 to 8,295,034.6808 kJ/mol after an
OpenMMTools-style state round-trip. The old periodic pucker flag hid this by
applying minimum-image displacements inside a bonded custom force, which is not
the intended model.

The corrected System declares the 135 omitted topology edges as zero-energy
HarmonicBondForce connectivity bonds, reports one 216-particle molecule, and
keeps pucker ordinary and nonperiodic. The same round-trip then changes energy
by only 1e-12 kJ/mol.

OpenMM itself is behaving as documented. Context.getMolecules() groups particles
connected by bonds or constraints, and enforcePeriodicBox=True translates each
complete molecule into one box. OpenMM CustomCompoundBondForce documentation
also says periodic bonded displacements are usually inappropriate. The failure
was that Cranberry did not declare all of its bonded topology in the System,
not that OpenMM failed to preserve a bond.

OpenMMTools still makes the consequential choice of recycling an imaged State as
the subsequent move input. That is reasonable for a System with a correct
molecule graph. It exposed Cranberry missing connectivity because the old System
invited OpenMM to image base groups independently. This is therefore a Cranberry
System-construction bug exposed by an ordinary OpenMMTools operation, not an
OpenMMTools PBC bug.
## Reproduction

The commands used the following output files:

- benchmarks/results/timestep-nve-screen-rtx2060.json
- benchmarks/results/timestep-nve-confirm-rtx2060.json
- benchmarks/results/timestep-nvt-screen-rtx2060.json
- benchmarks/results/timestep-nvt-confirm-1l2x-rtx2060.json
- benchmarks/results/timestep-ghmc-screen-rtx2060.json

The frame-level files are local study artifacts and are not intended for source
## External model sources
- 3SPN.2 methods: https://pmc.ncbi.nlm.nih.gov/articles/PMC3808442/
- 1CPN production input: https://github.com/lequieu/1cpn-model/blob/master/inputs/in.1cpn
- 1CPN NVE test: https://github.com/lequieu/1cpn-model/blob/master/test/integ_tests/1cpn_dinucl_nve/in.1cpn
- iConRNA script: https://github.com/lslumass/iConRNA/blob/main/scripts/run.py
- RACER methods: https://pmc.ncbi.nlm.nih.gov/articles/PMC5385882/
- GHMC timestep study: https://pmc.ncbi.nlm.nih.gov/articles/PMC6208357/
