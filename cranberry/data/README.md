# CRANBERRY Package Data

These runtime assets are distributed with `cranberry-rna`. CRANBERRY-authored model files and generated fixtures are released under MIT. Source coordinate records obtained from the Protein Data Bank are available under the wwPDB CC0 dedication; accession attribution is retained below.

## Model Assets

| File | SHA-256 | Provenance |
| --- | --- | --- |
| `forcefields/cranberry-v1-alpha.1.h5` | `f9285bb07f45c539957c76fd5b6313680b1f0254b0675b990e16a4fe3a1833a3` | Canonical merged CRANBERRY alpha parameter bundle. Bonded, angular, dihedral, and sugar-pucker terms came from legacy `6spn.h5`. Nonbonded, WCA, spline, stacking, and pairing terms came from `cranberry0.2.19.h5`. Angle spring constants include the legacy `0.1` runtime scaling. |
| `xml/cranberry.xml` | `14c53d546dd1b2067a9be177be004f56848e66aa45b37bec89198910e6f7d2ab` | Canonical CRANBERRY topology and virtual-site definitions, migrated unchanged from legacy `6spn.vs.xml`. |

Source hashes:

- legacy `6spn.h5`: `e80b5fc5a490420b93d8abf4bbcd722a7410824a5f5e47308f12e78b6c283627`
- legacy `cranberry0.2.19.h5`: `28431d43c8e2f9d531f0b93c41b97cc4599211c36ddbfadc0edc00eca4ce5cab`

## Example Structures

| File | SHA-256 | Source and transformation |
| --- | --- | --- |
| `examples/157d_cg_vs_conect.pdb` | `78b6c654a40027fb13e3728d7fecb6f0241d1a9d1425ff62026a32e9382493dd` | Coarse-grained from PDB 157D. |
| `examples/1l2x_cg_vs_conect.pdb` | `7edecacea0a123906bb52a280695994c3481adc40c42c0eef1f3736386af008c` | Coarse-grained from PDB 1L2X. |
| `examples/1zih_cg_vs_conect.pdb` | `53df9c2253e35c808a2e0e170fad80265b541b007bb082de8f48aaced0de1706` | Coarse-grained from PDB 1ZIH. The atomistic regression input is retained under `examples/aa/1zih/`. |
| `examples/2MI0_cg_vs_conect.pdb` | `b8dcbb5e3f13da1c90d456d08970d661a37c53fec9b3a22ebe44fffaca6cf16e` | Coarse-grained NMR structure from PDB 2MI0. |
| `examples/2ntCG_cg_vs_conect.pdb` | `6e4d99f57d03cb1ef0295b416353dfa70908c0c7def415b15f786eb57706795e` | Two-residue fixture extracted from the 157D-derived coarse-grained structure. |
| `examples/5ml7_cg_vs_conect.pdb` | `6e7a68542e218b08fa5492dbda8ce2997986986cd3c1ab3160cd0893ad080699` | Coarse-grained from PDB 5ML7. |
| `examples/ggcGCAAgcc_cg_vs_conect.pdb` | `341f701020c33e4044cd998782b5c127f82da044dbafa424fc7ade5913de420a` | Coarse-grained from PDB 1ZIH after removing the terminal G/U residues to obtain `GGCGCAAGCC`. |
| `examples/ggcGCAAgcc_extended_cg_vs_conect.pdb` | `2559f86ec114b8c5a8c2dc326e61c6a6041b655c03da2da68d981d4e7a923441` | Sequence `GGCGCAAGCC` generated as a single-stranded RNA fiber model with the Web 3DNA server, then coarse-grained with the legacy CRANBERRY workflow. |
| `examples/rU40_cg_vs_conect.pdb` | `9c8b52efdd0b7ab3f049a2b79c7ba8a4ed435dd79cf1359ea98c0636791415ce` | Sequence U40 generated as a single-stranded RNA fiber model with the Web 3DNA server, then coarse-grained with the legacy CRANBERRY workflow. |

The generated CRANBERRY fixtures were prepared by Yiheng Wu. Where canonical atomistic inputs are available, the current `cranberry cg` command is the preferred route for new structures.

## External Sources

- Protein Data Bank entries are identified by accession and DOI in the repository's `THIRD_PARTY_NOTICES.md`.
- The extended single-stranded structures were generated with the Web 3DNA fiber builder in ssRNA mode. The historical web-server version was not recorded.
- Web 3DNA citation: S. Li, W. K. Olson, and X.-J. Lu, "Web 3DNA 2.0 for the analysis, visualization, and modeling of 3D nucleic acid structures," *Nucleic Acids Research* 47, W26-W34 (2019), https://doi.org/10.1093/nar/gkz394.
- 3DNA citation: X.-J. Lu and W. K. Olson, "3DNA: a software package for the analysis, rebuilding and visualization of three-dimensional nucleic acid structures," *Nucleic Acids Research* 31, 5108-5121 (2003), https://doi.org/10.1093/nar/gkg680.

## Reproducibility

Packaged files are immutable within a released package version. A scientifically meaningful parameter or fixture change requires a new package/model version, updated hashes and provenance, and regression validation.
