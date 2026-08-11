from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from openmm import app, unit
import openmm as mm

from cranberry.data import data_path

FORCE_GROUP_NAMES = (
    "bond",
    "angle",
    "dihedral",
    "pucker",
    "stacking35",
    "stacking55",
    "stacking33",
    "pairing",
    "wca",
    "spline",
    "electrostatic",
)

# Keep legacy group IDs where they matter for migration logs. These IDs are an
# internal detail; public reporting should use names.
FORCE_GROUP_IDS = {
    "bond": 1,
    "angle": 2,
    "dihedral": 3,
    "wca": 5,
    "electrostatic": 7,
    "spline": 9,
    "pucker": 10,
    "stacking35": 11,
    "stacking55": 12,
    "stacking33": 13,
    "pairing": 14,
}

RESTYPE_TO_INDEX = {"A": 0, "U": 1, "G": 2, "C": 3}
STACKING_COMPONENT_NAMES = ("stacking35", "stacking55", "stacking33")
FUSED_STACKING_FORCE_NAME = "stacking"
STACKING_SCALE_PARAMETERS = {
    name: f"{name}_scale" for name in STACKING_COMPONENT_NAMES
}
_PAIRING_CUTOFF_NM = 0.8
# createSystem() does not know the eventual OpenMM platform, so this is a
# conservative, platform-independent bound on explicit pair storage.
_PAIRING_COMPOUND_MAX_BONDS = 8192
BASE_TYPE_TO_CG_NAMES = {
    "A": ("R1", "A1", "A2"),
    "U": ("Y1", "U1", "Y2"),
    "G": ("R1", "G1", "G2"),
    "C": ("Y1", "C1", "Y2"),
}
HIGH_RES_SPLINE_TYPES = {
    "P": 0,
    "S3": 1,
    "S2": 2,
    "A1": 3,
    "A2": 4,
    "G1": 5,
    "G2": 6,
    "U1": 7,
    "C1": 8,
    "R1": 9,
    "Y1": 10,
    "Y2": 11,
    "BC": -1,
    "BN": -1,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameter_file: str
    xml_file: str
    description: str

    @property
    def parameter_path(self) -> Path:
        return data_path(self.parameter_file)

    @property
    def xml_path(self) -> Path:
        return data_path(self.xml_file)


_MODEL_REGISTRY = {
    "cranberry-v1-alpha.1": ModelSpec(
        name="cranberry-v1-alpha.1",
        parameter_file="forcefields/cranberry-v1-alpha.1.h5",
        xml_file="xml/cranberry.xml",
        description="CRANBERRY v1 alpha model bundle.",
    ),
}
_DEFAULT_MODEL = "cranberry-v1-alpha.1"


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def default_model_name() -> str:
    return _DEFAULT_MODEL


def get_model_spec(name: str = "default") -> ModelSpec:
    if name == "default":
        name = _DEFAULT_MODEL
    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(["default", *available_models()])
        raise ValueError(f"Unknown CRANBERRY model {name!r}. Available models: {known}") from exc


@dataclass(frozen=True)
class _TopologyData:
    atom_names: list[str]
    residue_names: list[str]
    residue_indices: list[int]
    chain_indices: list[int]
    bonds: list[tuple[int, int]]
    adjacency: dict[int, set[int]]
    virtual_site_summaries: list[tuple[tuple[int, str, int, int, str, int], tuple[int, str, int, int, str, int]]]
    real_site_ids: set[int]
    real_or_cog_site_ids: set[int]


def prepare_periodic_positions(topology: app.Topology, positions, box_padding=2.0 * unit.nanometer):
    """Center positions in a generated cubic periodic box and attach it to topology."""

    return prepare_common_periodic_positions(topology, (positions,), box_padding)[0]


def prepare_common_periodic_positions(
    topology: app.Topology,
    position_sets,
    box_padding=2.0 * unit.nanometer,
):
    """Center multiple conformations independently in one shared cubic box."""

    position_sets = tuple(position_sets)
    if not position_sets:
        raise ValueError("at least one position set is required")
    box_sizes = [
        _periodic_box_size_and_translation(positions, box_padding)[0]
        for positions in position_sets
    ]
    box_size = max(
        box_sizes,
        key=lambda size: size.value_in_unit(unit.nanometer),
    )
    _set_cubic_box_vectors(topology, box_size)
    half_box_nm = box_size.value_in_unit(unit.nanometer) / 2.0
    centered = []
    for positions in position_sets:
        coordinates = positions.value_in_unit(unit.nanometer)
        center_nm = 0.5 * (np.min(coordinates, axis=0) + np.max(coordinates, axis=0))
        translation = (-center_nm + half_box_nm) * unit.nanometer
        centered.append(positions + translation)
    return tuple(centered)


def _configure_periodic_box(topology: app.Topology, positions, box_padding) -> None:
    box_size, _translation = _periodic_box_size_and_translation(positions, box_padding)
    _set_cubic_box_vectors(topology, box_size)


def _periodic_box_size_and_translation(positions, box_padding):
    padding = _as_quantity(box_padding, unit.nanometer).in_units_of(unit.nanometer)
    coordinates = positions.value_in_unit(unit.nanometer)
    span = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
    box_size_nm = float(np.max(span)) + 2.0 * padding.value_in_unit(unit.nanometer)
    if box_size_nm <= 0:
        raise ValueError("periodic box size must be positive")
    center_nm = 0.5 * (np.min(coordinates, axis=0) + np.max(coordinates, axis=0))
    translation_nm = -center_nm + box_size_nm / 2.0
    return box_size_nm * unit.nanometer, translation_nm * unit.nanometer


def _set_cubic_box_vectors(topology: app.Topology, box_size) -> None:
    zero = 0.0 * unit.nanometer
    topology.setPeriodicBoxVectors(((box_size, zero, zero), (zero, box_size, zero), (zero, zero, box_size)))


def validate_periodic_box_cutoffs(system: mm.System) -> None:
    """Fail early when a periodic cutoff is larger than half the box length."""

    vectors = system.getDefaultPeriodicBoxVectors()
    if vectors is None:
        return
    lengths_nm = []
    for vector in vectors:
        components_nm = vector.value_in_unit(unit.nanometer)
        lengths_nm.append(float(np.sqrt(sum(component * component for component in components_nm))))
    half_min_box_nm = 0.5 * min(lengths_nm)
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        get_cutoff = getattr(force, "getCutoffDistance", None)
        get_method = getattr(force, "getNonbondedMethod", None)
        cutoff_nm = None
        if get_cutoff is not None and get_method is not None:
            method = get_method()
            periodic_method = getattr(force, "CutoffPeriodic", None)
            if periodic_method is not None and method == periodic_method:
                cutoff_nm = get_cutoff().value_in_unit(unit.nanometer)
        elif (
            isinstance(force, mm.CustomCompoundBondForce)
            and force.getName() == "pairing"
            and force.usesPeriodicBoundaryConditions()
        ):
            cutoff_nm = _PAIRING_CUTOFF_NM
        if cutoff_nm is None:
            continue
        if cutoff_nm > half_min_box_nm:
            name = force.getName() or type(force).__name__
            raise ValueError(
                "Periodic box is too small for force cutoff: "
                f"force {name!r} cutoff is {cutoff_nm:.6g} nm, but half the smallest box length is "
                f"{half_min_box_nm:.6g} nm. Increase --box-padding or use a larger periodic box."
            )


class CranberryForceField:
    """OpenMM-style force-field object for the canonical CRANBERRY model."""

    def __init__(self, model: str = "default"):
        self.spec = get_model_spec(model)
        self._xml = app.ForceField(str(self.spec.xml_path))
        self._params = _load_parameters(self.spec.parameter_path)

    def createSystem(
        self,
        topology: app.Topology,
        *,
        positions=None,
        temperature=298 * unit.kelvin,
        salt_concentration=150 * unit.millimolar,
        enabled_forces: Iterable[str] | None = None,
        periodic: bool = False,
        box_padding=2.0 * unit.nanometer,
    ) -> mm.System:
        """Create an OpenMM System from a canonical CRANBERRY CG topology."""

        if periodic:
            if positions is None:
                raise ValueError("positions are required when periodic=True")
            _configure_periodic_box(topology, positions, box_padding)
        enabled = set(FORCE_GROUP_NAMES if enabled_forces is None else enabled_forces)
        unknown = enabled - set(FORCE_GROUP_NAMES)
        if unknown:
            raise ValueError(f"Unknown force names: {', '.join(sorted(unknown))}")

        system = self._xml.createSystem(topology)
        data = _collect_topology(topology)

        sugar_indices: dict[str, list] = {"bond": [], "angle": [], "dihedral": []}
        angle_indices_all: list[tuple[int, int, int]] = []
        dihedral_indices: list[tuple[int, int, int, int]] = []

        if "bond" in enabled:
            self._add_bonds(system, data, sugar_indices)
        if "angle" in enabled:
            angle_indices_all = self._add_angles(system, data, sugar_indices)
        if "dihedral" in enabled:
            dihedral_indices = self._add_dihedrals(system, data, sugar_indices, angle_indices_all)
        if "pucker" in enabled:
            self._add_sugar_pucker(system, topology, periodic=periodic)
        stacking_components = tuple(
            name for name in STACKING_COMPONENT_NAMES if name in enabled
        )
        if stacking_components:
            self._add_stacking(
                system, data, stacking_components, periodic=periodic
            )
        if "pairing" in enabled:
            self._add_pairing(system, data, periodic=periodic)
        if "wca" in enabled:
            self._add_wca(system, data, periodic=periodic)
        if "spline" in enabled:
            self._add_spline(system, data, base_idx_lim=3, periodic=periodic)
        if "electrostatic" in enabled:
            self._add_debye_huckel(system, data, temperature, salt_concentration, periodic=periodic)

        return system

    def _add_bonds(self, system: mm.System, data: _TopologyData, sugar_indices: dict[str, list]) -> None:
        force = mm.HarmonicBondForce()
        represented_edges = set()
        for i, j in data.bonds:
            idi, idj = data.residue_indices[i], data.residue_indices[j]
            nmi, nmj = data.atom_names[i], data.atom_names[j]
            if idi <= idj:
                name = f"{nmi}-{nmj}"
                postfix = f"0{idj - idi}"
            else:
                name = f"{nmj}-{nmi}"
                postfix = f"0{idi - idj}"
            found = _find_parameter(self._params["bond"], name, postfix)
            if found is not None:
                length, k = found[0]
                if np.isinf(k):
                    represented_edges.add(frozenset((i, j)))
                    system.addConstraint(i, j, float(length) * unit.nanometer)
                else:
                    force.addBond(i, j, float(length), float(k))
                    represented_edges.add(frozenset((i, j)))
            elif _find_parameter(self._params["sugar_bonded"], name, postfix) is not None:
                sugar_indices["bond"].append((i, j))
        # Custom bonded terms do not contribute molecule connectivity. Declare
        # every remaining topology edge as zero-energy bond connectivity.
        for i, j in data.bonds:
            if frozenset((i, j)) not in represented_edges:
                force.addBond(i, j, 0.0, 0.0)
        force.setForceGroup(FORCE_GROUP_IDS["bond"])
        force.setName("bond")
        system.addForce(force)

    def _add_angles(self, system: mm.System, data: _TopologyData, sugar_indices: dict[str, list]) -> list[tuple[int, int, int]]:
        force = mm.HarmonicAngleForce()
        angle_indices_all: list[tuple[int, int, int]] = []
        for center in range(len(data.atom_names)):
            for pre, post in combinations(sorted(data.adjacency[center]), 2):
                angle_indices_all.append((pre, center, post))

        for i, j, k in angle_indices_all:
            idi, idj, idk = data.residue_indices[i], data.residue_indices[j], data.residue_indices[k]
            if idi <= idk:
                name = f"{data.atom_names[i]}-{data.atom_names[j]}-{data.atom_names[k]}"
                postfix = f"0{idj - idi}{idk - idi}"
            else:
                name = f"{data.atom_names[k]}-{data.atom_names[j]}-{data.atom_names[i]}"
                postfix = f"0{idj - idk}{idi - idk}"
            found = _find_parameter(self._params["angle"], name, postfix)
            if found is not None:
                theta0, k_angle = found[0]
                force.addAngle(i, j, k, float(theta0), float(k_angle))
            elif _find_parameter(self._params["sugar_bonded"], name, postfix) is not None:
                sugar_indices["angle"].append((i, j, k))
        force.setForceGroup(FORCE_GROUP_IDS["angle"])
        force.setName("angle")
        system.addForce(force)
        return angle_indices_all

    def _add_dihedrals(
        self,
        system: mm.System,
        data: _TopologyData,
        sugar_indices: dict[str, list],
        angle_indices_all: list[tuple[int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        del angle_indices_all
        force = mm.CustomCompoundBondForce(
            4,
            "(sin(theta1))^3*(sin(theta2))^3*(U);"
            "U=k1*(1+cos(phi-a1));"
            "theta1=angle(p1,p2,p3);"
            "theta2=angle(p2,p3,p4);"
            "phi=dihedral(p1,p2,p3,p4);",
        )
        force.addPerBondParameter("a1")
        force.addPerBondParameter("k1")
        dihedral_indices: list[tuple[int, int, int, int]] = []
        for n1, n2 in data.bonds:
            for pre in data.adjacency[n1] - {n2}:
                for post in data.adjacency[n2] - {n1}:
                    if pre == post:
                        continue
                    dihedral_indices.append((pre, n1, n2, post))

        for i, j, k, l in dihedral_indices:
            idi, idj, idk, idl = (data.residue_indices[x] for x in (i, j, k, l))
            if idi <= idl:
                name = f"{data.atom_names[i]}-{data.atom_names[j]}-{data.atom_names[k]}-{data.atom_names[l]}"
                postfix = f"0{idj - idi}{idk - idi}{idl - idi}"
            else:
                name = f"{data.atom_names[l]}-{data.atom_names[k]}-{data.atom_names[j]}-{data.atom_names[i]}"
                postfix = f"0{idk - idl}{idj - idl}{idi - idl}"
            found = _find_parameter(self._params["dihedral"], name, postfix)
            if found is not None:
                force.addBond([i, j, k, l], [float(found[0][0]), float(found[0][1])])
            elif _find_parameter(self._params["sugar_bonded"], name, postfix) is not None:
                sugar_indices["dihedral"].append((i, j, k, l))
        force.setForceGroup(FORCE_GROUP_IDS["dihedral"])
        force.setName("dihedral")
        if force.getNumBonds() > 0:
            system.addForce(force)
        return dihedral_indices

    def _add_wca(self, system: mm.System, data: _TopologyData, *, periodic: bool) -> None:
        force = mm.CustomNonbondedForce(
            "(4*eps*((sig/r)^12-(sig/r)^6)+eps)*step(A*sig-r);A=2^(1/6);sig=0.5*(sig1+sig2)"
        )
        force.addPerParticleParameter("sig")
        force.addGlobalParameter("eps", 5 * unit.kilojoule_per_mole)
        force.setCutoffDistance(0.4 * unit.nanometer)
        force.setNonbondedMethod(force.CutoffPeriodic if periodic else force.CutoffNonPeriodic)
        wca = self._params["wca_6spn"]
        for name in data.atom_names:
            sigma = 0.3 if name in {"BC", "BN"} else float(wca[name])
            force.addParticle([sigma * unit.nanometer])
        force.addInteractionGroup(data.real_or_cog_site_ids, data.real_or_cog_site_ids)
        force.createExclusionsFromBonds(data.bonds, 3)
        force.setForceGroup(FORCE_GROUP_IDS["wca"])
        force.setName("wca")
        system.addForce(force)

    def _add_spline(self, system: mm.System, data: _TopologyData, base_idx_lim: int, *, periodic: bool) -> None:
        y_nodes = self._params["spline_y_nodes"]
        x_lim = self._params["spline_x_lim"]
        y_nodes, x_lim = _legacy_compatible_spline_tables(y_nodes, x_lim, base_idx_lim)
        n_types_x, n_types_y, n_nodes = y_nodes.shape
        expr = (
            "active*U(type1,type2,node);"
            f"active=step({base_idx_lim - 0.5}-min(type1,type2));"
            f"node=(r-rmin(type1,type2))*{n_nodes - 1}/(rmax(type1,type2)-rmin(type1,type2))"
        )
        force = mm.CustomNonbondedForce(expr)
        force.addTabulatedFunction(
            "U",
            mm.Continuous3DFunction(
                n_types_x, n_types_y, n_nodes,
                y_nodes.transpose(2, 1, 0).ravel(),
                0, n_types_x - 1, 0, n_types_y - 1, 0, n_nodes - 1,
            ),
        )
        force.addTabulatedFunction(
            "rmin", mm.Discrete2DFunction(n_types_x, n_types_y, x_lim[:, :, 0].T.ravel())
        )
        force.addTabulatedFunction(
            "rmax", mm.Discrete2DFunction(n_types_x, n_types_y, x_lim[:, :, 1].T.ravel())
        )
        force.addPerParticleParameter("type")
        for name in data.atom_names:
            force.addParticle([HIGH_RES_SPLINE_TYPES[name]])
        force.addInteractionGroup(data.real_site_ids, data.real_site_ids)
        force.createExclusionsFromBonds(data.bonds, 3)
        force.setCutoffDistance(2 * unit.nanometer)
        force.setNonbondedMethod(force.CutoffPeriodic if periodic else force.CutoffNonPeriodic)
        force.setForceGroup(FORCE_GROUP_IDS["spline"])
        force.setName("spline")
        system.addForce(force)

    def _add_debye_huckel(self, system: mm.System, data: _TopologyData, temperature, salt_concentration, *, periodic: bool) -> None:
        temperature = _as_quantity(temperature, unit.kelvin)
        salt_concentration = _as_quantity(salt_concentration, unit.millimolar)
        e = 249.4 - 0.788 * (temperature / unit.kelvin) + 7.2e-4 * (temperature / unit.kelvin) ** 2
        a = 1 - 0.2551 * (salt_concentration / unit.molar) + 5.151e-2 * (salt_concentration / unit.molar) ** 2 - 6.889e-3 * (salt_concentration / unit.molar) ** 3
        dielectric = e * a
        kb = unit.BOLTZMANN_CONSTANT_kB
        na = unit.AVOGADRO_CONSTANT_NA
        ec = 1.60217653e-19 * unit.coulomb
        pv = 8.8541878176e-12 * unit.farad / unit.meter
        debye_length = np.sqrt(dielectric * pv * kb * temperature / (2.0 * na * ec**2 * salt_concentration))
        debye_length = debye_length.in_units_of(unit.nanometer)
        denominator = 4 * np.pi * pv * dielectric / (na * ec**2)
        denominator = denominator.in_units_of(unit.kilojoules_per_mole**-1 * unit.nanometer**-1)
        force = mm.CustomNonbondedForce("energy; energy=q1*q2*exp(-r/dh_length)/(denominator*r);")
        force.addPerParticleParameter("q")
        force.addGlobalParameter("dh_length", debye_length)
        force.addGlobalParameter("denominator", denominator)
        force.setCutoffDistance(min(5 * unit.nanometer, 4 * debye_length))
        force.setNonbondedMethod(force.CutoffPeriodic if periodic else force.CutoffNonPeriodic)
        phosphate_ids = []
        for index, name in enumerate(data.atom_names):
            if name == "P":
                force.addParticle([-0.6])
                phosphate_ids.append(index)
            else:
                force.addParticle([0.0])
        force.addInteractionGroup(phosphate_ids, phosphate_ids)
        force.createExclusionsFromBonds(data.bonds, 3)
        force.setForceGroup(FORCE_GROUP_IDS["electrostatic"])
        force.setName("electrostatic")
        system.addForce(force)

    def _add_sugar_pucker(self, system: mm.System, topology: app.Topology, *, periodic: bool) -> None:
        descriptors = [name for name in self._params["sugar_sigmoid"] if name != "intercept"]
        expression = self._sugar_pucker_expression(descriptors)
        force = mm.CustomCompoundBondForce(6, expression)
        force.addGlobalParameter("a0", 2 * np.log(99) / 108)
        force.addGlobalParameter("P0", 90)
        force.addGlobalParameter("kg", float(np.ravel(self._params["sugar_gaussian"]["k"])[0]))
        force.addGlobalParameter("a1", float(np.ravel(self._params["sugar_gaussian"]["sharpness"])[0]))
        force.addGlobalParameter("P1", float(np.ravel(self._params["sugar_gaussian"]["mean"])[0]))
        force.addGlobalParameter("dU_C2endo", float(np.ravel(self._params["sugar_dU"])[0]))
        self._add_sugar_global_parameters(force, descriptors, suffix="")
        force.addGlobalParameter("intercept", float(self._params["sugar_sigmoid"]["intercept"]))
        for i, descriptor in enumerate(descriptors):
            force.addGlobalParameter(f"w{i}", float(self._params["sugar_sigmoid"][descriptor]))

        terminal_force = self._terminal_sugar_force(descriptors)
        for chain in topology.chains():
            for residue_offset, residue in enumerate(chain.residues()):
                atom_by_name = {atom.name: atom.index for atom in residue.atoms()}
                names = ["P", "S3", "S2", *BASE_TYPE_TO_CG_NAMES[residue.name]]
                if "P" in atom_by_name:
                    force.addBond([atom_by_name[name] for name in names])
                elif residue_offset == 0:
                    terminal_names = ["S3", "S2", *BASE_TYPE_TO_CG_NAMES[residue.name]]
                    terminal_force.addBond([atom_by_name[name] for name in terminal_names])
                else:
                    raise ValueError(f"Residue {residue.index} is missing P outside a 5-prime terminus")
        force.setForceGroup(FORCE_GROUP_IDS["pucker"])
        force.setName("pucker")
        # Pucker is a bonded internal coordinate.  Its participating atoms are
        # connected in the ordinary bonded force above, so OpenMMTools images
        # them as one molecule and this force must use ordinary displacements.
        force.setUsesPeriodicBoundaryConditions(False)
        system.addForce(force)
        if terminal_force.getNumBonds() > 0:
            terminal_force.setForceGroup(FORCE_GROUP_IDS["pucker"])
            terminal_force.setName("pucker")
            system.addForce(terminal_force)

    def _terminal_sugar_force(self, descriptors: list[str]) -> mm.CustomCompoundBondForce:
        descriptors_wo_phosphate = [descriptor for descriptor in descriptors if "P" not in descriptor]
        expression = self._sugar_u_expression(descriptors_wo_phosphate, state="3", phosphate=False)
        force = mm.CustomCompoundBondForce(5, expression)
        self._add_sugar_global_parameters(force, descriptors_wo_phosphate, suffix="_5", c2=False)
        return force

    def _sugar_pucker_expression(self, descriptors: list[str]) -> str:
        expression = "((1-sigmoid)*U3+sigmoid*(U2+dU_C2endo))*(1-gaussian); "
        expression += "sigmoid=1/(1+exp(-a0*(P-P0))); "
        expression += "gaussian=kg*exp(-a1*(P-P1)^2); "
        expression += "P=" + self._phase_angle_expression(descriptors)
        expression += "U3=" + self._sugar_u_expression(descriptors, state="3")
        expression += "U2=" + self._sugar_u_expression(descriptors, state="2")
        return expression

    def _phase_angle_expression(self, descriptors: list[str]) -> str:
        expression = "".join(f"dof{i}*w{i}+" for i in range(len(descriptors)))
        expression += "intercept; "
        for i, descriptor in enumerate(descriptors):
            expression += f"dof{i}={_dof_expression(descriptor)}; "
        return expression

    def _sugar_u_expression(self, descriptors: list[str], *, state: str, phosphate: bool = True) -> str:
        terms = []
        dof_to_index = {descriptor: i for i, descriptor in enumerate(descriptors)}
        parameter_suffix = "" if phosphate else "_5"
        for descriptor, i in dof_to_index.items():
            parts = descriptor.split("-")
            if len(parts) == 4:
                terms.append(self._cbt_dihedral_term(descriptor, i, dof_to_index, state=state, phosphate=phosphate))
            elif len(parts) == 5:
                if parts[-1] == "sin":
                    terms.append(self._cbt_dihedral_term("-".join(parts[:-1]), i, dof_to_index, state=state, phosphate=phosphate))
            else:
                terms.append(f"0.5*k{i}_C{state}{parameter_suffix}*(dof{i}-b{i}_C{state}{parameter_suffix})^2")
        expression = "+".join(terms) + "; "
        for i, descriptor in enumerate(descriptors):
            parts = descriptor.split("-")
            if len(parts) == 5 and parts[-1] == "cos":
                continue
            descriptor_for_expr = "-".join(parts[:-1]) if len(parts) == 5 else descriptor
            expression += f"dof{i}={_dof_expression(descriptor_for_expr, phosphate=phosphate)}; "
        return expression

    def _cbt_dihedral_term(self, descriptor: str, i: int, dof_to_index: dict[str, int], *, state: str, phosphate: bool) -> str:
        angle1, angle2 = _angles_from_dihedral(descriptor)
        angle1_index = _angle_descriptor_index(angle1, dof_to_index)
        angle2_index = _angle_descriptor_index(angle2, dof_to_index)
        suffix = "" if phosphate else "_5"
        term = f"k{i}_C{state}{suffix}*(1+cos(dof{i}-b{i}_C{state}{suffix}))"
        if angle1_index is None:
            term += f"*(sin({_dof_expression(angle1, phosphate=phosphate)}))^3"
        else:
            term += f"*(sin(dof{angle1_index}))^3"
        if angle2_index is None:
            term += f"*(sin({_dof_expression(angle2, phosphate=phosphate)}))^3"
        else:
            term += f"*(sin(dof{angle2_index}))^3"
        return term

    def _add_sugar_global_parameters(
        self,
        force: mm.CustomCompoundBondForce,
        descriptors: list[str],
        *,
        suffix: str,
        c2: bool = True,
    ) -> None:
        for i, descriptor in enumerate(descriptors):
            parts = descriptor.split("-")
            if len(parts) == 5:
                if parts[-1] == "sin":
                    descriptor = "-".join(parts[:-1])
                else:
                    continue
            key = f"{_dof_with_postfix(descriptor)}-3"
            c3 = self._params["sugar_bonded"][key]
            force.addGlobalParameter(f"b{i}_C3{suffix}", float(c3[0]))
            force.addGlobalParameter(f"k{i}_C3{suffix}", float(c3[1]))
            if c2:
                key = f"{_dof_with_postfix(descriptor)}-2"
                c2_params = self._params["sugar_bonded"][key]
                force.addGlobalParameter(f"b{i}_C2{suffix}", float(c2_params[0]))
                force.addGlobalParameter(f"k{i}_C2{suffix}", float(c2_params[1]))

    def _add_stacking(
        self,
        system: mm.System,
        data: _TopologyData,
        components: tuple[str, ...],
        *,
        periodic: bool,
    ) -> None:
        energy_terms = []
        component_definitions = []
        tables = []
        table_parameter_names = (
            "U0",
            "r0",
            "dr0",
            "ctheta0",
            "cpsi0",
            "r_sens",
            "theta_sens",
        )
        for component in components:
            stacking_type = component.removeprefix("stacking")
            term, definitions = _stacking_component_expression(stacking_type)
            energy_terms.append(term)
            component_definitions.append(definitions)
            para = self._params[component]
            values = para["para"] * para["para0"]
            matrices = (
                values[:, :, 0],
                values[:, :, 1],
                values[:, :, 2],
                np.cos(values[:, :, 3]),
                np.cos(values[:, :, 4]),
                values[:, :, 5],
                values[:, :, 6],
            )
            prefix = f"s{stacking_type}"
            tables.extend(
                (f"{prefix}_{name}_mat", matrix.ravel().tolist())
                for name, matrix in zip(table_parameter_names, matrices)
            )

        expr = (
            "+".join(energy_terms)
            + ";"
            + "".join(component_definitions)
            + "cos_normal_psi=select(sin(D2D1A1)*sin(D1A1A2),"
            + "cos_normal_psi_full,cos_normal_psi_partial);"
            + "cos_normal_psi_full=sin(D2D1A1)*sin(D1A1A2)*cos(phi)"
            + "-cos(D2D1A1)*cos(D1A1A2);"
            + "cos_normal_psi_partial=-cos(D2D1A1)*cos(D1A1A2);"
            + "r=distance(d1,a1);D2D1A1=angle(d2,d1,a1);"
            + "D1A1A2=angle(d1,a1,a2);phi=dihedral(d2,d1,a1,a2)"
        )
        force = mm.CustomHbondForce(expr)
        for component in components:
            force.addGlobalParameter(
                STACKING_SCALE_PARAMETERS[component], 1.0
            )
        for name, values in tables:
            force.addTabulatedFunction(
                name, mm.Discrete2DFunction(4, 4, values)
            )
        force.addPerAcceptorParameter("a_type")
        force.addPerDonorParameter("d_type")
        force.setCutoffDistance(0.7 * unit.nanometer)
        force.setNonbondedMethod(force.CutoffPeriodic if periodic else force.CutoffNonPeriodic)
        for i, (cog, normal) in enumerate(data.virtual_site_summaries):
            cog_idx, restype, *_ = cog
            normal_idx = normal[0]
            force.addDonor(cog_idx, normal_idx, -1, [RESTYPE_TO_INDEX[restype]])
            force.addAcceptor(cog_idx, normal_idx, -1, [RESTYPE_TO_INDEX[restype]])
            force.addExclusion(i, i)
        force.setForceGroup(FORCE_GROUP_IDS[components[0]])
        force.setName(FUSED_STACKING_FORCE_NAME)
        system.addForce(force)

    def _add_pairing(self, system: mm.System, data: _TopologyData, n_exclude_pairing: int = 1, *, periodic: bool) -> None:
        geom_pairs = self._params["pairing_geompairs"]
        para = self._params["pairing_para"] * self._params["pairing_para0"]
        rows_by_donor: dict[str, dict[str, list[tuple[int, str, np.ndarray]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for i, pair in enumerate(geom_pairs):
            rows_by_donor[pair[0]][pair[3]].append((i, pair[2], para[i]))

        # Choose the representation once while constructing the System. The
        # compound path exposes candidate pairs as parallel work, while the
        # H-bond path supplies a scalable neighbor list. Enumeration stops at
        # limit+1, and a one-channel/one-slot case stays on H-bond because it
        # has no launch-fusion benefit. This choice never changes during MD.
        compound_bonds = _pairing_compound_bonds(
            data,
            rows_by_donor,
            n_exclude_pairing,
            max_bonds=_PAIRING_COMPOUND_MAX_BONDS,
        )
        active_type_pairs = {
            type_pair for _, type_pair in compound_bonds
        }
        active_donor_types = {
            donor_type for donor_type, _ in active_type_pairs
        }
        max_active_slots = max(
            (
                len(rows_by_donor[donor_type][acceptor_type])
                for donor_type, acceptor_type in active_type_pairs
            ),
            default=0,
        )
        use_compound = (
            len(compound_bonds) <= _PAIRING_COMPOUND_MAX_BONDS
            and (
                len(active_donor_types) > 1
                or max_active_slots > 1
            )
        )
        if use_compound:
            self._add_compound_pairing(
                system, rows_by_donor, compound_bonds, periodic=periodic
            )
            return

        self._add_hbond_pairing(
            system,
            data,
            rows_by_donor,
            n_exclude_pairing,
            periodic=periodic,
        )

    def _add_compound_pairing(
        self,
        system: mm.System,
        rows_by_donor,
        compound_bonds,
        *,
        periodic: bool,
    ) -> None:
        active_type_pairs = {
            type_pair for _, type_pair in compound_bonds
        }
        slot_count = max(
            len(rows_by_donor[donor_type][acceptor_type])
            for donor_type, acceptor_type in active_type_pairs
        )
        parameter_prefix = "pairing_compound"
        force = mm.CustomCompoundBondForce(
            6,
            _donor_packed_pairing_expr(
                slot_count,
                parameter_prefix,
                particle_names=("p1", "p2", "p3", "p4", "p5", "p6"),
                cutoff_nm=_PAIRING_CUTOFF_NM,
            ),
        )
        parameter_names = (*_PACKED_PAIRING_PARAMETER_NAMES, "psi_sign")
        for slot in range(slot_count):
            for name in parameter_names:
                force.addPerBondParameter(
                    _pairing_parameter_name(name, slot, parameter_prefix)
                )

        zero_parameters = _packed_pairing_parameters(np.zeros(13))
        values_by_pair = {}
        for donor_type, rows_by_acceptor in rows_by_donor.items():
            for acceptor_type, raw_rows in rows_by_acceptor.items():
                if (donor_type, acceptor_type) not in active_type_pairs:
                    continue
                rows = [
                    (orientation, _packed_pairing_parameters(row))
                    for _, orientation, row in raw_rows
                ]
                values_by_pair[(donor_type, acceptor_type)] = [
                    float(
                        _pairing_slot_parameter(
                            rows, slot, name, zero_parameters
                        )
                    )
                    for slot in range(slot_count)
                    for name in parameter_names
                ]

        for particle_indices, type_pair in compound_bonds:
            force.addBond(particle_indices, values_by_pair[type_pair])
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(FORCE_GROUP_IDS["pairing"])
        force.setName("pairing")
        system.addForce(force)

    def _add_hbond_pairing(
        self,
        system: mm.System,
        data: _TopologyData,
        rows_by_donor,
        n_exclude_pairing: int,
        *,
        periodic: bool,
    ) -> None:
        added_force = False
        for donor_type_index, (d_type, rows_by_acceptor) in enumerate(rows_by_donor.items()):
            donor_residues = [entry for entry in data.virtual_site_summaries if entry[0][1] == d_type]
            acceptor_types = set(rows_by_acceptor)
            acceptor_residues = [
                entry for entry in data.virtual_site_summaries if entry[0][1] in acceptor_types
            ]
            if not donor_residues or not acceptor_residues:
                continue

            slot_count = max(len(rows) for rows in rows_by_acceptor.values())
            parameter_prefix = f"pairing_d{donor_type_index}"
            force = mm.CustomHbondForce(
                _donor_packed_pairing_expr(slot_count, parameter_prefix)
            )
            force.setCutoffDistance(_PAIRING_CUTOFF_NM * unit.nanometer)
            force.setNonbondedMethod(force.CutoffPeriodic if periodic else force.CutoffNonPeriodic)

            packed_rows = {
                restype: [
                    (orientation, _packed_pairing_parameters(row))
                    for _, orientation, row in rows
                ]
                for restype, rows in rows_by_acceptor.items()
            }
            zero_parameters = _packed_pairing_parameters(np.zeros(13))
            for slot in range(slot_count):
                for name in (*_PACKED_PAIRING_PARAMETER_NAMES, "psi_sign"):
                    force.addPerAcceptorParameter(
                        _pairing_parameter_name(name, slot, parameter_prefix)
                    )

            donor_metadata: list[tuple[int, int]] = []
            acceptor_metadata: list[tuple[int, int]] = []
            for cog, normal in donor_residues:
                cog_idx, _, resid, chainid, _, b1_idx = cog
                force.addDonor(cog_idx, normal[0], b1_idx, [])
                donor_metadata.append((chainid, resid))
            for cog, normal in acceptor_residues:
                cog_idx, restype, resid, chainid, _, b1_idx = cog
                rows = packed_rows.get(restype, ())
                acceptor_parameters = []
                for slot in range(slot_count):
                    for name in (*_PACKED_PAIRING_PARAMETER_NAMES, "psi_sign"):
                        value = _pairing_slot_parameter(
                            rows, slot, name, zero_parameters
                        )
                        acceptor_parameters.append(float(value))
                force.addAcceptor(
                    cog_idx, normal[0], b1_idx, acceptor_parameters
                )
                acceptor_metadata.append((chainid, resid))
            for donor_index, (donor_chainid, donor_resid) in enumerate(donor_metadata):
                for acceptor_index, (acceptor_chainid, acceptor_resid) in enumerate(acceptor_metadata):
                    if donor_chainid == acceptor_chainid and abs(acceptor_resid - donor_resid) <= n_exclude_pairing:
                        force.addExclusion(donor_index, acceptor_index)

            force.setForceGroup(FORCE_GROUP_IDS["pairing"])
            force.setName("pairing")
            system.addForce(force)
            added_force = True

        if not added_force:
            force = mm.CustomHbondForce("0")
            force.setCutoffDistance(_PAIRING_CUTOFF_NM * unit.nanometer)
            force.setNonbondedMethod(force.CutoffPeriodic if periodic else force.CutoffNonPeriodic)
            force.setForceGroup(FORCE_GROUP_IDS["pairing"])
            force.setName("pairing")
            system.addForce(force)


def _dof_with_postfix(name: str) -> str:
    return f"{name}-" + "0" * len(name.split("-"))


def _dof_expression(name: str, phosphate: bool = True) -> str:
    if phosphate:
        name_to_index = {"P": 1, "S3": 2, "S2": 3, "B1": 4, "B2": 5, "B3": 6}
    else:
        name_to_index = {"S3": 1, "S2": 2, "B1": 3, "B2": 4, "B3": 5}
    parts = name.split("-")
    mode = None
    if parts[-1] in {"sin", "cos"}:
        mode = parts[-1]
        parts = parts[:-1]
    indices = [name_to_index[part] for part in parts]
    if len(indices) == 2:
        return f"distance(p{indices[0]}, p{indices[1]})"
    if len(indices) == 3:
        return f"angle(p{indices[0]}, p{indices[1]}, p{indices[2]})"
    if len(indices) == 4:
        expr = f"dihedral(p{indices[0]}, p{indices[1]}, p{indices[2]}, p{indices[3]})"
        return f"{mode}({expr})" if mode else expr
    raise ValueError(f"Unsupported sugar descriptor {name!r}")


def _angles_from_dihedral(name: str) -> tuple[str, str]:
    p1, p2, p3, p4 = name.split("-")
    return f"{p1}-{p2}-{p3}", f"{p2}-{p3}-{p4}"


def _angle_descriptor_index(angle: str, dof_to_index: dict[str, int]) -> int | None:
    if angle in dof_to_index:
        return dof_to_index[angle]
    reversed_angle = "-".join(angle.split("-")[::-1])
    return dof_to_index.get(reversed_angle)


def _legacy_compatible_spline_tables(
    y_nodes: np.ndarray, x_lim: np.ndarray, base_idx_lim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pack spline tables while preserving legacy duplicate-name resolution."""

    effective_y = np.array(y_nodes, copy=True)
    effective_x = np.array(x_lim, copy=True)
    active_pairs = [
        (i, j)
        for i, j in product(range(y_nodes.shape[0]), range(y_nodes.shape[1]))
        if i < base_idx_lim or j < base_idx_lim
    ]

    # Legacy names omitted a separator: U110 means both (1, 10) and (11, 0).
    # OpenMM resolved references to the last table registered under each name.
    source_by_name = {f"U{i}{j}": (i, j) for i, j in active_pairs}
    for i, j in active_pairs:
        source_i, source_j = source_by_name[f"U{i}{j}"]
        effective_y[i, j] = y_nodes[source_i, source_j]
        effective_x[i, j] = x_lim[source_i, source_j]
    return effective_y, effective_x


def _load_parameters(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as h5:
        params: dict[str, object] = {
            "bond": _read_named_group(h5["bond/para"]),
            "angle": _read_named_group(h5["angle/para"]),
            "dihedral": _read_named_group(h5["dihedral/para"]),
            "sugar_bonded": _read_named_group(h5["sugar/bonded/para"]),
            "sugar_sigmoid": _read_named_group(h5["sugar/sigmoid/para"]),
            "sugar_gaussian": _read_named_group(h5["sugar/gaussian/para"]),
            "sugar_dU": h5["sugar/C3C2/dU"][()],
            "wca_6spn": {name: h5[f"wca/6spn/{name}"][()] for name in h5["wca/6spn"]},
            "spline_y_nodes": h5["spline/para/y_nodes"][()],
            "spline_x_lim": h5["spline/para/x_lim"][()],
            "pairing_geompairs": [x.decode() if isinstance(x, bytes) else str(x) for x in h5["pairing/geompairs"][()]],
            "pairing_para": h5["pairing/para"][()],
            "pairing_para0": h5["pairing/para0"][()],
        }
        for name in ("stacking35", "stacking55", "stacking33"):
            params[name] = {"para": h5[f"{name}/para"][()], "para0": h5[f"{name}/para0"][()]}
        return params


def _read_named_group(group) -> dict[str, np.ndarray]:
    return {name: group[name][()] for name in group.keys()}


def _find_parameter(group: dict[str, np.ndarray], name: str, postfix: str):
    full_name = f"{name}-{postfix}"
    if full_name in group:
        return group[full_name], full_name
    if set(postfix) == {"0"}:
        reversed_name = "-".join(name.split("-")[::-1])
        reversed_full_name = f"{reversed_name}-{postfix[::-1]}"
        if reversed_full_name in group:
            return group[reversed_full_name], reversed_full_name
    return None


def _collect_topology(topology: app.Topology) -> _TopologyData:
    atoms = list(topology.atoms())
    atom_names = [atom.name for atom in atoms]
    residue_names = [atom.residue.name for atom in atoms]
    residue_indices = [atom.residue.index for atom in atoms]
    chain_indices = [atom.residue.chain.index for atom in atoms]
    bonds = [(bond.atom1.index, bond.atom2.index) for bond in topology.bonds()]
    adjacency = {atom.index: set() for atom in atoms}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)

    summaries = []
    virtual_ids = set()
    normal_ids = set()
    cog_ids = set()
    for residue in topology.residues():
        restype = residue.name
        b1_name = BASE_TYPE_TO_CG_NAMES[restype][0]
        atom_by_name = {atom.name: atom.index for atom in residue.atoms()}
        cog_idx = atom_by_name["BC"]
        normal_idx = atom_by_name["BN"]
        b1_idx = atom_by_name[b1_name]
        virtual_ids.update([cog_idx, normal_idx])
        cog_ids.add(cog_idx)
        normal_ids.add(normal_idx)
        summaries.append(
            (
                (cog_idx, restype, residue.index, residue.chain.index, "cog", b1_idx),
                (normal_idx, restype, residue.index, residue.chain.index, "normal", b1_idx),
            )
        )
    all_ids = set(range(len(atoms)))
    return _TopologyData(
        atom_names=atom_names,
        residue_names=residue_names,
        residue_indices=residue_indices,
        chain_indices=chain_indices,
        bonds=bonds,
        adjacency=adjacency,
        virtual_site_summaries=summaries,
        real_site_ids=all_ids - virtual_ids,
        real_or_cog_site_ids=all_ids - normal_ids,
    )


def _as_quantity(value, target_unit):
    if hasattr(value, "unit"):
        return value.in_units_of(target_unit)
    return value * target_unit


def _stacking_component_expression(
    stacking_type: str,
) -> tuple[str, str]:
    if stacking_type == "35":
        g1_sign, g2_sign, g3_sign = "-", "", ""
    elif stacking_type == "55":
        g1_sign, g2_sign, g3_sign = "", "", "-"
    else:
        g1_sign, g2_sign, g3_sign = "-", "-", "-"
    prefix = f"s{stacking_type}"
    energy = (
        f"stacking{stacking_type}_scale*"
        f"{prefix}_U0_mat(d_type,a_type)*{prefix}_fr*"
        f"{prefix}_g1*{prefix}_g2*{prefix}_g3"
    )
    definitions = (
        f"{prefix}_fr=0.5*(tanh({prefix}_r_sens*"
        f"(abs(r-{prefix}_r0)-{prefix}_dr0))-1);"
        f"{prefix}_g1=0.5*(tanh({prefix}_theta_sens*"
        f"({g1_sign}cos(D2D1A1)-{prefix}_ctheta0))+1);"
        f"{prefix}_g2=0.5*(tanh({prefix}_theta_sens*"
        f"({g2_sign}cos(D1A1A2)-{prefix}_ctheta0))+1);"
        f"{prefix}_g3=0.5*(tanh({prefix}_theta_sens*"
        f"({g3_sign}cos_normal_psi-{prefix}_cpsi0))+1);"
        f"{prefix}_r0={prefix}_r0_mat(a_type,d_type);"
        f"{prefix}_dr0={prefix}_dr0_mat(a_type,d_type);"
        f"{prefix}_ctheta0={prefix}_ctheta0_mat(a_type,d_type);"
        f"{prefix}_cpsi0={prefix}_cpsi0_mat(a_type,d_type);"
        f"{prefix}_r_sens={prefix}_r_sens_mat(a_type,d_type);"
        f"{prefix}_theta_sens={prefix}_theta_sens_mat(a_type,d_type);"
    )
    return energy, definitions


_PACKED_PAIRING_PARAMETER_NAMES = (
    "Up",
    "r0",
    "dr0",
    "theta0",
    "dtheta0",
    "psi0",
    "phi1",
    "dphi1",
    "phi2",
    "dphi2",
    "r_sens",
    "theta_sens",
    "phi_sens",
)


def _packed_pairing_parameters(row: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(_PACKED_PAIRING_PARAMETER_NAMES, row)
    }


def _pairing_slot_parameter(rows, slot, name, zero_parameters):
    if slot >= len(rows):
        return 1.0 if name == "psi_sign" else zero_parameters[name]
    orientation, parameters = rows[slot]
    if name == "psi_sign":
        return 1.0 if orientation == "+" else -1.0
    return parameters[name]


def _pairing_compound_bonds(
    data, rows_by_donor, n_exclude_pairing, *, max_bonds
):
    residues_by_type = {
        restype: [
            entry
            for entry in data.virtual_site_summaries
            if entry[0][1] == restype
        ]
        for restype in RESTYPE_TO_INDEX
    }
    bonds = []
    for donor_type, rows_by_acceptor in rows_by_donor.items():
        for acceptor_type in rows_by_acceptor:
            for donor_cog, donor_normal in residues_by_type[donor_type]:
                d_cog, _, d_residue, d_chain, _, d_b1 = donor_cog
                for acceptor_cog, acceptor_normal in residues_by_type[
                    acceptor_type
                ]:
                    a_cog, _, a_residue, a_chain, _, a_b1 = acceptor_cog
                    if (
                        d_chain == a_chain
                        and abs(a_residue - d_residue)
                        <= n_exclude_pairing
                    ):
                        continue
                    bonds.append(
                        (
                            (
                                d_cog,
                                donor_normal[0],
                                d_b1,
                                a_cog,
                                acceptor_normal[0],
                                a_b1,
                            ),
                            (donor_type, acceptor_type),
                        )
                    )
                    if len(bonds) > max_bonds:
                        return bonds
    return bonds


def _donor_packed_pairing_expr(
    slot_count: int,
    parameter_prefix: str,
    *,
    particle_names=("d1", "d2", "d3", "a1", "a2", "a3"),
    cutoff_nm: float | None = None,
) -> str:
    energy_terms = []
    component_terms = []

    for slot in range(slot_count):
        def parameter(name: str) -> str:
            return _pairing_parameter_name(name, slot, parameter_prefix)
        energy_terms.append(
            f"{parameter('Up')}*fr{slot}*g1_{slot}*g2_{slot}*"
            f"g3_{slot}*g4_{slot}*h{slot}"
        )
        component_terms.append(
            f"fr{slot}=0.5*(tanh({parameter('r_sens')}*"
            f"(abs(r-{parameter('r0')})-{parameter('dr0')}))-1); "
            f"g1_{slot}=0.5*(tanh({parameter('theta_sens')}*"
            f"(cos(D2D1A1-{parameter('theta0')})-"
            f"cos({parameter('dtheta0')})))+1); "
            f"g2_{slot}=0.5*(tanh({parameter('theta_sens')}*"
            f"(cos(D1A1A2-{parameter('theta0')})-"
            f"cos({parameter('dtheta0')})))+1); "
            f"g3_{slot}=0.5*(tanh({parameter('phi_sens')}*"
            f"(cos(D3D1D2A1-{parameter('phi1')})-"
            f"cos({parameter('dphi1')})))+1); "
            f"g4_{slot}=0.5*(tanh({parameter('phi_sens')}*"
            f"(cos(A3A1A2D1-{parameter('phi2')})-"
            f"cos({parameter('dphi2')})))+1); "
            f"h{slot}=0.5*(tanh({parameter('theta_sens')}*"
            f"({parameter('psi_sign')}*cos_normal_psi-"
            f"cos({parameter('psi0')})))+1); "
        )
    energy = "+".join(energy_terms)
    if cutoff_nm is not None:
        energy = f"step({cutoff_nm}-r)*({energy})"
    d1, d2, d3, a1, a2, a3 = particle_names
    return (
        energy
        + "; "
        + "".join(component_terms)
        + "cos_normal_psi=select(sin(D2D1A1)*sin(D1A1A2), "
        + "cos_normal_psi_full, cos_normal_psi_partial); "
        + "cos_normal_psi_full=sin(D2D1A1)*sin(D1A1A2)*"
        + "cos(D2D1A1A2)-cos(D2D1A1)*cos(D1A1A2); "
        + "cos_normal_psi_partial=-cos(D2D1A1)*cos(D1A1A2); "
        + f"r=distance({d1},{a1}); "
        + f"D2D1A1=angle({d2},{d1},{a1}); "
        + f"D1A1A2=angle({d1},{a1},{a2}); "
        + f"D2D1A1A2=dihedral({d2},{d1},{a1},{a2}); "
        + "D3D1D2A1=select(sin(D1D2A1),"
        + f"dihedral({d3},{d1},{d2},{a1}),0); "
        + "A3A1A2D1=select(sin(A1A2D1),"
        + f"dihedral({a3},{a1},{a2},{d1}),0); "
        + f"D1D2A1=angle({d1},{d2},{a1}); "
        + f"A1A2D1=angle({a1},{a2},{d1}); "
    )


def _pairing_parameter_name(name: str, index: int, parameter_prefix: str) -> str:
    return f"{parameter_prefix}_{name}{index}" if parameter_prefix else f"{name}{index}"
