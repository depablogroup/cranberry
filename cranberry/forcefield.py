from __future__ import annotations

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
        if get_cutoff is None or get_method is None:
            continue
        method = get_method()
        periodic_method = getattr(force, "CutoffPeriodic", None)
        if periodic_method is None or method != periodic_method:
            continue
        cutoff_nm = get_cutoff().value_in_unit(unit.nanometer)
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
        if "stacking35" in enabled:
            self._add_stacking(system, data, "35", periodic=periodic)
        if "stacking55" in enabled:
            self._add_stacking(system, data, "55", periodic=periodic)
        if "stacking33" in enabled:
            self._add_stacking(system, data, "33", periodic=periodic)
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
                    system.addConstraint(i, j, float(length) * unit.nanometer)
                else:
                    force.addBond(i, j, float(length), float(k))
            elif _find_parameter(self._params["sugar_bonded"], name, postfix) is not None:
                sugar_indices["bond"].append((i, j))
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
        n_types = y_nodes.shape[0]
        expr = "0"
        active_pairs: list[tuple[int, int]] = []
        for i, j in product(range(n_types), range(n_types)):
            if i < base_idx_lim or j < base_idx_lim:
                expr += f"+delta(type1-{i})*delta(type2-{j})*U{i}{j}(r)"
                active_pairs.append((i, j))
        force = mm.CustomNonbondedForce(expr)
        for i, j in active_pairs:
            force.addTabulatedFunction(
                f"U{i}{j}",
                mm.Continuous1DFunction(y_nodes[i, j], float(x_lim[i, j, 0]), float(x_lim[i, j, 1])),
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
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)
        if terminal_force.getNumBonds() > 0:
            terminal_force.setForceGroup(FORCE_GROUP_IDS["pucker"])
            terminal_force.setName("pucker")
            terminal_force.setUsesPeriodicBoundaryConditions(periodic)
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

    def _add_stacking(self, system: mm.System, data: _TopologyData, stacking_type: str, *, periodic: bool) -> None:
        para = self._params[f"stacking{stacking_type}"]
        values = para["para"] * para["para0"]
        mats = [values[:, :, i].ravel().tolist() for i in range(7)]
        if stacking_type == "35":
            g1_sign, g2_sign, g3_sign = "-", "", ""
        elif stacking_type == "55":
            g1_sign, g2_sign, g3_sign = "", "", "-"
        else:
            g1_sign, g2_sign, g3_sign = "-", "-", "-"
        expr = (
            "U0_mat(d_type, a_type) * fr * g1 * g2 * g3; "
            "fr=1/2*(tanh(r_sens*(abs(r-r0)-dr0))-1); r=distance(d1, a1); "
            f"g1=1/2*(tanh(theta_sens*({g1_sign}cos(D2D1A1)-cos(theta0)))+1); "
            f"g2=1/2*(tanh(theta_sens*({g2_sign}cos(D1A1A2)-cos(theta0)))+1); "
            f"g3=1/2*(tanh(theta_sens*({g3_sign}cos_normal_psi-cos(psi0)))+1); "
            "cos_normal_psi=select(sin(D2D1A1)*sin(D1A1A2), cos_normal_psi_full, cos_normal_psi_partial); "
            "cos_normal_psi_full=sin(D2D1A1)*sin(D1A1A2)*cos(phi)-cos(D2D1A1)*cos(D1A1A2); "
            "cos_normal_psi_partial=-cos(D2D1A1)*cos(D1A1A2); "
            "D2D1A1=angle(d2, d1, a1); D1A1A2=angle(d1, a1, a2); phi=dihedral(d2, d1, a1, a2); "
            "r0=r0_mat(a_type, d_type); dr0=dr0_mat(a_type, d_type); theta0=theta0_mat(a_type, d_type); "
            "psi0=psi0_mat(a_type, d_type); r_sens=r_sens_mat(a_type, d_type); theta_sens=theta_sens_mat(a_type, d_type);"
        )
        force = mm.CustomHbondForce(expr)
        for name, vals in zip(["U0_mat", "r0_mat", "dr0_mat", "theta0_mat", "psi0_mat", "r_sens_mat", "theta_sens_mat"], mats):
            force.addTabulatedFunction(name, mm.Discrete2DFunction(4, 4, vals))
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
        group_name = f"stacking{stacking_type}"
        force.setForceGroup(FORCE_GROUP_IDS[group_name])
        force.setName(group_name)
        system.addForce(force)

    def _add_pairing(self, system: mm.System, data: _TopologyData, n_exclude_pairing: int = 1, *, periodic: bool) -> None:
        geom_pairs = self._params["pairing_geompairs"]
        para = self._params["pairing_para"] * self._params["pairing_para0"]
        expr = ""
        for i, pair in enumerate(geom_pairs):
            d_type_id, a_type_id = RESTYPE_TO_INDEX[pair[0]], RESTYPE_TO_INDEX[pair[3]]
            expr += f"Up{i}*fr{i}*g1_{i}*g2_{i}*g3_{i}*g4_{i}*h{i}*delta(d_type-{d_type_id})*delta(a_type-{a_type_id})*(1-is_excluded)+"
        expr = expr.rstrip("+") + "; "
        for i, pair in enumerate(geom_pairs):
            expr += _pairing_component_expr(i, pair[2])
        expr += (
            "cos_normal_psi=select(sin(D2D1A1)*sin(D1A1A2), cos_normal_psi_full, cos_normal_psi_partial); "
            "cos_normal_psi_full=sin(D2D1A1)*sin(D1A1A2)*cos(D2D1A1A2)-cos(D2D1A1)*cos(D1A1A2); "
            "cos_normal_psi_partial=-cos(D2D1A1)*cos(D1A1A2); "
            "D2D1A1=angle(d2, d1, a1); D1A1A2=angle(d1, a1, a2); D2D1A1A2=dihedral(d2, d1, a1, a2); "
            "D3D1D2A1=select(sin(D1D2A1), dihedral(d3, d1, d2, a1), 0); "
            "A3A1A2D1=select(sin(A1A2D1), dihedral(a3, a1, a2, d1), 0); "
            "D1D2A1=angle(d1, d2, a1); A1A2D1=angle(a1, a2, d1); "
            f"is_excluded=step({n_exclude_pairing}-abs(a_id-d_id))*delta(a_chainid-d_chainid);"
        )
        force = mm.CustomHbondForce(expr)
        for parameter in ["a_type", "a_id", "a_chainid"]:
            force.addPerAcceptorParameter(parameter)
        for parameter in ["d_type", "d_id", "d_chainid"]:
            force.addPerDonorParameter(parameter)
        force.setCutoffDistance(0.8 * unit.nanometer)
        force.setNonbondedMethod(force.CutoffPeriodic if periodic else force.CutoffNonPeriodic)
        for i, row in enumerate(para):
            names = ["Up", "r0_", "dr0_", "theta0_", "dtheta0_", "psi0_", "phi1_", "dphi1_", "phi2_", "dphi2_", "r_sens", "theta_sens", "phi_sens"]
            for name, value in zip(names, row):
                force.addGlobalParameter(f"{name}{i}", float(value))
        for i, (cog, normal) in enumerate(data.virtual_site_summaries):
            cog_idx, restype, resid, chainid, _, b1_idx = cog
            normal_idx = normal[0]
            values = [RESTYPE_TO_INDEX[restype], resid, chainid]
            force.addDonor(cog_idx, normal_idx, b1_idx, values)
            force.addAcceptor(cog_idx, normal_idx, b1_idx, values)
            force.addExclusion(i, i)
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


def _pairing_component_expr(index: int, orientation: str) -> str:
    expr = (
        f"fr{index}=1/2*(tanh(r_sens{index}*(abs(distance(d1, a1)-r0_{index})-dr0_{index}))-1); "
        f"g1_{index}=1/2*(tanh(theta_sens{index}*(cos(abs(D2D1A1-theta0_{index}))-cos(dtheta0_{index})))+1); "
        f"g2_{index}=1/2*(tanh(theta_sens{index}*(cos(abs(D1A1A2-theta0_{index}))-cos(dtheta0_{index})))+1); "
        f"g3_{index}=1/2*(tanh(phi_sens{index}*(cos(abs(D3D1D2A1-phi1_{index}))-cos(dphi1_{index})))+1); "
        f"g4_{index}=1/2*(tanh(phi_sens{index}*(cos(abs(A3A1A2D1-phi2_{index}))-cos(dphi2_{index})))+1); "
    )
    if orientation == "+":
        return expr + f"h{index}=1/2*(tanh(theta_sens{index}*(cos_normal_psi-cos(psi0_{index})))+1); "
    return expr + f"h{index}=1/2*(tanh(theta_sens{index}*(-cos_normal_psi-cos(psi0_{index})))+1); "
