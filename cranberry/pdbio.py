from __future__ import annotations

from pathlib import Path

from openmm import Vec3, app, unit


def write_pdb_with_conect(path: str | Path, topology: app.Topology, positions) -> None:
    path = Path(path)
    atom_serials = {atom: atom.index + 1 for atom in topology.atoms()}
    coords = positions.value_in_unit(unit.angstrom)
    lines = []

    for chain in topology.chains():
        for residue in chain.residues():
            for atom in residue.atoms():
                serial = atom_serials[atom]
                x, y, z = _vec3_xyz(coords[atom.index])
                lines.append(
                    _format_atom_line(
                        serial=serial,
                        atom_name=atom.name,
                        residue_name=residue.name,
                        chain_id=chain.id,
                        residue_id=residue.id,
                        x=x,
                        y=y,
                        z=z,
                        element=_pdb_element(atom),
                    )
                )

    for bond in topology.bonds():
        lines.append(f"CONECT{atom_serials[bond.atom1]:5d}{atom_serials[bond.atom2]:5d}")

    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _format_atom_line(*, serial, atom_name, residue_name, chain_id, residue_id, x, y, z, element):
    chain_id = (chain_id or " ")[:1]
    residue_name = residue_name[:3]
    atom_name = atom_name[:4]
    residue_serial = _residue_serial(residue_id)
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} {chain_id}{residue_serial:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
    )


def _pdb_element(atom: app.Atom) -> str:
    if atom.element is None:
        return ""
    return atom.element.symbol.upper()[:2]


def _residue_serial(residue_id) -> int:
    try:
        return int(residue_id)
    except (TypeError, ValueError):
        return 0


def _vec3_xyz(vec) -> tuple[float, float, float]:
    if isinstance(vec, Vec3):
        return vec.x, vec.y, vec.z
    return tuple(vec)
