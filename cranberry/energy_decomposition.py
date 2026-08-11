from __future__ import annotations

import openmm as mm

from cranberry.forcefield import (
    FORCE_GROUP_IDS,
    FORCE_GROUP_NAMES,
    FUSED_STACKING_FORCE_NAME,
    STACKING_COMPONENT_NAMES,
    STACKING_SCALE_PARAMETERS,
)


def present_force_group_names(system: mm.System) -> list[str]:
    names_by_group = {
        system.getForce(index).getForceGroup(): system.getForce(index).getName()
        for index in range(system.getNumForces())
    }
    fused = fused_stacking_force(system)
    fused_components = (
        set(fused_stacking_components(fused)) if fused is not None else set()
    )
    return [
        name
        for name in FORCE_GROUP_NAMES
        if name in fused_components
        or names_by_group.get(FORCE_GROUP_IDS[name]) == name
    ]


def force_group_energy(
    context: mm.Context,
    system: mm.System,
    name: str,
):
    fused = fused_stacking_force(system)
    if name in STACKING_COMPONENT_NAMES and fused is not None:
        return _fused_stacking_component_energy(context, fused, name)
    return context.getState(
        getEnergy=True, groups={FORCE_GROUP_IDS[name]}
    ).getPotentialEnergy()


def fused_stacking_force(system: mm.System):
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if force.getName() == FUSED_STACKING_FORCE_NAME:
            return force
    return None


def fused_stacking_components(force) -> tuple[str, ...]:
    parameter_names = {
        force.getGlobalParameterName(index)
        for index in range(force.getNumGlobalParameters())
    }
    return tuple(
        name
        for name in STACKING_COMPONENT_NAMES
        if STACKING_SCALE_PARAMETERS[name] in parameter_names
    )


def _fused_stacking_component_energy(
    context: mm.Context,
    force,
    selected: str,
):
    components = fused_stacking_components(force)
    if selected not in components:
        raise ValueError(f"Stacking component {selected!r} is not present")
    previous = {
        name: context.getParameter(STACKING_SCALE_PARAMETERS[name])
        for name in components
    }
    try:
        for name in components:
            context.setParameter(
                STACKING_SCALE_PARAMETERS[name],
                1.0 if name == selected else 0.0,
            )
        return context.getState(
            getEnergy=True, groups={force.getForceGroup()}
        ).getPotentialEnergy()
    finally:
        for name, value in previous.items():
            context.setParameter(STACKING_SCALE_PARAMETERS[name], value)
