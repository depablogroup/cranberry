# Output Files

`cranberry md` writes these files by default into `--output-dir`, which defaults to the current directory:

- `output.dcd`: trajectory written by OpenMM `DCDReporter`
- `log`: OpenMM state log with step, time, potential energy, temperature, and speed
- `detailed.log`: total potential energy plus named force-group components
- `args.json`: machine-readable record of the MD command settings
- `checkpoint.chk`: OpenMM checkpoint written after the run
- `final.pdb`: final coordinates after virtual-site positions are recomputed

`detailed.log` uses OpenMM-style quoted headers with units:

```text
#"Step","Time (ps)","Potential Energy (kJ/mole)","bond (kJ/mole)",...
```

MD overwrites these default outputs unless `--no-overwrite` is passed. REMD output conventions are planned separately because REMD will follow OpenMMTools-style no-overwrite behavior.
