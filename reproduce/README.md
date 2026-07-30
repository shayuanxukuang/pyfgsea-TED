# TED BIB companion reproduction entry

`verify_and_reproduce_figures.py` starts from a built companion, not from the
mutable analysis workspace.

```powershell
python reproduce/verify_and_reproduce_figures.py `
  G:\release-assets\ted-bib-companion-v1.1.0
```

The default command verifies the outer archive manifest, all package members,
the nested 2,400-output and stability manifests, the 480-task contract, the
focused 81-test evidence, and the four Figure 3 plus seven Figure 5 source
tables and final PDF/PNG hashes. This is a source-to-final-figure integrity
check; it does not claim that a renderer ran.

To request an actual redraw:

```powershell
python reproduce/verify_and_reproduce_figures.py `
  G:\release-assets\ted-bib-companion-v1.1.0 --redraw
```

Redrawing is enabled only if the core archive contains an explicit
`reproduction/FIGURE_RENDERERS.json` contract and every referenced script. If
the renderer is not packaged, the command reports
`not_available_renderer_contract_not_packaged`, executes nothing, and exits
with status 3.

The v1.1.0 core allowlist packages `FIGURE_RENDERERS.json`,
`render_bib_figures.py`, and the Python 3.11 dependency lock. The renderer
redraws Figure 3 from its four frozen source tables and Figure 5 from its seven
source tables plus the frozen RNA, protein-outcome, and replication status
JSON records. It validates the manuscript headline values and performs
nonblank PNG pixel QA. Generated and reference hashes are both reported, but
PDF or PNG byte identity is not required across Matplotlib/font/runtime
environments.
