# Upstream provenance

The active document-generation pipeline in this repository is a direct copy of
[`chloeli-15/model_spec_midtraining`](https://github.com/chloeli-15/model_spec_midtraining)
at commit:

```text
e8288a84912ba32af68ad15f2e52a7c1b4e81891
```

Its `safety-tooling` submodule is retained as a Git submodule at the revision pinned by that
commit:

```text
8b93cc293083a8d85b16c4b190da0e10fcc9b9b6
```

The imported upstream files have not been adapted for the previous Hillclimb traits. In
particular, no Hillclimb trait specification, generation configuration, prompt, or generated
document has been added to the active pipeline. The files already under `spec/paper/` are the
authors' upstream paper specifications.

The prior Hillclimb implementation is preserved under
`archive/pre_upstream_reset_20260813/`. Its last active snapshot is Git commit `823b2f5`.
