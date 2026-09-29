# Hierarchical RadZero Planning

This directory is the working home for the Anatomy -> Finding -> Global
Interpretation project built on RadZero.

## Documents

- [TODO.md](TODO.md): prioritized implementation checklist.
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md): ordered implementation and verification stages.
- [DECISIONS.md](DECISIONS.md): architecture decisions and unresolved questions.
- [LABEL_CONTRACT.md](LABEL_CONTRACT.md): target shapes and unknown-label rules.
- [TRAINING_STAGES.md](TRAINING_STAGES.md): stage ownership and teacher forcing.
- [DATA_AUDIT.md](DATA_AUDIT.md): available inputs and current data blockers.
- `exp/cxr_pt/hierarchy_contract.py`: reproducible label-contract builder.
- `exp/cxr_pt/hierarchy_dataset.py`: manifest builder, dataset, and collator.
- `exp/cxr_pt/hierarchy_metrics.py`: validity-aware hierarchy metrics.
- Original research proposal: `/home/user/gxq/idea.md`.

Keep experimental results and generated datasets outside Git. Record their paths,
configurations, and conclusions in this directory instead.
