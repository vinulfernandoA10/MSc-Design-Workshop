# Thesis project: calibration under distribution shift (MedMNIST)

M.Sc. AI thesis (UrFU). The student has no software-engineering background: explain in plain language,
give copy-paste commands, and keep code minimal. Start with README.md (findings, figures, file table).

## Status (2026-10-24)
Experiments are complete: 4 datasets (pneumonia, derma, blood, organc MNIST, 28px) x 4 methods
(cross-entropy, label smoothing, focal loss, MC Dropout) x 5 seeds, checkpoint selected by val NLL,
50 epochs, evaluated clean + 5 corruptions x 5 severities. Results: `results_<data>_nll.md`; pooled
tests: `significance_pneumoniamnist_dermamnist_bloodmnist_organcmnist_nll.md`; figures in `figures/`.
Next phase: literature review and thesis writing.

## Environment
- Python venv: `C:\Users\Vinul\.venvs\thesis\Scripts\python.exe` (GPU: GTX 1050 Ti, 4 GB, throttles when hot).
- Run every script from this folder (`cd C:\Users\Vinul\Desktop\UrFU\Research\Thesis;` first): paths are relative.
- In PowerShell, plain `python` opens the Microsoft Store alias; always use the venv path.
- Training started from a Claude session gets killed after ~10-15 min. Anything longer: give the student
  the command to run in their own PowerShell window. Training is resumable; `run_all.py` skips finished runs.
- `runs/` (models, logits, logs) is gitignored; everything else is on GitHub `vinulfernandoA10/MSc-Design-Workshop`, branch `main`.

## Pitfalls already found
- PneumoniaMNIST and OrganCMNIST have a val/test class-balance mismatch; DermaMNIST/BloodMNIST do not.
- Plain accuracy under heavy corruption is inflated by majority-class collapse (DermaMNIST): use balanced accuracy.
- Reliability-diagram ECE (fig 3) is pooled over seeds and corruptions, so it is lower than table ECE.
- AI-generated text must be disclosed per the programme's academic-integrity policy: label drafts as such.
