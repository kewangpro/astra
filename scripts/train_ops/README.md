Train-ops scripts relocated from Ensemble (Astra is the training center).

Mission training uses:

```bash
python -m astrarl.cli <grpo|agrpo|opd> --use-case ensemble --workspace …
python -m astratune.cli <sft|dpo|distill|rft|star|rlcd> --use-case ensemble --workspace …
```

These scripts are manual operator helpers only.
