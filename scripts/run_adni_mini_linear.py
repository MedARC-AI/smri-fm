"""Run the standard ADNI linear probe against the pinned ADNI-mini snapshot."""

from evaluation.adni_mini import load_adni_mini_eval
from evaluation.main_linear import cli
from evaluation.tasks import adni


adni.load_adni_eval = load_adni_mini_eval


if __name__ == "__main__":
    cli()
