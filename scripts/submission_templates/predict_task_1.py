from pathlib import Path
import argparse
import nibabel as nib, torch
from omegaconf import OmegaConf
from asparagus_bridge.models_smri_mae import SmriMaeClsRegBackbone
from evaluation.models.smri_mae import SmriMaeTransform

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FOMO25 Task 1 - Infarct Classification"
    )

    # Input paths for each modality
    parser.add_argument("--flair", type=str, help="Path to T2 FLAIR image")
    parser.add_argument("--adc", type=str, help="Path to ADC image")
    parser.add_argument("--dwi", type=str, help="Path to DWI image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, help="Path to SWI image (optional)")

    # Output path for predictions
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save output .txt file"
    )

    return parser.parse_args()

def strip_prefix(sd, pre='model.'): return {k[len(pre):]:v for k,v in sd.items() if k.startswith(pre)}
def local_path(path): return Path(__file__).parent/path

def predict(flair, adc, dwi, output, t2s=None, swi=None):
    cfg = OmegaConf.create(local_path('config.yaml').read_text())
    cfg = OmegaConf.to_container(cfg.model._cls_net, resolve=True)
    model = SmriMaeClsRegBackbone(input_channels=1, output_channels=2, **cfg) # todo: out channel correct?
    ckpt = torch.load(local_path('checkpoint.pth'), map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(strip_prefix(ckpt))
    model.half().eval()
    tfm = SmriMaeTransform(img_size=cfg.get('img_size', (160,160,160)))
    batch = tfm(nib.load(flair))
    with torch.no_grad(): pred = model(batch['image'][None]).softmax(-1)[0,1].item()
    return pred

def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Get prediction probability
    probability = predict(**vars(args))

    # Save probability in a text file called <subject_id>.txt
    subject_id = Path(args.output).stem  # Extract subject ID from output path
    output_file = Path(args.output).parent / f"{subject_id}.txt"
    Path(output_file).write_text(f'{probability:.3f}')

    return 0


if __name__ == "__main__":
    exit(main())
