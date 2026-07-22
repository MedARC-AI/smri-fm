from pathlib import Path
import nibabel as nib, torch
from omegaconf import OmegaConf
from fastcore.script import call_parse
from asparagus_bridge.models_smri_mae import SmriMaeClsRegBackbone
from evaluation.models.smri_mae import SmriMaeTransform

def strip_prefix(sd, pre='model.'): return {k[len(pre):]:v for k,v in sd.items() if k.startswith(pre)}
def local_path(path): return Path(__file__).parent/path
def get_cfg():
    yaml = local_path('config.yaml').read_text()
    return OmegaConf.to_container(OmegaConf.create(yaml).model._cls_net, resolve=True)

@call_parse
def predict(
    output:str=None, # path to save output .txt file
    flair :str=None, # path to T2 FLAIR image
    adc   :str=None, # path to ADC image
    dwi   :str=None, # path to DWI image
    t2s   :str=None, # path to T2* image (optional)
    swi   :str=None, # path to SWI image (optional)
):
    (device,cuda) = ('cuda',True) if torch.cuda.is_available() else ('cpu',False)
    cfg = get_cfg()
    model = SmriMaeClsRegBackbone(input_channels=1, output_channels=2, **cfg) # todo: out channel correct?
    ckpt = torch.load(local_path('checkpoint.pth'), map_location=device, weights_only=False)['state_dict']
    model.load_state_dict(strip_prefix(ckpt))
    model.eval().to(device)
    tfm = SmriMaeTransform(img_size=cfg.get('img_size', (160,160,160)))
    x = tfm(nib.load(flair))
    x = x['image'][None].to(device)
    if cuda: model.half()
    x = x.half() if cuda else x.float()
    with torch.no_grad(): pred = model(x).softmax(-1)[0,1].item()
    # Save prediction ot file
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    subject_id = Path(output).stem  # extract subject ID from output path
    (Path(output).parent/f'{subject_id}.txt').write_text(f'{pred:.3f}')
