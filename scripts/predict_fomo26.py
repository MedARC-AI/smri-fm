# inference script for fomo26 tasks
# `build_fomo26_submission.sh` copies this file into the submission image file

# todo:
# - deal w/ multiple inputs
# - add task to container.run

from pathlib import Path
import os, numpy as np, torch, nibabel as nib
from omegaconf import OmegaConf as OC
from fastcore.basics import AttrDict
from fastcore.script import call_parse
from asparagus_bridge.models_smri_mae import SmriMaeClsRegBackbone, SmriMaeSegBackbone
from evaluation.models.smri_mae import SmriMaeTransform

nets = dict(cls=SmriMaeClsRegBackbone, reg=SmriMaeClsRegBackbone, seg=SmriMaeSegBackbone, emb=SmriMaeClsRegBackbone)
keys = dict(cls='_cls_net',            reg='_reg_net',            seg='_seg_net',         emb='_cls_net')
tasks = AttrDict({
    1: AttrDict(kind='cls', inp=['flair','adc','dwi','t2s','swi'], out_ch=2),
    2: AttrDict(kind='seg', inp=['flair','dwi','t2s','swi'],       out_ch=2),
    3: AttrDict(kind='reg', inp=['t1'],                            out_ch=1),
    4: AttrDict(kind='seg', inp=['t2'],                            out_ch=3),
    5: AttrDict(kind='cls', inp=['t1'],                            out_ch=2),
    6: AttrDict(kind='emb', inp=['input'],                         out_ch=2),
    7: AttrDict(kind='emb', inp=['input'],                         out_ch=2)})

def strip_prefix(sd, pre='model.'): return {k[len(pre):]:v for k,v in sd.items() if k.startswith(pre)}
def local_path(path): return Path(__file__).parent/path
def get_cfg(key):
    yaml = local_path('config.yaml').read_text()
    return OC.to_container(getattr(OC.create(yaml).model, key), resolve=True)

@call_parse
def main(
    output:str=None, # path to save output
    flair :str=None, # path to T2 FLAIR image
    adc   :str=None, # path to ADC image
    dwi   :str=None, # path to DWI image
    t2s   :str=None, # path to T2* image
    swi   :str=None, # path to SWI image
    t1    :str=None, # path to T1 image
    t2    :str=None, # path to T2 image
    input :str=None, # generic input (tasks 6/7)
):
    t = tasks[int(os.environ['FOMO_TASK'])]
    device,cuda = ('cuda',True) if torch.cuda.is_available() else ('cpu',False)
    # load and prepare model
    cfg = get_cfg(keys[t.kind])
    model = nets[t.kind](input_channels=1, output_channels=t.out_ch, **cfg)
    ckpt = torch.load(local_path('checkpoint.pth'), map_location=device, weights_only=False)['state_dict']
    model.load_state_dict(strip_prefix(ckpt))
    model.eval().to(device)
    if cuda: model.half()
    # load and prepare input
    tfm = SmriMaeTransform(img_size=cfg.get('img_size', (160,160,160)))
    paths = dict(flair=flair, adc=adc, dwi=dwi, t2s=t2s, swi=swi, t1=t1, t2=t2, input=input)
    img = nib.load(paths[t.inp[0]]) # todo: currently taking only 1st file
    x = tfm(img)['image'][None].to(device)
    x = x.half() if cuda else x.float()
    # run model and write prediction
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        if t.kind=='cls':
            pred = model(x).softmax(-1)[0,1].item()
            Path(output).write_text(f'{pred:.3f}')
        elif t.kind=='reg':
            pred = model(x)[0,0].item()
            Path(output).write_text(f'{pred:.3f}')
        elif t.kind=='seg':
            mask = model(x)[0].argmax(0).cpu().numpy().astype(np.uint8)
            nib.save(nib.Nifti1Image(mask, img.affine, img.header), output)
        elif t.kind=='emb':
            emb = model._encode(x)[0].flatten().cpu().numpy()
            np.save(output, emb)
