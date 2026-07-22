"""
build_full_dataset.py
---------------------
Construye el dataset COMPLETO del contador general uniendo:
  - Mayo (labeling_subset): imagenes revisadas (difieren de la auto-etiqueta).
  - Videos nuevos (lote_nuevos_gabriel + lote_nuevos_companero): 120 imagenes.

Split POR FUENTE y contiguo/temporal (sin data leakage). Cada fuente
(video de mayo, fotos de mayo, IMG_0177, IMG_0178) aporta a train/val/test,
de modo que el TEST mide generalizacion en todas las condiciones.

Salidas:
  data/dataset_full/images|labels/{train,val,test}/
  data/dataset_full/data.yaml
  data/counts/conteo_full_test.csv   (image_name, real_count, source)
"""
from __future__ import annotations
import re, shutil
from pathlib import Path

IMG_EXTS={'.jpg','.jpeg','.png'}
ROOT=Path('.')
OUT=Path('data/dataset_full')
MAYO_SUBSET=Path('data/labeling_subset')
MAYO_ORIG=Path('data/labeling')
NEW_LOTES=[Path('data/labeling_collab/lote_nuevos_gabriel'),
           Path('data/labeling_collab/lote_nuevos_companero')]
TRAIN,VAL=0.8,0.1


def source_of(name:str)->str:
    if name.startswith('IMG_0177'): return 'vid_IMG_0177'
    if name.startswith('IMG_0178'): return 'vid_IMG_0178'
    m=re.match(r'(VID_\d+_\d+)',name)
    if m: return m.group(1)
    return 'foto_mayo'

def frame_index(name:str)->int:
    m=re.search(r'_frame_(\d+)',name)
    return int(m.group(1)) if m else 0

def nbox(txt:Path)->int:
    return len([l for l in txt.read_text(encoding='utf-8').splitlines() if l.strip()]) if txt.exists() else 0

def contiguous(items,tr,va):
    n=len(items); n_tr=int(round(n*tr)); n_va=int(round(n*va))
    if n>=3:
        n_tr=min(n_tr,n-2); n_va=max(n_va,1)
    return items[:n_tr], items[n_tr:n_tr+n_va], items[n_tr+n_va:]


def main():
    # recolectar (img, txt, source)
    groups={}
    # mayo revisadas
    n_mayo=0
    for img in MAYO_SUBSET.glob('*.jpg'):
        t=img.with_suffix('.txt'); o=MAYO_ORIG/f'{img.stem}.txt'
        if not t.exists(): continue
        if o.exists() and t.read_text()==o.read_text(): continue  # no revisada
        groups.setdefault(source_of(img.name),[]).append((img,t)); n_mayo+=1
    # videos nuevos
    n_new=0
    for lote in NEW_LOTES:
        for img in lote.glob('IMG_*.jpg'):
            t=img.with_suffix('.txt')
            if not t.exists(): continue
            groups.setdefault(source_of(img.name),[]).append((img,t)); n_new+=1

    # limpiar salida
    for sp in ('train','val','test'):
        for sub in ('images','labels'):
            d=OUT/sub/sp
            if d.exists(): shutil.rmtree(d)
            d.mkdir(parents=True,exist_ok=True)

    tally={'train':0,'val':0,'test':0}; test_rows=[]; total_box=0
    for src,items in groups.items():
        items=sorted(items,key=lambda p:(frame_index(p[0].name),p[0].name))
        tr,va,te=contiguous(items,TRAIN,VAL)
        for split,chunk in (('train',tr),('val',va),('test',te)):
            for img,txt in chunk:
                shutil.copy2(img,OUT/'images'/split/img.name)
                shutil.copy2(txt,OUT/'labels'/split/txt.name)
                tally[split]+=1; total_box+=nbox(txt)
                if split=='test':
                    test_rows.append((img.name,nbox(txt),src))

    # data.yaml
    (OUT/'data.yaml').write_text(
        f'path: {OUT.resolve().as_posix()}\n'
        'train: images/train\nval: images/val\ntest: images/test\n'
        'nc: 1\nnames:\n  0: alevin\n',encoding='utf-8')
    # csv test
    cd=Path('data/counts'); cd.mkdir(parents=True,exist_ok=True)
    with (cd/'conteo_full_test.csv').open('w',encoding='utf-8') as f:
        f.write('image_name,real_count,source\n')
        for n,c,s in test_rows: f.write(f'{n},{c},{s}\n')

    print('='*56)
    print('DATASET COMPLETO (contador general)')
    print('='*56)
    print(f'Mayo revisadas: {n_mayo} | Videos nuevos: {n_new} | TOTAL: {n_mayo+n_new}')
    print(f'Cajas totales: {total_box}')
    print('\nPor fuente:')
    for src,items in sorted(groups.items()):
        print(f'  {src:<18} {len(items)} imgs')
    print(f'\nSplit: train={tally["train"]} val={tally["val"]} test={tally["test"]}')
    print(f'Test por fuente:')
    from collections import Counter
    for s,c in sorted(Counter(r[2] for r in test_rows).items()):
        print(f'  {s:<18} {c}')
    print(f'\ndata.yaml -> {OUT/"data.yaml"}')

if __name__=='__main__':
    main()
