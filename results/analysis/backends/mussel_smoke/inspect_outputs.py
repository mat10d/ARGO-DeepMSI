import sys, h5py, numpy as np, torch
def walk(p):
    with h5py.File(p) as f:
        print("== ", p, "root attrs:", dict(f.attrs))
        def v(name, o):
            if isinstance(o, h5py.Dataset):
                print(f"  {name}: shape={o.shape} dtype={o.dtype!r} chunks={o.chunks} compression={o.compression}")
                for k, a in o.attrs.items(): print(f"     attr {k} = {a!r} ({type(a).__name__})")
                if o.ndim and o.shape[0]: print("     first rows:", o[:3].tolist() if o.dtype.kind!='V' else o[:1].tobytes()[:16])
            else:
                print(f"  group {name} attrs={dict(o.attrs)}")
        f.visititems(v)
        if "coords" in f:
            c=f["coords"][:]; print("  coords min", c.min(0), "max", c.max(0), "unique dx", np.unique(np.diff(np.unique(c[:,0])))[:5])
for p in sys.argv[1:]:
    if p.endswith(".h5"): walk(p)
    else:
        t=torch.load(p, weights_only=True); print("== ", p, type(t).__name__, getattr(t,'shape',None), getattr(t,'dtype',None))
