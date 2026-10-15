import sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("core", "/home/user/Fedlearning/experiments/core.py")
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
def P(m): return sum(p.numel() for p in m.parameters())
for ds,(inch,lat) in {"mnist":(3,16),"cifar10":(10,50)}.items():
    D=inch*256
    proto=torch.zeros(10,inch,16,16)
    cv=core.PaperCVAE(inch,lat); pc=core.ProtoCVAE(inch,lat,proto); pcr=core.ProtoCVAE(inch,lat,proto,rows=True); km=core.KMEGen(inch,lat,proto)
    print(ds,"D",D,"CVAE",P(cv),"PCVAE",P(pc),"PCR",P(pcr),"KMEGen",P(km), "label_proj", cv.label_proj.numel())
    tot=0; er=0; ds_=[]
    for g,w in enumerate(core.HEADER_WIDTHS):
        h=core.PaperHeader(w,inch,ds!="mnist")
        d=h.fc.in_features; ds_.append(d)
        body=P(h)-P(h.fc)
        tot+=P(h); er+=10*d+d*d+10
        print("  head",g,w,"params",P(h),"body",body,"d_g",d,"ER",10*d+d*d+10, "ER_sym", 10*d+d*(d+1)//2+10)
    print(" sum heads",tot,"sum ER",er)
    fe=core.PaperFE(1 if ds=="mnist" else 3, "mnist" if ds=="mnist" else "cifar10"); print(" FE",P(fe))
