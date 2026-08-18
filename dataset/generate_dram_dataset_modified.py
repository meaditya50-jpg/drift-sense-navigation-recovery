import os, csv, json, time, random, argparse, math
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import numpy as np
import cv2
from tqdm import tqdm

SEARCH_W = SEARCH_H = 1000
REF_W = REF_H = 100
TARGET_W = TARGET_H = 10

def sem_edge(image, alpha):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    m = float(g.max())
    if m > 0: g /= m
    return np.uint8(np.clip(image.astype(np.float32) + alpha * g * 255, 0, 255))

def sem_noise(image, shot_scale, read_std, blur_sigma):
    a = image.astype(np.float64) / 255.0
    poisson = np.random.poisson(np.maximum(a * shot_scale, 0)) / shot_scale
    gauss = np.random.normal(0, read_std / 255.0, image.shape)
    out = np.uint8(np.clip((poisson + gauss) * 255, 0, 255))
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0,0), blur_sigma)
    return out

def motion_blur(image, length, angle):
    if length <= 1: return image
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    t = math.radians(angle)
    for i in range(length):
        d = i - c
        x, y = int(round(c + math.cos(t)*d)), int(round(c + math.sin(t)*d))
        if 0 <= x < length and 0 <= y < length: k[y,x] = 1
    if k.sum(): k /= k.sum()
    return cv2.filter2D(image, -1, k)

def illum(image, gain, offset):
    return np.uint8(np.clip(image.astype(np.float32)*gain + offset, 0, 255))

def render_dram(width, height, px, py, vr, line=120, via=220):
    a = np.zeros((height,width), np.uint8)
    tx, ty = max(1, int(px*.22)), max(1, int(py*.22))
    x = 0
    while x < width:
        cv2.line(a, (int(x),0), (int(x),height-1), int(line), tx)
        x += px
    y = 0
    while y < height:
        cv2.line(a, (0,int(y)), (width-1,int(y)), int(line), ty)
        y += py
    x = 0
    while x < width:
        y = 0
        while y < height:
            cv2.circle(a, (int(x),int(y)), max(1,int(vr)), int(via), -1)
            y += py
        x += px
    return a

def add_signature(patch, seed):
    r = random.Random(seed)
    a = patch.copy()
    h,w = a.shape
    cx,cy = w//2,h//2
    mode = r.choice(["bright","dark","double","hgap","vgap","combined"])
    if mode in ["bright","double","combined"]: cv2.circle(a,(cx,cy),1,255,-1)
    if mode == "dark": cv2.circle(a,(cx,cy),1,40,-1)
    if mode in ["double","combined"]: cv2.circle(a,(min(w-1,cx+3),cy),1,245,-1)
    if mode in ["hgap","combined"]: cv2.line(a,(max(0,cx-3),cy),(min(w-1,cx+3),cy),10,1)
    if mode in ["vgap","combined"]: cv2.line(a,(cx,max(0,cy-3)),(cx,min(h-1,cy+3)),10,1)
    return a

def add_defect(a, r, p=.1):
    if r.random() >= p: return a
    out = a.copy(); h,w = out.shape
    x,y = r.randrange(w), r.randrange(h)
    mode = r.choice(["missing","gap","weak","extra"])
    if mode == "missing": cv2.circle(out,(x,y),r.choice([1,2]),10,-1)
    elif mode == "gap": cv2.rectangle(out,(max(0,x-3),y),(min(w-1,x+3),min(h-1,y+1)),10,-1)
    elif mode == "weak": cv2.circle(out,(x,y),r.choice([1,2]),90,-1)
    else: cv2.circle(out,(x,y),1,255,-1)
    return out

def place(dst, patch, cx, cy):
    ph,pw = patch.shape
    x0,y0 = int(round(cx-pw/2)), int(round(cy-ph/2))
    x1,y1 = x0+pw,y0+ph
    ix0,iy0,ix1,iy1 = max(0,x0),max(0,y0),min(dst.shape[1],x1),min(dst.shape[0],y1)
    if ix0 >= ix1 or iy0 >= iy1: return
    dst[iy0:iy1,ix0:ix1] = patch[iy0-y0:iy1-y0, ix0-x0:ix1-x0]

def near_match(target, r):
    d = target.copy()
    mode = r.choice(["brightness","blur","dot","line"])
    if mode == "brightness":
        d = np.uint8(np.clip(d.astype(np.float32)*r.uniform(.88,1.12),0,255))
    elif mode == "blur":
        d = cv2.GaussianBlur(d,(0,0),r.uniform(.2,.7))
    elif mode == "dot":
        d[r.randrange(d.shape[0]), r.randrange(d.shape[1])] = r.choice([30,220,255])
    else:
        yy = r.randrange(d.shape[0]); cv2.line(d,(0,yy),(d.shape[1]-1,yy),r.choice([40,210]),1)
    return d

def add_distractors(search,target,gx,gy,r,p=.65):
    if r.random() >= p: return 0,0
    used=[(gx,gy)]; exact=r.randint(1,3); near=r.randint(1,3)
    def pt(md):
        for _ in range(100):
            x,y=r.uniform(35,965),r.uniform(35,965)
            if all(math.hypot(x-u,y-v)>=md for u,v in used): return x,y
        return r.uniform(35,965),r.uniform(35,965)
    for _ in range(exact):
        x,y=pt(100); place(search,target,x,y); used.append((x,y))
    for _ in range(near):
        x,y=pt(80); place(search,near_match(target,r),x,y); used.append((x,y))
    return exact,near

def rotate_point(x,y,ang):
    cx=cy=499.5
    dx,dy=x-cx,y-cy; t=math.radians(ang)
    return math.cos(t)*dx-math.sin(t)*dy+cx, math.sin(t)*dx+math.cos(t)*dy+cy

def worker(task):
    idx, outdir, shard_size, master_seed, hard_p = task
    seed = (master_seed + idx*1000003) % (2**32-1)
    np.random.seed(seed); random.seed(seed); r=random.Random(seed)

    shard=idx//shard_size
    sdir=os.path.join(outdir,f"shard_{shard:04d}"); os.makedirs(sdir,exist_ok=True)

    px,py=r.uniform(18,30),r.uniform(18,30)
    base=render_dram(SEARCH_W,SEARCH_H,px,py,r.uniform(2.5,5),r.uniform(105,145),r.uniform(195,235))

    gx,gy=r.randint(30,970),r.randint(30,970)

    # TRUE HIGH-MAGNIFICATION REFERENCE:
    # crop 100x100 from physical target neighborhood.
    ref=base[max(0,gy-50):min(1000,gy+50),max(0,gx-50):min(1000,gx+50)]
    ref=cv2.copyMakeBorder(ref,0,max(0,100-ref.shape[0]),0,max(0,100-ref.shape[1]),cv2.BORDER_CONSTANT,value=0)[:100,:100]
    ref=add_signature(ref,seed+777)

    # Reference has lower degradation.
    ref_ang=r.uniform(-2.5,2.5); ref_scale=r.uniform(.96,1.04)
    M=cv2.getRotationMatrix2D((49.5,49.5),ref_ang,ref_scale)
    ref=cv2.warpAffine(ref,M,(100,100),borderMode=cv2.BORDER_CONSTANT,borderValue=10)
    ref=sem_edge(ref,r.uniform(.35,.70))
    ref=sem_noise(ref,r.uniform(35,55),r.uniform(4,9),r.uniform(0,.7))
    ref=illum(ref,r.uniform(.95,1.05),r.uniform(-5,5))

    # CRITICAL FIX:
    # Reference 100x100 -> target approximately 10x10 inside 1000x1000 search.
    tscale=r.uniform(.80,1.20)
    tw=max(6,round(10*tscale)); th=max(6,round(10*tscale))
    target=cv2.resize(add_signature(ref,seed+991),(tw,th),interpolation=cv2.INTER_AREA)
    target_ang=r.uniform(-2.5,2.5)
    M=cv2.getRotationMatrix2D((tw/2,th/2),target_ang,r.uniform(.96,1.04))
    target=cv2.warpAffine(target,M,(tw,th),borderMode=cv2.BORDER_CONSTANT,borderValue=10)
    target=sem_edge(target,r.uniform(.2,.45))
    # Independent target/search noise.
    target=sem_noise(target,r.uniform(20,40),r.uniform(3,7),r.uniform(0,.45))

    search=base.copy()
    place(search,target,gx,gy)

    exact,near=add_distractors(search,target,gx,gy,r,hard_p)

    search=sem_edge(search,r.uniform(.45,.85))
    search=sem_noise(search,r.uniform(12,25),r.uniform(12,24),r.uniform(.25,1.35))
    if r.random()<.65: search=motion_blur(search,r.choice([3,5,7]),r.uniform(0,180))
    search=illum(search,r.uniform(.85,1.15),r.uniform(-12,12))

    # Simulated global navigation/stage error.
    stage_ang=r.uniform(-.40,.40)
    M=cv2.getRotationMatrix2D((499.5,499.5),stage_ang,1.0)
    search=cv2.warpAffine(search,M,(1000,1000),borderMode=cv2.BORDER_CONSTANT,borderValue=10)
    gx,gy=rotate_point(gx,gy,stage_ang)

    dx,dy=r.uniform(-8,8),r.uniform(-8,8)
    shifted=np.full_like(search,10)
    sx=max(0,int(round(-dx))); sy=max(0,int(round(-dy)))
    tx=max(0,int(round(dx))); ty=max(0,int(round(dy)))
    cw=min(1000-sx,1000-tx); ch=min(1000-sy,1000-ty)
    if cw>0 and ch>0: shifted[ty:ty+ch,tx:tx+cw]=search[sy:sy+ch,sx:sx+cw]
    search=shifted; gx+=dx; gy+=dy
    gx=float(np.clip(gx,0,999)); gy=float(np.clip(gy,0,999))

    ref_path=os.path.join(sdir,f"ref_{idx:07d}.png")
    search_path=os.path.join(sdir,f"search_{idx:07d}.png")
    cv2.imwrite(ref_path,ref,[cv2.IMWRITE_PNG_COMPRESSION,3])
    cv2.imwrite(search_path,search,[cv2.IMWRITE_PNG_COMPRESSION,3])

    return {"id":idx,"style":"DRAM",
            "ref":os.path.relpath(ref_path,outdir).replace("\\","/"),
            "search":os.path.relpath(search_path,outdir).replace("\\","/"),
            "x":round(gx,4),"y":round(gy,4),
            "reference_width":100,"reference_height":100,
            "search_width":1000,"search_height":1000,
            "nominal_scale":10,"target_scale":round(tscale,5),
            "target_rotation_deg":round(target_ang,5),
            "stage_rotation_deg":round(stage_ang,5),
            "stage_translation_x":round(dx,5),"stage_translation_y":round(dy,5),
            "exact_duplicate_count":exact,"near_match_count":near,
            "pitch_x":round(px,5),"pitch_y":round(py,5)}

def chunks(it,n):
    it=iter(it)
    while True:
        c=list(islice(it,n))
        if not c: return
        yield c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--count",type=int,default=100000)
    ap.add_argument("--output",default="./dram_dataset_100k")
    ap.add_argument("--shard-size",type=int,default=1000)
    ap.add_argument("--workers",type=int,default=max(1,(os.cpu_count() or 4)-1))
    ap.add_argument("--seed",type=int,default=20260814)
    ap.add_argument("--hard-probability",type=float,default=.65)
    ap.add_argument("--queue-chunk",type=int,default=256)
    a=ap.parse_args()

    if not 0<=a.hard_probability<=1: raise ValueError("--hard-probability must be 0..1")
    os.makedirs(a.output,exist_ok=True)
    print(f"Generating {a.count} DRAM pairs -> {os.path.abspath(a.output)}")
    print("Reference: 100x100 | Search: 1000x1000 | Target: ~10x10 | PNG storage")

    labels=os.path.join(a.output,"labels.csv")
    fields=["id","style","ref","search","x","y","reference_width","reference_height",
            "search_width","search_height","nominal_scale","target_scale",
            "target_rotation_deg","stage_rotation_deg","stage_translation_x",
            "stage_translation_y","exact_duplicate_count","near_match_count",
            "pitch_x","pitch_y"]

    tasks=((i,a.output,a.shard_size,a.seed,a.hard_probability) for i in range(1,a.count+1))
    start=time.time(); done=0
    with open(labels,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for chunk in chunks(tasks,max(a.queue_chunk,a.workers*4)):
                for row in ex.map(worker,chunk,chunksize=1):
                    w.writerow(row); done+=1
                    if done%5000==0 or done==a.count:
                        rate=done/max(time.time()-start,1e-9)
                        eta=(a.count-done)/max(rate,1e-9)/60
                        print(f"[{done}/{a.count}] {rate:.2f} pairs/s | ETA {eta:.1f} min")

    # JSON index for convenience.
    with open(labels,encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    with open(os.path.join(a.output,"labels.json"),"w",encoding="utf-8") as f:
        json.dump(rows,f,indent=2)

    with open(os.path.join(a.output,"dataset_config.json"),"w",encoding="utf-8") as f:
        json.dump({"architecture":"DRAM","pairs":a.count,
                   "reference":[100,100],"search":[1000,1000],
                   "nominal_scale":10,"hard_probability":a.hard_probability,
                   "seed":a.seed,"storage":"PNG"},f,indent=2)

    print(f"Done. {done} pairs in {(time.time()-start)/60:.2f} minutes.")
    print(f"Labels: {os.path.abspath(labels)}")

if __name__=="__main__":
    main()
