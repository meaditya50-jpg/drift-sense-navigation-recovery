import os
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

TEST_CSV = r"D:\SemiconIndia\DriftSense\unified_dataset\test.csv"
CHECKPOINT = r"D:\SemiconIndia\DriftSense\checkpoints_zoom_v3\best.pt"
OUTPUT_CSV = r"D:\SemiconIndia\DriftSense\v3_test_results.csv"

SEARCH_SIZE = 1000
REF_SIZE = 100
COARSE_INPUT = 500
TARGET_SCALE_COARSE = 5
TOP_K = 12
ROI_SIZE = 160
BATCH_SIZE = 1
NUM_WORKERS = 0
FINE_THRESHOLD = 0.65
MAX_FINE_SHIFT_PX = 32.0

class DatasetV3(Dataset):
    def __init__(self, csv_file):
        self.rows = []
        with open(csv_file, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        print(f"Loaded {len(self.rows):,} test samples")
    def __len__(self):
        return len(self.rows)
    @staticmethod
    def load_gray(path):
        a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(a).unsqueeze(0)
    def __getitem__(self, i):
        r = self.rows[i]
        ref = self.load_gray(r["ref_path"])
        search = self.load_gray(r["search_path"])
        target = torch.tensor([float(r["x"])/SEARCH_SIZE, float(r["y"])/SEARCH_SIZE], dtype=torch.float32)
        return ref, search, target, i

class DSConv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        return x

class ReferenceEncoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.full_branch = nn.Sequential(DSConv(1,16,2), DSConv(16,32,2), DSConv(32,48,1), DSConv(48,channels,1))
        self.target_branch = nn.Sequential(DSConv(1,32,1), DSConv(32,channels,1))
        self.full_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.target_proj = nn.Conv2d(channels, channels, 1, bias=False)
    def forward(self, reference):
        full = self.full_branch(reference)
        target_view = F.interpolate(reference, size=(TARGET_SCALE_COARSE,TARGET_SCALE_COARSE), mode="area")
        target_features = self.target_proj(self.target_branch(target_view))
        full_embedding = F.normalize(F.adaptive_avg_pool2d(self.full_proj(full),1), dim=1)
        target_embedding = F.normalize(F.adaptive_avg_pool2d(target_features,1), dim=1)
        return full, target_features, full_embedding, target_embedding

class SearchEncoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(DSConv(1,16,2), DSConv(16,32,1), DSConv(32,48,1), DSConv(48,channels,1))
    def forward(self,x): return self.net(x)

class CoarseLocator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.search_projection = nn.Conv2d(channels,channels,1,bias=False)
        self.target_projection = nn.Conv2d(channels,channels,1,bias=False)
        self.full_reference_projection = nn.Conv2d(channels,channels,1,bias=False)
        self.head = nn.Sequential(
            nn.Conv2d(channels,32,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32,16,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16,1,1))
    def forward(self, target_features, target_embedding, full_embedding, search_features):
        search_map = F.normalize(self.search_projection(search_features), dim=1)
        target_map = F.normalize(self.target_projection(target_features), dim=1)
        target_template = F.adaptive_avg_pool2d(target_map,1)
        local_correlation = (search_map * target_template).sum(dim=1, keepdim=True)
        global_reference = F.normalize(self.full_reference_projection(full_embedding), dim=1)
        global_correlation = (search_map * global_reference).sum(dim=1, keepdim=True)
        return self.head(search_features) + 3.0*local_correlation + global_correlation

class FineMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.search_branch = nn.Sequential(DSConv(1,24,2), DSConv(24,40,2), DSConv(40,56,2), DSConv(56,72,2))
        self.reference_branch = nn.Sequential(DSConv(1,24,2), DSConv(24,40,2), DSConv(40,56,2), DSConv(56,72,2))
        self.fusion = nn.Sequential(DSConv(144,96,1), DSConv(96,96,1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(96,64), nn.ReLU(inplace=True), nn.Dropout(0.10))
        self.confidence = nn.Linear(64,1)
        self.offset = nn.Linear(64,2)
    def forward(self, roi, reference):
        ref = F.interpolate(reference, (10,10), mode="area")
        ref = F.interpolate(ref, (ROI_SIZE,ROI_SIZE), mode="bilinear", align_corners=False)
        sf = self.search_branch(roi)
        rf = self.reference_branch(ref)
        f = self.fusion(torch.cat([sf,rf], dim=1))
        f = self.fc(self.pool(f).flatten(1))
        return self.confidence(f).squeeze(1), torch.tanh(self.offset(f))

class AdaptiveZoomDriftSenseV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference_encoder = ReferenceEncoder()
        self.search_encoder = SearchEncoder()
        self.coarse_locator = CoarseLocator()
        self.fine_matcher = FineMatcher()
    def coarse_logits(self, reference, search):
        search_small = F.interpolate(search, (COARSE_INPUT,COARSE_INPUT), mode="area")
        _, tf, fe, te = self.reference_encoder(reference)
        sf = self.search_encoder(search_small)
        return self.coarse_locator(tf,te,fe,sf)

def topk_candidates(logits,k=TOP_K):
    b,_,h,w = logits.shape
    values, idx = torch.topk(logits.flatten(1), k=min(k,h*w), dim=1)
    ys=(idx//w).float()/max(h-1,1); xs=(idx%w).float()/max(w-1,1)
    return values, torch.stack([xs,ys],dim=-1)

def crop_rois(search, centers):
    b,c,h,w=search.shape; k=centers.shape[1]; half=ROI_SIZE//2; allb=[]
    for bi in range(b):
        rb=[]
        for ki in range(k):
            cx=int(round(float(centers[bi,ki,0])*SEARCH_SIZE)); cy=int(round(float(centers[bi,ki,1])*SEARCH_SIZE))
            x0,y0=cx-half,cy-half; x1,y1=x0+ROI_SIZE,y0+ROI_SIZE
            pl,pt,pr,pb=max(0,-x0),max(0,-y0),max(0,x1-w),max(0,y1-h)
            img=search[bi:bi+1]
            if pl or pt or pr or pb:
                img=F.pad(img,(pl,pr,pt,pb),value=0); x0+=pl; x1+=pl; y0+=pt; y1+=pt
            roi=img[:,:,y0:y1,x0:x1]
            if roi.shape[-2:]!=(ROI_SIZE,ROI_SIZE):
                fixed=torch.zeros((1,c,ROI_SIZE,ROI_SIZE),device=search.device,dtype=search.dtype)
                hh=min(ROI_SIZE,roi.shape[-2]); ww=min(ROI_SIZE,roi.shape[-1])
                fixed[:,:,:hh,:ww]=roi[:,:,:hh,:ww]; roi=fixed
            rb.append(roi)
        allb.append(torch.cat(rb,dim=0))
    return torch.stack(allb,dim=0)

def center_distance(xy):
    x=xy[...,0]*SEARCH_SIZE; y=xy[...,1]*SEARCH_SIZE
    return torch.sqrt((x-SEARCH_SIZE/2)**2+(y-SEARCH_SIZE/2)**2)

def select_candidate(coarse_centers, fine_centers, conf, coarse_scores):
    b,k,_=coarse_centers.shape; out=[]
    for i in range(b):
        valid=conf[i]>=FINE_THRESHOLD
        if valid.any():
            ids=torch.where(valid)[0]; d=center_distance(fine_centers[i,ids]); best=ids[torch.argmin(d)]
            out.append(fine_centers[i,best])
        else:
            out.append(coarse_centers[i,torch.argmax(coarse_scores[i])])
    return torch.stack(out)

@torch.no_grad()
def predict(model, ref, search):
    logits=model.coarse_logits(ref,search)
    cs, cc=topk_candidates(logits)
    rois=crop_rois(search,cc); b,k,c,h,w=rois.shape
    flat_rois=rois.reshape(b*k,c,h,w)
    flat_ref=ref[:,None].expand(b,k,1,REF_SIZE,REF_SIZE).reshape(b*k,1,REF_SIZE,REF_SIZE)
    clogits,off=model.fine_matcher(flat_rois,flat_ref)
    conf=torch.sigmoid(clogits).reshape(b,k); off=off.reshape(b,k,2)
    fine=torch.clamp(cc + off*(MAX_FINE_SHIFT_PX/SEARCH_SIZE),0,1)
    return select_candidate(cc,fine,conf,cs), conf

def main():
    print("="*70)
    print("DRIFT-SENSE V3 — FINAL TEST")
    print("="*70)
    if not os.path.exists(TEST_CSV): raise FileNotFoundError(TEST_CSV)
    if not os.path.exists(CHECKPOINT): raise FileNotFoundError(CHECKPOINT)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))

    ds=DatasetV3(TEST_CSV)
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=False)
    model=AdaptiveZoomDriftSenseV3().to(device)

    ckpt=torch.load(CHECKPOINT,map_location=device)
    print("Checkpoint epoch:",ckpt.get("epoch"))
    print("Checkpoint validation metrics:",ckpt.get("metrics",{}))
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    errors=[]; results=[]; fine_used=0

    for bi,(ref,search,target,idx) in enumerate(loader):
        ref=ref.to(device); search=search.to(device); target=target.to(device)
        with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
            pred,conf=predict(model,ref,search)

        e=torch.sqrt((((pred-target)*SEARCH_SIZE)**2).sum(dim=1))
        errors.extend(e.cpu().numpy())

        confmax=conf.max(dim=1).values
        fine_used += int((confmax>=FINE_THRESHOLD).sum().item())

        p=pred.cpu().numpy()*SEARCH_SIZE
        t=target.cpu().numpy()*SEARCH_SIZE
        ee=e.cpu().numpy()
        cc=confmax.cpu().numpy()

        for j in range(len(idx)):
            results.append({
                "index":int(idx[j]),
                "pred_x":float(p[j,0]),
                "pred_y":float(p[j,1]),
                "true_x":float(t[j,0]),
                "true_y":float(t[j,1]),
                "error_px":float(ee[j]),
                "confidence":float(cc[j])
            })

        if bi % 500 == 0 or bi == len(loader)-1:
            arr=np.asarray(errors)
            processed=len(errors)
            print(f"Processed {processed:,}/{len(ds):,} | <=10px {np.mean(arr<=10)*100:.2f}% | mean {arr.mean():.2f}px")

    errors=np.asarray(errors)
    print("\n"+"="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Samples: {len(errors):,}")
    print(f"Mean error: {errors.mean():.3f} px")
    print(f"Median error: {np.median(errors):.3f} px")
    print(f"P90: {np.percentile(errors,90):.3f} px")
    print(f"P95: {np.percentile(errors,95):.3f} px")
    for n in [1,3,5,10,20,50,100]:
        print(f"<= {n:>3} px: {np.mean(errors<=n)*100:.2f}%")
    print(f"Fine stage used: {100*fine_used/max(len(ds),1):.2f}%")

    with open(OUTPUT_CSV,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["index","pred_x","pred_y","true_x","true_y","error_px","confidence"])
        writer.writeheader(); writer.writerows(results)

    print("Detailed results:",OUTPUT_CSV)

if __name__=="__main__":
    main()
