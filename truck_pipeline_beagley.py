"""
Truck Load Volume Estimator - BeagleY-AI
yolo_inf.py pipeline + sequential camera capture
Output: 4-panel image matching yolo_inf output
"""
import sys
if '/usr/lib/python3.12/site-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3.12/site-packages')

import cv2, math, time, json, os, threading
import numpy as np
import onnxruntime as ort
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from ultrasonic_sensor import UltrasonicSystem
    ULTRASONIC_AVAILABLE = True
except ImportError:
    ULTRASONIC_AVAILABLE = False

CONFIG = {
    "MODELS": {
        "SIDE_TRUCKBED":    "side_view_slim.onnx",
        "TOP_LOADING_AREA": "best_loading_area_detector_slim.onnx",
        "TOP_MATERIAL":     "best_material_detector_slim.onnx",
        "DEPTH":            "depth_anything_slim.onnx",
    },
    "TOP_CAM_DEV":  "/dev/video-usb-cam1",
    "SIDE_CAM_DEV": "/dev/video-usb-cam0",
    "CAPTURE_W": 320, "CAPTURE_H": 240,
    "TOP_CAM_DURATION": 10, "SIDE_CAM_DURATION": 10,
    # yolo_inf.py calibration values
    "TOP_VIEW_PIXEL_AREA_CM2":  0.000751,   # measured: 175x190px = 5x5cm @ 32cm height
    "TOP_VIEW_CAMERA_HEIGHT_M": 0.32,       # camera height above ground
    "SIDE_VIEW_PIXEL_AREA_CM2": 0.000343,   # measured: 280x260px = 5x5cm @ 21cm distance
    "FALLBACK_BED_HEIGHT_M":    0.035,   # truck bed 3.5cm above ground
    "FALLBACK_TOP_BASELINE_M":  0.285,   # 32cm - 3.5cm bed height
    "YOLO_INPUT_SIZE": 640,
    "DEPTH_INPUT_H": 518, "DEPTH_INPUT_W": 518,
    "CONF_THRESH": {"AREA": 0.4, "MATERIAL": 0.3, "SIDE_BED": 0.5},
    "IOU_THRESH": 0.45,
    "TRUCK_PRESENCE_RATIO": 0.7,
    "STREAM_PORT": 8080,
    "STREAM_JPEG_QUALITY": 80,
    "DENSITIES": {
        'sand':1600,'gravel':1680,'timber':700,'brick':1900,'soil':1200,
        'stone':2500,'steel':7850,'cement brick':2100,'copper':8960,
        'carton':120,'machinery':1500,'big_bag':600,'barrel':400,
        'tarp':200,'default':500
    },
}

MATERIAL_NAMES = {
    0:'Sand',1:'Gravel',2:'Timber',3:'bag',4:'barrel',5:'big_bag',
    6:'cement brick',7:'blue-tag',8:'building structure',9:'cables',
    10:'carton',11:'copper',12:'crane',13:'cylinder',14:'excavator',
    15:'hidden-single-bar',16:'hole-single-bar',
    17:'intermediate_bulk_container',18:'load carrying crane',
    19:'machinery',20:'metal_strip',21:'pallet',22:'person',
    23:'person with PPE',24:'pile casing',25:'pipeline',
    26:'plastic-tube',27:'plastic_bucket',28:'red-tag',29:'secant pile',
    30:'single-bar',31:'soil',32:'square_package',33:'steel',
    34:'steel_coil',35:'stone',36:'tarp',37:'Brick',38:'transport_barrel'
}

_DEPTH_MEAN = np.array([0.485,0.456,0.406],dtype=np.float32)
_DEPTH_STD  = np.array([0.229,0.224,0.225],dtype=np.float32)

class SharedState:
    def __init__(self):
        self.lock=threading.Lock(); self.frame_out=None
        self.results={}; self.status="Waiting..."
state = SharedState()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        p=self.path.split("?")[0]
        if p=="/": self._html()
        elif p=="/stream/output": self._jpeg()
        elif p=="/results":
            with state.lock: payload=json.dumps(state.results,indent=2).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",len(payload)); self.end_headers()
            self.wfile.write(payload)
        else: self.send_error(404)
    def _jpeg(self):
        with state.lock: data=state.frame_out
        if data is None:
            blank=np.full((480,1280,3),30,np.uint8)
            with state.lock: msg=state.status
            cv2.putText(blank,msg,(20,240),cv2.FONT_HERSHEY_SIMPLEX,0.8,(180,180,180),2)
            _,buf=cv2.imencode(".jpg",blank); data=buf.tobytes()
        self.send_response(200); self.send_header("Content-Type","image/jpeg")
        self.send_header("Content-Length",len(data))
        self.send_header("Cache-Control","no-cache,no-store"); self.end_headers()
        try: self.wfile.write(data)
        except: pass
    def _html(self):
        self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
        h=("<!DOCTYPE html><html><head><title>Truck Load Estimator</title>"
           "<style>body{background:#111;color:#eee;font-family:monospace;text-align:center;margin:0;padding:15px}"
           "h2{color:#0df}img{width:95%;max-width:1280px;border:2px solid #333;border-radius:6px}"
           "#st{color:#fa0;font-size:0.95em}#res{margin:10px auto;padding:12px;background:#1a1a1a;"
           "display:inline-block;text-align:left;border-radius:4px;border:1px solid #333;"
           "font-size:0.9em;min-width:360px;white-space:pre}</style>"
           "<script>function rf(){var t=Date.now();"
           "document.getElementById('img').src='/stream/output?t='+t;"
           "fetch('/results').then(r=>r.json()).then(d=>{"
           "document.getElementById('res').innerText=JSON.stringify(d,null,2);"
           "document.getElementById('st').innerText=d.status||'';}).catch(()=>{});}"
           "setInterval(rf,2000);</script></head><body>"
           "<h2>Truck Load Estimator - BeagleY-AI</h2>"
           "<p id='st'>Initialising...</p><p style='color:#888;font-size:0.8em'>Refreshes every 2s</p>"
           "<img id='img' src='/stream/output' alt='Output'><br><br>"
           "<pre id='res'>Waiting...</pre></body></html>")
        self.wfile.write(h.encode())

def _enc(bgr):
    _,buf=cv2.imencode(".jpg",bgr,[cv2.IMWRITE_JPEG_QUALITY,CONFIG["STREAM_JPEG_QUALITY"]])
    return buf.tobytes()

def _build_session(path):
    if not os.path.exists(path): print(f"   MISS {path}"); return None
    try:
        s=ort.InferenceSession(path,providers=["CPUExecutionProvider"])
        print(f"   OK  {os.path.basename(path)}"); return s
    except Exception as e: print(f"   FAIL {path}: {e}"); return None

def _letterbox(img,t=640):
    h,w=img.shape[:2]; sc=t/max(h,w); nh,nw=int(round(h*sc)),int(round(w*sc))
    c=np.full((t,t,3),114,np.uint8); pt,pl=(t-nh)//2,(t-nw)//2
    c[pt:pt+nh,pl:pl+nw]=cv2.resize(img,(nw,nh)); return c,sc,pt,pl

def _nms(boxes,scores,thr):
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        if order.size==1: break
        inter=(np.minimum(x2[i],x2[order[1:]])-np.maximum(x1[i],x1[order[1:]])).clip(0)*\
              (np.minimum(y2[i],y2[order[1:]])-np.maximum(y1[i],y1[order[1:]])).clip(0)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order=order[1:][iou<=thr]
    return keep

def _yolo(sess,img,conf,iou,sz=640):
    oh,ow=img.shape[:2]; lb,sc,pt,pl=_letterbox(img,sz)
    blob=cv2.cvtColor(lb,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    blob=blob.transpose(2,0,1)[np.newaxis]
    raw=sess.run(None,{sess.get_inputs()[0].name:blob})[0]
    pred=raw[0]
    if pred.shape[0]<pred.shape[1]: pred=pred.T
    oc=pred[:,4:].max(axis=1); mk=oc>=conf
    if not mk.any(): return []
    pred=pred[mk]; oc=oc[mk]; ci=pred[:,4:].argmax(axis=1)
    cx,cy,bw,bh=pred[:,0],pred[:,1],pred[:,2],pred[:,3]
    x1=np.clip((cx-bw/2-pl)/sc,0,ow); y1=np.clip((cy-bh/2-pt)/sc,0,oh)
    x2=np.clip((cx+bw/2-pl)/sc,0,ow); y2=np.clip((cy+bh/2-pt)/sc,0,oh)
    boxes=np.stack([x1,y1,x2,y2],axis=1); keep=_nms(boxes,oc,iou)
    return [{"x1":int(boxes[i,0]),"y1":int(boxes[i,1]),
             "x2":int(boxes[i,2]),"y2":int(boxes[i,3]),
             "conf":float(oc[i]),"cls":int(ci[i])} for i in keep]

class TruckVolumePipeline:
    def __init__(self):
        print("\n=== Loading Models ===")
        self.sess={}
        for k,p in CONFIG["MODELS"].items(): self.sess[k]=_build_session(p)
        if not self.sess.get("DEPTH"): print("ERROR: depth model required"); sys.exit(1)
        ds=self.sess["DEPTH"]; self._di=ds.get_inputs()[0].name
        sh=ds.get_inputs()[0].shape
        self._dh=CONFIG["DEPTH_INPUT_H"] if not isinstance(sh[2],int) else sh[2]
        self._dw=CONFIG["DEPTH_INPUT_W"] if not isinstance(sh[3],int) else sh[3]
        print(f"   Depth: {self._dh}x{self._dw}\n")

    def _density(self,lbl):
        return CONFIG["DENSITIES"].get(lbl.lower(),CONFIG["DENSITIES"]["default"])

    def _height_map(self,bgr):
        oh,ow=bgr.shape[:2]
        rgb=cv2.cvtColor(cv2.resize(bgr,(self._dw,self._dh)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        rgb=(rgb-_DEPTH_MEAN)/_DEPTH_STD; blob=rgb.transpose(2,0,1)[np.newaxis]
        raw=self.sess["DEPTH"].run(None,{self._di:blob})[0].squeeze()
        raw=cv2.resize(raw.astype(np.float32),(ow,oh),interpolation=cv2.INTER_LINEAR)
        raw=np.nan_to_num(raw,nan=0.)
        mn,mx=raw.min(),raw.max()
        if mx-mn<1e-6: return np.zeros((oh,ow),np.float32)
        return ((raw-mn)/(mx-mn)*CONFIG["TOP_VIEW_CAMERA_HEIGHT_M"]).astype(np.float32)

    def analyze_side(self,img):
        """Exact yolo_inf.py analyze_side_view_metric logic"""
        vis=img.copy(); bed_m=CONFIG["FALLBACK_BED_HEIGHT_M"]
        sess=self.sess.get("SIDE_TRUCKBED")
        if sess:
            dets=_yolo(sess,img,CONFIG["CONF_THRESH"]["SIDE_BED"],CONFIG["IOU_THRESH"],CONFIG["YOLO_INPUT_SIZE"])
            if dets:
                det=max(dets,key=lambda d:(d["x2"]-d["x1"])*(d["y2"]-d["y1"]))
                x1,y1,x2,y2=det["x1"],det["y1"],det["x2"],det["y2"]
                h=img.shape[0]; gy=h-5; by=y2
                bed_px=max(0,gy-by)
                # yolo_inf.py formula: 1px = sqrt(SIDE_VIEW_PIXEL_AREA_CM2) cm
                bed_m=(bed_px*math.sqrt(CONFIG["SIDE_VIEW_PIXEL_AREA_CM2"]))/100.
                cv2.line(vis,(0,gy),(vis.shape[1],gy),(0,0,255),3)
                cv2.line(vis,(x1,by),(x2,by),(0,255,0),3)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(255,165,0),2)
                cv2.putText(vis,f"Base H = {bed_m:.2f}m",(10,h-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
            else:
                cv2.putText(vis,f"Bed H: {bed_m:.3f}m (fallback)",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        return vis, bed_m

    def calc_volume(self,top,bed_m):
        """Exact yolo_inf.py calculate_volume_metric logic"""
        vis=top.copy(); hm=self._height_map(top)
        oh,ow=top.shape[:2]; mask=np.zeros((oh,ow),np.uint8)
        mat="None Detected"; crop=None
        # 1. Loading area
        s=self.sess.get("TOP_LOADING_AREA")
        if s:
            dets=_yolo(s,top,CONFIG["CONF_THRESH"]["AREA"],CONFIG["IOU_THRESH"],CONFIG["YOLO_INPUT_SIZE"])
            if dets:
                det=max(dets,key=lambda d:(d["x2"]-d["x1"])*(d["y2"]-d["y1"]))
                x1,y1,x2,y2=det["x1"],det["y1"],det["x2"],det["y2"]
                mask[y1:y2,x1:x2]=1; crop=(x1,y1,x2,y2)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(vis,f"Load Area ({det['conf']:.2f})",
                            (x1,max(y1-8,12)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
        # 2. Material
        s=self.sess.get("TOP_MATERIAL")
        if crop and s:
            cx1,cy1,cx2,cy2=crop
            dets=_yolo(s,top[cy1:cy2,cx1:cx2],CONFIG["CONF_THRESH"]["MATERIAL"],
                       CONFIG["IOU_THRESH"],CONFIG["YOLO_INPUT_SIZE"])
            if dets:
                best=max(dets,key=lambda d:d["conf"])
                mat=MATERIAL_NAMES.get(best["cls"],f"Class {best['cls']}")
                bx1=best["x1"]+cx1;by1=best["y1"]+cy1
                bx2=best["x2"]+cx1;by2=best["y2"]+cy1
                cv2.rectangle(vis,(bx1,by1),(bx2,by2),(0,255,0),2)
                cv2.putText(vis,mat,(bx1,max(by1-8,12)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        # 3. Volume (exact yolo_inf formula)
        if not mask.any(): mask=np.ones((oh,ow),np.uint8)
        net=np.maximum(0.,hm*mask-bed_m)
        net=np.nan_to_num(net,nan=0.)
        px_m2=CONFIG["TOP_VIEW_PIXEL_AREA_CM2"]*1e-4   # cm^2 -> m^2
        vol=float(np.sum(net)*px_m2)
        if np.isnan(vol) or np.isinf(vol): vol=0.
        # 4. Heatmap
        if net.max()>1e-6:
            norm=(net/(net.max()+1e-6)*255).astype(np.uint8)
            heat=cv2.applyColorMap(norm,cv2.COLORMAP_JET)
        else:
            heat=np.zeros_like(top)
        return vis, heat, vol, mat

    def build_panel(self,side_vis,top_vis,heatmap,vol,mat,bed_m,density):
        """Build 4-panel output matching yolo_inf.py matplotlib layout"""
        W,H=640,480
        def rsz(i): return cv2.resize(i,(W,H))
        sp=rsz(side_vis); tp=rsz(top_vis); hp=rsz(heatmap)

        # Info panel
        info=np.full((H,W,3),25,np.uint8)
        wkg=vol*density
        lines=[
            f"MATERIAL: {mat}","",
            "VOLUME CALCULATION",
            "-------------------",
            "Top View Calibration:",
            f"  1 px = {CONFIG['TOP_VIEW_PIXEL_AREA_CM2']} cm2 (Area)",
            f"  Cam H = {CONFIG['TOP_VIEW_CAMERA_HEIGHT_M']} m","",
            "Side View Calibration:",
            f"  1 px = {CONFIG['SIDE_VIEW_PIXEL_AREA_CM2']} cm2 (Area)",
            f"  Baseline = {bed_m:.3f} m","",
            "FINAL VOLUME:",
            f"  {vol:.3f} m3",
            f"  ({vol*1e6:.2f} cm3)","",
            f"DENSITY: {density} kg/m3",
            f"WEIGHT:  {wkg:.3f} kg",
            f"         ({wkg*1000:.1f} g)",
        ]
        y=30
        for l in lines:
            c=(0,255,100) if l.startswith("MATERIAL") else \
              (0,220,255) if l.startswith("FINAL") or l.startswith("WEIGHT") else \
              (200,200,200)
            cv2.putText(info,l,(15,y),cv2.FONT_HERSHEY_SIMPLEX,0.52,c,1,cv2.LINE_AA)
            y+=22

        def title(p,t):
            cv2.rectangle(p,(0,0),(W,28),(40,40,40),-1)
            cv2.putText(p,t,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,255),1,cv2.LINE_AA)
            return p

        sp=title(sp,f"Side View (Base H = {bed_m:.2f}m)")
        tp=title(tp,f"Material: {mat}")
        hp=title(hp,"Load Height Map (Net)")
        info=title(info,"VOLUME ESTIMATION OUTPUT")

        return np.vstack([np.hstack([sp,tp]),np.hstack([hp,info])])


class Calibration:
    def __init__(self,us):
        self.us=us
        self.bed_m=CONFIG["FALLBACK_BED_HEIGHT_M"]
        self.top_m=CONFIG["FALLBACK_TOP_BASELINE_M"]
    def run(self):
        if self.us:
            r=self.us.calibrate(samples=10)
            if r.get("top_baseline_m",0)>0:
                self.top_m=r["top_baseline_m"]
                self.bed_m=r["bed_height_m"]
                print(f"   top_baseline: {self.top_m:.4f}m  bed: {self.bed_m:.4f}m")
            else: print("   Sensor failed -> fallbacks")
        else: print("   No ultrasonic -> fallbacks")
        print("Calibration complete.\n")
    def truck_present(self):
        if self.us: return self.us.truck_is_present(CONFIG["TRUCK_PRESENCE_RATIO"])
        return True


def _capture(dev,w,h,dur,label):
    with state.lock: state.status=f"{label}: capturing {dur}s..."
    print(f"   {label} ({dev}) {dur}s ...")
    cap=cv2.VideoCapture(dev,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1); time.sleep(0.5)
    for _ in range(15): cap.read(); time.sleep(0.05)
    best,score,count=None,-1,0; te=time.time()+dur
    while time.time()<te:
        ret,f=cap.read()
        if ret and f is not None:
            s=cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
            if s>score: score=s; best=f.copy()
            count+=1; print(f"\r   {label}: {count}f score={score:.1f} {te-time.time():.1f}s",end="",flush=True)
        time.sleep(0.05)
    cap.release(); print(f"\n   {label}: done score={score:.1f}")
    return best


def main():
    print("="*58)
    print("  Truck Load Estimator - BeagleY-AI")
    print("="*58)

    us=None
    if ULTRASONIC_AVAILABLE:
        try: us=UltrasonicSystem(); print("Ultrasonic ready")
        except Exception as e: print(f"Ultrasonic failed: {e}")

    calib=Calibration(us)
    if us: input("\nPlace EMPTY TRUCK. Press ENTER to calibrate ...")
    calib.run()

    pipeline=TruckVolumePipeline()

    server=HTTPServer(("0.0.0.0",CONFIG["STREAM_PORT"]),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    ip="192.168.137.231"; port=CONFIG["STREAM_PORT"]
    print(f"\nReady! Browser -> http://{ip}:{port}\n")

    n=0
    try:
        while True:
            n+=1
            print(f"\n{'='*58}\n  TRUCK #{n}\n{'='*58}")
            with state.lock: state.status=f"Truck #{n} - waiting..."
            input("\nPosition LOADED truck. Press ENTER ...")

            print("\nPHASE 1: Top camera ...")
            top=_capture(CONFIG["TOP_CAM_DEV"],CONFIG["CAPTURE_W"],
                         CONFIG["CAPTURE_H"],CONFIG["TOP_CAM_DURATION"],"TOP CAM")
            if top is None: print("Top cam failed"); continue

            print("\nPHASE 2: Side camera ...")
            side=_capture(CONFIG["SIDE_CAM_DEV"],CONFIG["CAPTURE_W"],
                          CONFIG["CAPTURE_H"],CONFIG["SIDE_CAM_DURATION"],"SIDE CAM")
            if side is None:
                side=np.zeros((CONFIG["CAPTURE_H"],CONFIG["CAPTURE_W"],3),np.uint8)

            print("\nPHASE 3: Computing ...")
            with state.lock: state.status="Computing..."

            side_vis,bed_m=pipeline.analyze_side(side)
            top_vis,heatmap,vol,mat=pipeline.calc_volume(top,bed_m)
            density=pipeline._density(mat)
            wkg=vol*density

            out=pipeline.build_panel(side_vis,top_vis,heatmap,vol,mat,bed_m,density)

            result={
                "material":mat,
                "volume_m3":round(vol,5),
                "volume_cm3":round(vol*1e6,2),
                "weight_kg":round(wkg,3),
                "weight_g":round(wkg*1000,1),
                "bed_h_m":round(bed_m,4),
                "density_kgm3":density,
                "timestamp":time.strftime("%H:%M:%S"),
                "status":f"Truck #{n} result ready",
            }

            with state.lock:
                state.frame_out=_enc(out)
                state.results=result
                state.status=f"Truck #{n} result ready"

            print(f"\n{'='*58}")
            print(f"  RESULT #{n}: {mat}")
            print(f"  Volume : {vol:.4f} m3  ({vol*1e6:.2f} cm3)")
            print(f"  Weight : {wkg:.3f} kg  ({wkg*1000:.1f} g)")
            print(f"  Bed H  : {bed_m:.4f} m")
            print(f"  View   : http://{ip}:{port}")
            print(f"{'='*58}")

            if input("\nAnother truck? (y/n): ").strip().lower()!='y':
                break
    except KeyboardInterrupt: print("\nStopped.")
    finally: server.shutdown()

if __name__=="__main__": main()
