import json
M = json.load(open("web_model.json"))
HTML = r'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diabetes Risk Estimator — live demo</title>
<style>
 :root{--bg:#0b0f17;--panel:#121826;--panel2:#0e1420;--line:#1f2a3c;--ink:#e8eef9;--mut:#8aa0bf;
   --teal:#2dd4bf;--accent:#38bdf8;--warn:#ffb454;--lo:#27d08a;--mid:#ffb454;--hi:#ff5d6c;}
 *{box-sizing:border-box}
 body{margin:0;background:radial-gradient(1100px 650px at 75% -10%,#0e2a30 0%,var(--bg) 55%);color:var(--ink);
   font:16px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;-webkit-font-smoothing:antialiased}
 .wrap{max-width:900px;margin:0 auto;padding:36px 20px 80px}
 .eyebrow{letter-spacing:.16em;text-transform:uppercase;font-size:12px;color:var(--teal);font-weight:700}
 h1{font-size:29px;margin:.25em 0 .15em;line-height:1.15}
 .sub{color:var(--mut);max-width:64ch}
 .disc{margin-top:14px;border-left:3px solid var(--warn);background:#1a160c;padding:10px 14px;border-radius:0 10px 10px 0;color:#ecd9b0;font-size:13.5px}
 .grid{display:grid;grid-template-columns:1.25fr .9fr;gap:18px;margin-top:22px}
 @media(max-width:760px){.grid{grid-template-columns:1fr}}
 .card{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:16px;padding:20px 22px;box-shadow:0 20px 50px -30px #000}
 .row{margin:13px 0}
 .row .lab{display:flex;justify-content:space-between;font-size:13.5px;color:var(--mut);margin-bottom:5px}
 .row .lab b{color:var(--ink);font-variant-numeric:tabular-nums;font-size:15px}
 input[type=range]{width:100%;accent-color:var(--teal);height:5px}
 .presets{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px}
 .pz{font-size:12.5px;border:1px solid var(--line);background:#0c1422;color:var(--mut);padding:6px 11px;border-radius:999px;cursor:pointer}
 .pz:hover{border-color:var(--teal);color:var(--ink)}
 .gauge{text-align:center;padding:6px 0 2px}
 .big{font-size:54px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
 .band{font-weight:700;letter-spacing:.04em;margin-top:8px;padding:6px 12px;border-radius:999px;display:inline-block;font-size:13.5px}
 .meter{height:12px;border-radius:999px;background:#0a0f1a;border:1px solid var(--line);overflow:hidden;margin:16px 0 6px}
 .fill{height:100%;transition:width .25s,background .25s}
 .scale{display:flex;justify-content:space-between;font-size:11.5px;color:var(--mut)}
 h2{font-size:15px;margin:18px 0 8px;color:var(--mut);font-weight:600;letter-spacing:.02em;text-transform:uppercase}
 .imp{display:flex;flex-direction:column;gap:7px}
 .ib{display:grid;grid-template-columns:130px 1fr;align-items:center;gap:9px;font-size:13px;color:var(--mut)}
 .ib .t{height:14px;background:#0a0f1a;border:1px solid var(--line);border-radius:5px;overflow:hidden}
 .ib .f{height:100%;background:linear-gradient(90deg,#2dd4bf,#38bdf8)}
 .foot{margin-top:24px;color:var(--mut);font-size:13px;text-align:center}
 a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}
</style></head>
<body><div class="wrap">
 <div class="eyebrow">health-risk-prediction · live demo</div>
 <h1>Diabetes risk estimator</h1>
 <p class="sub">The same <b>calibrated logistic-regression</b> model from the repo (held-out ROC-AUC 0.81, Brier 0.165), running entirely in your browser. Move the sliders; the probability is calibrated, so 30% really means ~30%.</p>
 <div class="disc">⚠️ <b>Educational project on synthetic, Pima-style data — not a medical device and not medical advice.</b> Do not use for real clinical decisions.</div>

 <div class="grid">
  <div class="card">
   <div class="presets" id="presets"></div>
   <div id="sliders"></div>
  </div>
  <div class="card">
   <div class="gauge">
     <div class="big" id="pct">—</div>
     <div class="band" id="band">—</div>
   </div>
   <div class="meter"><div class="fill" id="fill"></div></div>
   <div class="scale"><span>0%</span><span>50%</span><span>100%</span></div>
   <h2>What drives the model</h2>
   <div class="imp" id="imp"></div>
  </div>
 </div>

 <div class="foot">Calibrated logistic regression (5-fold Platt scaling) · features match the Pima Indians dataset · <a href="https://github.com/danielduongg/health-risk-prediction" target="_blank">source, model card &amp; fairness audit →</a></div>
</div>
<script>
const M = __MODEL__;
const F = M.features, R = M.ranges;
const LABELS = ["Pregnancies","Glucose (mg/dL)","Blood pressure (mm Hg)","Skin thickness (mm)","Insulin (μU/mL)","BMI","Diabetes pedigree","Age"];
const PRESETS = {
  "Average adult":      [2,117,72,23,80,32,0.47,33],
  "Higher-risk":        [6,168,78,35,180,41,0.9,52],
  "Lower-risk":         [1,92,66,20,60,24,0.2,26],
  "Lean & young":       [0,95,64,18,55,22,0.2,24]
};
let vals = PRESETS["Average adult"].slice();

function risk(x){
  let out=0;
  for(const m of M.members){
    let s=m.intercept;
    for(let i=0;i<F.length;i++){
      const xi = (x[i]===''||isNaN(x[i]))? m.median[i] : x[i];
      s += ((xi - m.mean[i])/m.scale[i]) * m.coef[i];
    }
    out += 1/(1+Math.exp(m.a*s + m.b));
  }
  return out/M.members.length;
}
function bandFor(p){
  if(p<0.20) return ["LOW RISK","var(--lo)","rgba(39,208,138,.15)"];
  if(p<0.50) return ["MODERATE RISK","var(--mid)","rgba(255,180,84,.15)"];
  return ["HIGHER RISK","var(--hi)","rgba(255,93,108,.15)"];
}
function colFor(p){ return p<0.20?"#27d08a":p<0.5?"#ffb454":"#ff5d6c"; }

function render(){
  const p = risk(vals);
  document.getElementById('pct').textContent=(p*100).toFixed(0)+'%';
  document.getElementById('pct').style.color=colFor(p);
  const [t,c,bg]=bandFor(p);
  const b=document.getElementById('band'); b.textContent=t; b.style.color=c; b.style.background=bg;
  const fl=document.getElementById('fill'); fl.style.width=(p*100).toFixed(1)+'%'; fl.style.background=colFor(p);
}
const sl=document.getElementById('sliders');
F.forEach((f,i)=>{
  const [mn,mx,df,st]=R[i];
  const row=document.createElement('div'); row.className='row';
  row.innerHTML=`<div class="lab"><span>${LABELS[i]}</span><b id="v${i}"></b></div>
    <input type="range" id="s${i}" min="${mn}" max="${mx}" step="${st}" value="${vals[i]}">`;
  sl.appendChild(row);
  const inp=row.querySelector('input');
  inp.addEventListener('input',()=>{vals[i]=parseFloat(inp.value);document.getElementById('v'+i).textContent=fmt(vals[i],st);render();});
});
function fmt(v,st){return st<1? v.toFixed(st<0.05?2:1): Math.round(v);}
function setVals(arr){vals=arr.slice();F.forEach((f,i)=>{const s=document.getElementById('s'+i);s.value=vals[i];document.getElementById('v'+i).textContent=fmt(vals[i],R[i][3]);});render();}

const pc=document.getElementById('presets');
Object.keys(PRESETS).forEach(k=>{const c=document.createElement('span');c.className='pz';c.textContent=k;c.onclick=()=>setVals(PRESETS[k]);pc.appendChild(c);});

const imp=document.getElementById('imp');
const entries=Object.entries(M.importance).sort((a,b)=>b[1]-a[1]);const mx=entries[0][1];
imp.innerHTML=entries.map(([k,v])=>`<div class="ib"><div>${k}</div><div class="t"><div class="f" style="width:${(v/mx*100).toFixed(0)}%"></div></div></div>`).join('');

setVals(vals);
</script>
</body></html>'''
HTML = HTML.replace("__MODEL__", json.dumps(M))
open("index.html","w").write(HTML)
print("wrote index.html", round(len(HTML)/1024,1), "KB")
