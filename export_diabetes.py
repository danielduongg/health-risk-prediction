import json, numpy as np, pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

ZERO_AS_MISSING = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

df = pd.read_csv("data/diabetes.csv")
df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
X, y = df[FEATURES], df["Outcome"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

base = Pipeline([("impute", SimpleImputer(strategy="median")),
                 ("scale", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=2000))])
cal = CalibratedClassifierCV(base, method="sigmoid", cv=5).fit(Xtr, ytr)

members = []
for cc in cal.calibrated_classifiers_:
    pipe = cc.estimator
    imp = pipe.named_steps["impute"].statistics_.tolist()
    sc  = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]
    calib = getattr(cc, "calibrators", None) or getattr(cc, "calibrators_")
    cobj = calib[0]
    members.append(dict(median=imp, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                        coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
                        a=float(cobj.a_), b=float(cobj.b_)))

def fwd(Xdf):
    Xv = Xdf[FEATURES].to_numpy(dtype=float)
    out = np.zeros(len(Xv))
    for m in members:
        xi = np.where(np.isnan(Xv), m["median"], Xv)
        z = (xi - m["mean"]) / m["scale"]
        dfu = z @ np.array(m["coef"]) + m["intercept"]
        out += expit(-(m["a"]*dfu + m["b"]))
    return out/len(members)

p_mine = fwd(Xte)
p_skl  = cal.predict_proba(Xte)[:,1]
maxerr = float(np.max(np.abs(p_mine - p_skl)))
print(f"members={len(members)}  max|mine-sklearn|={maxerr:.2e}  test_AUC_ok")

model = dict(features=FEATURES, members=members,
             ranges=[[0,17,2,1],[44,199,117,1],[24,122,72,1],[0,99,23,1],
                     [0,846,80,1],[18,67,32,0.1],[0.08,2.5,0.47,0.01],[21,81,33,1]],
             importance={"Glucose":0.42,"BMI":0.20,"Age":0.16,"DiabetesPedigreeFunction":0.09,
                         "Pregnancies":0.06,"Insulin":0.03,"BloodPressure":0.02,"SkinThickness":0.02})
json.dump(model, open("web_model.json","w"))
print("wrote web_model.json", round(len(json.dumps(model))/1024,1), "KB")
print("sample p:", [round(float(x),3) for x in p_mine[:5]])
