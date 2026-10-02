import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
g = pd.read_csv('grouped_data.csv'); g['publishedAt'] = pd.to_datetime(g['publishedAt']); g['trending_date'] = pd.to_datetime(g['trending_date'])
d = pd.DataFrame(index=g.index); d['lv'] = np.log1p(g.view_count); d['lc'] = np.log1p(g.comment_count); d['ld'] = np.log1p(g.dislikes)
d['c_v'] = g.comment_count/(g.view_count+1); d['d_v'] = g.dislikes/(g.view_count+1); d['d_c'] = g.dislikes/(g.comment_count+1)
d['days'] = (g.trending_date-g.publishedAt).dt.days; d['pyear'] = g.p_year; d['hour'] = g.publishedAt.dt.hour; d['dow'] = g.publishedAt.dt.dayofweek
d['cd'] = g.comments_disabled.astype(int); d['rd'] = g.ratings_disabled.astype(int); d['cat'] = g.categoryId; d['dis_zero'] = (g.dislikes==0).astype(int)
d['ch_n'] = g.groupby('channelId').video_id.transform('count'); d['ch_lv'] = g.groupby('channelId').view_count.transform(lambda s: np.log1p(s).mean())
y = g.likes; v = g.view_count.values; r_all = np.log((y.values+1)/(v+1)); ch = g.channelId.values
Xtr, Xte, ytr, yte = train_test_split(d, y, test_size=0.25, random_state=3)
tr, te = Xtr.index.values, Xte.index.values; gm = r_all[tr].mean()
def enc(fi, ai): s = pd.DataFrame({'c': ch[fi], 'r': r_all[fi]}).groupby('c').r.agg(['sum','count']); a = pd.Series(ch[ai]); return ((a.map(s['sum']).fillna(0).values+gm*5)/(a.map(s['count']).fillna(0).values+5), a.map(s['count']).fillna(0).values)
oof = np.zeros(len(tr)); oofc = np.zeros(len(tr)); folds = list(KFold(5, shuffle=True, random_state=3).split(tr))
_ = [(oof.__setitem__(b, enc(tr[a], tr[b])[0]), oofc.__setitem__(b, enc(tr[a], tr[b])[1])) for a, b in folds]
te_e, te_c = enc(tr, te); Xtr2 = Xtr.assign(ch_rate=oof, ch_tr_n=oofc); Xte2 = Xte.assign(ch_rate=te_e, ch_tr_n=te_c)
models = {'Random Forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=3), 'LightGBM': LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=3), 'XGBoost': XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=3)}
preds = {n: (v[te]+1)*np.exp(m.fit(Xtr2, r_all[tr]).predict(Xte2))-1 for n, m in models.items()}
preds['Average of 3'] = np.mean(list(preds.values()), axis=0)
out = pd.DataFrame({n: [r2_score(yte, p), r2_score(np.log1p(yte), np.log1p(np.clip(p, 0, None))), mean_absolute_error(yte, p)] for n, p in preds.items()}, index=['R2 on raw likes','R2 on log1p(likes)','MAE raw']).T
out.round(4)
