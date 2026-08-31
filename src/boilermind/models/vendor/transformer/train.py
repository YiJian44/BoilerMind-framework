"""
Train Transformer (original paper implementation) for steam mass flow prediction.
Mass flow M is the direct target; volume flow V is computed from M + true P,T.
Uses the working code from 第一篇代码/对照组/transformer对照决定版/
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from transformer import transformer as Transformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_PAST, OFFSET = 20, 40
D_MODEL, D_FF, NUM_LAYERS, NUM_HEADS = 64, 48, 6, 12
DROPOUT, TOP_K, BATCH_SIZE = 0.5, 5, 512
LR, EPOCHS, PATIENCE = 0.005, 1000, 10
SEED = 42
DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'shortperiod_new.csv')
EXCEL = os.path.join(os.path.dirname(__file__), '..', 'data', 'boiler_181var.xlsx')
SAVE = os.path.join(os.path.dirname(__file__), 'transformer_weights.pth')
R = 0.461526  # steam gas constant kJ/(kg*K)

def compute_vol(mass_flow_ts, pressure_MPa, temp_C):
    """M (t/h) -> V (m3/s) via ideal gas law."""
    m_kg_s = mass_flow_ts * 1000 / 3600
    return m_kg_s * R * (temp_C + 273.15) / (pressure_MPa * 1000)

def train():
    torch.manual_seed(SEED); np.random.seed(SEED)
    raw = pd.read_csv(DATA, header=None).values.astype(np.float64)
    X, Y = raw[:, :30], raw[:, 30:31]
    # Load P and T from original Excel (same row order as shortperiod_new.csv)
    df = pd.read_excel(EXCEL)
    P_all = df.iloc[:, 3 + 0].values.astype(np.float64)   # Var1 = pressure MPa
    T_all = df.iloc[:, 3 + 8].values.astype(np.float64)   # Var9 = temperature C

    scaler = MinMaxScaler()
    # Fit scaler on training portion only (first 70% of raw rows, before windowing)
    train_rows = int(len(X) * 0.7)
    scaler.fit(np.hstack([X[:train_rows], Y[:train_rows]]))
    ds = scaler.transform(np.hstack([X, Y]))

    xs, ys, idxs = [], [], []
    for i in range(N_PAST, len(ds) - OFFSET + 1):
        xs.append(ds[i-N_PAST:i, :30]); ys.append(ds[i+OFFSET-1, 30])
        idxs.append(i + OFFSET - 1)
    wX, wY = np.array(xs), np.array(ys)
    wIdx = np.array(idxs)
    te, ve = int(len(wX)*0.7), int(len(wX)*0.8)
    teIdx = wIdx[ve:]  # row indices for test target positions

    tx=torch.FloatTensor(wX[:te]).to(DEVICE); ty=torch.FloatTensor(wY[:te]).unsqueeze(-1).unsqueeze(-1).to(DEVICE)
    vx=torch.FloatTensor(wX[te:ve]).to(DEVICE); vy=torch.FloatTensor(wY[te:ve]).unsqueeze(-1).unsqueeze(-1).to(DEVICE)
    ex=torch.FloatTensor(wX[ve:]).to(DEVICE); ey=torch.FloatTensor(wY[ve:]).unsqueeze(-1).unsqueeze(-1).to(DEVICE)
    print(f'Train:{tx.shape} Val:{vx.shape} Test:{ex.shape}')
    ld=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),batch_size=BATCH_SIZE,shuffle=False)

    model=Transformer(N_PAST, 1, D_MODEL, D_FF, NUM_HEADS, NUM_LAYERS, DROPOUT, TOP_K).to(DEVICE)
    crit=nn.MSELoss();opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=5,gamma=0.5)

    class ES:
        def __init__(s,p=7):s.c=0;s.b=float('inf');s.p=p;s.es=False
        def __call__(s,vl,m):
            if vl<s.b:s.b=vl;s.c=0;torch.save(m.state_dict(),SAVE)
            else:s.c+=1
            if s.c>=s.p:s.es=True
    es=ES(PATIENCE)

    print(f'Training Transformer (n_past={N_PAST}, lr={LR})...')
    for ep in range(EPOCHS):
        model.train();ls=[]
        for bx,by in ld:
            out=model(bx,by);loss=10*crit(out,by)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();ls.append(loss.item())
        model.eval()
        with torch.no_grad():
            z=torch.zeros_like(vy);vl=(10*crit(model(vx,z),vy)).item()
        sch.step();es(vl,model)
        if ep%10==0:print(f'  Ep{ep:4d}|Loss{sum(ls)/len(ls):.4f}|Val{vl:.4f}')
        if es.es:print(f'  Early stop@{ep}');break

    model.load_state_dict(torch.load(SAVE));model.eval()
    with torch.no_grad():
        o=torch.ones_like(ey);pp=model(ex,o).cpu().numpy().squeeze();tt=ey.cpu().numpy().squeeze()

    ymin,ymax=scaler.data_min_[30],scaler.data_max_[30]
    pr,tr=pp*(ymax-ymin)+ymin,tt*(ymax-ymin)+ymin  # denormalized mass flow t/h

    # Mass flow metrics
    mse=mean_squared_error(tr,pr);rmse=np.sqrt(mse);mae=mean_absolute_error(tr,pr);r2=r2_score(tr,pr)
    mape=np.mean(np.abs((tr-pr)/(tr+1e-8)))*100

    # Volume flow metrics (M_pred + true P,T — aligned with HGB/MTGNN)
    P_test, T_test = P_all[teIdx], T_all[teIdx]
    V_pred = compute_vol(pr, P_test, T_test)
    V_true = compute_vol(tr, P_test, T_test)
    v_rmse = np.sqrt(((V_pred - V_true)**2).mean())
    v_mae = np.abs(V_pred - V_true).mean()
    v_range = V_true.max() - V_true.min()

    print(f'\nTransformer: M  MSE={mse:.1f} RMSE={rmse:.1f} MAE={mae:.1f} MAPE={mape:.2f}% R2={r2:.4f}')
    print(f'Transformer: V  RMSE={v_rmse:.4f} MAE={v_mae:.4f} m3/s (V range={v_range:.2f}, MAE/Vrange={v_mae/v_range*100:.1f}%)')
    torch.save({'model_state':model.state_dict(),'scaler':scaler,
                'metrics':{'M_mse':mse,'M_rmse':rmse,'M_mae':mae,'M_mape':mape,'M_r2':r2,
                           'V_rmse':v_rmse,'V_mae':v_mae,'V_range':v_range}}, SAVE)
    return model, scaler

if __name__=='__main__': train()
