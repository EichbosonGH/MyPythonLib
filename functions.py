import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

###
def check_df(df: pd.DataFrame):
    '''
	Ausgabe Anzahl Zeilen x Spalten und Größeninfos
	'''  
    try:
        nrows,ncols = df.shape
    except ValueError:
        nrows,ncols = df.index.size,1
    size  = df.__sizeof__()
    print(f'>>> {nrows:,} x {ncols:,} = {size/1024**2:2.1f} MB ({size/max(nrows,1):2.0f} Byte/Zeile)')

### 
def check_ar(array):
    nrows = array.shape[0]
    try:
        ncols = array.shape[1]
    except IndexError:
        ncols = 0
    size  = array.nbytes/1024**2
    print(f'>>> {nrows:,} x {ncols:,} = {size:2.1f} MB ({size*1024**2/(nrows+1e-8):2.0f} Byte/Zeile)')
    
###
def metrics(y_true,y_pred,Norm=1,Print=True,Return=False):
    #
    N      = y_true.size    
    ME     = (y_true-y_pred).mean()/Norm
    MAE    = sklearn.metrics.mean_absolute_error(y_true,y_pred)/Norm
    MSE    = sklearn.metrics.mean_squared_error(y_true,y_pred)/Norm
    RMSE   = np.sqrt(MSE)
    R2     = sklearn.metrics.r2_score(y_true,y_pred)
    #
    if Print:
        print(f'>>> N = {N:,} ; ME = {ME:2.3%} ; MAE = {MAE:2.3%} ; RMSE = {RMSE:2.3%} ; R2 = {R2:2.3%}')
    #
    if Return:
        return {'N':N,'ME':ME,'MAE':MAE,'RMSE':RMSE,'R2':R2}
    else:
        pass
    
###
def feat_imp(Predictor,Names=None):
    #
    try:
        importance = Predictor.feature_importances_
    except:
        print('>>> keine "feature_importances_" vorhanden!')
        return
    try:
        names = Predictor.feature_names_
    except:
        names = Names
    #
    res = pd.DataFrame(data={'importance':importance},index=names)
    res['importance'] = res.importance/res.importance.sum()
    res.sort_values('importance',inplace=True,ascending=False)
    #
    return res
    
###
def mem_lookup(List: list=[],n: int=10):
    mem = dict()
    for x in List:
        try: 
            mem[x] = eval(x+'.__sizeof__()/1024**2')            
            print(x,mem[x])
        except:
            pass        
    #
    mem = pd.Series(mem,dtype=float,name='Size (MB)').sort_values(ascending=False).to_frame()
    return mem
    #return mem.head(n)

###
def toy(List: list=[],n: int=10):
    for x in List:
        print(x)    