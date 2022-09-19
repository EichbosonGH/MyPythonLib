import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
    
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
def knee_point(y):
    ''' 
    Finde Knee point
    '''
    x=np.r_[0:1:y.size*1j]
    xy=np.r_[[x,y]].T
    XY = sklearn.preprocessing.MinMaxScaler().fit_transform(xy)
    return np.sum(XY,axis=1).argmin()

###
def toy(List: list=[],n: int=10):
    for x in List:
        print(x)   