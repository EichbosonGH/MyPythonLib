import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
import matplotlib.pyplot as plt

###
def Gini_plot(X,Y=None,dq=0.1,Return=False,Draw=True):
    ''' 
    Gini Plot mit Quantilen zur Berechnung.
			dq = None : Diskreter Gini Plot ohne Einteilung in Quantile
    '''
    if Y is None: Y = X
    # Dataframe erstellen    
    tmp = pd.DataFrame({'x':np.array(X),'y':np.array(Y)})
    # Percentilranking nach x
    tmp['Rank'] = tmp.x.rank(ascending=True,method='first',pct=True)
    tmp.sort_values('Rank',ascending=True,inplace=True,ignore_index=True)
    # Quantile erstellen
    if dq==None:
        dq = 1/tmp.x.size + 1e-14 # +1e-14 für Ungenauigkeit bei float64: resolution = 1e-15
    qs  = np.r_[0:1+dq/2:dq]
    # normierte Summe und kumulierte Summe
    ys = [tmp[tmp.Rank.between(qs[i],qs[i]+dq,inclusive='right')].y.sum()/tmp.y.sum() for i in range(qs.size-1)]
    ys = np.array(ys)
    ycs = np.r_[0,ys].cumsum()
    #
    AUC   = sklearn.metrics.auc(qs,ycs)
    AUC_0 = sklearn.metrics.auc(qs,qs)
    GUK = (AUC-AUC_0)/AUC_0
		# >> Der GUK ist die signierte relative Abweichung des gesehenen AUC mit der Erwartung AUC_0
		# >> AUC_0 = das Integral über die Identische Funktion im Def.Bereich
		# >> GUK = (AUC-AUC_0)/AUC_0 = {0: Gleichverteilung, >0: AUC > AUC_0 im DB }
    #
    if Draw:
        plt.plot([qs.min(),qs.max()],[0,1],'k--',lw=0.5,label='Gleichverteilung kumuliert')
        plt.plot(qs,ycs,'C1.--',lw=0.5,ms=5.0,label=f'Gini-Kurve: GUK={GUK:2.1%}')
        #
        plt.stairs(ys,qs,fill=True,alpha=0.5,ec='C1',color='C1',label='Quantilanteil')
        plt.stairs([dq],[qs.min(),qs.max()],ec='k',lw=0.5,ls='-',label='Gleichverteilung')
        plt.legend()
        plt.xlabel('Quantil nach X')
    #        
    if Return:
        return {'dq':dq,'qs':qs,'x':x,'y':y,'ycs':ycs,'AUC':AUC,'GUK':GUK}
    else:
        pass

###
def rang_plot(ds,topn=10,dropna=False):
    # Zählstatistik
    tmp = ds.value_counts(dropna=dropna).rename('Nevt').to_frame()
    # rel. Häufigkeit und Rang
    tmp['Frac'] = tmp.Nevt/tmp.Nevt.sum()
    tmp['Rang'] = tmp.Nevt.rank(ascending=False,method='first',pct=False).astype('int32')
    # Relative Entropy berechnen
    rel_entr = scipy.stats.entropy(tmp.Frac)/(scipy.stats.entropy(np.ones_like(tmp.Frac))+1e-8)
    # Plot der Rangkurve
    plt.plot(tmp.Rang,tmp.Frac,'k.--',ms=5,lw=0.5,label=f'Rang plot (U={tmp.index.size:,d})')
    # Top N
    for x in tmp.index[:topn]:
        r = tmp.loc[x].Rang
        f = tmp.loc[x].Frac
        plt.plot(r,f,'.',label=f'{r:1.0f}. "{x}" ({f:2.1%})')
    # Log Skala falls N>500
    if tmp.index.size>500:
        plt.xscale('log')
    # Dekoration
    plt.legend(fontsize=7)
    plt.title(f'{ds.name} (rel. Entr. = {rel_entr:2.1%})')
    plt.xlabel('Rang')
    plt.ylabel('Anteil')
    #
    return None

###
def draw_silhouette(DistM,Label,gap:int=1):
    '''
    Zeichne Silhouette Plot (https://de.wikipedia.org/wiki/Silhouettenkoeffizient)
    
    Input
    -----
    DistM: array-like mit shape = (n_samples,n_samples),
           Symmetrische Distanzmatrix
           
    Label: array-like mit Länge = n_samples
           Array der Labels je Datenpunkte.
    
    gap:   int,
           Lücke zwischen Clustern in Anzahl Schrittweite.
    
    Output
    ------
    (None)    
    '''
    sil_val = sklearn.metrics.silhouette_samples(DistM,Label,metric='precomputed')
    tmp = pd.DataFrame({'label':Label,'score':sil_val})
    tmp.sort_values(['label','score'],ascending=[True,False],inplace=True,ignore_index=True)
    # select all but noise
    tmp = tmp[tmp.label>-1].reset_index(drop=True)
    # Loop über labels
    _x = tmp.index.values
    for i in tmp.label.unique():
        sel = (tmp.label==i)
        color = plt.cm.turbo(i/tmp.label.nunique())
        plt.fill_between(x=_x-(1-gap)*i,y1=0,y2=tmp.score,where=sel,facecolor=color,lw=0.0)
        # plt.bar(x=_x[sel]+gap*i,height=tmp[sel].score,bottom=0,width=1,color=color) # Barplot für Testzwecke
    # Dekoration
    title = f'Silhouette: Min={tmp.score.min():2.1f}, Median = {tmp.score.median():2.2f}, Max = {tmp.score.max():2.2f}'
    plt.title(title,fontsize=10)
    plt.tick_params(bottom=True,labelbottom=False)
    plt.ylabel('Silhouette Score')
    # 
    return