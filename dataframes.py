import re
import numpy as np
import pandas as pd
         

###
def check_df(df: pd.DataFrame):
    '''
    Ausgabe Anzahl Zeilen x Spalten und Größeninfos.
    '''
    try:
        nrows,ncols = df.shape
    except ValueError:
        nrows,ncols = df.index.size,1
    size  = df.__sizeof__()
    print(f'>>> {nrows:,} x {ncols:,} = {size/1024**2:2.1f} MB ({size/max(nrows,1):2.0f} Byte/Zeile)')
    
###
def check_cols_generic(df: pd.DataFrame):
    '''
    Generische Statistik über ein Pandas DataFrame.
    '''
    # Grundlegende Checks
    if type(df) is not pd.DataFrame: return 'kein pandas.Dataframe'
    if df.index.size==0: return 'Leeres Dataframe'
    #
    nrows,ncols = df.shape
    Bytes = df.__sizeof__()
    sorted_index = df.index.equals(df.index.sort_values())
    print('-'*99)
    print(f'>> N_rows    = {nrows:,d}')
    print(f'>> N_cols    = {ncols:,d}')    
    print(f'>> Size(MB)  = {Bytes/1024**2:2.1f}')
    print(f'>> Bytes/row = {Bytes/nrows:2.1f}')
    dupl = df.duplicated().sum() if nrows<500e6 else -1
    print(f'>> N_Duplikate = {dupl:,d}') 
    print(f'>> N_benachbarte_Duplikate = {(df.shift(-1)==df).all(axis=1).sum():,d}')
    print(f'>> Index ist sortiert: {sorted_index}')
    print('-'*99)
    # Ein paar Metriken
    mode_0  = df.mode().loc[0] # Häufigste Ausprägung
    n_uniq  = df.nunique(dropna=True)
    n_notna = df.notna().sum()
    multip  = n_notna/n_uniq # Multiplizität: durchschn. Häufigkeit
    n_isna  = df.isna().sum()
    p_isna  = df.isna().mean()
    mem_row = df.memory_usage(index=False,deep=True)/nrows  # Bytes pro Spalte
    Dtypes  = df.dtypes
    # Ergebnistabelle
    tmp = pd.DataFrame({'dtype'    : Dtypes,
                        'Mode'     : mode_0,
                        'N_unique' : n_uniq,
                        'Multi'    : multip,
                        'N_isna'   : n_isna,
                        'P_isna'   : p_isna,
                        'Bytes/row': mem_row,
                       })
    #
    return tmp.astype({'N_unique' : int,
                       'Multi'    : float,
                       'N_isna'   : int,
                       'P_isna'   : float,
                       'Bytes/row': float,
                      }).round(2)


###
def check_cols_cats(df: pd.DataFrame,**kwargs):
    '''
    Spalteninformationen wenn Spalten als "category" definert.
    '''
    ### Ausgabetabelle
    tab = pd.DataFrame(columns=['Modus',
                                'N_unique',
                                'U_Alpha',
                                'U_Numeric',
                                'U_NotAlnum',
                                'U_HasWs',
                                'spec_char',
                                'min_len',
                                'max_len',
                                'U_length(1-10)'])
    ### Loop über Spalten
    print(df.columns.size,end=':')
    for i,col in enumerate(df.columns):
        #
        print(f'{i+1}',end=',')
        #
        if df[col].isna().all(): 
            tab.loc[col] = np.nan
        #
        else:
            cats = df[col].cat.categories
            #
            tab.loc[col,'Modus']      = df[col].mode(dropna=True)[0]
            tab.loc[col,'N_unique']   = cats.size
            tab.loc[col,'U_Alpha']    = int(cats.map(str.isalpha).values.sum())
            tab.loc[col,'U_Numeric']  = int(cats.map(str.isnumeric).values.sum())
            tab.loc[col,'U_NotAlnum'] = int((cats.map(str.isalnum)==False).sum())
            tab.loc[col,'U_HasWs']    = int(cats.map(lambda x: x.count(' ')>0).values.sum())
            tab.loc[col,'min_len']    = int(cats.map(len).min())
            tab.loc[col,'max_len']    = int(cats.map(len).max())
            chars = ''.join(set(''.join(cats)))
            tab.loc[col,'spec_char']      = re.findall('[^a-zA-Z0-9\ ]',chars)[0:10]
            tab.loc[col,'U_length(1-10)'] = np.histogram(cats.map(len),bins=np.r_[1:11:1])[0]
    return tab