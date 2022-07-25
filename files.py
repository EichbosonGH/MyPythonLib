import pandas as pd

###
def check_file_format(path: str,nrows: int=5,sep: str=',',encoding: str='utf-8'):
    '''
    Ausgabe der ersten <nrows> Zeilen einer Datei mit Separator <sep>.
    '''
    #
    with open(path,mode='r',encoding=encoding) as file:
        for i in range(5):
            x = file.readline()
            z = x.split(sep=sep)
            print(f'{i}: {len(z)} Felder')
            print(x)

            
###
def check_csv_tables(files: list,**kwargs):
    '''
    Funktion um allgemeine Eigenschaften einer Liste von 
    CSV Dateien mit Tabelleninhalt zu sammeln.
    '''
    # Standardparameter zum Laden der CSV Dateien
    encoding = kwargs.get('encoding','latin9')
    dtype    = kwargs.get('dtype','category')
    nrows    = kwargs.get('nrows',None)
    decimal  = kwargs.get('decimal',',')
    quoting  = kwargs.get('quoting',0)    
    sep      = kwargs.get('sep',';')
    # Ergebnistabelle
    tabs = pd.DataFrame()
    # Anzahl der Tabellen
    print(f'{len(files)} Files: ',end='')
    # Iteration
    for i,file in enumerate(files):        
        #
        print(f'{i}',end=',')
        # 
        try: 
            df = pd.read_csv(file,
                             sep=sep,
                             encoding=encoding,
                             dtype=dtype,
                             nrows=nrows,
                             decimal=decimal,
                             low_memory=False, # lädt ggf. schneller, alles Daten sind konsistent ein Datentyp
                             quoting=quoting)
        except Exception as ex: 
            print(f"\n>> Exception in {file} : {ex}")
            continue
        #
        if df.index.size==0:
            continue
        #
        sizeof = df.__sizeof__()
        nrows  = df.index.size
        ncols  = df.columns.size
        tabs.loc[i,'file']      = file.split('/')[-1]
        tabs.loc[i,'rows']      = nrows
        tabs.loc[i,'cols']      = ncols
        tabs.loc[i,'size(MB)']  = sizeof/1024**2
        tabs.loc[i,'Bytes/row'] = sizeof/nrows
        tabs.loc[i,'n_leer']    = df.isna().sum().sum()
        tabs.loc[i,'p_leer']    = df.isna().sum().sum()/df.size
        tabs.loc[i,'n_dupl']    = df.duplicated().sum() if nrows<500e6 else -1
        tabs.loc[i,'p_dupl']    = tabs.loc[i,'n_dupl']/df.index.size
        #
        del df
        #
    return tabs.astype({'rows':int,
                        'cols':int,
                        'size(MB)':int,
                        'Bytes/row':int,
                        'n_leer':int,
                        'n_dupl':int,
                       })