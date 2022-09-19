import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
       
###
def mem_lookup(List,n:int=10):
    '''
    Liste der <n> speicherintensivsten Nutzervariablen.
    
    -----
    Input
    -----
    List: list,
          Lister der Variablen, z.B. List=%who_ls
    n: integer,
       Länge der Top Liste

    ------
    Output
    ------
    pandas.Series: Top Liste der Nutzervariablen mit Speicherbedarf in MB.
    
    --------
    Beispiel
    --------
    ### import in den globalen Namespace notwendig
    from gruenberg.system import mem_lookup
    mem_lookup(list(locals().keys()))
    '''
    res = pd.Series(dtype=float)
    i=0
    for x in List:
        try:
            res.loc[x] = eval(f'{x}.__sizeof__()')/1024**2
        except:
            i+=1
    #
    print(f'>> Unbeachtet: {i:,}')
    return res.sort_values(ascending=False)