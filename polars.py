def cols_check(df: pl.DataFrame):
    '''
    Generische Statistik über ein Polars DataFrame.
    Sollte für alle Dataframes funktionieren, unabhängig vom Datentyp.
    '''
    # Grundlegende Checks
    if not isinstance(df,pl.DataFrame): 
        return 'kein polars.Dataframe'
    if df.is_empty(): 
        return 'Leeres Dataframe'
    #
    small_check(df)
    #
    tmp = {       
        "Column" :[col.name for col in df.iter_columns()], 
        "Dtype"  :[str(col.dtype.base_type()) for col in df.iter_columns()],
        "N_null" :[col.null_count() for col in df.iter_columns()],
        "Kard"   :[col.n_unique() if not col.dtype.is_numeric() else None for col in df.iter_columns()],        
        # absolute Entropie, je größer desto ausgeglichener die Besetzung
        "Entr"   :[col.value_counts()["count"].entropy(base=2,normalize=True) 
                   if not col.dtype.is_numeric() 
                   else None for col in df.iter_columns()],
        "Mode"   :[col.mode()[0] if not col.dtype.is_numeric() else None for col in df.iter_columns()],
        "Min"    :[col.min() for col in df.iter_columns()],
        "Max"    :[col.max() for col in df.iter_columns()],
    }
    #
    tmp = pl.DataFrame(tmp,strict=False).with_columns(pl.selectors.float().round(2),
                                                    #   pl.selectors.object().cast(pl.String),
                                                      )
    #
    return tmp
