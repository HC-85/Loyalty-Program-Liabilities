import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import random
from os import listdir
from pathlib import Path
import re
from itertools import product
from decimal import Decimal
from tqdm import tqdm
import pdb


def encode_map(df, col, dic):
    tmp_keys = df[col].unique()
    if len(dic)==0:
        mapping = dict(zip(tmp_keys, range(len(tmp_keys))))
    else:
        new_keys = set(tmp_keys).difference(set(dic.keys()))
        for new_key in new_keys:
            dic[new_key] = len(dic)
        mapping = dic
    return mapping


def invert_map(dic):
    return dict(zip(dic.values(), dic.keys()))


def reindex(df, col, ind_map):
    df.loc[:, col] = df[col].ind_map(invert_map(ind_map))
    new_ind_map = encode_map(df, col, ind_map)
    df.loc[:,col] = df[col].ind_map(new_ind_map)
    df = df.reset_index(drop=True)
    return df, new_ind_map


def force_sum(dictionary, key, sum_val):
    if dictionary.get(key) is None:
        return sum_val
    else:
        return dictionary[key] + sum_val


def force_concat(df_base: str, df_smoll):
    if globals().get(df_base) is None:
        return df_smoll
    else:
        return pd.concat([globals().get(df_base), df_smoll])


def try_concat(df_base, df_smoll):
    try:
        df_big = pd.concat([df_base, df_smoll])
        return df_big
    except:
        return df_smoll


def crear_transacciones(directorio_entrada, directorio_salida, untwist = True, decimal = True, CEMEX_ID = 54635, cols = ["ID Transacción", "Fecha", "Tarjeta", "ID Comercio", "Importe Bonificación", "Importe Redención", 'Campaña']):
    print('Actualizando transacciones...')
    df_trans = pd.DataFrame(columns = cols)
    
    map_campana = {}
    for item in tqdm(listdir(directorio_entrada)):
        if re.match('^Transacciones.*\.csv$',item) is not None:
            tmp_trans = pd.read_csv(Path(directorio_entrada, item), usecols = cols)
            map_campana = encode_map(tmp_trans, 'Campaña', map_campana)
            tmp_trans.loc[:, 'Campaña'] = tmp_trans['Campaña'].map(map_campana)
            df_trans = pd.concat([df_trans, tmp_trans])
    
    df_trans.loc[:, 'Fecha'] = pd.to_datetime(df_trans['Fecha'])

    df_trans = df_trans.sort_values('Fecha')\
        .rename(columns={'Tarjeta': 'ID_Tarjeta', 'Campaña': 'ID_Campaña', 'ID Comercio': 'ID_Comercio', 'ID Transacción': 'ID_Transaccion',  ' Importe Bonificación ': 'Importe Bonificación', ' Importe Redención ': 'Importe Redención'})\
            .reset_index(drop=True)
    
    fixes = str.maketrans({' ':'', ',':'', '(':'', ')':'', '$':'', '-': '0'})
    df_trans['Importe Bonificación'] = df_trans['Importe Bonificación'].apply(lambda x: str(x).translate(fixes))
    df_trans['Importe Redención'] = df_trans['Importe Redención'].apply(lambda x: str(x).translate(fixes))

    num_mask = df_trans['Importe Bonificación'].apply(lambda x: True if re.search('[0-9]', x) is not None else False)
    df_trans = df_trans.loc[num_mask, :]
    num_mask = df_trans['Importe Redención'].apply(lambda x: True if re.search('[0-9]', x) is not None else False)
    df_trans = df_trans.loc[num_mask, :]
    df_trans = df_trans.reset_index(drop=True)

    df_trans['Importe Bonificación'] = pd.to_numeric(df_trans['Importe Bonificación'])
    df_trans['Importe Redención'] = pd.to_numeric(df_trans['Importe Redención'])
    
    if decimal:
        df_trans.loc[:, 'Importe Bonificación'] = df_trans['Importe Bonificación'].apply(lambda x: Decimal(str(x)))
        df_trans.loc[:, 'Importe Redención'] = df_trans['Importe Redención'].apply(lambda x: Decimal(str(x)))

    # Correcciones
    df_trans.loc[df_trans['ID_Transaccion'].isna(), 'ID_Comercio'] = CEMEX_ID
    df_trans.loc[df_trans['ID_Transaccion'].isna(), 'ID_Transaccion'] = -1
    df_trans.loc[:, 'ID_Transaccion'] = df_trans['ID_Transaccion'].astype('int64')

    puntos_extra = pd.read_csv(Path(directorio_entrada, 'puntos_extra.csv'), index_col=0)
    puntos_extra_set = set(puntos_extra[~puntos_extra['Agente'].isna()]['Campaña'].map(map_campana))
    df_trans.loc[df_trans['ID_Campaña'].isin(puntos_extra_set), "ID_Comercio"] = CEMEX_ID

    # Plot twist
    if untwist:
        plot_twist = set([4561, 4603, 4632, 4875])
        df_trans.loc[df_trans['ID_Comercio'].isin(plot_twist), "ID_Comercio"] = 30156

    return df_trans, map_campana


def crear_vencimientos(directorio_entrada, directorio_salida, cols = ['Fecha', 'ID_Tarjeta', ' Importe '], decimal=True):
    print('Actualizando vencimientos...')
    df_ven = pd.DataFrame(columns = cols)

    for item in tqdm(listdir(directorio_entrada)):
        if re.match('^Dinero_vencido.*\.csv$',item) is not None:
            tmp_ven = pd.read_csv(Path(directorio_entrada, item), usecols = cols)
            df_ven = pd.concat([df_ven, tmp_ven])
    #Limpieza
    df_ven = df_ven.rename(columns={' Importe ':'Importe'})
    fixes = str.maketrans({' ':'', ',':'', '(':'', ')':'', '$':''})
    num_mask = df_ven['Importe'].apply(lambda x: True if re.search('[0-9]', x) is not None else False)
    df_ven = df_ven.loc[num_mask, :]
    df_ven['Importe'] = pd.to_numeric(df_ven['Importe'].apply(lambda x: x.translate(fixes)))

    df_ven.loc[:, 'Fecha'] = pd.to_datetime(df_ven['Fecha'])
    df_ven = df_ven\
        .sort_values('Fecha')\
        .drop_duplicates()\
        .reset_index(drop=True)

    if decimal:
        df_ven.loc[:, 'Importe'] = df_ven['Importe'].apply(lambda x: Decimal(str(x)))
    return df_ven
    

def crear_registros(directorio_entrada, directorio_salida, cols = ['ID_Comercio', 'ID_Tarjeta']):
    print('Actualizando registros de tarjetas...')
    df_reg_tar = pd.DataFrame(columns = cols)
    for item in listdir(directorio_entrada):
        if re.match('^Base_tarjeta', item) is not None:
            tmp_reg_tar = pd.read_csv(Path(directorio_entrada, item), names = cols, header=0)
            df_reg_tar = pd.concat([df_reg_tar, tmp_reg_tar])
    return df_reg_tar


def sustituir_reposiciones(trans, ven, reg, max_fecha, directorio_entrada = r'.\archivos_entrada', directorio_salida = r'.\archivos_intermedios', cols = ['Fecha', 'Tarjeta antigua', 'Tarjeta nueva']):
    max_fecha = datetime.strptime(max_fecha, "%d/%m/%Y")
    df_repos = pd.DataFrame(columns = cols)
    for item in listdir(directorio_entrada):
        if re.match('^Reposiciones.*csv$', item) is not None:
            tmp_repos = pd.read_csv(Path(directorio_entrada, item), names = cols, header=0, index_col=0)
            df_repos = pd.concat([df_repos, tmp_repos])

    df_repos.loc[:, 'Fecha'] = pd.to_datetime(df_repos['Fecha'])
    df_repos = df_repos.sort_values('Fecha', ascending = True).reset_index(drop=True)
    df_repos = df_repos.query('Fecha <= @max_fecha')
    df_repos = df_repos[df_repos['Tarjeta antigua'] != df_repos['Tarjeta nueva']]

    set_repos = set(df_repos['Tarjeta antigua'])
    mapitamapita = dict(zip(df_repos['Tarjeta antigua'],  df_repos['Tarjeta nueva']))

    for df in [trans, ven, reg]:
        sub = df.query('ID_Tarjeta in @set_repos')['ID_Tarjeta'].copy()
        while len(sub)>0:
            sub = sub.map(mapitamapita)
            df.loc[sub.index, 'ID_Tarjeta'] = sub
            sub = sub[sub.isin(set_repos)].copy()
    assert(len(ven.query('ID_Tarjeta in @set_repos')) == 0)

    return trans, ven, reg


def actualizar_archivos(max_fecha, trans=True, ven=True, reg=True, reposiciones = True, directorio_entrada = r'.\archivos_entrada', directorio_salida = r'.\archivos_intermedios'):
    if trans:
        df_trans, map_campana = crear_transacciones(directorio_entrada, directorio_salida)
    if ven:
        df_ven = crear_vencimientos(directorio_entrada, directorio_salida)
    if reg:
        df_reg = crear_registros(directorio_entrada, directorio_salida)
    if reposiciones:
        df_trans, df_ven, df_reg = sustituir_reposiciones(df_trans, df_ven, df_reg, max_fecha)

    guardar(parquets = [(df_trans, 'trans'),(df_ven, 'ven'), (df_reg, 'registro_tarjetas')], pickles= [(map_campana, 'map_campana')], directorio_salida = directorio_salida)


def cargar_archivos(directorio_entrada = r'.\archivos_intermedios'):
    trans = pd.read_parquet(rf'{directorio_entrada}\trans.parquet')
    ven = pd.read_parquet(rf'{directorio_entrada}\ven.parquet')
    reg_tar = pd.read_parquet(rf'{directorio_entrada}\registro_tarjetas.parquet')
    with open(rf'{directorio_entrada}\map_campana.pkl', 'rb') as f:
        map_campana = pickle.load(f)
    return ven, trans, reg_tar, map_campana


def fechas_filtrar(ven, trans, min_fecha, max_fecha):
    min_fecha = datetime.strptime(min_fecha, "%d/%m/%Y")
    max_fecha = datetime.strptime(max_fecha, "%d/%m/%Y")
    tarjs = ven.query('(Fecha >= @min_fecha) & (Fecha <= @max_fecha)')['ID_Tarjeta'].reset_index(drop=True)
    ven = ven.query('(ID_Tarjeta in @tarjs) & (Fecha <= @max_fecha)').reset_index(drop=True)
    trans = trans.query('(ID_Tarjeta in @tarjs) &(Fecha <= @max_fecha)').reset_index(drop=True)
    return ven, trans


def muestrear(ven, m, mseed = 1):
    random.seed(mseed)
    tarjs = random.sample(sorted(set(ven.ID_Tarjeta)), m)
    return ven.query('ID_Tarjeta in @tarjs').reset_index(drop=True)


def tarj_filtrar_trans(ven, trans):
    tarj_set = set(ven.ID_Tarjeta)
    new_trans = trans.query('ID_Tarjeta in @tarj_set').reset_index(drop=True)
    return new_trans


def ven_filtrar_trans(ven, trans): 
    ven = ven.sort_values('Fecha')
    ven_tmp = ven.reset_index()
    ult_ven_ind = ven_tmp.groupby('ID_Tarjeta')[['index']].max().values
    mask = np.zeros(len(ven), bool)
    mask[ult_ven_ind] = True
    ult_ven = ven.iloc[mask]
    ult_ven_comp = ven.iloc[~mask]
    penult_ven = ult_ven_comp\
        .sort_values('Fecha')\
        .groupby('ID_Tarjeta').last()\
        .Fecha

    trans_relevantes = []
    for tarjeta in tqdm(ult_ven.ID_Tarjeta.values):
        try:
            penult_fecha = penult_ven.loc[tarjeta]
        except:
            penult_fecha = datetime.strptime('01/01/2000', "%d/%m/%Y")
        ult_fecha = ult_ven.query('ID_Tarjeta == @tarjeta')['Fecha'].iloc[0]
        sub_trans = trans.query('(ID_Tarjeta == @tarjeta) & (Fecha >= @penult_fecha) & (Fecha < @ult_fecha)')
        trans_relevantes.extend(sub_trans.index.values)
    mask = np.zeros(len(trans), bool)
    mask[trans_relevantes] = True
    trans = trans[mask].reset_index(drop=True)
    return ult_ven, trans


def calcular_devoluciones(ven, trans, reg_tar, directorio_salida=r'.\archivos_salida', exportar=True, mostrar_error=True, merge_index=False):
    devoluciones = {}
    devuelto = [Decimal(0), Decimal(0)]
    errs = []

    with open(Path(directorio_salida, 'excepciones.txt'), 'w') as excepciones:
        for tarjeta in tqdm(ven['ID_Tarjeta']):
            flag = 0
            #----------Monitoreo de errores--------------
            devuelto[1] = sum(list(devoluciones.values()))
            if sum(devuelto) != Decimal(0):
                errs.append((devuelto[0] + din_ven - devuelto[1]))
            try:
                if (abs((devuelto[0] + din_ven) - devuelto[1])>Decimal(str(1E-6))) & (devuelto[0] != Decimal(0)):
                    excepciones.write(f'!!! Error mayor a tolerancia !!! [Tarjeta: {tar_pasada}]\n')
                    excepciones.write(f'Valor esperado: {devuelto[0] + din_ven}\n')
                    excepciones.write(f'Valor calculado: {devuelto[1]}\n')
                    excepciones.write(f'=============================================\n')
            except:
                pass
            devuelto[0] = devuelto[1]
            tar_pasada = tarjeta
            #--------------------------------------------

            din_ven = ven.query('ID_Tarjeta==@tarjeta')['Importe'].iloc[0]
            sub_trans = trans.query('ID_Tarjeta==@tarjeta')
            comercios = list(set(sub_trans['ID_Comercio']))

            # Si la tarjeta expirada no tiene transacciones, se devuelve al comercio donde se registró
            if len(comercios)==0:
                
                no_trans_sub = reg_tar.query('ID_Tarjeta == @tarjeta')
                if len(no_trans_sub) == 0: # Si no esta en el registro, se asigna a 'no reg'
                    comercio = 'no reg'
                    excepciones.write(f'{tarjeta}: no reg\n')
                else:
                    comercio = no_trans_sub['ID_Comercio'].values[0] 
                    #excepciones.write(f'{tarjeta}: no trans\n')     
                comercios = [comercio]
            
            ind_tup = [*product([tarjeta], comercios)]
            ind = pd.MultiIndex.from_tuples(ind_tup, names=["ID_Tarjeta", "ID_Comercio"])
            devs_df_tmp = pd.DataFrame(index = ind, columns=['bonificado', 'redimido', 'total_vencido', '%', 'dinero_comercio'])

            bonifs = {}
            redims = {}
            balances = {}
            for comercio in comercios:
                sub_sub_trans = sub_trans.query('ID_Comercio==@comercio')
                bonif = sub_sub_trans['Importe Bonificación'].sum()
                redim = sub_sub_trans['Importe Redención'].sum()
                balance = bonif - redim
                bonifs[comercio] = bonif
                redims[comercio] = redim
                balances[comercio] = balance
            deuda = sum(balances.values())

            def pctg(balances, comercio, deuda, comercios):
                return Decimal(1/len(comercios))

            if (len(comercios)!=0) & (sub_trans['Importe Bonificación'].sum()==Decimal(0)):
                #excepciones.write(f'{tarjeta}: no bonif\n')
                flag = 1
                
            if all(np.array(list(balances.values()))<Decimal(0)):
                #excepciones.write(f'{tarjeta}: all neg ({len(comercios)})\n')
                if len(comercios)>1:
                    for comercio in comercios:
                        sub_sub_trans = sub_trans.query('ID_Comercio==@comercio')
                        balance = sub_sub_trans['Importe Bonificación'].sum()
                        balances[comercio] = balance
                    deuda = sum(balances.values())
                else:
                    flag = 1
            if deuda==Decimal(0):
                #excepciones.write(f'{tarjeta}: no deuda*\n')
                if len(comercios) == 1:
                    flag = 1
            
            if flag == 0:
                if any(np.array(list(balances.values()))>Decimal(0)):
                    for comercio in balances.keys():
                        if balances[comercio]<Decimal(0):
                            balances[comercio] = Decimal(0)
                    deuda = sum(balances.values())

                def pctg(balances, comercio, deuda, comercios):
                    return Decimal(balances[comercio]/deuda)

            for comercio in comercios:
                pctg_ = pctg(balances, comercio, deuda, comercios)
                devolucion = din_ven * pctg_
                bonif = bonifs[comercio]
                redim = redims[comercio]
                devs_df_tmp.loc[(tarjeta, comercio),('bonificado','redimido','total_vencido', '%', 'dinero_comercio')] = (bonif, redim, din_ven, pctg_, devolucion)
                devoluciones[comercio] = force_sum(devoluciones, comercio, devolucion)

            try:
                devs_df = pd.concat([devs_df, devs_df_tmp])
            except:
                devs_df = devs_df_tmp


    #---------------------------------------------------------------------------------------------------------------------------------------#
    res = pd.Series(devoluciones).reset_index().rename(columns={'index': 'Comercio', 0: 'Dinero'}).sort_values('Dinero', ascending=False)

    if exportar:
        devs_df.index = devs_df.index.set_levels(list(map(str, devs_df.index.levels[0].values)), level = 0)
        if merge_index is False:
            devs_df = devs_df.reset_index()
            res.to_excel(Path(directorio_salida,r'devoluciones_detalles.csv'), index=0)
        else:
            devs_df.to_excel(Path(directorio_salida, r'devoluciones_detalles.xlsx'), index=0)
        res.to_excel(Path(directorio_salida, r'devoluciones.csv'), index=0)

    if mostrar_error:    
        print(f"Dinero vencido: {ven['Importe'].sum()}")
        print(f"Dinero devuelto: {res['Dinero'].sum()}")
        print(f"Error: {sum(errs)}")

    return res, devs_df


def guardar(parquets, pickles, directorio_salida):
    # [(var1, filename1), (var2, filename2), ...]
    for par in parquets:
        par[0].to_parquet(path=Path(directorio_salida, f'{par[1]}.parquet'))

    for pic in pickles:
        with open(Path(directorio_salida, f'{pic[1]}.pkl'), 'wb') as f:
            pickle.dump(pic[0], f)