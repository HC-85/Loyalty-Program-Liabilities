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


def safe_sum(dictionary, key, sum_val):
    if dictionary.get(key) is None:
        return sum_val
    else:
        return dictionary[key] + sum_val


def safe_concat(df_base: str, df_smoll):
    if locals().get(df_base) is None:
        return df_smoll
    else:
        return pd.concat([locals().get(df_base), df_smoll])


def crear_transacciones(directorio_entrada, directorio_salida, decimal = True, CEMEX_ID = 54635, cols = ["ID Transacción", "Fecha", "Tarjeta", "ID Comercio", "Importe Bonificación", "Importe Redención", 'Campaña']):
    print('Actualizando transacciones...')
    df_trans = pd.DataFrame(columns = cols)

    map_campana = {}
    for item in tqdm(listdir(directorio_entrada)):
        if item[0:13]=="Transacciones":
            tmp_trans = pd.read_csv(Path(directorio_entrada, item), usecols = cols)
            map_campana = encode_map(tmp_trans, 'Campaña', map_campana)
            tmp_trans.loc[:, 'Campaña'] = tmp_trans['Campaña'].map(map_campana)
            df_trans = pd.concat([df_trans, tmp_trans])

    df_trans.loc[:, 'Fecha'] = pd.to_datetime(df_trans['Fecha'])

    df_trans = df_trans.sort_values('Fecha')\
        .rename(columns={'Tarjeta': 'ID_Tarjeta', 'Campaña': 'ID_Campaña', 'ID Comercio': 'ID_Comercio', 'ID Transacción': 'ID_Transaccion'})\
            .reset_index(drop=True)

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
    
    # Exportacion
    with open(Path(directorio_salida, 'map_campana.pkl'), 'wb') as f:
        pickle.dump(map_campana, f)

    df_trans.to_parquet(path=Path(directorio_salida, 'trans.parquet'))


def crear_vencimientos(directorio_entrada, directorio_salida, cols = ['Fecha', 'ID_Tarjeta', ' Importe '], decimal=True):
    print('Actualizando vencimientos...')
    df_ven = pd.read_csv(Path(directorio_entrada, 'Dinero_vencido.csv'), usecols = cols)
    
    #Limpieza
    df_ven = df_ven.rename(columns={' Importe ':'Importe'}).\
        drop(index=[824356, 824369, 824370]).\
            reset_index(drop=True)
    fixes = str.maketrans({' ':'', ',':'', '(':'', ')':'', '$':''})
    df_ven['Importe'] = pd.to_numeric(df_ven['Importe'].apply(lambda x: x.translate(fixes)))

    df_ven.loc[:, 'Fecha'] = pd.to_datetime(df_ven['Fecha']).\
        sort_values('Fecha').\
            drop_duplicates()

    if decimal:
        df_ven.loc[:, 'Importe'] = df_ven['Importe'].apply(lambda x: Decimal(str(x)))
    # Exportacion
    df_ven.to_parquet(path=Path(directorio_salida, 'din_ven.parquet'))
    

def crear_registros(directorio_entrada, directorio_salida, cols = ['ID_Comercio', 'ID_Tarjeta']):
    print('Actualizando registros de tarjetas...')
    df_reg_tar = pd.DataFrame(columns = cols)
    for item in listdir(directorio_entrada):
        if re.match('^Base_tarjeta', item) is not None:
            tmp_reg_tar = pd.read_csv(Path(directorio_entrada, item), names = cols, header=0)
            df_reg_tar = pd.concat([df_reg_tar, tmp_reg_tar])
    # Exportacion
    df_reg_tar = df_reg_tar.set_index('ID_Tarjeta')
    df_reg_tar.to_parquet(path=Path(directorio_salida, 'registro_tarjetas.parquet'))


def actualizar_archivos(trans=True, ven=True, reg=True, directorio_entrada = r'.\archivos_entrada', directorio_salida = r'.\archivos_intermedios'):
    if trans:
        crear_transacciones(directorio_entrada, directorio_salida)
    if ven:
        crear_vencimientos(directorio_entrada, directorio_salida)
    if reg:
        crear_registros(directorio_entrada, directorio_salida)


def cargar_archivos(directorio_entrada = r'.\archivos_intermedios'):
    trans = pd.read_parquet(rf'{directorio_entrada}\trans.parquet')
    ven = pd.read_parquet(rf'{directorio_entrada}\din_ven.parquet')
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
    return ven.loc[random.sample(sorted(ven.index), m)].reset_index(drop=True)


def tarj_filtrar_trans(ven, trans):
    return trans.query('ID_Tarjeta in @ven.ID_Tarjeta').reset_index(drop=True)


def ven_filtrar_trans(ven, trans): 
    assert (all(ven.index.values == [*range(len(ven))])) & (all(trans.index.values == [*range(len(trans))]))

    ven = ven.reset_index()
    ult_ven = ven.groupby('ID_Tarjeta')['Fecha'].max()\
        .reset_index().set_index(['ID_Tarjeta', 'Fecha'])\
            .join(ven.set_index(['ID_Tarjeta', 'Fecha']))\
                .reset_index()

    mask = np.ones(len(ven), bool)
    mask[ult_ven.index] = False
    ult_ven_comp = ven.loc[mask]
    penult_ven = ult_ven_comp.sort_values('Fecha')\
        .groupby('ID_Tarjeta').last()\
            .Fecha

    ult_ven = ult_ven.drop(columns='index')
    
    trans_relevantes = []
    for tarjeta in tqdm(ult_ven.ID_Tarjeta.values):
        try:
            penult_fecha = penult_ven.loc[tarjeta]
        except:
            penult_fecha = datetime.strptime('01/01/2000', "%d/%m/%Y")
        ult_fecha = ult_ven.query('ID_Tarjeta == @tarjeta')['Fecha'].iloc[0]
        sub_trans = trans.query('ID_Tarjeta == @tarjeta & Fecha > @penult_fecha & Fecha < @ult_fecha')
        trans_relevantes.extend(sub_trans.index.values)

    mask = np.zeros(len(trans), bool)
    mask[trans_relevantes] = True
    trans = trans[mask].reset_index(drop=True)
    return ult_ven, trans


def calcular_devoluciones(ven, trans, reg_tar, directorio_salida=r'.\archivos_salida', exportar=True, mostrar_resultados=True):
    devoluciones = {}
    devuelto = [Decimal(0), Decimal(0)]
    errs = []

    with open(Path(directorio_salida, 'excepciones.txt'), 'w') as excepciones:
        for tarjeta in tqdm(ven['ID_Tarjeta']):
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

            din_ven = ven.query('ID_Tarjeta==@tarjeta')['Importe'].iloc[0]
            sub_trans = trans.query('ID_Tarjeta==@tarjeta')
            comercios = list(set(sub_trans['ID_Comercio']))

            if len(comercios)==0:
                excepciones.write(f'{tarjeta}: no trans\n')
                devolucion = din_ven
                if reg_tar.get(tarjeta) is None:
                    comercio = 'no reg'
                    excepciones.write(f'{tarjeta}: no reg\n')
                else:
                    comercio = reg_tar.loc[tarjeta][0]       
                comercios = [comercio]

                ind_tup = [*product([tarjeta], comercios)]
                ind = pd.MultiIndex.from_tuples(ind_tup, names=["ID_Tarjeta", "ID_Comercio"])
                devs_df_tmp = pd.DataFrame(index = ind, columns=['total_vencido', '%', 'dinero_comercio'])
                for comercio in comercios:
                    devs_df_tmp.loc[(tarjeta, comercio),('total_vencido', '%', 'dinero_comercio')] = zip((din_ven, 1/len(comercios), devolucion)) 
                    devoluciones[comercio] = safe_sum(devoluciones, comercio, devolucion)

                devs_df = safe_concat('devs_df', devs_df_tmp)

                continue

            ind_tup = [*product([tarjeta], comercios)]
            ind = pd.MultiIndex.from_tuples(ind_tup, names=["ID_Tarjeta", "ID_Comercio"])
            devs_df_tmp = pd.DataFrame(index = ind, columns=['total_vencido', '%', 'dinero_comercio'])

            if sub_trans['Importe Bonificación'].sum()==Decimal(0):
                excepciones.write(f'{tarjeta}: no bonif\n')
                devolucion = din_ven/len(comercios)

                for comercio in comercios:
                    devs_df_tmp.loc[(tarjeta, comercio),('total_vencido', '%', 'dinero_comercio')] = zip((din_ven, 1/len(comercios), devolucion)) 
                    devoluciones[comercio] = safe_sum(devoluciones, comercio, devolucion)

                devs_df = safe_concat('devs_df', devs_df_tmp)
                continue

            if len(comercios) == 1:
                devolucion = din_ven
                for comercio in comercios:
                    devs_df_tmp.loc[(tarjeta, comercio),('total_vencido', '%', 'dinero_comercio')] = zip((din_ven, 1/len(comercios), devolucion))
                    devoluciones[comercio] = safe_sum(devoluciones, comercio, devolucion)                

                devs_df = safe_concat('devs_df', devs_df_tmp)
                continue

            balances = {}
            for comercio in comercios:
                sub_sub_trans = sub_trans.query('ID_Comercio==@comercio')
                balance = sub_sub_trans['Importe Bonificación'].sum() - sub_sub_trans['Importe Redención'].sum()
                balances[comercio] = balance

            if any(np.array(list(balances.values()))>Decimal(0)):
                for comercio in balances.keys():
                    if balances[comercio]<Decimal(0):
                        balances[comercio] = Decimal(0)

            elif all(np.array(list(balances.values()))<Decimal(0)):
                excepciones.write(f'{tarjeta}: all neg\n')
                devolucion = din_ven/len(comercios)
                for comercio in comercios:
                    devs_df_tmp.loc[(tarjeta, comercio),('total_vencido', '%', 'dinero_comercio')] = zip((din_ven, 1/len(comercios), devolucion))
                    devoluciones[comercio] = safe_sum(devoluciones, comercio, devolucion)
                
                devs_df = safe_concat('devs_df', devs_df_tmp)
                continue

            deuda = sum(balances.values())

            if deuda==Decimal(0):
                devolucion = din_ven/len(comercios)
                for comercio in comercios:  
                    devs_df_tmp.loc[(tarjeta, comercio),('total_vencido', '%', 'dinero_comercio')] = zip((din_ven, 1/len(comercios), devolucion))    
                    devoluciones[comercio] = safe_sum(devoluciones, comercio, devolucion)
                    
                devs_df = safe_concat('devs_df', devs_df_tmp)
                continue

            for comercio in balances.keys():
                devolucion = din_ven*(balances[comercio]/deuda)
                devs_df_tmp.loc[(tarjeta, comercio),('total_vencido', '%', 'dinero_comercio')] = zip((din_ven, balances[comercio]/deuda, devolucion))
                devoluciones[comercio] = safe_sum(devoluciones, comercio, devolucion)

            devs_df = safe_concat('devs_df', devs_df_tmp)

    #---------------------------------------------------------------------------------------------------------------------------------------#
    res = pd.Series(devoluciones).reset_index().rename(columns={'index': 'Comercio', 0: 'Dinero'}).sort_values('Dinero', ascending=False)

    if exportar:
        devs_df.index = devs_df.index.set_levels(list(map(str, devs_df.index.levels[0].values)), level = 0)
        devs_df.to_excel(Path(directorio_salida, r'.\devoluciones_detalles.xlsx'))
        res.to_csv(Path(directorio_salida,r'\devoluciones.csv'), encoding='utf-8-sig')

    if mostrar_resultados:    
        print(f"Dinero vencido: {ven['Importe'].sum()}")
        print(f"Dinero devuelto: {res['Dinero'].sum()}")
        print(f"Error: {sum(errs)}")

    return res, devs_df


def guardar(res, ven, devs, trans, directorio_salida = r'.\archivos_intermedios'):
    with open(Path(directorio_salida, 'results.pkl'), 'wb') as f:
        pickle.dump(res, f)

    with open(Path(directorio_salida, 'ult_ven.pkl'), 'wb') as f:
        pickle.dump(ven, f)

    with open(Path(directorio_salida, 'devs.pkl'), 'wb') as f:
        pickle.dump(devs, f)

    with open(Path(directorio_salida, 'trans.pkl'), 'wb') as f:
        pickle.dump(trans, f)