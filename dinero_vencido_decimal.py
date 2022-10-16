from utils_test import *
import argparse
from datetime import datetime
min_fecha = '01/07/2021'
max_fecha = '31/12/2021'


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("min_fecha", help="Fecha de inicio para cálculos (dd/mm/yyyy)")
parser.add_argument("max_fecha", help="Fecha de término para cálculos (dd/mm/yyyy)")
parser.add_argument("--actualizar",'-a', action="store_true", help="Actualiza los archivos usados para el calculo")
parser.add_argument("--muestreo",'-m', type=int, default=0, help="Numero de muestras para realizar una prueba de funcionamiento")
parser.add_argument("--guardar",'-g', action="store_true", help="Guardar archivos con variables procesadas")


argvars = vars(parser.parse_args())


if __name__ == "__main__":
    if argvars['actualizar']:
        actualizar_archivos()
        with open(r'.\archivos_intermedios\ultima_actualizacion.txt', 'w', encoding='utf-8') as file:
            dt = datetime.now()
            file.write(f'Última actualización: {dt.day}/{dt.month}/{dt.year} a las {dt.hour}:{dt.minute} horas')
    else:
        with open(r'.\archivos_intermedios\ultima_actualizacion.txt', 'r', encoding='utf-8') as file:
            ult_act = file.read()
        print(ult_act)

    print('Cargando archivos...')
    ven, trans, reg_tar, map_campana = cargar_archivos()
    
    print('Filtrando...')
    fechas_filtrar(ven, trans, argvars['min_fecha'], argvars['max_fecha'])
    
    if argvars['muestreo']!=0:
        ven = muestrear(ven, argvars['muestreo'])

    trans = tarj_filtrar_trans(ven, trans)

    ven, trans = ven_filtrar_trans(ven, trans)
    print('Calculando devoluciones...')
    res, devs = calcular_devoluciones(ven, trans, reg_tar)

    if argvars['guardar']:
        guardar(res, ven, devs, trans)
