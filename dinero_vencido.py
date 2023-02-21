from utils import *
import argparse
from datetime import datetime
min_fecha = '01/07/2021'
max_fecha = '31/12/2021'


def main(min_fecha, max_fecha, actualizar, muestreo, no_guardar):
    if actualizar:
        actualizar_archivos(max_fecha)
        with open(r'.\archivos_intermedios\ultima_actualizacion.txt', 'w', encoding='utf-8') as file:
            dt = datetime.now()
            file.write(f'Última actualización de archivos: {dt.day}/{dt.month}/{dt.year} a las {dt.hour}:{dt.minute:02d} horas')
    else:
        with open(r'.\archivos_intermedios\ultima_actualizacion.txt', 'r', encoding='utf-8') as file:
            ult_act = file.read()
        print(ult_act)

    print('Cargando archivos...')
    ven, trans, reg_tar, map_campana = cargar_archivos()
    
    print('Filtrando...')
    ven, trans = fechas_filtrar(ven, trans, min_fecha, max_fecha)
    
    if muestreo!=0:
        ven = muestrear(ven, muestreo)

    trans = tarj_filtrar_trans(ven, trans)
    
    ven, trans = ven_filtrar_trans(ven, trans)
    print('Calculando devoluciones...')
    res, devs = calcular_devoluciones(ven, trans, reg_tar)

    if no_guardar:
        guardar([(trans, 'trans_proc'), (ven, 'ven_proc')], [(devs, 'devs'), (res, 'results')], r'.\archivos_intermedios')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("min_fecha", help="Fecha de inicio para cálculos (dd/mm/yyyy)")
parser.add_argument("max_fecha", help="Fecha de término para cálculos (dd/mm/yyyy)")
parser.add_argument("--actualizar",'-a', action="store_true", help="Actualiza los archivos usados para el calculo")
parser.add_argument("--muestreo",'-m', type=int, default=0, help="Numero de muestras para realizar una prueba de funcionamiento")
parser.add_argument("--no_guardar", action="store_false", help="No guardar archivos de variables procesadas")

argvars = vars(parser.parse_args())
min_fecha, max_fecha, actualizar, muestreo, no_guardar = argvars.values()

if __name__ == "__main__":
    
    main(min_fecha, max_fecha, actualizar, muestreo, no_guardar)