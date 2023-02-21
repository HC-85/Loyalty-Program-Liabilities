Para preparar los archivos a analizar:
- Dentro de la carpeta "archivos_entrada" introducir como CSVs el/los archivo(s) de:
    - Transacciones ("Transacciones...")
    - Vencimientos ("Dinero vencido...")
    - Registro de tarjetahabientes ("Base_tarjeta...")
    - Reposiciones de tarjetas ("Reposiciones...")

Para utilizar la herramienta:
- Abra la terminal y cambia el directorio a donde se encuentra "dinero_vencido.py"
- Active el entorno "din_venv": "\din_venv\Scripts\Activate.ps1"
- Corra en la terminal: python dinero_vencido.py [fecha de inicio] [fecha de fin]

Por ejemplo, si deseamos realizar el analisis para el periodo Julio-Diciembre 2021, correriamos:
>> python dinero_vencido.py 01/07/2021 31/12/2021