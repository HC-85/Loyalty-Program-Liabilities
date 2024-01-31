# Loyalty Program Liabilities
This project was done as part of an internship at CEMEX sales department.

Calculates liabilities due to expired points for a loyalty program.

## Procedure
1. **Provide inside [Input Files] "archivos_entrada" CSVs files containing**:
    - [Transactions] Transacciones ("Transacciones...")
    - [Expirations] Vencimientos ("Dinero vencido...")
    - [User Base] Registro de tarjetahabientes ("Base_tarjeta...")
    - [Replacements] Reposiciones de tarjetas ("Reposiciones...")

2. **Open the terminal and navigate to the folder containing "dinero_vencido.py"**:
3. **Activate the environment "din_venv"**: 
```bash
    \din_venv\Scripts\Activate.ps1
```
4. **Run in terminal with the desired dates**:
```bash
    python dinero_vencido.py [start date] [end date]
```
For example, if we wish to analize from July 2021 to December 2021, we'd run:
```bash
    python dinero_vencido.py 01/07/2021 31/12/2021
```
