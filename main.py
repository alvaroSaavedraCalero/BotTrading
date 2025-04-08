
import logging
import os
from datetime import datetime
from models.builder import crear_entrenar_modelo, cargar_modelo, guardar_modelo
from backtestingService.engine import realizar_backtest
from reports.exporter import exportar_resultados_excel, generar_dataframe_resultados, mostrar_resultados
from binanceService.api import obtener_datos_binance
from utils.enumerados import Modelo

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    simbolo = "BTCUSDC"
    intervalo = "15m"
    periodo = "3 months ago UTC"
    ruta_modelos = rf"C:\Users\Álvaro\OneDrive\Escritorio\InfoRecursosBots\ModelosEntrenados"
    ruta_modelo_def = os.path.join(ruta_modelos, "modelo_entrenado_RANDOM_FOREST_BTCUSDC_2025-04-04_09-08-06.joblib")
    
    # XGB / GradientBoosting / RandomForest
    modelo = crear_entrenar_modelo(ruta_modelo=None, simbolo=simbolo, temporalidad=intervalo, periodo=periodo, modelo=Modelo.XGB)
    
    #modelo = cargar_modelo(ruta=ruta_modelo_def)
    
    df_backtest = obtener_datos_binance(simbolo=simbolo, intervalo=intervalo, periodo=periodo)
    
    balance, trades, balance_hist = realizar_backtest(df=df_backtest, model=modelo)
    mostrar_resultados(balance_final=balance, historial_trades=trades)
    
    df_resultados = generar_dataframe_resultados(trades)
    
    fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta = rf'C:\Users\Álvaro\OneDrive\Escritorio\InfoRecursosBots\ResultadosBots'
    nombre_archivo = rf"ResultadosBot5_temp{intervalo}_durante{periodo}_simbolo{simbolo}_{fecha_hora}.xlsx"
    archivo_final = os.path.join(ruta, nombre_archivo)
    exportar_resultados_excel(trades_df=df_resultados, balance_final=balance, archivo_excel=archivo_final)