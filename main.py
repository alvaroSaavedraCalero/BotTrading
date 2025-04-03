
import logging
import os
from datetime import datetime
from models.builder import crear_entrenar_modelo, cargar_modelo
from backtestingService.engine import realizar_backtest
from reports.exporter import exportar_resultados_excel, generar_dataframe_resultados, mostrar_resultados
from binanceService.api import obtener_datos_binance

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    simbolo = "BTCUSDC"
    intervalo = "15m"
    periodo = "2 month ago UTC"
    ruta_modelos = rf"C:\Users\Álvaro\OneDrive\Escritorio\InfoRecursosBots\ModelosEntrenados"
    ruta_modelo_def = os.path.join(ruta_modelos, "modelo_entrenado_RandomForest_BTCUSDC_2025-04-03_09-49-23.joblib")
    
    modelo = cargar_modelo(ruta=ruta_modelo_def)
    # XGB / GradientBoosting / RandomForest
    #modelo = crear_entrenar_modelo(ruta_modelo=None, simbolo=simbolo, temporalidad=intervalo, periodo=periodo, nombre_modelo='RandomForest')
    #df_modelo = obtener_datos_binance(simbolo=simbolo, intervalo=intervalo, periodo=periodo)   
    
    #modelo = entrenar_modelo(model=modelo, df=df_modelo)
    
    df_backtest = obtener_datos_binance(simbolo=simbolo, intervalo=intervalo, periodo=periodo)
    balance, trades, balance_hist = realizar_backtest(df=df_backtest, model=modelo)
    
    mostrar_resultados(balance_final=balance, historial_trades=trades)
    
    df_resultados = generar_dataframe_resultados(trades)
    
    fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta = rf'C:\Users\Álvaro\OneDrive\Escritorio\InfoRecursosBots\ResultadosBots'
    nombre_archivo = rf"ResultadosBot5_temp{intervalo}_durante{periodo}_simbolo{simbolo}_{fecha_hora}.xlsx"
    archivo_final = os.path.join(ruta, nombre_archivo)
    exportar_resultados_excel(trades_df=df_resultados, balance_final=balance, archivo_excel=archivo_final)