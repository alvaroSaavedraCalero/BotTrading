# main.py

import logging
import os
import sys # Import sys para poder salir si falla algo crítico
from datetime import datetime
import pandas as pd # Importar pandas
from typing import List, Tuple, Any, Optional, Dict # Importar tipos

# Importar funciones y Enum (asumiendo tipos definidos en ellos)
from models.builder import crear_entrenar_modelo, cargar_modelo
# Asegurarse que TradeTuple está definido o importado consistentemente
# from backtestingService.engine import realizar_backtest, TradeTuple # Si está definido ahí
# O definirlo aquí si es más conveniente:
from pandas import Timestamp # Asumiendo fechas como Timestamp
TradeTuple = Tuple[str, float, float, Timestamp, Timestamp]
# ---
from backtestingService.engine import realizar_backtest
from reports.exporter import exportar_resultados_excel, generar_dataframe_resultados, mostrar_resultados
from binanceService.api import obtener_datos_binance
from utils.enumerados import Modelo, TargetMethod # Importar TargetMethod también


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Asegurar que salga por consola estándar
        # Podrías añadir logging.FileHandler aquí si quieres guardar a archivo
    ]
)

if __name__ == '__main__':
    logging.info("--- Iniciando Proceso Principal del Bot ---")
    try:
        # Limpiar pantalla (opcional)
        os.system('cls' if os.name == 'nt' else 'clear')

        # --- Parámetros ---
        simbolo: str = "BTCUSDC"
        intervalo: str = "15m"
        periodo_entrenamiento: str = "6 month ago UTC" # Separar periodo de entreno y backtest
        periodo_backtest: str = "1 month ago UTC"
        ruta_modelos: str = rf"F:\_PERSONAL\_Bot Trading\Machine Learning"
        # Ejemplo de ruta para cargar un modelo específico (si se usara)
        # ruta_modelo_a_cargar: Optional[str] = os.path.join(ruta_modelos, "nombre_especifico.joblib")
        ruta_modelo_a_cargar: Optional[str] = None # Poner None para entrenar siempre uno nuevo

        tipo_modelo_enum: Modelo = Modelo.XGB # Elegir el modelo a entrenar/usar
        metodo_target: TargetMethod = TargetMethod.ORIGINAL # Elegir método de target
        params_target: Dict[str, Any] = {} # Parámetros para el método de target (ej. {'umbral': 0.0015})

        # --- Entrenamiento o Carga del Modelo ---
        logging.info(f"Iniciando creación/entrenamiento de modelo: {tipo_modelo_enum.name}")
        # crear_entrenar_modelo retorna Optional[Any]
        modelo: Optional[Any] = crear_entrenar_modelo(
            ruta_modelo=ruta_modelo_a_cargar, # Usar la variable definida arriba
            simbolo=simbolo,
            temporalidad=intervalo,
            periodo=periodo_entrenamiento, # Usar periodo de entrenamiento
            modelo_enum=tipo_modelo_enum,
            target_method=metodo_target, # Pasar método
            target_params=params_target    # Pasar parámetros
        )

        # Salir si el modelo no se pudo crear/entrenar
        if modelo is None:
            logging.error("❌ No se pudo crear o entrenar el modelo. Terminando ejecución.")
            sys.exit(1) # Salir con código de error

        logging.info(f"✅ Modelo {tipo_modelo_enum.name} listo para usar.")

        # --- Obtención de Datos para Backtest ---
        logging.info(f"Obteniendo datos para backtesting ({periodo_backtest})...")
        # obtener_datos_binance retorna pd.DataFrame
        df_backtest: pd.DataFrame = obtener_datos_binance(
            simbolo=simbolo,
            intervalo=intervalo,
            periodo=periodo_backtest # Usar periodo de backtest
        )

        # Salir si no hay datos para backtest
        if df_backtest.empty:
            logging.error("❌ No se pudieron obtener datos para el backtesting. Terminando ejecución.")
            sys.exit(1)

        logging.info(f"✅ Datos para backtesting ({len(df_backtest)} filas) obtenidos.")

        # --- Ejecución del Backtest ---
        logging.info("Iniciando backtesting...")
        # realizar_backtest retorna Tuple[float, List[TradeTuple], List[float]]
        balance: float; trades: List[TradeTuple]; balance_hist: List[float]
        balance, trades, balance_hist = realizar_backtest(df=df_backtest, model=modelo)
        logging.info("✅ Backtesting finalizado.")

        # --- Generación de Reportes ---
        if not trades:
             logging.warning("⚠️ No se generaron trades durante el backtest. No se crearán reportes detallados.")
        else:
             logging.info("Generando reportes...")
             # Mostrar resultados en consola (usa el balance real del backtest)
             mostrar_resultados(balance_final_real=balance, historial_trades=trades)

             # Generar DataFrame para Excel (usa cálculo interno SIN comisiones para balance por operación)
             # generar_dataframe_resultados retorna pd.DataFrame
             df_resultados: pd.DataFrame = generar_dataframe_resultados(trades)

             # Exportar a Excel (usa el cálculo interno SIN comisiones para stats finales)
             fecha_hora: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
             # Considerar usar config.REPORTS_DIR si se definió
             ruta_reportes: str = rf'F:\_PERSONAL\_Bot Trading\ResultadosBots'
             os.makedirs(ruta_reportes, exist_ok=True) # Asegurar que la carpeta exista
             nombre_archivo: str = rf"Resultados_{tipo_modelo_enum.name}_{simbolo}_{intervalo}_{fecha_hora}.xlsx"
             archivo_final: str = os.path.join(ruta_reportes, nombre_archivo)

             # exportar_resultados_excel retorna None
             # Pasamos 'balance' (el real con comisiones) aunque internamente use el recalculado para las stats
             exportar_resultados_excel(trades_df=df_resultados, balance_final_real=balance, archivo_excel=archivo_final)
             logging.info("✅ Reportes generados.")

        logging.info("--- Proceso Principal Finalizado ---")

    except Exception as e_main:
        logging.exception(f"❌ Ocurrió un error inesperado en la ejecución principal: {e_main}")
        sys.exit(1) # Salir con error en caso de excepción no controlada