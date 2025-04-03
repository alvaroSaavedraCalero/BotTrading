
import pandas as pd
from config.config import INITIAL_CAPITAL, RISK_PERCENT, RR_RATIO
import logging
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Font, Alignment

# Funcion para generar un DataFrame con los resultados de los trades realizados
def generar_dataframe_resultados(historial_trades):
    trades_list = []
    balance_actual = INITIAL_CAPITAL
    for idx, (resultado, entrada, salida, fecha_entrada, fecha_salida) in enumerate(historial_trades, start=1):
        tipo, res = resultado.split('_')
        if res == 'GANANCIA':
            balance_actual += balance_actual * RISK_PERCENT * RR_RATIO
        else:
            balance_actual -= balance_actual * RISK_PERCENT
        operacion = {
            "Operación": idx,
            "Tipo": tipo,
            "Fecha Entrada": fecha_entrada.strftime("%Y-%m-%d %H:%M"),
            "Fecha Salida": fecha_salida.strftime("%Y-%m-%d %H:%M"),
            "Precio Entrada": round(entrada, 2),
            "Stop Loss": round(salida if res == 'PERDIDA' else entrada * (1 - RISK_PERCENT), 2),
            "Take Profit": round(salida if res == 'GANANCIA' else entrada * (1 + RISK_PERCENT * RR_RATIO), 2),
            "Resultado": res,
            "Balance Tras Operación": round(balance_actual, 2)
        }
        trades_list.append(operacion)
    return pd.DataFrame(trades_list, columns=[
        "Operación", "Tipo", "Fecha Entrada", "Fecha Salida",
        "Precio Entrada", "Stop Loss", "Take Profit", "Resultado", "Balance Tras Operación"])
    
    
    


# Funcion para exportar los resultados a un archivo Excel
def exportar_resultados_excel(trades_df, balance_final, archivo_excel="resultados_backtesting.xlsx"):
    trades_df.to_excel(archivo_excel, sheet_name='Backtesting', index=False, startrow=1)
    wb = load_workbook(archivo_excel)
    ws = wb['Backtesting']
    # Título del reporte
    ws['A1'] = 'Resultados del Backtesting'
    ws['A1'].font = Font(size=16, bold=True)
    ws.merge_cells('A1:I1')
    ws['A1'].alignment = Alignment(horizontal='center')
    # Formato a cabeceras
    header_fill = PatternFill("solid", fgColor="4F81BD")
    header_font = Font(bold=True, color="FFFFFF")
    for col in range(1, len(trades_df.columns) + 1):
        cell = ws.cell(row=2, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center')
    # Aplicar formato condicional para las filas
    for row in range(3, ws.max_row + 1):
        tipo_cell = ws[f'B{row}']
        resultado_cell = ws[f'H{row}']
        precio_entrada_cell = ws[f'E{row}']
        stop_loss_cell = ws[f'F{row}']
        take_profit_cell = ws[f'G{row}']
        balance_cell = ws[f'I{row}']
        # Tipo: LONG (verde) o SHORT (rojo)
        tipo_value = tipo_cell.value
        if tipo_value:
            if str(tipo_value).upper() == "LONG":
                tipo_cell.fill = PatternFill("solid", fgColor="C6EFCE")
                tipo_cell.font = Font(color="006100", bold=True)
            elif str(tipo_value).upper() == "SHORT":
                tipo_cell.fill = PatternFill("solid", fgColor="FFC7CE")
                tipo_cell.font = Font(color="9C0006", bold=True)
        # Precio Entrada, Stop Loss y Take Profit
        for celda_num in [precio_entrada_cell, stop_loss_cell, take_profit_cell]:
            celda_num.number_format = '#,##0.00'
            celda_num.alignment = Alignment(horizontal='center')
        # Precio Entrada en azul oscuro
        precio_entrada_cell.font = Font(color="1F497D", bold=True)
        # Stop Loss rojo, Take Profit verde (solo texto)
        stop_loss_cell.font = Font(color="9C0006", bold=True)
        take_profit_cell.font = Font(color="006100", bold=True)
        # Resultado (GANANCIA verde, PERDIDA rojo)
        if resultado_cell.value == 'GANANCIA':
            resultado_cell.fill = PatternFill("solid", fgColor="C6EFCE")  
            resultado_cell.font = Font(color="006100", bold=True)
        else:
            resultado_cell.fill = PatternFill("solid", fgColor="FFC7CE")
            resultado_cell.font = Font(color="9C0006", bold=True)
        # Balance con formato moneda en euros
        balance_cell.number_format = '#,##0.00 €'
    # Ajuste automático del ancho de columnas
    for i, col in enumerate(ws.iter_cols(min_row=2, max_row=ws.max_row), start=1):
        column = get_column_letter(i)
        max_length = max(len(str(cell.value) if cell.value else "") for cell in col)
        ws.column_dimensions[column].width = max_length + 3
    # Añadir estadísticas al final correctamente
    ultima_fila = ws.max_row + 2
    tp_count = (trades_df['Resultado'] == 'GANANCIA').sum()
    sl_count = (trades_df['Resultado'] == 'PERDIDA').sum()
    total_trades = len(trades_df)
    estadisticas = [
        ["📈 Estadísticas del rendimiento", ""],
        ["Trades realizados:", total_trades],
        ["Take Profit alcanzado:", f"{tp_count} trades ({(tp_count / total_trades * 100):.2f}%)"],
        ["Stop Loss alcanzado:", f"{sl_count} trades ({(sl_count / total_trades) * 100:.2f}%)"],
        ["Balance Inicial:", f"{INITIAL_CAPITAL:.2f} €"],
        ["Balance Final:", f"{balance_final:.2f} €"],
        ["Rentabilidad total:", f"{((balance_final - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.2f}%"]
    ]
    for idx, (etiqueta, valor) in enumerate(estadisticas, start=ultima_fila):
        ws[f'A{idx}'] = etiqueta
        ws[f'B{idx}'] = valor
        ws[f'A{idx}'].font = Font(bold=True, size=12)
        ws[f'B{idx}'].alignment = Alignment(horizontal='left')
    wb.save(archivo_excel)
    logging.info(f"✅ Excel generado exitosamente en: '{archivo_excel}'") 
    
    
    
# Función para mostrar los resultados de forma tabular y clara
def mostrar_resultados(balance_final, historial_trades):
    trades_list = []
    balance_actual = INITIAL_CAPITAL
    tp_count = 0
    sl_count = 0

    for idx, (resultado, entrada, salida, fecha_entrada, fecha_salida) in enumerate(historial_trades, start=1):
        tipo, res = resultado.split('_')
        if res == 'GANANCIA':
            balance_actual += balance_actual * RISK_PERCENT * RR_RATIO
            tp_count += 1
        else:
            balance_actual -= balance_actual * RISK_PERCENT
            sl_count += 1

        operacion = {
            "Operación": idx,
            "Tipo": tipo,
            "Fecha Entrada": fecha_entrada.strftime("%Y-%m-%d %H:%M"),
            "Fecha Salida": fecha_salida.strftime("%Y-%m-%d %H:%M"),
            "Precio Entrada": round(entrada, 2),
            "Stop Loss": round(salida if res == 'PERDIDA' else entrada * (1 - RISK_PERCENT), 2),
            "Take Profit": round(salida if res == 'GANANCIA' else entrada * (1 + RISK_PERCENT * RR_RATIO), 2),
            "Resultado": res,
            "Balance Tras Operación": round(balance_actual, 2)
        }

        trades_list.append(operacion)

    trades_df = pd.DataFrame(trades_list, columns=[
        "Operación", "Tipo", "Fecha Entrada", "Fecha Salida",
        "Precio Entrada", "Stop Loss", "Take Profit", "Resultado", "Balance Tras Operación"
    ])

    logging.info("\nResultados del Backtesting:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logging.info(trades_df.to_string(index=False))

    total_trades = len(historial_trades)
    porcentaje_tp = (tp_count / total_trades) * 100 if total_trades > 0 else 0
    porcentaje_sl = (sl_count / total_trades) * 100 if total_trades > 0 else 0

    logging.info("\n📈 Estadísticas del rendimiento:")
    logging.info(f"- Trades realizados: {total_trades}")
    logging.info(f"- Take Profit alcanzado: {tp_count} trades ({porcentaje_tp:.2f}%)")
    logging.info(f"- Stop Loss alcanzado: {sl_count} trades ({porcentaje_sl:.2f}%)")
    logging.info(f"- Balance inicial: ${INITIAL_CAPITAL}")
    logging.info(f"- Balance final: ${balance_final:.2f}")
    rentabilidad = ((balance_final - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    logging.info(f"- Rentabilidad total: {rentabilidad:.2f}%")

