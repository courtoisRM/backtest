import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib
matplotlib.use('Agg')  # Backend que no requiere GUI
import matplotlib.pyplot as plt
import backtrader as bt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración
ticker = "QQQ"
period = "2y"  # Período más largo para mejor análisis
interval = "4h"

print("Descargando datos...")
# Descarga de datos
data = yf.download(tickers=ticker, period=period, interval=interval, group_by='ticker', progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(1)

data = data.dropna(subset=['Close', 'High', 'Low', 'Volume'])

print("Calculando indicadores...")
# Indicadores técnicos mejorados
data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()

# VWAP CORREGIDO - Reset diario
data['Date'] = data.index.date
data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
data['Volume_Price'] = data['Typical_Price'] * data['Volume']

# Calcular VWAP diario
vwap_data = []
for date in data['Date'].unique():
    day_data = data[data['Date'] == date].copy()
    day_data['Cumulative_VP'] = day_data['Volume_Price'].cumsum()
    day_data['Cumulative_Vol'] = day_data['Volume'].cumsum()
    day_data['VWAP'] = day_data['Cumulative_VP'] / day_data['Cumulative_Vol']
    vwap_data.append(day_data)

data = pd.concat(vwap_data).sort_index()

# Indicador adicional: EMA para filtro de tendencia
data['EMA_50'] = ta.trend.EMAIndicator(close=data['Close'], window=50).ema_indicator()

# Eliminar filas con NaN
data = data.dropna()

print(f"Datos preparados: {len(data)} registros")
print(f"Período: {data.index[0]} a {data.index[-1]}")

class CustomData(bt.feeds.PandasData):
    """Feed personalizado con indicadores"""
    lines = ('RSI', 'VWAP', 'EMA_50')
    params = (
        ('RSI', -1),
        ('VWAP', -1),
        ('EMA_50', -1),
    )

class ImprovedVWAPRSIStrategy(bt.Strategy):
    """Estrategia VWAP-RSI Mejorada"""
    
    params = (
        ('rsi_oversold', 25),
        ('rsi_overbought', 75),
        ('rsi_extreme', 80),
        ('stop_loss_pct', 0.04),    # 4% stop loss
        ('take_profit_pct', 0.10),   # 10% take profit
        ('min_volume', 500000),      # Volumen mínimo
        ('use_trend_filter', True),  # Filtro de tendencia
        ('print_logs', True),        # Control de logging
    )
    
    def __init__(self):
        self.rsi = self.datas[0].RSI
        self.vwap = self.datas[0].VWAP
        self.ema_50 = self.datas[0].EMA_50
        
        # Variables para tracking
        self.entry_price = 0
        self.stop_price = 0
        self.target_price = 0
        self.order = None
        
        # Para estadísticas
        self.trades_won = 0
        self.trades_lost = 0
        self.total_pnl = 0
        self.trade_history = []
        
    def log(self, txt, dt=None):
        """Logging personalizado"""
        if self.params.print_logs:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}, {txt}')
    
    def notify_order(self, order):
        """Notificación de órdenes"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'COMPRA EJECUTADA - Precio: {order.executed.price:.2f}, '
                        f'Tamaño: {order.executed.size}, Comisión: {order.executed.comm:.2f}')
                self.entry_price = order.executed.price
                self.stop_price = self.entry_price * (1 - self.params.stop_loss_pct)
                self.target_price = self.entry_price * (1 + self.params.take_profit_pct)
            else:
                self.log(f'VENTA EJECUTADA - Precio: {order.executed.price:.2f}, '
                        f'Tamaño: {order.executed.size}, Comisión: {order.executed.comm:.2f}')
                
                # Calcular P&L
                if self.entry_price > 0:
                    pnl = (order.executed.price - self.entry_price) / self.entry_price * 100
                    self.total_pnl += pnl
                    
                    trade_info = {
                        'entry_price': self.entry_price,
                        'exit_price': order.executed.price,
                        'pnl_pct': pnl,
                        'date': self.datas[0].datetime.date(0)
                    }
                    self.trade_history.append(trade_info)
                    
                    if pnl > 0:
                        self.trades_won += 1
                        self.log(f'✅ GANANCIA: {pnl:.2f}%')
                    else:
                        self.trades_lost += 1
                        self.log(f'❌ PÉRDIDA: {pnl:.2f}%')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Orden Cancelada/Rechazada')
        
        self.order = None
    
    def next(self):
        """Lógica principal de la estrategia"""
        
        # Verificar si hay orden pendiente
        if self.order:
            return
        
        current_price = self.data.close[0]
        current_rsi = self.rsi[0]
        current_vwap = self.vwap[0]
        current_ema = self.ema_50[0]
        current_volume = self.data.volume[0]
        
        # Verificar datos válidos
        if np.isnan(current_rsi) or np.isnan(current_vwap) or np.isnan(current_ema):
            return
        
        if not self.position:
            # === LÓGICA DE ENTRADA ===
            
            # Condiciones básicas
            price_below_vwap = current_price < current_vwap
            oversold = current_rsi < self.params.rsi_oversold
            good_volume = current_volume > self.params.min_volume
            
            # Filtro de tendencia (opcional)
            trend_ok = True
            if self.params.use_trend_filter:
                trend_ok = current_price > current_ema  # Solo comprar en tendencia alcista
            
            # Señal de entrada
            if price_below_vwap and oversold and good_volume and trend_ok:
                # Calcular tamaño de posición (95% del capital)
                size = int(self.broker.getcash() * 0.02 / current_price)
                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(f'🔵 SEÑAL COMPRA - RSI: {current_rsi:.1f}, '
                            f'Precio: {current_price:.2f}, VWAP: {current_vwap:.2f}')
        
        else:
            # === LÓGICA DE SALIDA ===
            
            # Stop Loss
            if current_price <= self.stop_price:
                self.order = self.sell()
                self.log(f'🛑 STOP LOSS - Precio: {current_price:.2f}')
                
            # Take Profit
            elif current_price >= self.target_price:
                self.order = self.sell()
                self.log(f'🎯 TAKE PROFIT - Precio: {current_price:.2f}')
                
            # Salida por condiciones técnicas
            elif current_price > current_vwap and current_rsi > self.params.rsi_overbought:
                self.order = self.sell()
                self.log(f'📈 SALIDA TÉCNICA - RSI: {current_rsi:.1f}')
                
            # Salida por RSI extremo (independiente de VWAP)
            elif current_rsi > self.params.rsi_extreme:
                self.order = self.sell()
                self.log(f'🔴 RSI EXTREMO - RSI: {current_rsi:.1f}')
    
    def stop(self):
        """Al finalizar la estrategia"""
        total_trades = self.trades_won + self.trades_lost
        win_rate = (self.trades_won / total_trades * 100) if total_trades > 0 else 0
        
        print('\n' + '=' * 50)
        print('RESULTADOS DE LA ESTRATEGIA:')
        print('=' * 50)
        print(f'Total trades: {total_trades}')
        print(f'Trades ganadores: {self.trades_won}')
        print(f'Trades perdedores: {self.trades_lost}')
        print(f'Win rate: {win_rate:.1f}%')
        
        if self.trade_history:
            pnls = [trade['pnl_pct'] for trade in self.trade_history]
            avg_win = np.mean([pnl for pnl in pnls if pnl > 0]) if self.trades_won > 0 else 0
            avg_loss = np.mean([pnl for pnl in pnls if pnl < 0]) if self.trades_lost > 0 else 0
            
            print(f'Ganancia promedio: {avg_win:.2f}%')
            print(f'Pérdida promedio: {avg_loss:.2f}%')
            if avg_loss < 0:
                print(f'Ratio Ganancia/Pérdida: {abs(avg_win/avg_loss):.2f}')
        print('=' * 50)

def run_backtest():
    """Función principal para ejecutar el backtest"""
    
    # Configurar Cerebro
    cerebro = bt.Cerebro()
    
    # Añadir estrategia con parámetros optimizados
    cerebro.addstrategy(
        ImprovedVWAPRSIStrategy,
        rsi_oversold=55,      # Menos restrictivo (era 25)
        rsi_overbought=60,    # Salida más temprana
        rsi_extreme=70,       # RSI extremo menos restrictivo
        stop_loss_pct=0.02,   # 4% stop loss
        take_profit_pct=0.04, # 8% take profit (menos ambicioso)
        use_trend_filter=False, # Desactivar filtro contradictorio
        print_logs=True       # Activar logs para ver qué pasa
    )
    
    # Añadir datos
    cerebro.adddata(CustomData(dataname=data))
    
    # Configuración del broker
    initial_capital = 100000.0
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% comisión
    
    # Añadir analizadores
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print("=" * 60)
    print(f"BACKTESTING ESTRATEGIA VWAP-RSI MEJORADA - {ticker}")
    print("=" * 60)
    print(f"Capital inicial: ${initial_capital:,.2f}")
    print("Ejecutando backtest...")
    
    # Ejecutar backtest
    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    profit = final_value - initial_capital
    profit_pct = profit / initial_capital * 100
    
    print(f"Capital final: ${final_value:,.2f}")
    print(f"Ganancia total: ${profit:,.2f} ({profit_pct:.2f}%)")
    
    # Mostrar análisis detallado
    print("\n" + "=" * 40)
    print("ANÁLISIS DETALLADO:")
    print("=" * 40)
    
    # Sharpe Ratio
    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe.get('sharperatio', None)
        if sharpe_ratio is not None:
            print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        else:
            print("Sharpe Ratio: No disponible")
    except:
        print("Sharpe Ratio: Error en cálculo")
    
    # Drawdown
    try:
        drawdown = strat.analyzers.drawdown.get_analysis()
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
        print(f"Max Drawdown: {max_dd:.2f}%")
    except:
        print("Max Drawdown: Error en cálculo")
    
    # Trade Analysis
    try:
        trade_analysis = strat.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        lost_trades = trade_analysis.get('lost', {}).get('total', 0)
        
        if total_trades > 0:
            print(f"Total de operaciones: {total_trades}")
            print(f"Operaciones ganadoras: {won_trades}")
            print(f"Operaciones perdedoras: {lost_trades}")
            print(f"Win Rate: {won_trades/total_trades*100:.1f}%")
            
            # P&L promedio
            won_pnl = trade_analysis.get('won', {}).get('pnl', {})
            lost_pnl = trade_analysis.get('lost', {}).get('pnl', {})
            
            if won_pnl and 'average' in won_pnl:
                avg_win = won_pnl['average']
                print(f"Ganancia promedio: ${avg_win:.2f}")
            
            if lost_pnl and 'average' in lost_pnl:
                avg_loss = abs(lost_pnl['average'])
                print(f"Pérdida promedio: ${avg_loss:.2f}")
                
                if 'average' in won_pnl and avg_loss > 0:
                    print(f"Ratio Ganancia/Pérdida: {won_pnl['average']/avg_loss:.2f}")
        else:
            print("No se ejecutaron operaciones")
    except Exception as e:
        print(f"Error en análisis de trades: {e}")
    
    # Comparación con Buy & Hold
    initial_price = data['Close'].iloc[0]
    final_price = data['Close'].iloc[-1]
    buy_hold_return = (final_price / initial_price - 1) * 100
    
    print(f"\n📊 COMPARACIÓN:")
    print(f"Buy & Hold return: {buy_hold_return:.2f}%")
    print(f"Strategy return: {profit_pct:.2f}%")
    print(f"Outperformance: {profit_pct - buy_hold_return:.2f}%")
    
    # Crear gráfico básico sin mostrar
    try:
        print("\nGenerando gráfico...")
        fig = cerebro.plot(style='candlestick', barup='green', bardown='red', 
                          plotdist=0.1, figsize=(15, 10))
        # Guardar gráfico con ruta absoluta
        import os
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'backtest_results.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {file_path}")
    except Exception as e:
        print(f"Error al generar gráfico: {e}")
    
    return {
        'final_value': final_value,
        'profit': profit,
        'profit_pct': profit_pct,
        'buy_hold_return': buy_hold_return,
        'strategy': strat
    }

# Ejecutar backtest
if __name__ == "__main__":
    try:
        results = run_backtest()
        print("\n✅ Backtest completado exitosamente!")
    except Exception as e:
        print(f"❌ Error durante el backtest: {e}")
        import traceback
        traceback.print_exc()
