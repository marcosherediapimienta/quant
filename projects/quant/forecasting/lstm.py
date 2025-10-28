# -*- coding: utf-8 -*-
"""
LSTM para Señales de Compra/Venta (Buy/Sell) con Sentimiento
-------------------------------------------------------------
Pipeline completo de trading con redes LSTM usando PyTorch:
- Descarga de datos con yfinance
- Features técnicas (RSI, MACD, Bollinger, etc.)
- Features de sentimiento (VADER/FinBERT)
- Etiquetado para clasificación (Buy/Sell/Hold)
- Entrenamiento de LSTM
- Backtest simple

Requisitos:
    pip install torch torchvision numpy pandas yfinance scikit-learn matplotlib
    pip install vaderSentiment transformers sentencepiece  # Opcional: para sentimiento
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise SystemExit("Instala PyTorch: pip install torch torchvision")

# Intentar importar VADER para sentimiento
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False
    print("VADER no disponible. Instala: pip install vaderSentiment")

# Intentar importar FinBERT para sentimiento financiero
try:
    from transformers import pipeline as transform_pipeline
    _HAS_FINBERT = True
except ImportError:
    _HAS_FINBERT = False
    print("FinBERT no disponible. Instala: pip install transformers")

# ================================ CONFIGURACIÓN ================================
# Ticker para descargar datos
TICKER = 'AAPL'  # Puedes cambiar por cualquier ticker de Yahoo Finance

# Parámetros de datos
START_DATE = "2010-01-01"
END_DATE = None  # Hasta hoy

# Parámetros del modelo
LOOKBACK = 60  # Días hacia atrás para predecir (ventana de secuencia)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Ratio de train/test
TRAIN_SIZE = 0.8

# Usar sentimiento (requiere vaderSentiment instalado)
USE_SENTIMENT = True

# Device (GPU si está disponible)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

# ================================ FUNCIONES DE SENTIMIENTO ================================

def fetch_recent_news_yf(ticker: str, start: str, end: str):
    """
    Descarga noticias recientes de Yahoo Finance.
    Nota: yfinance solo proporciona noticias recientes, no histórico profundo.
    
    Args:
        ticker: Ticker del activo
        start: Fecha inicio
        end: Fecha fin
        
    Returns:
        Lista de dicts con 'date' y 'title'
    """
    try:
        stock = yf.Ticker(ticker)
        news = getattr(stock, 'news', [])
        
        items = []
        start_dt = pd.to_datetime(start).date()
        end_dt = pd.to_datetime(end).date() if end else datetime.now().date()
        
        for item in news:
            # Extraer fecha
            pub_time = item.get('providerPublishTime') or item.get('publishTime')
            if pub_time:
                date = datetime.fromtimestamp(pub_time).date()
            else:
                continue
            
            # Filtrar por rango de fechas
            if start_dt <= date <= end_dt:
                title = item.get('title', '')
                if title:
                    items.append({'date': date, 'title': title})
        
        print(f"Noticias encontradas: {len(items)}")
        return items
    
    except Exception as e:
        print(f"Error descargando noticias: {e}")
        return []

def score_sentiment_daily(headlines: list) -> pd.Series:
    """
    Calcula sentimiento diario usando VADER y opcionalmente FinBERT.
    
    Args:
        headlines: Lista de dicts con 'date' y 'title'
        
    Returns:
        Serie con sentimiento diario [-1, 1]
    """
    if not headlines:
        return pd.Series(dtype=float)
    
    # Inicializar analizadores
    vader = SentimentIntensityAnalyzer() if _HAS_VADER else None
    
    finbert = None
    if _HAS_FINBERT and torch.cuda.is_available():
        try:
            finbert = transform_pipeline(
                "sentiment-analysis",
                model="yiyanghkust/finbert-tone",
                device=0
            )
        except Exception:
            finbert = None
    
    # Agrupar por día
    daily_scores = defaultdict(list)
    
    for item in headlines:
        text = item['title'][:300]  # Limitar longitud
        date = item['date']
        
        scores = []
        
        # VADER
        if vader:
            vader_score = vader.polarity_scores(text)['compound']
            scores.append(vader_score)
        
        # FinBERT (si está disponible)
        if finbert:
            try:
                result = finbert(text)[0]
                label = result['label'].upper()
                confidence = result['score']
                
                # Mapear a escala [-1, 1]
                if 'POSITIVE' in label:
                    score = confidence
                elif 'NEGATIVE' in label:
                    score = -confidence
                else:
                    score = 0.0
                
                scores.append(score)
            except Exception:
                pass
        
        # Promedio de todos los scores
        if scores:
            daily_scores[date].append(np.mean(scores))
    
    # Promedio por día
    daily_means = {d: np.mean(vals) for d, vals in daily_scores.items()}
    
    # Convertir a Serie
    sentiment_series = pd.Series(daily_means)
    if not sentiment_series.empty:
        sentiment_series.index = pd.to_datetime(sentiment_series.index)
        sentiment_series = sentiment_series.sort_index()
        sentiment_series.name = 'Sentiment_Daily'
    
    print(f"Sentimiento calculado para {len(sentiment_series)} días")
    return sentiment_series

def add_sentiment_feature(df_prices: pd.DataFrame, ticker: str, start: str, end=None) -> pd.DataFrame:
    """
    Añade feature de sentimiento al DataFrame de precios.
    
    Args:
        df_prices: DataFrame con precios
        ticker: Ticker del activo
        start: Fecha inicio
        end: Fecha fin
        
    Returns:
        DataFrame con columna 'Sentiment_Daily' añadida
    """
    if not USE_SENTIMENT or not _HAS_VADER:
        df_prices['Sentiment_Daily'] = 0.0
        print("Sentimiento deshabilitado o no disponible")
        return df_prices
    
    # Descargar noticias
    headlines = fetch_recent_news_yf(ticker, start, end)
    
    if not headlines:
        df_prices['Sentiment_Daily'] = 0.0
        print("No se encontraron noticias, sentimiento = 0")
        return df_prices
    
    # Calcular sentimiento
    sentiment_series = score_sentiment_daily(headlines)
    
    if sentiment_series.empty:
        df_prices['Sentiment_Daily'] = 0.0
    else:
        # Alinear con índices de precios
        df_prices = df_prices.copy()
        df_prices = df_prices.join(sentiment_series, how='left')
        df_prices['Sentiment_Daily'] = df_prices['Sentiment_Daily'].fillna(0.0)
    
    print(f"Feature 'Sentiment_Daily' añadido (rango: [{df_prices['Sentiment_Daily'].min():.3f}, {df_prices['Sentiment_Daily'].max():.3f}])")
    return df_prices

# ================================ FUNCIONES DE DATOS ================================

def download_data(ticker: str, start: str, end=None) -> pd.DataFrame:
    """
    Descarga datos históricos usando yfinance
    
    Args:
        ticker: Ticker del activo (ej: 'AAPL')
        start: Fecha inicio (ej: '2010-01-01')
        end: Fecha fin (None = hasta hoy)
    
    Returns:
        DataFrame con OHLCV
    """
    print(f"Descargando datos de {ticker} desde {start}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    
    if df.empty:
        raise ValueError(f"No se pudieron descargar datos para {ticker}")
    
    print(f"Datos descargados: {len(df)} observaciones")
    return df

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features técnicas: RSI, MACD, SMA, volatilidad, etc.
    
    Args:
        df: DataFrame con precios OHLCV
    
    Returns:
        DataFrame con features adicionales
    """
    df = df.copy()
    close = df['Close']
    
    # 1. Retornos
    df['Returns'] = close.pct_change()
    df['Log_Returns'] = np.log(close / close.shift(1))
    
    # 2. Media móvil simple (SMA)
    df['SMA_5'] = close.rolling(5).mean()
    df['SMA_10'] = close.rolling(10).mean()
    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()
    
    # 3. RSI (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 5. Bandas de Bollinger
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (close - df['BB_Lower']) / df['BB_Width']
    
    # 6. Volatilidad (desviación estándar de retornos)
    df['Volatility'] = df['Returns'].rolling(20).std() * np.sqrt(252)
    
    # 7. ROC (Rate of Change)
    df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
    
    # 8. Volumen relativo
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # 9. ADX aproximado (Average Directional Index simplificado)
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    atr = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
    df['ADX_plus'] = (plus_dm.rolling(14).mean() / atr) * 100
    df['ADX_minus'] = (minus_dm.rolling(14).mean() / atr) * 100
    
    # Eliminar NaNs iniciales
    df = df.dropna()
    
    print(f"Features técnicas agregadas: {len(df.columns)} columnas")
    return df

def create_labels(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02) -> np.ndarray:
    """
    Crea etiquetas para clasificación:
    - Buy (1): precio sube más de threshold en forward_days
    - Sell (-1): precio baja más de threshold en forward_days
    - Hold (0): resto
    
    Args:
        df: DataFrame con features
        forward_days: Días hacia adelante para evaluar
        threshold: Umbral de cambio de precio (ej: 0.02 = 2%)
    
    Returns:
        Array de etiquetas (-1, 0, 1)
    """
    close = df['Close'].values
    labels = np.zeros(len(close))
    
    for i in range(len(close) - forward_days):
        current_price = close[i]
        future_price = close[i + forward_days]
        
        change_pct = (future_price - current_price) / current_price
        
        if change_pct > threshold:
            labels[i] = 1  # Buy
        elif change_pct < -threshold:
            labels[i] = -1  # Sell
        else:
            labels[i] = 0  # Hold
    
    print(f"Etiquetas creadas - Buy: {np.sum(labels==1)}, Sell: {np.sum(labels==-1)}, Hold: {np.sum(labels==0)}")
    return labels

def prepare_sequences(data: pd.DataFrame, labels: np.ndarray, lookback: int) -> tuple:
    """
    Prepara secuencias para LSTM (sliding window)
    
    Args:
        data: DataFrame con features
        labels: Array de etiquetas
        lookback: Longitud de la ventana
    
    Returns:
        X: Array de secuencias (n_samples, lookback, n_features)
        y: Array de etiquetas
    """
    # Seleccionar features (excluir OHLCV y etiquetas)
    feature_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits', 'Returns']]
    
    if not feature_cols:
        raise ValueError("No se encontraron features adecuadas")
    
    X_data = data[feature_cols].values
    y_data = labels.copy()
    
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_data) - lookback):
        X_sequences.append(X_data[i:i+lookback])
        y_sequences.append(y_data[i+lookback-1])  # Etiqueta del último día de la ventana
    
    return np.array(X_sequences), np.array(y_sequences)

# ================================ CLASES PYTORCH ================================

class LSTMDataset(Dataset):
    """Dataset para PyTorch DataLoader"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y + 1)  # Convertir -1,0,1 a 0,1,2
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    """Clasificador LSTM para señales de trading"""
    
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2, num_classes: int = 3):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Capas fully connected
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Tomar la última salida de la secuencia
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# ================================ ENTRENAMIENTO ================================

def train_model(model, train_loader, val_loader, epochs, device):
    """Entrena el modelo LSTM"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Época [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    
    return model, train_losses, val_losses

# ================================ BACKTEST ================================

def run_backtest(model, test_loader, test_dates, test_prices, device):
    """Ejecuta backtest simple con el modelo entrenado"""
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    predictions = np.array(predictions) - 1  # Convertir 0,1,2 de vuelta a -1,0,1
    
    # Simulación de trading
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    positions = []
    portfolio_values = []
    
    buy_price = 0
    
    for i, pred in enumerate(predictions):
        if i >= len(test_prices):
            break
        
        current_price = test_prices[i]
        
        if pred == 1 and shares == 0 and cash >= current_price:  # Buy signal
            shares = cash / current_price
            buy_price = current_price
            cash = 0
            positions.append(('BUY', test_dates[i], current_price))
        
        elif pred == -1 and shares > 0:  # Sell signal
            cash = shares * current_price
            profit = shares * (current_price - buy_price)
            positions.append(('SELL', test_dates[i], current_price, profit))
            shares = 0
            buy_price = 0
        
        # Calcular valor de cartera
        portfolio_value = cash + shares * current_price
        portfolio_values.append(portfolio_value)
    
    # Finalizar posición si queda abierta
    if shares > 0 and len(test_prices) > 0:
        final_price = test_prices[-1]
        final_value = shares * final_price
        portfolio_values[-1] = final_value
    
    final_value = portfolio_values[-1] if portfolio_values else initial_cash
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    print(f"\n=== RESULTADOS BACKTEST ===")
    print(f"Inversión inicial: ${initial_cash:,.2f}")
    print(f"Valor final: ${final_value:,.2f}")
    print(f"Retorno total: {total_return:.2f}%")
    print(f"Total de señales: Buy={sum(1 for p in positions if p[0]=='BUY')}, Sell={sum(1 for p in positions if p[0]=='SELL')}")
    
    return predictions, positions, portfolio_values

# ================================ VISUALIZACIÓN ================================

def plot_results(df, labels, predictions, portfolio_values):
    """Grafica resultados del trading"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    dates = df.index[LOOKBACK:len(df)-1]
    
    # 1. Precios y señales reales vs predichas
    ax1 = axes[0]
    prices = df['Close'].values[LOOKBACK:len(df)-1]
    ax1.plot(dates, prices, 'b-', label='Precio', linewidth=1.5)
    
    # Señales reales
    real_labels = labels[LOOKBACK:]
    buy_idx = np.where(real_labels == 1)[0]
    sell_idx = np.where(real_labels == -1)[0]
    if len(buy_idx) > 0:
        ax1.scatter(dates[buy_idx], prices[buy_idx], color='green', marker='^', s=100, label='Real: Buy', alpha=0.6)
    if len(sell_idx) > 0:
        ax1.scatter(dates[sell_idx], prices[sell_idx], color='red', marker='v', s=100, label='Real: Sell', alpha=0.6)
    
    # Señales predichas (solo primeras para no saturar)
    pred_buy = np.where(predictions[:len(real_labels)] == 1)[0][::5]
    pred_sell = np.where(predictions[:len(real_labels)] == -1)[0][::5]
    if len(pred_buy) > 0:
        ax1.scatter(dates[pred_buy], prices[pred_buy], color='lightgreen', marker='^', s=50, label='Pred: Buy', edgecolors='green', linewidths=1)
    if len(pred_sell) > 0:
        ax1.scatter(dates[pred_sell], prices[pred_sell], color='salmon', marker='v', s=50, label='Pred: Sell', edgecolors='red', linewidths=1)
    
    ax1.set_ylabel('Precio')
    ax1.set_title(f'Precios y Señales - {TICKER}')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Valores de portfolio
    ax2 = axes[1]
    ax2.plot(dates[:len(portfolio_values)], portfolio_values, 'g-', linewidth=2, label='Valor Portfolio')
    ax2.axhline(y=10000, color='gray', linestyle='--', label='Inversión Inicial')
    ax2.set_ylabel('Valor ($)')
    ax2.set_title('Evolución del Portfolio')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución de señales predichas
    ax3 = axes[2]
    unique, counts = np.unique(predictions[:len(real_labels)], return_counts=True)
    ax3.bar(unique, counts, color=['red', 'gray', 'green'])
    ax3.set_xticks([-1, 0, 1])
    ax3.set_xticklabels(['Sell', 'Hold', 'Buy'])
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Señales Predichas')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('lstm_signals_results.png', dpi=300, bbox_inches='tight')
    print("\nGráfico guardado como: lstm_signals_results.png")
    plt.close()

# ================================ MAIN ================================

def main():
    print("\n=== LSTM PARA SEÑALES DE TRADING ===")
    print(f"Ticker: {TICKER}")
    print(f"Dispositivo: {DEVICE}")
    
    # 1. Descargar datos
    df = download_data(TICKER, START_DATE, END_DATE)
    
    # 1.5 Añadir sentimiento (opcional)
    if USE_SENTIMENT and _HAS_VADER:
        df = add_sentiment_feature(df, TICKER, START_DATE, END_DATE)
    
    # 2. Agregar features técnicas
    df = add_technical_features(df)
    
    # 3. Crear etiquetas
    labels = create_labels(df, forward_days=5, threshold=0.02)
    
    # 4. Preparar secuencias
    X, y = prepare_sequences(df, labels, LOOKBACK)
    
    print(f"\nDatos preparados:")
    print(f"  - Forma de X: {X.shape}")
    print(f"  - Forma de y: {y.shape}")
    print(f"  - Clases: {np.unique(y)}")
    
    # 5. Split train/val/test
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    # Train/Val split temporal
    split_idx = int(n_samples * TRAIN_SIZE)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:n_samples]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    # Crear datasets
    train_dataset = LSTMDataset(X_train, y_train)
    val_dataset = LSTMDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nSplit de datos:")
    print(f"  - Entrenamiento: {len(X_train)} muestras")
    print(f"  - Validación: {len(X_val)} muestras")
    
    # 6. Crear y entrenar modelo
    input_size = X.shape[2]
    model = LSTMClassifier(input_size=input_size, hidden_size=50, num_layers=2, num_classes=3).to(DEVICE)
    
    print(f"\nModelo creado:")
    print(f"  - Features: {input_size}")
    print(f"  - Parámetros totales: {sum(p.numel() for p in model.parameters())}")
    
    print("\nEntrenando modelo...")
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, DEVICE)
    
    # 7. Backtest en validation set
    print("\nEjecutando backtest...")
    test_dates = df.index[LOOKBACK+split_idx:LOOKBACK+n_samples-1].values
    test_prices = df['Close'].values[LOOKBACK+split_idx:LOOKBACK+n_samples-1]
    
    predictions, positions, portfolio_values = run_backtest(model, val_loader, test_dates, test_prices, DEVICE)
    
    # 8. Visualización
    print("\nGenerando gráficos...")
    plot_results(df, labels, predictions, portfolio_values)
    
    print("\n=== COMPLETADO ===")

if __name__ == "__main__":
    main()