import os

def download_ticker(ticker, start, end, filename=None):
    """
    Downloads historical stock data for a given ticker symbol from Yahoo Finance.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.
    filename (str): Optional. The name of the file to save the data. If None, no file is saved.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data.
    """
    import yfinance as yf
    import pandas as pd

    # Descargar datos históricos
    data = yf.download(ticker, start=start, end=end)

    # Verificar si hay datos
    if data.empty:
        raise ValueError(f"No se encontraron datos para el ticker {ticker} entre {start} y {end}")

    if filename is not None:
        # Crear la ruta completa en la carpeta data que está un nivel arriba
        path = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
        # Resetear índice para tener 'Date' como columna
        data = data.reset_index()
        # Reordenar columnas de forma estándar
        cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        data = data[cols].copy()
        
        # Formato fecha limpio
        data['Date'] = pd.to_datetime(data['Date']).dt.date

        # Guardar en la ruta especificada
        data.to_csv(path, index=False, header=True)

    return data