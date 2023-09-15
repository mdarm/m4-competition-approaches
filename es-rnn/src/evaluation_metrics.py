import numpy as np


def smape(forecasted, actual):
    """
    Compute Symmetric Mean Absolute Percentage Error (sMAPE)
    """
    return 100 * np.mean(2 * np.abs(forecasted - actual) / (np.abs(forecasted) + np.abs(actual)))


def mase(forecasted, actual):
    """
    Compute Mean Absolute Scaled Error (MASE)
    """
    n = len(actual)
    d = np.abs(np.diff(actual)).sum() / (n - 1)
    errors = np.abs(forecasted - actual)
    return errors.mean() / d


def owa(forecasted, actual, seasonality=1):
    """
    Compute Overall Weighted Average (OWA) based on sMAPE and MASE
    """
    smape_value = smape(forecasted, actual)
    mase_value = mase(forecasted, actual)
    
    naive_forecast = np.roll(actual, seasonality)
    smape_naive = smape(naive_forecast[seasonality:], actual[seasonality:])
    mase_naive = mase(naive_forecast[seasonality:], actual[seasonality:])
    
    owa_value = 0.5 * (smape_value / smape_naive) + 0.5 * (mase_value / mase_naive)

    print('OWA: {} '.format(np.round(owa_value, 3)))
    print('SMAPE: {} '.format(np.round(smape_naive, 3)))
    print('MASE: {} '.format(np.round(mase_naive, 3)))    

    return owa_value 
