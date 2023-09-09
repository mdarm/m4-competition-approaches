import numpy as np
import torch
import torch.nn as nn


class HoltWintersNoTrend(nn.Module):
    
    def __init__(self, init_a=0.1, init_g=0.1, slen=12):
        super(HoltWintersNoTrend, self).__init__()
        
        # Smoothing parameters
        self.alpha = nn.Parameter(torch.tensor(init_a))
        self.gamma = nn.Parameter(torch.tensor(init_g))
        
        # Initial parameters
        self.init_season = nn.Parameter(torch.tensor(np.random.random(size=slen)))
        
        # Season length used to pick appropriate past season step
        self.slen = slen
        
        # Sigmoid used to normalize the parameters to be between 0 and 1 if needed
        self.sig = nn.Sigmoid()
        
    def forward(self, series, series_shifts, n_preds=8, rv=False):
        batch_size = series.shape[0]
        init_season_batch = self.init_season.repeat(batch_size).view(batch_size, -1)
        
        # Use roll to allow for our random input shifts
        seasonals = torch.stack([torch.roll(j, int(rol)) for j, rol in zip(init_season_batch, series_shifts)]).float()
        
        # Convert to a list to avoid inplace tensor changes
        seasonals = [x.squeeze() for x in torch.split(seasonals, 1, dim=1)]
        
        result = []
        
        if rv:
            value_list = []
            season_list = []
        
        for i in range(series.shape[1] + n_preds):
            if i == 0:
                smooth = series[:, 0]
                result.append(smooth)
                if rv:
                    value_list.append(smooth)
                    season_list.append(seasonals[i % self.slen])
            else:
                smooth_prev = smooth
                season_prev = seasonals[i % self.slen]
                smooth = self.alpha * (series[:, i] - season_prev) + (1 - self.alpha) * smooth_prev
                seasonals.append(self.gamma * (series[:, i] - smooth) + (1 - self.gamma) * season_prev)
                
                result.append(smooth + seasonals[i % self.slen])
                
                if rv:
                    value_list.append(smooth)
                    season_list.append(seasonals[i % self.slen])
        
        if rv:
            return torch.stack(result, dim=1), torch.stack(value_list, dim=1), torch.stack(season_list, dim=1)
        else:
            return torch.stack(result, dim=1)[:, -n_preds:]


class ESRNN(nn.Module):
    
    def __init__(self, hidden_size=16, slen=12, pred_len=12):
        super(ESRNN, self).__init__()
        
        self.hw = HoltWintersNoTrend(init_a=0.1, init_g=0.1)
        self.rnn = nn.GRU(hidden_size=hidden_size, input_size=1, batch_first=True)
        self.lin = nn.Linear(hidden_size, pred_len)
        self.pred_len = pred_len
        self.slen = slen
        
    def forward(self, series, shifts):
        batch_size = series.shape[0]
        result, smoothed_value, smoothed_season = self.hw(series, shifts, rv=True, n_preds=0)
        
        de_season = series - smoothed_season
        de_level = de_season - smoothed_value
        noise = torch.randn(de_level.shape[0], de_level.shape[1])
        noisy = de_level  # +noise
        noisy = noisy.unsqueeze(2)
        
        feature = self.rnn(noisy)[1].squeeze()
        pred = self.lin(feature)
        
        season_forecast = [smoothed_season[:, i % self.slen] for i in range(self.pred_len)]
        season_forecast = torch.stack(season_forecast, dim=1)
        
        return smoothed_value[:, -1].unsqueeze(1) + season_forecast + pred
