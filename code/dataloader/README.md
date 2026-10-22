# Wind forecasting DataLoader

The loader constructs samples for provincial 24-hour wind-power forecasting:

- input 1: previous 72 hours of wind power, shape `[72, 31]`;
- input 2: next 24 hours of weather forecasts, shape `[24, 31, 6]`;
- target: next 24 hours of wind power, shape `[24, 31]`.

By default, the complete history, forecast origin, and target must all fall
inside 2019. Forecasts are initialized at 08:00 and 20:00 Beijing time.

```python
from dataloader import create_wind_dataloader

train_loader = create_wind_dataloader(batch_size=32, shuffle=True)

for batch in train_loader:
    history_wind = batch["history_wind"]
    future_weather = batch["future_weather"]
    target_wind = batch["target_wind"]
    # model(history_wind, future_weather) -> target_wind
```

Run the built-in data check from the project root:

```powershell
D:\anaconda\envs\torch_cuda\python.exe dataloader\wind_dataset.py
```
