Источник - https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales

Данные распаковать и поместить файлы .csv в ml-tabular-pfs/data/landing

## Обучение LightGBM

В каталоге с данными нужны `sales_train.csv`, `items.csv`, `item_categories.csv` и `shops.csv`.
Перед каждым обучением команда заново собирает месячную сетку Gold из транзакций.

```shell
uv run python -m pfs.training --data-dir data/landing --output-dir data/runs/run-01
```

Укажите новый каталог для каждого запуска. В нём сохраняются `train.csv` с сеткой Gold,
`model.txt` с моделью LightGBM и `metrics.json` с гиперпараметрами, RMSE на последнем
историческом месяце и номером целевого месяца прогноза.
Гиперпараметры можно переопределить флагами `--learning-rate`, `--num-leaves` и `--n-estimators`.

## HTTP-сервис и реестр моделей

Для локального запуска используйте один рабочий процесс API. Переменная `PFS_DATA_DIR`
задаёт каталог исходных CSV (по умолчанию `data/landing`), а `PFS_STATE_DIR` задаёт
каталог SQLite и артефактов MLflow (по умолчанию `data/service`).

```shell
uv run uvicorn pfs.service:create_service --factory --host 0.0.0.0 --port 8000 --workers 1
```

Откройте `http://localhost:8000/docs` и вызовите `POST /train`, например с
`{"n_estimators": 2}` для пробного запуска. Состояние доступно по `GET /train/{run_id}`.
Успешный запуск публикует модель и снимок данных в MLflow и назначает алиас `champion`.
Сервис использует этот алиас при следующем старте. Контракты описаны в [docs/queries.md](docs/queries.md).
