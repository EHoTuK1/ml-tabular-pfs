# Прогноз месячных продаж

HTTP API обучает модель LightGBM на данных Kaggle и прогнозирует месячные продажи для пар `(shop_id, item_id)`.
MLflow хранит параметры, метрики и версии моделей.

## Развёртывание на чистой машине

### 1. Получите код и подготовьте данные

Установите и запустите Docker Desktop в режиме Linux-контейнеров. Проверьте в PowerShell, что
`docker version` выводит раздел `Server`. [Установка Docker Desktop на Windows](https://docs.docker.com/desktop/setup/install/windows-install/)

Склонируйте репозиторий с файлами `Dockerfile` и `.dockerignore` или скопируйте его на компьютер.
Для команды `git clone` понадобится Git.

```powershell
git clone https://github.com/EHoTuK1/ml-tabular-pfs.git
Set-Location ml-tabular-pfs
```

Скачайте архив Kaggle и распакуйте его. [Данные Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data)

Положите четыре файла в одну папку.

- `sales_train.csv`
- `items.csv`
- `item_categories.csv`
- `shops.csv`

В примерах используется `data/landing`. Если CSV лежат в `data/bronze` или в папке вне проекта,
укажите фактический путь в `Resolve-Path` на шаге запуска. Gold-сетка собирается автоматически при обучении.

### 2. Соберите образ

Из корня проекта соберите образ `pfs:local`.

```powershell
docker build -t pfs:local .
```

### 3. Запустите API

```powershell
$csvDir = (Resolve-Path -LiteralPath '.\data\landing').Path

docker run --detach --rm --name pfs-api `
  --publish 127.0.0.1:8000:8000 `
  --mount "type=bind,source=$csvDir,target=/data/source,readonly" `
  --mount "type=volume,source=pfs-state,target=/data/state" `
  --env PFS_DATA_DIR=/data/source `
  --env PFS_STATE_DIR=/data/state `
  pfs:local
```

Docker подключает папку с CSV только для чтения. Том `pfs-state` хранит `mlflow.db`, версии моделей
и снимки данных. После `docker stop pfs-api` контейнер удаляется благодаря флагу `--rm`.
Повторите команду запуска с тем же томом и путём `/data/state`, чтобы сервис загрузил активную модель.

Если API должен быть доступен с другой машины в локальной сети, замените
`--publish 127.0.0.1:8000:8000` на `--publish 8000:8000` и разрешите входящие подключения
к порту `8000` в брандмауэре Windows. С другой машины используйте IP-адрес компьютера с Docker.

## Swagger и проверка API

Откройте [Swagger UI](http://127.0.0.1:8000/docs). Выберите нужный маршрут, нажмите **Try it out**,
при необходимости вставьте JSON и нажмите **Execute**.

1. Выполните `GET /health`. На пустом томе ответ будет `200 OK` с `model_loaded: false`.
2. Отправьте `POST /train`. Для быстрой проверки используйте `{"n_estimators": 2}`.
   Для обучения с параметрами из ноутбука используйте это тело запроса.

   ```json
   {
     "learning_rate": 0.05,
     "num_leaves": 84,
     "n_estimators": 215
   }
   ```

3. Скопируйте `run_id` из ответа `202 Accepted` и подставьте его в `GET /train/{run_id}`.
   Дождитесь `status: succeeded`. После этого `GET /health` вернёт `model_loaded: true`.
4. Отправьте `POST /predict` для одной или нескольких пар.

   ```json
   {
     "items": [
       {"shop_id": 5, "item_id": 5037},
       {"shop_id": 5, "item_id": 5320},
       {"shop_id": 2, "item_id": 5233}
     ]
   }
   ```

   Ответ `200 OK` содержит массив `predictions` с прогнозом `item_cnt_month` для каждой пары.

При ошибке запуска или обучения посмотрите журнал командой `docker logs --tail 100 pfs-api`.
Маршруты, параметры и коды ответов перечислены в контракте. [Контракт API](docs/queries.md)

## Просмотр запусков в MLflow

После запуска API поднимите MLflow UI в отдельном контейнере из того же образа.
Том `pfs-state` подключается к нему только для чтения.

```powershell
docker run --detach --rm --name pfs-mlflow-ui `
  --publish 127.0.0.1:5000:5000 `
  --mount "type=volume,source=pfs-state,target=/data/state,readonly" `
  --env MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS=false `
  pfs:local /app/.venv/bin/mlflow server `
  --backend-store-uri "sqlite:///file:/data/state/mlflow.db?mode=ro&uri=true" `
  --no-serve-artifacts --host 0.0.0.0 --port 5000 --workers 1
```

В [MLflow UI](http://127.0.0.1:5000/) выберите эксперимент `pfs-training`. В нём видны запуски,
гиперпараметры и метрика `validation_rmse`. В разделе зарегистрированных моделей модель `pfs-lightgbm`
хранит версии и алиас `champion`, по которому API загружает активную версию.
Остановить интерфейс можно командой `docker stop pfs-mlflow-ui`.

## Локальный запуск без Docker

Из корня проекта установите зависимости и запустите API с одним воркером.

```powershell
uv sync --locked
$env:PFS_DATA_DIR = (Resolve-Path -LiteralPath '.\data\landing').Path
$env:PFS_STATE_DIR = 'data/service'
uv run uvicorn pfs.service:create_service --factory --host 0.0.0.0 --port 8000 --workers 1
```

Чтобы обучить модель отдельной командой, укажите новый каталог результатов.

```powershell
uv run python -m pfs.training --data-dir data/landing --output-dir data/runs/run-01
```

В нём сохранятся `train.csv` с сеткой Gold, `model.txt` и `metrics.json` с гиперпараметрами,
RMSE и целевым месяцем. Параметры можно переопределить флагами `--learning-rate`, `--num-leaves`
и `--n-estimators`.
