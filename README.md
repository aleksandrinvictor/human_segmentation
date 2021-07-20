# Human segmentation

## Обучение моделей
1. Распаковать данные в папку data в корне проекта.<br/>
    data/<br/>
        &nbsp; &nbsp; &nbsp; &nbsp;train/<br/>
        &nbsp; &nbsp; &nbsp; &nbsp;train_mask/<br/>
        &nbsp; &nbsp; &nbsp; &nbsp;valid/<br/>
        &nbsp; &nbsp; &nbsp; &nbsp;valid_mask/<br/>
        &nbsp; &nbsp; &nbsp; &nbsp;test/<br/>
1. `docker build -t humseg .`
1. `docker run -v $(pwd):/workspaces/humseg --gpus all --shm-size 8G -it humseg /bin/bash`
1. `bash scripts/train.sh`
1. `bash scripts/predict.sh`

В папке results будут сгенерированы результаты и необходимые файлы.

В папке notebooks лежат ноутбуки exploration.ipynb и solution_description.ipynb с
исследованием данных и описанием решения соответственно.
