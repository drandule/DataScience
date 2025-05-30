# Результаты обучения модели

**Общее количество образцов:** 10 000  
**Разбиение на выборки:**
- Обучающая выборка: 7 000 (70%)
- Валидационная выборка: 1 500 (15%)
- Тестовая выборка: 1 500 (15%)

## Последние эпохи обучения

| Эпоха | Train Loss | Val Loss |
|-------|------------|----------|
| 91/100 | 0.3737 | 0.3764 |
| 92/100 | 0.3737 | 0.3763 |
| 93/100 | 0.3736 | 0.3764 |
| 94/100 | 0.3736 | 0.3764 |
| 95/100 | 0.3736 | 0.3764 |
| 96/100 | 0.3736 | 0.3764 |
| 97/100 | 0.3742 | 0.3765 |
| 98/100 | 0.3737 | 0.3764 |
| 99/100 | 0.3735 | 0.3765 |
| 100/100 | 0.3734 | 0.3765 |

## Результат на тестовой выборке

**Среднее значение IoU (Mean IoU):** `0.4904`

![Loss проекта](https://github.com/drandule/DataScience/blob/main/module_7/unet/loss_plot3.png)

## Для  образца

![Входная](https://github.com/drandule/DataScience/blob/main/module_7/unet/test/00001.png)

![предсказание](https://github.com/drandule/DataScience/blob/main/module_7/unet/prediction/00001.png)
