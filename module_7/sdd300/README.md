# SSD300 Model Training and Evaluation Results

## Training Log
Loaded checkpoint from epoch 1946.

Epoch: [1946][0/37] Batch Time 23.750 (23.750) Data Time 12.086 (12.086) Loss 1.9838 (1.9838)
Epoch: [1947][0/37] Batch Time 12.915 (12.915) Data Time 12.226 (12.226) Loss 1.7594 (1.7594)
Epoch: [1948][0/37] Batch Time 12.635 (12.635) Data Time 12.074 (12.074) Loss 1.8648 (1.8648)

Copy

## Evaluation Results
Evaluating: 100%|██████████| 2/2 [00:35<00:00, 17.88s/it]
{'platelets': 0.7960292100906372,
'rbc': 0.812813937664032,
'wbc': 0.9803023338317871}

Mean Average Precision (mAP): 0.863

Copy

### Performance Metrics

| Class      | AP Score   |
|------------|------------|
| Platelets  | 0.796      |
| RBC        | 0.813      |
| WBC        | 0.980      |
| **mAP**    | **0.863**  |



![input](https://github.com/drandule/DataScience/blob/main/module_7/sdd300/BloodImage_00400.jpg)
![predict](https://github.com/drandule/DataScience/blob/main/module_7/sdd300/BloodImage_00400_predict.jpg)

==========================================================================================================

![input](https://github.com/drandule/DataScience/blob/main/module_7/sdd300/BloodImage_00402.jpg)
![predict](https://github.com/drandule/DataScience/blob/main/module_7/sdd300/BloodImage_00402_predict.jpg)

==========================================================================================================

![input](https://github.com/drandule/DataScience/blob/main/module_7/sdd300/BloodImage_00403.jpg)
![predict](https://github.com/drandule/DataScience/blob/main/module_7/sdd300/BloodImage_00403_predict.jpg)

==========================================================================================================

