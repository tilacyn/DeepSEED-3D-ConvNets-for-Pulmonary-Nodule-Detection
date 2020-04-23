

## 17.04

Use detector<144> as a well trained model

## 18.04

train_6 - random=True

train_5 - random=False

## 19.04

0.025 - loss on luna

## 20.04

trained on luna, tested on luna 

{0: [411, 716, 512, 876, 13346],
 1: [383, 818, 514, 874, 6226],
 2: [313, 862, 501, 887, 2826],
 3: [267, 862, 514, 874, 1333],
 4: [190, 869, 511, 877, 540],
 5: [135, 872, 513, 875, 241],
 5.5: [110, 881, 507, 881, 155],
 6: [83, 852, 534, 854, 100]}
 
 trained on luna, tested on lidc
 
 {-1.5: [137, 13, 143, 157],
 -1: [121, 49, 152, 148],
 -0.5: [90, 90, 163, 137],
 0: [77, 88, 171, 129]}
 
 model path: baseline_1/detector_062.ckpt
 
 ## 22.04
 
 todo:
 
 1. provide real mask radius but not 5
 
 2. try train model on our custom lidc
 
 3. GAN ????
 
 4. roc auc 0.81
 
 5. 