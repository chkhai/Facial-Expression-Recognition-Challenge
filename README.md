# Facial Expression Recognition — FER-2013

## პროექტის მოკლე აღწერა
პროექტის მიზანია ადამიანის სახის გამოსახულებიდან ემოციის დადგენა.
ამოცანა არის მულტი კლასის კლასიფიკაციის პრობლემა. ინფუთი არის 48x48 ფოტო და სულ გვაქვს 7 ემოცია(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

## პროექტის სტრუქტურა

```
├── model_experiment.ipynb   # მთავარი ნოუთბუქი
├── submission.csv           # Test set predictions
├── best_residualcnn.pth     # საუკეთესო მოდელის weights
└── README.md               
```
მხოლოდ ერთი ნოუთბუქ ფაილი მაქვს, რადგან კეგლზე საბმიშენს ვერ ვაკეთებდი ამაშივე დავაგენერირე submission.csv და inference-ის ნოუთბუქი აღარ შევქმენი

## შეფასების მეტრიკა

ტრენინგოს დროს ვაკვირდებოდი train/val loss, train/val accuracy, overfit gap(train acc - val acc), საბოლოო მოდელის შესაფასებლად კი გამოვიყენე confusion matrix და თითოეული კლასისთვის Precision, Recall, F1.


## EDA

### მონაცემების სტრუქტურა

ქომფეთიშენის დატასეტში არის 35,887 სტრიქონი და 3 სვეტი(emotion, pixels, usage):

გამოვიყენე `train.csv` და `test.csv` ფაილები:

| Split      | Rows   |
|------------|--------|
| Train      | 28,709 |
| Test       | 7,178  |

### Class Distribution

Class distribution შევამოწმე ტრეინ სეტზე:
<img width="821" height="357" alt="image" src="https://github.com/user-attachments/assets/c2d73d57-941e-4c54-8ccd-34f0115d470e" />


| Emotion  | Count | % |
|----------|-------|---|
| Angry    | 3,995 | 13.91% |
| Disgust  | 436   | 1.52% |
| Fear     | 4,097 | 14.27% |
| Happy    | 7,215 | 25.13% |
| Sad      | 4,830 | 16.82% |
| Surprise | 3,171 | 11.05% |
| Neutral  | 4,965 | 17.29% |

ტრეინ სეტი დაბალანსებული არაა. Disgust კლასი Happy კლასზე 16.5-ჯერ ნაკლებია. ეს ნიშნავს, რომ მხოლოდ Accuracy ყოველთვის საკმარისი არ არის მოდელის შესაფასებლად.

### კლასების დაბალანსება — Class Weights

იმის გამო, რომ Disgust კატეგორია ძალიან ცოტაა, გამოვიყენე inverse-frequency class weights CrossEntropyLoss-ში:

| Emotion  | Weight |
|----------|--------|
| Angry    | 0.4477 |
| Disgust  | 4.5090 |
| Fear     | 0.4412 |
| Happy    | 0.2426 |
| Sad      | 0.3639 |
| Surprise | 0.6426 |
| Neutral  | 0.3530 |

ყოველი Disgust შეცდომა 4.5-ჯერ მეტად ისჯება ვიდრე Happy შეცდომა. ამან საბოლოოდ Disgust კლასზე 72.73% recall მოგვცა, მიუხედავად იმისა, რომ ის ტრეინ სეტის მხოლოდ 1.4%-ია.

### Duplicates და Data Quality

- **1,236 duplicate image** ამოვიღეთ pixel string matching-ით
- ყველა სურათი ზუსტად 48×48 (2,304 pixel)
- Missing values არ არის

### Pixel Values და Normalization

პიქსელების მნიშვნელობები [0, 255] შუალედშია. 1,000 random sample-ზე გამოვთვალეთ:

- Mean: `0.5077`
- Std: `0.2550`

ეს მნიშვნელობები ნორმალიზაციისთვის გამოვიყენე და უბრალოდ 0.5/0.5 არ ვქენი.

### Train/Validation Split
შემდეგ ტრეინ სეტი დავყავი ტრეინად და ვალიდაციად.
Stratified split 80/20:

| Split      | Samples |
|------------|---------|
| Train      | 21,978  |
| Validation | 5,495   |

Stratify პარამეტრი უზრუნველყოფს, რომ ორივე split-ში კლასების პროპორცია იდენტური იყოს.

---

## Data Augmentation

ტრეინინგზე მხოლოდ:

| Augmentation           | Value                        |
|------------------------|------------------------------|
| RandomHorizontalFlip   | p=0.5                        |
| RandomRotation         | ±10 degrees                  |
| RandomCrop             | 48×48 with padding=4         |
| ColorJitter            | brightness=0.2, contrast=0.2 |
| Normalize              | mean=0.5077, std=0.2550      |

ვალიდაციაზე და ტესტზე augmentation არ გამოვიყენე — მხოლოდ ნორმალიზაცია.

---

## Training Setup

| Setting        | Value                                      |
|----------------|--------------------------------------------|
| Batch size     | 64                                         |
| Optimizer      | Adam (lr=1e-3, weight_decay=1e-4)         |
| Loss           | CrossEntropyLoss + class weights           |
| Scheduler      | ReduceLROnPlateau (patience=3, factor=0.5) |
| Early Stopping | patience=7, min_delta=0.001               |
| Device         | Kaggle GPU T4                              |

---

## ტრენინგი
გავტესტე სხვადასხვა არქიტექტურები. თავიდან დავიწყე მარტივი CNN-ით და თანდათან გავართულე.

### TinyCNN | Val Acc: 41.02%

**არქიტექტურა:** 2 conv block, no BatchNorm, **196,167 params**

```
Conv(1→16) → ReLU → MaxPool
Conv(16→32) → ReLU → MaxPool
Flatten → Linear(4608→128) → ReLU → Dropout(0.5) → Linear(128→7)
```

**ანალიზი:**
პირველი baseline მოდელი. 5 ეპოქაზე გავუშვი CPU-ზე უბრალოდ გასატესტად. 41% accuracy მიიღო. მოდელი ანდერფიტშია, ის ძალიან მარტივია ემოციების ქომფლექსითის ამოსაცნობად. 2 conv block-ს კიდეების და რაღაც ტექსტურის სწავლა შეუძლია, მაგრამ ემოციებისთვის საჭიროა ადამიანის სახის ორგანოებს შორის (თვალები, პირი, წარბები) კანონზომიერების დაჭერა, რაც მეტ სიღრმეს მოითხოვს.

**Early Stopping:** არ გამომიყენებია აქ, მხოლოდ 5 ეპოქაზე გავრანე.

---

### DeepCNN | Val Acc: 52.81%

**არქიტექტურა:** 4 conv block, no BatchNorm, **1,571,591 params**

```
Conv(1→32) → ReLU → MaxPool
Conv(32→64) → ReLU → MaxPool
Conv(64→128) → ReLU → MaxPool
Conv(128→256) → ReLU → MaxPool
Flatten → Linear(2304→512) → ReLU → Dropout(0.5) → Linear(512→7)
```

**ანალიზი:**
სიღრმის გაზრდამ +11.79% accuracy მოგვცა TinyCNN-თან შედარებით. 30 ეპოქა გავუშვი, early stopping გამოვიყენე მაგრამ ტრენინგი არ შეწყდა, რადგან  მოდელი ბოლო ეპოქამდე უმჯობესდებოდა. train accuracy (51%) და val accuracy (52%) თითქმის იდენტურია, რაც ნიშნავს რომ მოდელი კვლავ underfitting-შია — BatchNorm-ის გარეშე gradient flow 4 block-ის გავლით არასტაბილურია.

---

### DeepCNN_BN | Val Acc: 62.04%

**არქიტექტურა:** 4 conv block + BatchNorm after every conv, **1,573,575 params**

```
[Conv → BatchNorm → ReLU → MaxPool] × 4
Flatten → Linear → BatchNorm → ReLU → Dropout(0.5) → Linear(→7)
```

**ანალიზი:**
ბეჩ ნორმალიზაციის დამატებამ +9.23% მოგვცა უბრალო DeepCNN-თან შედარებით, იგივე პარამეტრების რაოდენობით. BatchNorm ანორმალიზებს აქტივაციებს ლეიერებს შორის და აბალანსებს შიდა კოვარიაციულ წანაცვლებას. ეს კი გრადიენტის ნაკადის სტაბილიზაციას უზრუნველყიფს. კიდევ ერთი დადებითი ეფექტია კონვერგენციის სისწრაფე — DeepCNN-მა 20 ეპოქაზე მიაღწია 52%-ს, DeepCNN_BN-მა კი 4 ეპოქაზე. LR 28-ე ეპოქაზე შემცირდა და მაშინვე +2% accuracy მოგვცა, რაც ნიშნავს, რომ 30 ეპოქა საკმარისი არ იყო. early stopping აქაც არ გამოყენებულა.

---

### ResidualCNN | Val Acc: 64.26% (Best Scratch Model)

**არქიტექტურა:** 4 downsampling block + 3 residual block + BatchNorm, **1,961,991 params**

```
Entry: Conv(1→32) → BN → ReLU → MaxPool
ResidualBlock(32) → Down(32→64) → MaxPool
ResidualBlock(64) → Down(64→128) → MaxPool
ResidualBlock(128) → Down(128→256) → MaxPool
Flatten → Linear(2304→512) → BN → ReLU → Dropout(0.5) → Linear(512→7)
```

ResidualBlock:
```
x + [Conv → BN → ReLU → Conv → BN] → ReLU
```

**ანალიზი:**
residual cnn-მა +2.6% მოგვცა DeepCNN_BN-თან შედარებით. სქიფ ქონექშენები გრადიენტს პირდაპირ გზას უხსნის, რაც ადრეულ ლეიერებს უფრო ეფექტური სწავლის საშუალებას აძლევს. LR ორჯერ შემცირდა (epoch 30 და 37) და ყოველ ჯერზე ახალი best accuracy-ი დაფიქსირდა. Early stopping epoch 39-ზე გამოიყენა.

**Overfitting:** ნორმ — train (67%) vs val (64%), 3% gap
**Early Stopping:** დიახ — epoch 39-ზე

---

### ResidualCNN2 | Val Acc: 64.29%

**არქიტექტურა:** გაფართოებული ResidualCNN (48→96→192→256 channels), **4,020,871 params**

**ანალიზი:**
ResidualCNN-ზე 2-ჯერ მეტი პარამეტრი, მაგრამ შედეგი ოდნავ უარესი. ეს პირდაპირ ადასტურებს, რომ FER2013 (22K training image) ზედმეტად პატარაა 4M+ პარამეტრის მქონე მოდელისთვის — მეტმა პარამეტრმა პირიქით უფრო ოვერფიტი გამოიწვია და accuracy არ გაუმჯობესდა. 

**Overfitting:** მეტი ვიდრე ResidualCNN
**Early Stopping:** არ გამოიყენა

---

### ResidualCNN_NoDrop | Val Acc: 62.60%

**არქიტექტურა:** ResidualCNN without Dropout, **1,961,991 params**

**ანალიზი:**
დროფაუთის გარეშეც გავტესტე იგივე არქიტექტურა. დროფაუთის ამოღებამ -2.04% accuracy მოგვცა. Train/val gap გაიზარდა 3%-დან 5.3%-მდე, რაც ოვერფიტინგის გაზრდაზე მიუთითებს. BatchNorm მცირე რეგულარიზაციას იძლევა, მაგრამ ეს არ არის საკმარისი ამ დატასეტისთვის. Early stopping epoch 31-ზე გამოიყენა — 8 ეპოქით ადრე ვიდრე ResidualCNN-ზე, რაც ნიშნავს რომ მოდელმა უფრო ადრე შეწყვიტა გაუმჯობესება.

**Overfitting:** მეტი — train (67%) vs val (62%), 5.3% gap
**Early Stopping:** დიახ — epoch 31-ზე

---

### GAP_CNN | Val Acc: 58.11%

**არქიტექტურა:** 4 conv block + Global Average Pooling, **422,855 params**

```
[Conv → BN → ReLU → MaxPool] × 3
Conv → BN → ReLU (no pool)
GlobalAveragePool → Flatten
Linear(256→128) → BN → ReLU → Dropout(0.5) → Linear(128→7)
```

**ანალიზი:**
ყველაზე ეფექტური მოდელი პარამეტრების თვალსაზრისით. GAP ანაცვლებს დიდ FC layer-ს (2304→512, ~1.2M param) პატარათი (256→128, ~33K param), რის გამოც მთლიანი მოდელი მხოლოდ 422K პარამეტრს შეიცავს. მიუხედავად ამისა, DeepCNN-ს (1.57M) 4-ჯერ ნაკლები პარამეტრებით 5.3%-ით სჯობია. Val loss არასტაბილური იყო, ეპოქებს შორის ჰქონდა ხტომები, რაც მიუთითებს, რომ მოდელი LR-ისადმი სენსიტიურია.

**Overfitting:** მინიმალური — პატარა მოდელი
**Early Stopping:** არ გამოიყენა

---

### MultiScaleCNN | Val Acc: 63.40%

**არქიტექტურა:** Parallel 1×1, 3×3, 5×5 kernels concatenated, **1,880,039 params**

```
Entry Conv(1→32) → MaxPool
MultiScaleBlock(32→96): [1×1 branch | 3×3 branch | 5×5 branch] → concat
MaxPool
MultiScaleBlock(96→192): [1×1 branch | 3×3 branch | 5×5 branch] → concat
MaxPool
Conv(192→256) → MaxPool
Flatten → Linear → BN → ReLU → Dropout(0.5) → Linear(→7)
```

**ანალიზი:**
Parallel kernel sizes-ი Inception არქიტექტურის მთავარი იდეაა. 1x1 kernel არხების მიქსინგს შვება, 3x3 ლოკალურ ფიჩერებს სწავლობს, ხოლო 5x5 უფრო ფართო კონტესქტს ხედავს. ემოციების ამოცნობა ერთდროულად მოითხოვს როგორც წვრილი დეტალებს (მაგ. ტუჩის კუთხეები), ისე ზოგად სქრუქტურებს (მაგ. პირის ფორმა), რის გამოც ეს multi-scale მიდგომა სასარგებლოა. ამ მოდელმა DeepCNN_BN-ს 1.4%-ით აჯობა (62%-იანი სიზუსტით), თუმცა ResidualCNN-ის შედეგის გაუმჯობესება ვერ შეძლო. LR 35-ე ეპოქაზე შემცირდა, ხოლო best accuracy 36-ე ეპოქაზე დაფიქსირდა.

**Overfitting:** ნორმ — train (66%) vs val (63%)
**Early Stopping:** არ გამოიყენა

---

## Pretrained მოდელები
გავტესტე პრეტრეინ მოდელებიც, მაგრამ რატომღაც კარგი შედეგი ვერ მივიღე;(

### ResNet18

რადგან ჩვენი dataset-ის გამოსახულებები 1x48x48 ფორმატისაა, ხოლო ResNet18-ის input-ი 3x224x224 ზომას მოითხოვს, conv1 შევცვალეთ 3->1 არხისთვის (რანდომ ინიციალიზაციით), ხოლო ფინალური FC ლეიერი გადავაკეთე 7 კლასზე.

| Config | Val Acc |
|--------|---------|
| Frozen backbone | 26.72% |
| Full finetune (Adam lr=1e-4) | 59.60% |
| Full finetune (AdamW layerwise) | 59.34% |

**ანალიზი:** Frozen backbone - მა ძალიან ცუდი შედეგი მოგვცა(26.72%). მიზეზი: conv1 შეიცვალა რანდომ წონებით, მაგრამ backbone frozen დარჩა. შედეგად, random input layer სიგნალს გადასცემს frozen pretrained features-ს, რაც ნიშნავს, რომ ბექბოუნი ფაქტობრივად უსარგებლო ინფორმაციას ამუშავებს.

Full finetune-მაც კი ვერ აჯობა აქამდე საუკეთესო მოდელის შედეგს (64.26%). ამის სამი ძირითადი მიზეზი არსებობს

1. **Domain gap** — ImageNet: diverse 3-channel high-res objects; FER2013: 48×48 grayscale faces
2. **Grayscale conversion** — conv1 random weights-ით შეცვლა pretrained low-level features-ს ანადგურებს
3. **Resolution mismatch** — ResNet18 224×224-ისთვის არის შემუშავებული, 48×48 ძალიან პატარაა

**Early Stopping:** epoch ~30-ზე

### EfficientNet-B0

| Config | Val Acc |
|--------|---------|
| Full finetune (Adam lr=1e-4) | 54.27% |

**ანალიზი:** კიდევ სუსტი შედეგი. EfficientNet compound scaling 224×224-ზე არის ოპტიმიზირებული. 48×48-ზე spatial feature maps ძალიან სწრაფად კოლაფსდება ქსელში და ინფორმაციის უმეტესობა კლასიფიკატორამდე ვეღარ აღწევს.

---

## საერთო შედარება

| Rank | Model | Params | Val Acc |
|------|-------|--------|---------|
| 🥇 | ResidualCNN | 1.96M | 64.26% |
| 🥈 | ResidualCNN2 | 4.02M | 64.29% |
| 🥉 | MultiScaleCNN | 1.88M | 63.40% |
| 4 | ResidualCNN_NoDrop | 1.96M | 62.60% |
| 5 | DeepCNN_BN | 1.57M | 62.04% |
| 6 | ResNet18 (full finetune) | 11M | 59.60% |
| 7 | GAP_CNN | 422K | 58.11% |
| 8 | DeepCNN AdamW | 1.57M | 54.87% |
| 9 | EfficientNet-B0 | 5.3M | 54.27% |
| 10 | DeepCNN | 1.57M | 52.81% |
| 11 | TinyCNN | 196K | 41.02% |

---

## საბოლოო მოდელი და შედეგები

**საბოლოო მოდელად ავირჩიე ResidualCNN:**
- Adam optimizer, lr=1e-3, weight_decay=1e-4
- ReduceLROnPlateau scheduler
- Dropout=0.5 + BatchNorm
- Class weights imbalance-ისთვის
- Early stopping patience=7

**PrivateTest შედეგები (3,589 image):**

| Emotion  | Precision | Recall | F1   | Accuracy |
|----------|-----------|--------|------|----------|
| Angry    | 0.59      | 0.51   | 0.55 | 51.12%   |
| Disgust  | 0.40      | 0.73   | 0.52 | 72.73%   |
| Fear     | 0.45      | 0.42   | 0.44 | 41.86%   |
| Happy    | 0.89      | 0.84   | 0.86 | 83.85%   |
| Sad      | 0.53      | 0.50   | 0.51 | 49.66%   |
| Surprise | 0.72      | 0.82   | 0.77 | 82.21%   |
| Neutral  | 0.61      | 0.69   | 0.65 | 69.49%   |
| **Overall** | **0.65** | **0.65** | **0.65** | **64.67%** |

**Confusion Matrix:**

<img width="570" height="586" alt="image" src="https://github.com/user-attachments/assets/987e8b8c-0897-4e94-9f40-46f5c80f1ea4" />

- Happy და Surprise ყველაზე მარტივად ამოსაცნობი — ვიზუალურად მკვეთრად განსხვავებული
- Fear ყველაზე რთული (41.86%) — ხშირად ერევა Sad და Angry-ში
- Sad ხშირად ერევა Neutral-ში — სახის გამომეტყველება ძალიან ჰგავს
- Disgust-ს მაღალი recall (72.73%) class weights-ის წყალობით

---

## ძირითადი დასკვნები

**რამ გაამართლა:**
- **BatchNorm** იყო ყველაზე დიდი single improvement (+9.23%). გრადიენტის ფლოუს სტაბილიზაცია გადამწყვეტია ღრმა networks-ისთვის
- **Residual connections** დაამატეს +2.6% BatchNorm-ზე — skip connections ადრეული layers-ის სწავლას ამარტივებს
- **Class weights** Disgust recall-ს 72.73%-მდე მიიყვანეს მიუხედავად 1.4% representation-ისა
- **Data augmentation** გამოვიყენეთ ყველა მოდელის ტრენინგში (horizontal flip, rotation, crop, color jitter). DeepCNN-ზე no augmentation variant-ი ვცადეთ და train/val gap მნიშვნელოვნად გაიზარდა, რაც augmentation-ის სასარგებლო ეფექტზე მიუთითებს

**ყველაზე ეფექტური მოდელი:**
მიუხედავად იმისა რომ საუკეთესო შედეგი residual_cnn-მა აჩვენა, GAP_CNN 422K პარამეტრით 58.11% accuracy-ს აღწევს — DeepCNN-ს (1.57M) 4-ჯერ ნაკლები პარამეტრებით აჯობა. Global Average Pooling-ი ძალიან ეფექტური არჩევანია პარამეტრების შეზღუდვისას.

---

## Experiment Tracking

ყველა ექსპერიმენტი დალოგილია [wandb-ზე](https://wandb.ai/chkhai-free-university-of-tbilisi-/fer2013-experiments?nw=nwuserchkhai).

თითოეული run-ი შეიცავს Train/val loss, accuracy per epoch, Learning rate schedule

---
