# Clinical Data Dictionary: CardioSense AI

The CardioSense AI model is trained on the UCI Heart Disease Dataset. It uses 13 clinical features to predict cardiovascular risk.

---

## 1. Demographic & Primary Vitals

| Feature | Description | Unit | Range (Training) |
| :--- | :--- | :--- | :--- |
| `age` | Patient Age | Years | 29 - 77 |
| `sex` | Biological Sex | 1: Male, 0: Female | 0, 1 |
| `trestbps` | Resting Blood Pressure | mmHg | 94 - 200 |
| `chol` | Serum Cholesterol | mg/dl | 126 - 564 |

---

## 2. Chest Pain & Cardiac Response

### `cp` (Chest Pain Type)
Categorical mapping:
1.  **Typical Angina**: Chest pain related to reduced blood flow to the heart.
2.  **Atypical Angina**: Chest pain not typical of angina.
3.  **Non-anginal Pain**: Typically esophageal.
4.  **Asymptomatic**: No chest pain, but heart disease may still be present.

### `restecg` (Resting Electrocardiographic Results)
Categorical mapping:
0.  **Normal**
1.  **ST-T Wave Abnormality**: T wave inversions and/or ST elevation or depression.
2.  **Left Ventricular Hypertrophy**: Possible or definite LVH by Estes' criteria.

### `thalach` (Maximum Heart Rate Achieved)
Maximum heart rate achieved during exercise stress testing. (Range: 71 - 202 bpm)

### `exang` (Exercise Induced Angina)
Whether exercise triggers chest pain (1: Yes, 0: No).

---

## 3. Electrocardiogram (ECG) Markers

### `oldpeak` (ST Depression)
ST depression induced by exercise relative to rest. (Range: 0.0 - 6.2)

### `slope` (Slope of Peak Exercise ST Segment)
The slope of the ST segment during peak exercise:
1.  **Upsloping**
2.  **Flat**
3.  **Downsloping**

---

## 4. Vascular & Hematologic Markers

### `ca` (Major Vessels)
Number of major vessels (0-3) colored by fluoroscopy.

### `fbs` (Fasting Blood Sugar)
Is fasting blood sugar > 120 mg/dl? (1: True, 0: False)

### `thal` (Thalassemia)
A blood disorder characterized by less hemoglobin and fewer red blood cells than normal:
3.  **Normal**
6.  **Fixed Defect**
7.  **Reversible Defect**

---

## 5. Clinical Safety Thresholds

Our **Safety Engine** monitors these vitals for critical values:
- **Hypertensive Crisis**: `trestbps` > 180 mmHg.
- **Tachycardia Risk**: `thalach` > 190 bpm.
- **Critical ST Depression**: `oldpeak` > 4.0.
