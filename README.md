## Abstract
Anomaly detection in live trajectory data is a critical task for ensuring safety, security, and legality in global transport. Traditional anomaly detection methods often struggle with dynamic and evolving trajectory patterns, especially as systems must adapt to new scenarios over time due to increased traffic, geopolitical events or global warming. We propose a continual learning approach to detect anomalous activity in moving vessels. Unlike conventional static models, our method leverages continual learning to enable the model to learn from new data continuously and recognise specific behaviours dependent on position and recent movements.
We implement an adapter-based framework, Continual Learning for AIS Anomalies (CLAISA), that adapts to shifting behavioural environments in transportation, ensuring the system can identify novel and evolving patterns of anomalies, such as deviations from expected routes, irregular speed changes, or unusual local movements. Evaluations on synthetic maritime trajectory datasets spanning sparsely populated waters and heavily trafficked shipping lanes demonstrate that CLAISA achieves up to a $32\%$ decrease in error for trajectory forecasting and consistently outperforms benchmark methods in anomaly detection on synthetically generated datasets.

## Datasets and Preprocessing
The datasets used in the paper are bounded by the below times and spatial coordinates:
| Region              | Lat Range                | Lon Range                 | Min Date     | Max Date     |
|---------------------|--------------------------|----------------------------|--------------|--------------|
| Gulf of Mexico, USA | (27.0°, 30.0°)           | (-90.5°, -87.5°)           | 01/01/2018   | 20/03/2020   |
| Kattegat, Denmark   | (10.0°, 13.0°)           | (55.0°, 58.0°)             | 01/01/2018   | 31/12/2020   |
| North Sea, Denmark  | (6.30°, 9.30°)           | (55.0°, 57.5°)             | 01/01/2018   | 31/12/2020   |
| East Coast, USA     | (29.4°, 32.4°)           | (-81.3°, -78.3°)           | 01/01/2018   | 20/03/2020   |
| Piraeus, Greece     | (37.5°, 38.1°)           | (23.0°, 23.9°)             | 01/01/2018   | 31/12/2019   |
| Ushant, France      | (-6.60°, -3.60°)         | (46.9°, 49.9°)             | 01/01/2019   | 01/07/2019   |

Datasets may be downloaded externally and converted to a pkl file with csv2pkl.py. The pkl file must then be preprocessed with preprocessing.py before model training may begin.

## Model training
main.py begins model training on a preprocessed dataset. Example usage: python main.py --dataset gulfofmexico --adapters 10 --max_seqlen 120 --gpu 0.
