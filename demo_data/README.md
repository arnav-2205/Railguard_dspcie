# Demo files for the RailGuard upload box

Upload either CSV in the app: sidebar → **Upload CSV** → tab **Live demo** → **Start at 0, Play length 75** → **Start**.
Both files have the exact Rail-VIVID format (Timestamp, Channel1-6, Temperature, Humidity, Latitude, Longitude; 2 kHz; 150 000 rows; ~75 s; ~29 MB).

| File | What it is |
|---|---|
| `demo_clean.csv` | Unmodified real data: seconds 40-115 of test run `BtoA_25_3` (2025-04-11). |
| `demo_with_injected_events.csv` | The same excerpt plus 4 **synthetic** impact events (damped 200 Hz ring-down, added to Ch1-4) at 15.05 s (3.2 g), 30.05 s (3.2 g), 45.05 s (1.6 g), 60.05 s (3.2 g). Times are seconds from file start. |
| `expected_output_*.csv` | What the live pipeline produces per segment (time, score, class, healthy channels), for checking the app against. |

## Expected result in the app

`demo_clean.csv`: 291 segments, 288 NORMAL-LIKE, 3 REVIEW RECOMMENDED at t ≈ 66.8-67.3 s (a real, unexplained deviation in the data).

`demo_with_injected_events.csv`: 6 flagged segments (all REVIEW, none HIGH):
* 15.1 s - injected event **detected**
* 59.9 s and 60.2 s - injected event **detected** (the overlapping segments both catch it)
* 30.05 s (3.2 g) and 45.05 s (1.6 g) - **missed** (score stays below 2.69)
* 66.8-67.3 s - the same real deviation as in the clean file

So 2 of the 4 injected events are caught. That matches the measured sensitivity (about 49 % recall at 3.2 g, about 22 % at 1.6 g; decisions.md D16) - the positions were fixed in advance and not tuned to look good.

Thresholds: REVIEW >= 2.69, HIGH >= 5.00. Regenerate with `python scripts/make_demo_data.py`.
