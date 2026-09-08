# Data Quality Report — PhysioNet Apnea-ECG
Records checked: 70 (expected 70)
## Sampling rate
All records expected at 100 Hz. All OK.
## Missing values
No missing-sample sentinel values found in any record.
## Clipping / saturation
Records with clipped/saturated samples:
- a03: 3 clipped samples
- a05: 430 clipped samples
- a08: 8 clipped samples
- a10: 189 clipped samples
- a15: 2563 clipped samples
- a16: 1977 clipped samples
- a17: 32382 clipped samples
- a19: 1235 clipped samples
- a20: 9 clipped samples
- b02: 34117 clipped samples
- b04: 23 clipped samples
- b05: 28772 clipped samples
- c01: 43 clipped samples
- c02: 2 clipped samples
- c04: 25 clipped samples
- c09: 98 clipped samples
- c10: 74 clipped samples
- x01: 30 clipped samples
- x04: 183 clipped samples
- x05: 12 clipped samples
- x06: 54 clipped samples
- x07: 968 clipped samples
- x08: 116 clipped samples
- x10: 975 clipped samples
- x11: 1078 clipped samples
- x12: 49475 clipped samples
- x14: 21 clipped samples
- x16: 4 clipped samples
- x17: 168 clipped samples
- x18: 29 clipped samples
- x20: 83665 clipped samples
- x21: 38 clipped samples
- x22: 28 clipped samples
- x24: 38 clipped samples
- x26: 3 clipped samples
- x27: 20 clipped samples
- x28: 3990 clipped samples
- x29: 592 clipped samples
- x30: 127839 clipped samples
- x31: 1 clipped samples
- x35: 28 clipped samples
## Annotation alignment
Compares number of per-minute A/N annotations to floor(duration / 60s).
Records with mismatch > 1 minute:
- a01: 489 annotations vs 492 expected minutes
- a02: 528 annotations vs 530 expected minutes
- a03: 519 annotations vs 522 expected minutes
- a04: 492 annotations vs 496 expected minutes
- a09: 495 annotations vs 498 expected minutes
- a11: 466 annotations vs 468 expected minutes
- a14: 509 annotations vs 522 expected minutes
- a18: 489 annotations vs 492 expected minutes
- b02: 517 annotations vs 528 expected minutes
- x02: 469 annotations vs 472 expected minutes
- x12: 527 annotations vs 536 expected minutes
- x13: 506 annotations vs 508 expected minutes
- x14: 490 annotations vs 497 expected minutes
- x15: 498 annotations vs 520 expected minutes
- x17: 400 annotations vs 405 expected minutes
- x19: 487 annotations vs 490 expected minutes
- x23: 527 annotations vs 532 expected minutes
- x30: 511 annotations vs 534 expected minutes
## Class balance
Overall: 13064 apnea / 21249 normal minutes (38.1% apnea)
Learning set: 6514 apnea / 10531 normal (38.2% apnea)
Test set: 6550 apnea / 10718 normal (37.9% apnea)

Records with minority class < 10% (highly imbalanced):
- a01: 470 apnea / 19 normal (minority frac 3.9%)
- a04: 453 apnea / 39 normal (minority frac 7.9%)
- a12: 534 apnea / 43 normal (minority frac 7.4%)
- b01: 19 apnea / 468 normal (minority frac 3.9%)
- b04: 10 apnea / 419 normal (minority frac 2.3%)
- c01: 0 apnea / 484 normal (minority frac 0.0%)
- c02: 1 apnea / 501 normal (minority frac 0.2%)
- c03: 0 apnea / 454 normal (minority frac 0.0%)
- c04: 0 apnea / 482 normal (minority frac 0.0%)
- c05: 3 apnea / 463 normal (minority frac 0.6%)
- c06: 1 apnea / 467 normal (minority frac 0.2%)
- c07: 4 apnea / 425 normal (minority frac 0.9%)
- c08: 0 apnea / 513 normal (minority frac 0.0%)
- c09: 2 apnea / 466 normal (minority frac 0.4%)
- c10: 1 apnea / 430 normal (minority frac 0.2%)
- x03: 12 apnea / 453 normal (minority frac 2.6%)
- x04: 0 apnea / 482 normal (minority frac 0.0%)
- x06: 0 apnea / 450 normal (minority frac 0.0%)
- x11: 13 apnea / 444 normal (minority frac 2.8%)
- x17: 1 apnea / 399 normal (minority frac 0.2%)
- x18: 2 apnea / 457 normal (minority frac 0.4%)
- x22: 2 apnea / 480 normal (minority frac 0.4%)
- x24: 1 apnea / 428 normal (minority frac 0.2%)
- x27: 487 apnea / 11 normal (minority frac 2.2%)
- x29: 0 apnea / 470 normal (minority frac 0.0%)
- x31: 516 apnea / 41 normal (minority frac 7.4%)
- x33: 3 apnea / 470 normal (minority frac 0.6%)
- x34: 4 apnea / 471 normal (minority frac 0.8%)
- x35: 0 apnea / 483 normal (minority frac 0.0%)

## Excluded duplicate-session records
The following are **not** independent ECG records — they carry respiration/SpO2 channels for the *same* recording sessions as a01-a04, b01, c01-c03. `a01er` etc. even reference `a01.dat` directly rather than owning separate signal data. They must never be used as additional training/validation/test samples, and must never appear in a split alongside their base record:

- a01r
- a02r
- a03r
- a04r
- b01r
- c01r
- c02r
- c03r
- a01er
- a02er
- a03er
- a04er
- b01er
- c01er
- c02er
- c03er
