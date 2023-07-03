#! /bin/bash

awk 'BEGIN { ls = 100; } { arr[(NR % ls)] = $6; base[(NR % ls)] = $8; steps[(NR % ls)] = $10; remainings[(NR % ls)] = $12; if ((NR % ls) == 0) { sum = 0; sumBase = 0; sumSteps = 0; sumRemainings = 0; for (i = 0; i < ls; i += 1) { sum += arr[i]; sumBase += base[i]; sumSteps += steps[i]; sumRemainings += remainings[i]; } print "episodes:", NR*50, "loss:", sum / ls, "baseline loss:", sumBase / ls, "steps: ", sumSteps / ls, "remainings:", sumRemainings / ls; } }' a
