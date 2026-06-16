import random
import sys
import time

# ANSI colors
GREEN  = "\033[92m"
YELLOW = "\033[93m"
ORANGE = "\033[33m"
RED    = "\033[91m"
WHITE  = "\033[97m"
GRAY   = "\033[90m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
CLEAR  = "\033[2J\033[H"

BAR_WIDTH = 40

def color_bar(level):
    """level = 0.0 to 1.0"""
    filled = int(level * BAR_WIDTH)
    empty  = BAR_WIDTH - filled

    # color based on level
    if level < 0.4:
        color = GREEN
    elif level < 0.6:
        color = YELLOW
    elif level < 0.8:
        color = ORANGE
    else:
        color = RED

    bar = color + ("█" * filled) + GRAY + ("░" * empty) + RESET
    return bar, color, level * 100

def status_label(level):
    if level < 0.4:
        return GREEN + "CALM      " + RESET
    elif level < 0.6:
        return YELLOW + "ELEVATED  " + RESET
    elif level < 0.8:
        return ORANGE + "STRESSED  " + RESET
    else:
        return RED    + "DECEPTION " + RESET

cues = [
    ("BLINK RATE ", 0.15),
    ("MICRO EXPR ", 0.10),
    ("LIP COMPRESS", 0.10),
    ("GAZE AVERT ", 0.12),
    ("HEAD MOVEMENT", 0.08),
    ("HEART RATE ", 0.20),
    ("SKIN FLUSH ", 0.10),
    ("JAW TENSION", 0.10),
]

# Simulate rising stress over time
stress_curve = [
    0.1, 0.12, 0.13, 0.15, 0.18, 0.20,   # calm baseline
    0.22, 0.28, 0.33, 0.38,               # slight rise
    0.42, 0.48, 0.52, 0.57,               # yellow zone
    0.61, 0.66, 0.70, 0.73,               # orange zone
    0.78, 0.82, 0.87, 0.91, 0.95,        # red zone
    0.93, 0.89, 0.85,                     # slight drop
    0.91, 0.94, 0.97,                     # peak
]

try:
    for tick, base_stress in enumerate(stress_curve):
        sys.stdout.write(CLEAR)

        print(BOLD + WHITE + "╔══════════════════════════════════════════════════╗" + RESET)
        print(BOLD + WHITE + "║       VIDEO LIE DETECTOR — SIGNAL MONITOR       ║" + RESET)
        print(BOLD + WHITE + "╚══════════════════════════════════════════════════╝" + RESET)
        print()

        # simulate each cue with slight random noise
        cue_levels = []
        for name, sensitivity in cues:
            noise = random.uniform(-0.05, 0.05)
            level = min(1.0, max(0.0, base_stress * (1 + sensitivity * 10) + noise))
            cue_levels.append((name, level))

        for name, level in cue_levels:
            bar, color, pct = color_bar(level)
            label = status_label(level)
            print(f"  {WHITE}{name:<14}{RESET} {bar}  {color}{pct:5.1f}%{RESET}  {label}")

        print()
        print(BOLD + WHITE + "  ─────────────────────────────────────────────────" + RESET)

        # overall score
        overall = min(1.0, base_stress + random.uniform(-0.02, 0.02))
        overall_bar, overall_color, overall_pct = color_bar(overall)
        print(f"  {BOLD}{'OVERALL':<14}{RESET} {overall_bar}  {overall_color}{BOLD}{overall_pct:5.1f}%{RESET}  {status_label(overall)}")
        print(BOLD + WHITE + "  ─────────────────────────────────────────────────" + RESET)
        print()

        # AI note at bottom
        if overall < 0.4:
            note = GREEN + "  ► Subject appears calm. No significant cues detected." + RESET
        elif overall < 0.6:
            note = YELLOW + "  ► Elevated stress detected. Monitoring..." + RESET
        elif overall < 0.8:
            note = ORANGE + "  ► Multiple cues firing. High stress on current question." + RESET
        else:
            note = RED + BOLD + "  ► ALERT: Strong deception indicators. 5+ cues active." + RESET

        print(note)
        print()
        print(GRAY + f"  tick {tick+1}/{len(stress_curve)}   [press CTRL+C to stop]" + RESET)

        time.sleep(0.6)

    print()
    print(GREEN + "  Preview complete." + RESET)

except KeyboardInterrupt:
    print()
    print(GRAY + "  Stopped." + RESET)
