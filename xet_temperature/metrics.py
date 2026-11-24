import glob
import os
import time

FEATURES = ["cpu_percent", "mem_percent", "load_1min"]


def _cpu_times():
    with open("/proc/stat", encoding="utf-8") as f:
        parts = f.readline().split()[1:]
    values = [int(x) for x in parts]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


def cpu_percent(interval=1.0):
    """CPU usage in percent measured over `interval` seconds."""
    idle1, total1 = _cpu_times()
    time.sleep(interval)
    idle2, total2 = _cpu_times()
    total = total2 - total1
    if total <= 0:
        return 0.0
    return round(100.0 * (1.0 - (idle2 - idle1) / total), 2)


def mem_percent():
    """Used memory in percent."""
    info = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0])
    total = info["MemTotal"]
    available = info.get("MemAvailable", info.get("MemFree", 0))
    return round(100.0 * (total - available) / total, 2)


def read_real_temperature():
    """Return the real sensor temperature in C, or None if no sensor is exposed."""
    paths = glob.glob("/sys/class/thermal/thermal_zone*/temp")
    paths += glob.glob("/sys/class/hwmon/hwmon*/temp*_input")
    readings = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                readings.append(int(f.read().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return round(max(readings), 1) if readings else None


def current_metrics():
    """Dict with the model features measured right now."""
    return {
        "cpu_percent": cpu_percent(),
        "mem_percent": mem_percent(),
        "load_1min": round(os.getloadavg()[0], 2),
    }
