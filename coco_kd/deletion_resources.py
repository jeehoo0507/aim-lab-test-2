"""Read-only machine checks and limits applied only to this training process."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from dataclasses import dataclass


class ResourceStop(RuntimeError):
    """Keep the last completed epoch and allow an explicit restart."""


@dataclass
class Limits:
    gpu_memory_gib: float = 8.0
    min_gpu_free_gib: float = 3.0
    min_ram_available_gib: float = 12.0
    min_disk_free_gib: float = 15.0
    max_temperature_c: float = 78.0
    duty_cycle: float = 0.65
    max_session_minutes: float | None = 120.0
    check_interval_seconds: float = 5.0
    nice: int = 10

    def __post_init__(self):
        if not 0 < self.duty_cycle <= 1 or not 0 < self.max_temperature_c <= 90:
            raise ValueError("Invalid duty/temperature limit")
        if min(self.gpu_memory_gib, self.min_gpu_free_gib, self.min_ram_available_gib,
               self.min_disk_free_gib, self.check_interval_seconds) <= 0:
            raise ValueError("Resource limits must be positive")
        if self.max_session_minutes is not None and self.max_session_minutes <= 0:
            raise ValueError("Session limit must be positive or null (unlimited)")
        if not 0 <= self.nice <= 19:
            raise ValueError("nice must be 0..19")


def memory_info():
    entries = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        entries[name] = int(value.split()[0]) / 1024**2
    return {"total_gib": entries['MemTotal'], "available_gib": entries['MemAvailable']}


def gpu_info(index=0):
    fields = ('index', 'name', 'driver_version', 'memory.total', 'memory.used', 'memory.free',
              'utilization.gpu', 'temperature.gpu', 'power.draw', 'power.limit')
    run = subprocess.run(['nvidia-smi', '--query-gpu='+','.join(fields), '--format=csv,noheader,nounits',
                          '--id='+str(index)], text=True, capture_output=True, timeout=10)
    if run.returncode:
        raise RuntimeError((run.stderr or run.stdout).strip())
    row = next(csv.reader(io.StringIO(run.stdout)))
    raw = dict(zip(fields, [v.strip() for v in row]))
    def number(key):
        try:
            return float(raw[key])
        except (ValueError, KeyError):
            return None
    return {'index': index, 'name': raw['name'], 'driver': raw['driver_version'],
            'total_gib': number('memory.total')/1024, 'used_gib': number('memory.used')/1024,
            'free_gib': number('memory.free')/1024, 'temperature_c': number('temperature.gpu'),
            'utilization_percent': number('utilization.gpu'), 'power_w': number('power.draw'),
            'power_limit_w': number('power.limit')}


def hardware(path, gpu_index=0):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), 'unknown')
    result = {'hostname': os.uname().nodename, 'cpu': cpu, 'logical_cpus': os.cpu_count(),
              'ram': memory_info(), 'disk_free_gib': shutil.disk_usage(path).free/1024**3,
              'disk_path': str(path)}
    try:
        result['gpu'] = gpu_info(gpu_index)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        result['gpu_error'] = str(error)
    return result


def check_resources(info, limits, require_gpu=True):
    errors = []
    if info['ram']['available_gib'] < limits.min_ram_available_gib:
        errors.append('Available RAM is below the configured reserve')
    if info['disk_free_gib'] < limits.min_disk_free_gib:
        errors.append('Free disk space is below the configured reserve')
    if require_gpu:
        gpu = info.get('gpu')
        if not gpu:
            errors.append('GPU telemetry unavailable: '+info.get('gpu_error', 'unknown error'))
        else:
            if gpu['free_gib'] < limits.min_gpu_free_gib:
                errors.append('Free VRAM is below the desktop reserve')
            if gpu['temperature_c'] is None:
                errors.append('GPU temperature telemetry unavailable')
            elif gpu['temperature_c'] >= limits.max_temperature_c:
                errors.append('GPU temperature reached the configured stop threshold')
    return errors


class ResourceGuard:
    def __init__(self, limits, output, device=None):
        self.limits, self.output, self.device = limits, Path(output), device
        self.started = self.last_work = time.monotonic()
        self.last_check = -float('inf')
        self.stop_requested = False
        self.sleep_seconds = 0.0

    def tick(self, force=False):
        now = time.monotonic()
        if self.stop_requested:
            raise ResourceStop('Stop requested; replay unfinished epoch on next run')
        if self.limits.max_session_minutes is not None and now-self.started >= self.limits.max_session_minutes*60:
            raise ResourceStop('Session time limit reached')
        if force or now-self.last_check >= self.limits.check_interval_seconds:
            index = (self.device.index or 0) if self.device is not None else 0
            info = hardware(self.output, index)
            errors = check_resources(info, self.limits, require_gpu=self.device is not None and self.device.type == 'cuda')
            self.last_check = now
            if errors:
                raise ResourceStop('; '.join(errors))

    def pace(self):
        if self.device is not None and self.device.type == 'cuda':
            import torch
            torch.cuda.synchronize(self.device)
        active = max(0, time.monotonic()-self.last_work)
        delay = active*(1/self.limits.duty_cycle-1)
        end = time.monotonic()+delay
        while time.monotonic() < end:
            pause = min(0.5, end-time.monotonic())
            time.sleep(max(0, pause))
            self.sleep_seconds += max(0, pause)
            self.tick()
        self.last_work = time.monotonic()
        self.tick()

    def configure(self):
        if self.device is not None and self.device.type == 'cuda':
            import torch
            info = hardware(self.output, self.device.index or 0)
            errors = check_resources(info, self.limits)
            if errors:
                raise ResourceStop('; '.join(errors))
            gpu = info['gpu']
            cap = min(self.limits.gpu_memory_gib, gpu['free_gib']-self.limits.min_gpu_free_gib)
            if cap < 2:
                raise ResourceStop('Less than 2 GiB remains for this process after desktop reserve')
            torch.cuda.set_per_process_memory_fraction(cap/gpu['total_gib'], self.device)
            major, minor = torch.cuda.get_device_capability(self.device)
            if f'sm_{major}{minor}' not in torch.cuda.get_arch_list():
                raise RuntimeError('Installed PyTorch does not contain this GPU architecture; run setup_deletion.sh env')
        if self.limits.nice > 0:
            current = os.getpriority(os.PRIO_PROCESS, 0)
            if current < self.limits.nice:
                os.nice(self.limits.nice-current)
        self.tick(force=True)


def write_hardware_snapshot(path, output):
    info = hardware(output)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2, ensure_ascii=False)+'\n')
    return info
