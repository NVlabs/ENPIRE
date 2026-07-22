# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MotorCalibrator — torque calibration from collect_torque_data CSV.

Import:
    from enpire.env.forge.robot.systemid.motor_calibrator import MotorCalibrator

    cal = MotorCalibrator("robot/systemid/4310.csv", method="linreg")
    cmd = cal.get_cmd_from_torque(1.5)       # desired Nm → cmd to send
    est = cal.get_torque_from_feedback(0.92) # motor fb Nm → true torque estimate

Methods: linreg | poly2 | poly3 | lut | spline
"""

from pathlib import Path

import numpy as np

METHODS = ("linreg", "poly2", "poly3", "lut", "spline")


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


class MotorCalibrator:
    """Torque calibrator built from a collect_torque_data CSV.

    Fits two mappings:
      get_cmd_from_torque(torque_nm)     — sensor_nm  → cmd_nm
      get_torque_from_feedback(fb_nm)    — fb_nm      → sensor_nm
    """

    def __init__(self, csv_path: Path | str, method: str = "linreg") -> None:
        assert method in METHODS, f"method must be one of {METHODS}"
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        cmd, fb, sensor = data[:, 0], data[:, 1], data[:, 2]
        self.method = method
        self.get_cmd_from_torque, self.get_torque_from_feedback = \
            _fit(method, cmd, fb, sensor, verbose=True)


def _fit(method: str, cmd, fb, sensor, verbose: bool = False):
    """Return (get_cmd_from_torque, get_torque_from_feedback) callables."""

    if method == "lut":
        f1 = _lut_fit(sensor, cmd)
        f2 = _lut_fit(fb, sensor)
        return f1, f2

    if method == "spline":
        from scipy.interpolate import UnivariateSpline
        order = np.argsort(sensor); xs, ys = sensor[order], cmd[order]
        spl1 = UnivariateSpline(xs, ys, k=3, s=len(xs) * np.var(ys) * 0.01)
        order = np.argsort(fb);    xs, ys = fb[order], sensor[order]
        spl2 = UnivariateSpline(xs, ys, k=3, s=len(xs) * np.var(ys) * 0.01)
        if verbose:
            _print_r2("spline", "sensor→cmd", sensor, cmd,    lambda x: spl1(x))
            _print_r2("spline", "fb→sensor",  fb,     sensor, lambda x: spl2(x))
        return (lambda t: float(spl1(t)), lambda f: float(spl2(f)))

    deg = {"linreg": 1, "poly2": 2, "poly3": 3}[method]
    c1 = np.polyfit(sensor, cmd,    deg)
    c2 = np.polyfit(fb,     sensor, deg)
    if verbose:
        _print_r2(method, "sensor→cmd", sensor, cmd,    lambda x: np.polyval(c1, x))
        _print_r2(method, "fb→sensor",  fb,     sensor, lambda x: np.polyval(c2, x))
    return (lambda t, c=c1: float(np.polyval(c, t)),
            lambda f, c=c2: float(np.polyval(c, f)))


def _lut_fit(x, y):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xs_u, idx = np.unique(xs, return_index=True)
    ys_u = np.array([ys[idx[i]:idx[i+1]].mean() if i+1 < len(idx) else ys[idx[i]:].mean()
                     for i in range(len(idx))])
    return lambda v, xs=xs_u, ys=ys_u: float(np.interp(v, xs, ys))


def _print_r2(method, label, x, y_true, fn):
    r2 = _r2(y_true, fn(x))
    print(f"[cal] {method:8s}  {label}  R²={r2:.4f}")

