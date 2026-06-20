# Host-only calibration spike for a resonance-based nozzle probe
#
# This module reuses the existing resonance test machinery to vibrate an
# axis and collect accelerometer data, then analyzes it to help choose the
# parameters a future resonance probe would need: the excitation frequency,
# which accelerometer axis responds most strongly, and the amplitude drop
# that occurs on nozzle contact.  It requires NO microcontroller changes -
# all analysis happens on the host after each move completes.
#
# Copyright (C) 2026
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math
from . import shaper_calibrate
from .resonance_tester import (TestAxis, _parse_axis,
                               VibrationPulseTestGenerator,
                               ResonanceTestExecutor)

# Ignore the lowest frequencies when hunting for a resonance peak so the
# DC/drift portion of the spectrum does not win the argmax.
MIN_PEAK_FREQ = 5.

class ResonanceProbeCalibrate:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.config = config
        # Plain frequency-sweep generator (NOT the sweeping-vibrations one):
        # for finding a resonance at a fixed probe point we want only the small
        # fast vibration, not the large slow position sweep that
        # SweepingVibrationsTestGenerator superimposes for input shaping.
        self.generator = VibrationPulseTestGenerator(config)
        self.executor = ResonanceTestExecutor(config)
        self.move_speed = config.getfloat('move_speed', 50., above=0.)
        # Default accelerometer chip (overridable per command via CHIP=)
        self.default_chip = config.get('accel_chip', None)
        # Default fixed-frequency excitation parameters.  Kept deliberately
        # small: peak displacement is ~0.03 * accel_per_hz / freq (mm), and
        # this spike creates nozzle contact by hand, so small motion avoids
        # damaging the nozzle or platform.
        self.accel_per_hz = config.getfloat('accel_per_hz', 10., above=0.)
        self.max_accel_per_hz = config.getfloat('max_accel_per_hz', 30.,
                                                above=0.)
        # Safety: refuse to vibrate along Z by default.  A Z excitation drives
        # the nozzle vertically toward the platform, which is unsafe in the
        # manual (no auto-stop) spike workflow.
        self.allow_z = config.getboolean('allow_z_vibration', False)
        self.gcode = self.printer.lookup_object('gcode')
        self.gcode.register_command(
                "RESONANCE_PROBE_FIND_FREQ", self.cmd_FIND_FREQ,
                desc=self.cmd_FIND_FREQ_help)
        self.gcode.register_command(
                "RESONANCE_PROBE_MEASURE", self.cmd_MEASURE,
                desc=self.cmd_MEASURE_help)

    # -- helpers -----------------------------------------------------------

    # Reject excitation with a Z component unless explicitly allowed, so the
    # nozzle is never driven toward the platform during the manual spike.
    def _check_axis_safety(self, gcmd, axis):
        if axis.get_dir()[2] and not (self.allow_z
                                      or gcmd.get_int("ALLOW_Z", 0)):
            raise gcmd.error(
                "Refusing to vibrate along Z (would drive the nozzle toward"
                " the platform). Use a lateral AXIS, or set ALLOW_Z=1 /"
                " allow_z_vibration only if you understand the risk.")

    def _lookup_chip(self, gcmd):
        chip_name = gcmd.get("CHIP", self.default_chip)
        if not chip_name:
            raise gcmd.error("No accelerometer specified (set CHIP= or"
                             " accel_chip in the config section)")
        chip = self.printer.lookup_object(chip_name.strip(), None)
        if chip is None or not hasattr(chip, 'start_internal_client'):
            raise gcmd.error("'%s' is not a known accelerometer" % (chip_name,))
        return chip

    def _move_to_point(self, gcmd):
        point = gcmd.get("POINT", None)
        if point is None:
            return
        coords = point.split(',')
        if len(coords) != 3:
            raise gcmd.error("POINT must be 'x,y,z'")
        try:
            coords = [float(c.strip()) for c in coords]
        except ValueError:
            raise gcmd.error("POINT must be 'x,y,z' of floats")
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.manual_move(coords, self.move_speed)
        toolhead.wait_moves()

    # Generate a constant-frequency back-and-forth excitation.  Each entry is
    # a constant-acceleration segment of length 0.25/freq with alternating
    # sign, matching the format consumed by ResonanceTestExecutor.run_test.
    def _gen_fixed_freq(self, freq, accel, duration):
        # Each pulse is a (+a, -a) pair with the sign flipping between pulses
        # (as in VibrationPulseTestGenerator), so the toolhead oscillates
        # symmetrically instead of drifting in one direction.
        t_seg = .25 / freq
        n_pulse = max(1, int(round(duration / (2. * t_seg))))
        res = []
        sign = 1.
        t = 0.
        for _ in range(n_pulse):
            t += t_seg
            res.append((t, sign * accel, freq))
            t += t_seg
            res.append((t, -sign * accel, freq))
            sign = -sign
        return res

    # Vibrate along 'axis' using 'test_seq' while collecting accel samples.
    def _vibrate_and_collect(self, gcmd, chip, axis, test_seq):
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.wait_moves()
        toolhead.dwell(0.500)
        aclient = chip.start_internal_client()
        try:
            self.executor.run_test(test_seq, axis, gcmd)
        finally:
            aclient.finish_measurements()
        if not aclient.has_valid_samples():
            raise gcmd.error("accelerometer '%s' measured no data"
                             % (chip.name,))
        return aclient

    # Single-bin DFT: amplitude of the response at exactly 'freq' for each
    # chip axis, using the real (possibly non-uniform) sample timestamps.
    def _amplitude_at_freq(self, samples, freq):
        import numpy as np
        data = np.asarray(samples, dtype=np.float64)
        t = data[:, 0] - data[0, 0]
        ref = np.exp(-2j * np.pi * freq * t)
        n = len(t)
        amps = []   # single-frequency (DFT) amplitude at 'freq'
        envs = []   # broadband envelope mean|x-DC| (what the MCU detector sees)
        for col in (1, 2, 3):
            a = data[:, col] - data[:, col].mean()
            amps.append(2.0 / n * abs(np.sum(a * ref)))
            envs.append(float(np.mean(np.abs(a))))
        return amps, envs  # ([ax,ay,az], [ex,ey,ez])

    # -- commands ----------------------------------------------------------

    cmd_FIND_FREQ_help = ("Sweep frequencies and report the dominant resonance"
                          " peak per accelerometer axis")
    def cmd_FIND_FREQ(self, gcmd):
        chip = self._lookup_chip(gcmd)
        axis = _parse_axis(gcmd, gcmd.get("AXIS", "x").lower())
        self._check_axis_safety(gcmd, axis)
        self._move_to_point(gcmd)
        self.generator.prepare_test(gcmd, is_z=bool(axis.get_dir()[2]))
        test_seq = self.generator.gen_test()
        aclient = self._vibrate_and_collect(gcmd, chip, axis, test_seq)
        helper = shaper_calibrate.ShaperCalibrate(self.printer)
        data = helper.process_accelerometer_data(chip.name, aclient)
        freqs = data.freq_bins
        mask = freqs >= MIN_PEAK_FREQ
        best_axis = None
        best_power = -1.
        for name, psd in (('x', data.psd_x), ('y', data.psd_y),
                          ('z', data.psd_z)):
            masked = psd[mask]
            peak_idx = masked.argmax()
            peak_freq = freqs[mask][peak_idx]
            peak_power = masked[peak_idx]
            gcmd.respond_info(
                "accel %s-axis: peak at %.1f Hz (power %.3g)"
                % (name, peak_freq, peak_power))
            if peak_power > best_power:
                best_power = peak_power
                best_axis = (name, peak_freq)
        gcmd.respond_info(
            "Suggested ACCEL_AXIS=%s, FREQ=%.1f (strongest response to"
            " vibrating %s)" % (best_axis[0], best_axis[1], axis.get_name()))

    cmd_MEASURE_help = ("Vibrate at a fixed frequency and report the steady"
                        " state response amplitude per accelerometer axis")
    def cmd_MEASURE(self, gcmd):
        chip = self._lookup_chip(gcmd)
        axis = _parse_axis(gcmd, gcmd.get("AXIS", "x").lower())
        self._check_axis_safety(gcmd, axis)
        freq = gcmd.get_float("FREQ", above=1., maxval=300.)
        duration = gcmd.get_float("DURATION", 1.0, above=0.1, maxval=10.)
        accel_per_hz = gcmd.get_float("ACCEL_PER_HZ", self.accel_per_hz,
                                      above=0., maxval=self.max_accel_per_hz)
        self._move_to_point(gcmd)
        accel = accel_per_hz * freq
        test_seq = self._gen_fixed_freq(freq, accel, duration)
        aclient = self._vibrate_and_collect(gcmd, chip, axis, test_seq)
        output = gcmd.get("OUTPUT", None)
        if output:
            aclient.write_to_file(output)
            gcmd.respond_info("Wrote raw accelerometer data to %s" % (output,))
        amps, envs = self._amplitude_at_freq(aclient.get_samples(), freq)
        label = gcmd.get("LABEL", "")
        prefix = ("%s: " % label) if label else ""
        gcmd.respond_info(
            "%sDFT@%.1fHz x=%.2f y=%.2f z=%.2f | envelope(mean|x-DC|)"
            " x=%.2f y=%.2f z=%.2f"
            % (prefix, freq, amps[0], amps[1], amps[2],
               envs[0], envs[1], envs[2]))
        gcmd.respond_info(
            "Compare baseline vs contact: DFT = clean single-frequency signal,"
            " envelope = what the MCU detector currently uses")

def load_config(config):
    return ResonanceProbeCalibrate(config)
