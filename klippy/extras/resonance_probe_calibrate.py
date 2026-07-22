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
import bisect, math, random
from . import shaper_calibrate
from .resonance_probe import HaltingContactProbe, _gen_fixed_freq, \
    _plain
from .resonance_tester import (TestAxis, _parse_axis,
                               VibrationPulseTestGenerator,
                               ResonanceTestExecutor)

# Ignore the lowest frequencies when hunting for a resonance peak so the
# DC/drift portion of the spectrum does not win the argmax.
MIN_PEAK_FREQ = 5.

# gcmd wrapper that swallows respond_info (keeps get/error/etc.) so a swept test
# does not print its per-frequency progress line - used for the multi-point mesh
# survey, where ~35 lines/point would otherwise flood the console.
class _QuietGCmd:
    def __init__(self, gcmd):
        self._gcmd = gcmd
    def respond_info(self, msg, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return getattr(self._gcmd, name)

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
        self.gcode.register_command(
                "RESONANCE_PROBE_CALIBRATE", self.cmd_CALIBRATE,
                desc=self.cmd_CALIBRATE_help)
        self.gcode.register_command(
                "RESONANCE_PROBE_CONTACT", self.cmd_CONTACT,
                desc=self.cmd_CONTACT_help)
        self.gcode.register_command(
                "RESONANCE_PROBE_CHARACTERIZE_NOISE",
                self.cmd_CHARACTERIZE_NOISE,
                desc=self.cmd_CHARACTERIZE_NOISE_help)
        self.gcode.register_command(
                "RESONANCE_PROBE_RANK_FREQ", self.cmd_RANK_FREQ,
                desc=self.cmd_RANK_FREQ_help)
        self.gcode.register_command(
                "RESONANCE_PROBE_CALIBRATE_MESH", self.cmd_CALIBRATE_MESH,
                desc=self.cmd_CALIBRATE_MESH_help)
        self.gcode.register_command(
                "RESONANCE_PROBE_SURVEY_MESH", self.cmd_SURVEY_MESH,
                desc=self.cmd_SURVEY_MESH_help)
        self.gcode.register_command(
                "RESONANCE_PROBE_ZNOISE", self.cmd_ZNOISE,
                desc=self.cmd_ZNOISE_help)

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
        return _gen_fixed_freq(freq, accel, duration)

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

    # -- Z-motion noise diagnostic (temporary; remove before PR) -----------

    cmd_ZNOISE_help = (
        "Diagnostic: compare the accelerometer spectrum with Z stationary vs Z"
        " descending at probe speed, to test whether Z-axis motion injects noise"
        " (e.g. near a high candidate mode) that makes it a bad probe frequency")
    def cmd_ZNOISE(self, gcmd):
        import numpy as np
        chip = self._lookup_chip(gcmd)
        reps = gcmd.get_int("REPS", 1, minval=1, maxval=20)
        # Two report bands: the low mode (~48) and the high mode (~132) by
        # default, so we can see if noise concentrates in the high band.
        lo1 = gcmd.get_float("BAND1_LO", 40.)
        hi1 = gcmd.get_float("BAND1_HI", 56.)
        lo2 = gcmd.get_float("BAND2_LO", 120.)
        hi2 = gcmd.get_float("BAND2_HI", 145.)
        # MOVE=0: stationary-only (acoustic) mode - no motion at all, so it needs
        # no homing; use it to test airborne acoustic pickup (talk vs silence).
        move_on = gcmd.get_int("MOVE", 1)
        toolhead = self.printer.lookup_object('toolhead')
        helper = shaper_calibrate.ShaperCalibrate(self.printer)

        def bandpow(data, psd, lo, hi):
            f = data.freq_bins
            m = (f >= lo) & (f <= hi)
            return float(np.trapz(psd[m], f[m])) if m.any() else 0.

        # Guard against a too-short capture (few samples) that would stall the
        # PSD math - the cause of the earlier high-speed hang.
        def process(aclient):
            if (not aclient.has_valid_samples()
                    or len(aclient.get_samples()) < 200):
                raise gcmd.error("ZNOISE: too few accelerometer samples - use a"
                                 " longer DUR / larger DISTANCE / slower SPEED")
            return helper.process_accelerometer_data(chip.name, aclient)

        def report(tag, data):
            for name, psd in (('x', data.psd_x), ('y', data.psd_y),
                              ('z', data.psd_z)):
                pk = data.freq_bins[psd.argmax()]
                gcmd.respond_info(
                    "  %s accel-%s: peak %.1fHz  band[%.0f-%.0f]=%.3g "
                    " band[%.0f-%.0f]=%.3g"
                    % (tag, name, pk, lo1, hi1, bandpow(data, psd, lo1, hi1),
                       lo2, hi2, bandpow(data, psd, lo2, hi2)))

        def stationary_capture(dur):
            toolhead.wait_moves()
            toolhead.dwell(0.2)
            aclient = chip.start_internal_client()
            try:
                toolhead.dwell(dur)
                toolhead.wait_moves()
            finally:
                aclient.finish_measurements()
            return process(aclient)

        if not move_on:
            dur = gcmd.get_float("DUR", 3.0, above=0.5, maxval=30.)
            gcmd.respond_info(
                "ZNOISE stationary (acoustic) x%d @ %.1fs each; bands %.0f-%.0f"
                " and %.0f-%.0f Hz" % (reps, dur, lo1, hi1, lo2, hi2))
            for r in range(reps):
                report("STATIONARY %d" % (r + 1), stationary_capture(dur))
            return

        # Moving-comparison mode: needs a homed axis to descend.
        self._move_to_point(gcmd)
        speed = gcmd.get_float("SPEED", 0.2, above=0., maxval=25.)
        distance = gcmd.get_float("DISTANCE", 1.0, above=0.05)
        x, y, z0 = toolhead.get_position()[:3]
        # Hard floor so an in-air diagnostic never drives the nozzle into the bed.
        zmin = gcmd.get_float("ZMIN", -0.15)
        z_floor = max(z0 - distance, zmin)
        if z_floor >= z0:
            raise gcmd.error("ZNOISE: no room to descend (z0=%.3f floor=%.3f);"
                             " raise the nozzle or lower ZMIN" % (z0, z_floor))
        actual = z0 - z_floor
        dur = actual / speed
        lift = min(self.move_speed, 10.)

        def moving_capture():
            toolhead.wait_moves()
            toolhead.dwell(0.2)
            aclient = chip.start_internal_client()
            try:
                toolhead.manual_move([x, y, z_floor], speed)
                toolhead.wait_moves()
            finally:
                aclient.finish_measurements()
            return process(aclient)

        gcmd.respond_info(
            "ZNOISE at X%.1f Y%.1f: descend %.2fmm @ %.3fmm/s (%.1fs) vs a"
            " matched stationary dwell; bands %.0f-%.0f and %.0f-%.0f Hz"
            % (x, y, actual, speed, dur, lo1, hi1, lo2, hi2))
        for r in range(reps):
            if reps > 1:
                gcmd.respond_info("-- rep %d/%d --" % (r + 1, reps))
            stat = stationary_capture(dur)
            report("STATIONARY", stat)
            move = moving_capture()
            report("Z-MOVING  ", move)
            for name, ss, mm in (('x', stat.psd_x, move.psd_x),
                                 ('y', stat.psd_y, move.psd_y),
                                 ('z', stat.psd_z, move.psd_z)):
                s1, m1 = bandpow(stat, ss, lo1, hi1), bandpow(move, mm, lo1, hi1)
                s2, m2 = bandpow(stat, ss, lo2, hi2), bandpow(move, mm, lo2, hi2)
                gcmd.respond_info(
                    "  accel-%s move/stat ratio: low[%.0f-%.0f]=%.2fx "
                    " high[%.0f-%.0f]=%.2fx"
                    % (name, lo1, hi1, m1 / max(s1, 1e-9),
                       lo2, hi2, m2 / max(s2, 1e-9)))
            toolhead.manual_move([x, y, z0], lift)
            toolhead.wait_moves()

    # -- automatic calibration --------------------------------------------

    # Find the strongest resonance peak and the accelerometer axis that
    # responds to it (same analysis as FIND_FREQ).  Returns (accel_axis, freq).
    def _find_resonance(self, gcmd, chip, axis):
        self.generator.prepare_test(gcmd, is_z=bool(axis.get_dir()[2]))
        test_seq = self.generator.gen_test()
        aclient = self._vibrate_and_collect(gcmd, chip, axis, test_seq)
        helper = shaper_calibrate.ShaperCalibrate(self.printer)
        data = helper.process_accelerometer_data(chip.name, aclient)
        freqs = data.freq_bins
        mask = freqs >= MIN_PEAK_FREQ
        best = None
        for name, psd in (('x', data.psd_x), ('y', data.psd_y),
                          ('z', data.psd_z)):
            masked = psd[mask]
            peak_freq = freqs[mask][masked.argmax()]
            peak_power = masked.max()
            if best is None or peak_power > best[2]:
                best = (name, peak_freq, peak_power)
        gcmd.respond_info("Calibrate: resonance at %.1f Hz, strongest response"
                          " on accel %s-axis" % (best[1], best[0]))
        return best[0], best[1]

    # Sweep once and return the top-N candidate resonance peaks (local maxima)
    # on the strongest-responding accelerometer axis, not just the global argmax.
    # A machine can have several high-Q modes (e.g. ~47 and ~55 Hz here) and the
    # loudest IN-AIR peak is not necessarily the one that damps best ON CONTACT,
    # so we return several and let the caller rank them by contact damping.
    # Returns (accel_axis_name, [(freq, power), ...] strongest first).
    # Local-neighborhood window (Hz) used both to test how PROMINENT a local
    # maximum is (instead of one global-relative threshold) and to mask out the
    # single strongest ("principal") peak's own neighborhood when hunting for
    # other candidates.  A peak at only ~8% of the global max can still be a
    # real, useful resonance - this machine's ~135Hz mode damps on contact far
    # more reliably than its ~48Hz global-max mode - what matters is whether a
    # peak stands out from ITS OWN surroundings, not whether it rivals the
    # single loudest mode anywhere in the spectrum.  See resonance-nozzle-probe
    # memory, 2026-07-06 #5, for the on-machine failure (FREQ_END=200 still
    # missed the 131-142Hz modes) this replaces.
    PROMINENCE_WINDOW_HZ = 15.
    PROMINENCE_GUARD_HZ = 2.
    PROMINENCE_FACTOR = 3.
    # A relaxed second admission path for BROAD peaks: the sharp-peak
    # prominence test above can reject a real but wide resonance (its
    # shoulders extend further, so the local floor it's measured against
    # is relatively higher) even though a wide peak may be more resistant to
    # the frequency drift seen across sessions - see resonance-nozzle-probe
    # memory and task #20.  Admit a peak here that clears only a modest
    # prominence bar AND is clearly wider than a typical sharp mode, tagged
    # as "broad" in the log so sharp vs. wide candidates can be told apart in
    # the data this collects.
    PROMINENCE_FACTOR_BROAD = 1.5
    BROAD_MIN_WIDTH_HZ = 6.

    # A peak's prominence/local-max status can only be trusted if its FULL
    # +-PROMINENCE_WINDOW_HZ neighborhood was actually swept - a peak sitting
    # near the edge of the tested range (e.g. FREQ_END) has a truncated window
    # on that side, so the "floor" there is measured over only a sliver of
    # data and can look artificially low (inflating prominence), and there is
    # no way to tell a genuine local max from the rising shoulder of a much
    # bigger, unmeasured peak just past the edge.  Gate the new prominence-
    # based admission (and everything derived from it) on this, so a result
    # like "198.3Hz is prominent" 1.7Hz from a 200Hz FREQ_END is not trusted -
    # extending FREQ_END is the correct way to confirm/refute it, not treating
    # the truncated window as sufficient evidence.
    def _window_confirmed(self, fr, f0):
        return (f0 - self.PROMINENCE_WINDOW_HZ >= fr[0]
               and f0 + self.PROMINENCE_WINDOW_HZ <= fr[-1])

    # How far a peak at (f0, p0) stands out above its own local neighborhood
    # (excluding a small guard band around itself, so its own shoulder does not
    # inflate the floor estimate).  >>1 means a real, well-defined local mode;
    # ~1 means it is barely distinguishable from the surrounding noise floor.
    def _peak_prominence(self, fr, psd, f0, p0):
        lo = bisect.bisect_left(fr, f0 - self.PROMINENCE_WINDOW_HZ)
        hi = bisect.bisect_right(fr, f0 + self.PROMINENCE_WINDOW_HZ)
        glo = bisect.bisect_left(fr, f0 - self.PROMINENCE_GUARD_HZ)
        ghi = bisect.bisect_right(fr, f0 + self.PROMINENCE_GUARD_HZ)
        floor_vals = list(psd[lo:glo]) + list(psd[ghi:hi])
        floor = min(floor_vals) if floor_vals else p0
        return p0 / max(floor, 1e-9)

    # How WIDE (Hz) a peak at (f0, p0) is: the span, within its own
    # +-PROMINENCE_WINDOW_HZ neighborhood, over which the PSD stays at or
    # above the midpoint between the local floor and the peak height (a
    # half-max width relative to the LOCAL floor, since the absolute PSD
    # level varies by frequency).  Purely a DATA point, not a selection
    # criterion by itself - whether a wide peak resists the frequency drift
    # seen across sessions better than a sharp one (or is just weaker/
    # noisier) is the empirical question the user wants answered from real
    # measurements, not something to assume in the detection logic.
    def _peak_width_hz(self, fr, psd, f0, p0):
        lo = bisect.bisect_left(fr, f0 - self.PROMINENCE_WINDOW_HZ)
        hi = bisect.bisect_right(fr, f0 + self.PROMINENCE_WINDOW_HZ)
        glo = bisect.bisect_left(fr, f0 - self.PROMINENCE_GUARD_HZ)
        ghi = bisect.bisect_right(fr, f0 + self.PROMINENCE_GUARD_HZ)
        floor_vals = list(psd[lo:glo]) + list(psd[ghi:hi])
        floor = min(floor_vals) if floor_vals else 0.
        half = floor + 0.5 * (p0 - floor)
        i0 = min(max(bisect.bisect_left(fr, f0), lo), max(hi - 1, lo))
        j = i0
        while j > lo and psd[j - 1] >= half:
            j -= 1
        k = i0
        while k < hi - 1 and psd[k + 1] >= half:
            k += 1
        return float(fr[k] - fr[j])

    # All local maxima in psd(fr), strongest first, merged within ~3 Hz.
    def _local_maxima(self, fr, psd):
        pk = []
        for i in range(1, len(psd) - 1):
            if psd[i] > psd[i-1] and psd[i] >= psd[i+1]:
                pk.append((float(fr[i]), float(psd[i])))
        pk.sort(key=lambda x: -x[1])
        merged = []
        for fq, pw in pk:
            if all(abs(fq - c[0]) > 3.0 for c in merged):
                merged.append((fq, pw))
        return merged

    def _find_candidate_freqs(self, gcmd, chip, axis, n_peaks, peak_frac,
                              quiet=False):
        self.generator.prepare_test(gcmd, is_z=bool(axis.get_dir()[2]))
        test_seq = self.generator.gen_test()
        sweep_gcmd = _QuietGCmd(gcmd) if quiet else gcmd
        aclient = self._vibrate_and_collect(sweep_gcmd, chip, axis, test_seq)
        helper = shaper_calibrate.ShaperCalibrate(self.printer)
        data = helper.process_accelerometer_data(chip.name, aclient)
        freqs = data.freq_bins
        # Cap candidates to the actually-swept range (self.generator.freq_end,
        # set by prepare_test() above) - the PSD's FFT bins extend to Nyquist
        # regardless of what was driven, so an uncapped scan can surface a
        # peak far outside anything ever validated by sweeping, and later
        # promote it straight to live excitation (see memory: 307.8Hz
        # cross-coupling candidate crashed the MCU when driven for real,
        # while FREQ_END was only 135Hz).
        mask = (freqs >= MIN_PEAK_FREQ) & (freqs <= self.generator.freq_end)
        fr = freqs[mask]
        psds = {'x': data.psd_x[mask], 'y': data.psd_y[mask],
               'z': data.psd_z[mask]}
        # Primary axis: the one with the strongest overall response - still
        # used to seed the excitation-axis candidate list (unchanged).
        aname = max(psds, key=lambda k: psds[k].max())
        psd = psds[aname]
        merged = self._local_maxima(fr, psd)
        gmax_f, gmax = merged[0]
        gcmd.respond_info("Mode scan (accel %s-axis): local maxima %s" % (aname,
            ", ".join("%.1fHz(%.0f%%)" % (f, 100.*p/gmax) for f, p in merged[:5])))
        # Admit a peak if EITHER it is within peak_frac of the single loudest
        # mode (the original test - fine when the modes of interest really are
        # all comparably loud) OR it is locally PROMINENT (stands out well
        # above its own neighborhood, regardless of how it compares to the
        # global max elsewhere in the spectrum) AND its window was fully swept
        # (see _window_confirmed - a peak near a scan edge cannot be trusted by
        # prominence alone).
        chosen = []
        unconfirmed = []
        for fq, pw in merged:
            if pw >= peak_frac * gmax:
                chosen.append((fq, pw))
            elif not self._window_confirmed(fr, fq):
                unconfirmed.append(fq)
            elif self._peak_prominence(fr, psd, fq, pw) >= self.PROMINENCE_FACTOR:
                chosen.append((fq, pw))
            if len(chosen) >= n_peaks:
                break
        if not chosen:
            chosen = [merged[0]]
        if unconfirmed:
            gcmd.respond_info(
                "Mode scan: ignoring %s Hz - too close to the tested range's"
                " edge to confirm as a real peak vs. the rising shoulder of a"
                " mode past it; widen FREQ_START/FREQ_END to check"
                % ", ".join("%.1f" % f for f in unconfirmed))
        # Coarse band guarantee: if every admitted candidate still sits within
        # the principal peak's own neighborhood (e.g. a very noisy sweep where
        # prominence alone found nothing else), force in the strongest local
        # max found OUTSIDE that neighborhood, so a higher (or lower) mode
        # always gets a chance at the downstream driven-refine + contact-
        # damping test - which is the real arbiter of usefulness.  This scan
        # only needs to narrow the field, not make the final call.  Still
        # requires a confirmed window - an edge artifact should not be forced
        # in just because nothing else qualified.
        outside = [(fq, pw) for fq, pw in merged
                  if abs(fq - gmax_f) > self.PROMINENCE_WINDOW_HZ
                  and self._window_confirmed(fr, fq)]
        have_outside = any(abs(fq - gmax_f) > self.PROMINENCE_WINDOW_HZ
                           for fq, _ in chosen)
        if not have_outside and outside:
            chosen.append(outside[0])
        widths = {fq: self._peak_width_hz(fr, psd, fq, pw) for fq, pw in chosen}
        # Second, relaxed admission pass for BROAD peaks the sharp-peak test
        # above rejects (see PROMINENCE_FACTOR_BROAD/BROAD_MIN_WIDTH_HZ) - a
        # few extra candidates at most, so this can't blow up the downstream
        # contact-damping test count.  Tagged 'broad' in the log/width map so
        # sharp vs. wide candidates are distinguishable in the collected data.
        broad = []
        for fq, pw in merged:
            if any(abs(fq - c[0]) <= 3.0 for c in chosen):
                continue
            if not self._window_confirmed(fr, fq):
                continue
            if self._peak_prominence(fr, psd, fq, pw) < self.PROMINENCE_FACTOR_BROAD:
                continue
            w = self._peak_width_hz(fr, psd, fq, pw)
            if w >= self.BROAD_MIN_WIDTH_HZ:
                broad.append((fq, pw))
                widths[fq] = w
            if len(broad) >= 2:
                break
        chosen.extend(broad)
        gcmd.respond_info(
            "Mode scan: %d candidate(s) (>=%.0f%% of peak or >=%.0fx local"
            " prominence, +%d broad @>=%.0fx/%.0fHz wide): %s" % (
                len(chosen), peak_frac * 100., self.PROMINENCE_FACTOR,
                len(broad), self.PROMINENCE_FACTOR_BROAD,
                self.BROAD_MIN_WIDTH_HZ,
                ", ".join("%.1fHz(%.1fHz wide%s)"
                          % (f, widths[f], ", broad" if f in
                             [b[0] for b in broad] else "")
                          for f, p in chosen)))
        # Cross-coupling candidates: for each OTHER accelerometer axis, surface
        # its own strongest local max OUTSIDE the principal peak's neighborhood
        # (the excitation axis's own modes are already covered above) - a proof
        # of concept that a mode may damp on contact more clearly on a cross-
        # coupled axis than on the axis actually being driven.  Same edge-
        # confirmation requirement applies.
        for oname, opsd in psds.items():
            if oname == aname:
                continue
            omerged = [(fq, pw) for fq, pw in self._local_maxima(fr, opsd)
                      if abs(fq - gmax_f) > self.PROMINENCE_WINDOW_HZ
                      and self._window_confirmed(fr, fq)]
            if not omerged:
                continue
            ofq, opw = omerged[0]
            if any(abs(ofq - c[0]) <= 3.0 for c in chosen):
                continue
            chosen.append((ofq, opw))
            gcmd.respond_info(
                "  +cross-coupling candidate from %s-axis: %.1f Hz"
                % (oname, ofq))
        chosen.sort(key=lambda x: -x[1])
        return aname, chosen

    # Rank candidate modes by how strongly each DAMPS on bed contact relative to
    # its own noise (detectability = contact drop / air noise).  Reuses the dwell
    # characterization primitive (one level per frequency) around a known contact
    # Z.  The loudest in-air mode may not detect best; this picks the mode the
    # probe should actually use.  Returns [(freq, drop, noise, snr), ...] best
    # first, plus the chosen (best) frequency.
    def _rank_modes_by_damping(self, gcmd, chip, accel_axis, candidates,
                               contact_z, x0, y0, cap, z_min, warmup, lift,
                               up_margin, down_margin, cycles,
                               min_drop, target_noise):
        out_idx = {'x': 0, 'y': 1, 'z': 2}[accel_axis]
        axis = _parse_axis(gcmd, gcmd.get("AXIS", "x").lower())
        ranked = []
        for fq, _pw in candidates:
            amp = cap / (4. * math.pi**2 * fq)
            helper = HaltingContactProbe(
                self.printer, chip, out_idx, axis, fq, cap, amp, z_min,
                warmup, 0.06, 0.15, 5., 0.01, 0.05, 0.)
            # A single candidate's characterization can fail outright (e.g. no
            # axis got even one valid air/contact window - too few samples at
            # a very high frequency, or the accelerometer saw nothing usable)
            # rather than just reading "dirty".  That is itself a real,
            # reportable data point for a frequency-vs-detectability survey -
            # it must not kill the whole table (see memory: RANK_FREQ died
            # outright at 230Hz during a 5-250Hz center-point survey).
            # SWEEP amplitudes for every candidate, and judge each mode at ITS
            # OWN best amplitude.  Ranking on a single amplitude compared modes
            # under conditions that disadvantage some of them: the contact drop
            # is a ratio, so a mode that is excellent at accel_per_hz 60 can
            # read poorly when measured at 120 or 200.  Ranking from one
            # measurement is also the same single-shot fragility that made
            # amplitude selection flip 100/120/60 between runs, one level up -
            # and mode choice is the more consequential of the two, since it
            # decides which resonance is used at all.
            #
            # This is affordable: full calibration is a ONE-TIME setup step (a
            # 20-minute run is acceptable), and the expensive part per candidate
            # is the contact-find, which does not repeat across levels.
            levels = gcmd.get_int("RANK_LEVELS", 3, minval=1, maxval=8)
            try:
                m = helper.characterize_amplitude(
                    gcmd, x0, y0, contact_z, lift, up_margin, down_margin,
                    cycles, levels, min_drop, target_noise)
            except gcmd.error as e:
                gcmd.respond_info(
                    "Mode %.1fHz: characterization failed (%s) - recording as"
                    " no usable reading" % (fq, str(e)))
                ranked.append((fq, 0., 1., 0., '?'))
                continue
            drop = m['max_drop']
            noise = m['rel_noise']
            # Rank by the SAME live-headroom metric that selects the amplitude
            # and derives the halt floors (0.5*drop - noise*1.3 - 3pp), rather
            # than by drop/noise.  Using one definition of "detectable"
            # throughout means the mode that wins is the mode that will actually
            # have margin while descending - drop/noise could crown a mode whose
            # absolute drop is too small to clear its own floor.  Kept on the
            # same 0-scale sign convention: bigger is better, <=0 is unusable.
            head = m.get('headroom')
            if head is None:
                head = 0.5 * drop - (noise * 1.3 + 0.03)
            ranked.append((fq, drop, noise, head, m['axis']))
            all_axes = ", ".join(
                "%s=%.0f%%/%.1f%%" % (a, v['drop'] * 100., v['noise'] * 100.)
                for a, v in m['all_axes'].items())
            gcmd.respond_info(
                "Mode %.1fHz: best aph=%.0f drop=%.0f%% noise=%.1f%%"
                " headroom=%+.1fpp axis=%s (all axes drop%%/noise%%: %s)"
                % (fq, m['accel_per_hz'], drop * 100., noise * 100.,
                   head * 100., m['axis'], all_axes))
        ranked.sort(key=lambda r: -r[3])
        return ranked

    # Path A calibration.  ONE halting descent at the STARTING (=cap) amplitude
    # finds the contact height reliably - we start strong so the first detection
    # cannot fail and drive the nozzle past the bed.  Then, with the excitation
    # left running, characterize_amplitude oscillates Z in a small bounded band
    # around that now-known contact while stepping the amplitude DOWN, to find
    # the gentlest amplitude that still detects contact cleanly (no per-step
    # warm-up, bidirectional, bounded = safe).  Returns a dict (sensitivity,
    # halt, baseline, rel_noise, max_drop, accel_per_hz, contact_z).
    # known_drop: the contact drop the mode search actually MEASURED on the
    # winning mode (bounded touch/release sweep).  Used to sanity-bound the
    # air-derived halt floor - see the clamp below.
    def _calibrate_contact(self, gcmd, chip, axis, accel_axis, freq,
                           known_drop=None):
        out_idx = {'x': 0, 'y': 1, 'z': 2}[accel_axis]
        toolhead = self.printer.lookup_object('toolhead')
        # Hard descent floor.  The default assumes a standard paper-gauge bed
        # setup (z=0 ~ one paper thickness, ~0.1mm, above the bed), so true
        # contact is around z=-0.1 and a -0.2 floor sits just below it: if the
        # first detection ever misses, the descent can over-travel only ~0.1mm
        # before stopping, instead of gouging the bed.  REQUIRES that the bed is
        # not above z=0 (the paper setup guarantees this) - run RESONANCE_PROBE_
        # CALIBRATE only after a paper-gauge level.
        z_min = gcmd.get_float("CONTACT_ZMIN", -0.2)
        # First-contact descent speed: slow enough for an accurate contact Z (a
        # single descent at strong amplitude, so this is not the slow part).
        speed = gcmd.get_float("CONTACT_SPEED", 0.1, above=0., maxval=5.)
        warmup = gcmd.get_float("CONTACT_WARMUP", 0.5, above=0.05)
        # Bounded bidirectional amplitude-sweep parameters.
        up_margin = gcmd.get_float("CONTACT_UP", 0.10, above=0.)
        down_margin = gcmd.get_float("CONTACT_DOWN", 0.04, above=0.)
        cycles = gcmd.get_int("CONTACT_CYCLES", 4, minval=1)
        n_levels = gcmd.get_int("CONTACT_LEVELS", 5, minval=1)
        min_drop = gcmd.get_float("CONTACT_MIN_DROP", 0.10, above=0., below=1.)
        target_noise = gcmd.get_float("CONTACT_TARGET_NOISE", 0.06,
                                      above=0., below=1.)
        # Generous fine/halt thresholds for the first descent so it reliably
        # finds contact and halts.
        cal_sens, cal_halt = 0.06, 0.15
        # Drive the contact-find at the CONFIGURED accel_per_hz, not the cap.
        # A stronger excitation does NOT detect better: the contact drop is a
        # RATIO, and more drive energy makes the same damping a smaller fraction
        # of it.  Measured here: at accel_per_hz=120 the air amplitude is ~15000
        # and the live drop 8-9%, but at the 200 cap it is ~27000 and the drop
        # only 5-7% - below the air-derived halt floor, so the descent sails
        # past the bed and reports "no contact".  (The amplitude sweep below
        # independently picked 100 as the best level, same conclusion.)  Same
        # choice the mesh path already makes with its min(120, cap) candidates.
        cap = gcmd.get_float("ACCEL_PER_HZ",
                             min(self.accel_per_hz, self.max_accel_per_hz),
                             above=0., maxval=self.max_accel_per_hz)
        pos = toolhead.get_position()
        x0, y0, ceiling = pos[0], pos[1], pos[2]
        if ceiling <= z_min:
            raise gcmd.error("Calibrate: start Z %.3f is at/below the contact"
                             " floor %.3f; raise the nozzle" % (ceiling, z_min))
        distance = max(2.0, ceiling - z_min)
        lift = min(self.move_speed, 10.)
        # Overrun protection for the vibrating descent (see CALIBRATE_MESH /
        # _mode_low_first's docstring) - a high-frequency contact-find here is
        # exactly as prone to the "Timer too close" shutdown, so this needs the
        # same deeper MCU look-ahead.
        drip_time = gcmd.get_float("DRIP_TIME", 0.3, minval=0.) or None
        # Phase 1: reliable first contact at the STARTING (cap) amplitude.
        # A single live halt is not trusted as real contact on its own - the
        # per-axis independent live monitor (_HostResonanceEndstop) can fire
        # early on cross-axis noise/harmonic content at a mode never live-
        # descent-tested before (see resonance-nozzle-probe memory: a 142Hz
        # z-axis halt landed 0.5mm off from a moments-earlier scan, and its
        # own strongest-amplitude verification touch showed 0% drop on every
        # axis - proof it was never real contact, yet the old code silently
        # accepted it and computed sensitivity from the noisiest, weakest
        # level's coincidental "drop").  So: after every halt, immediately
        # verify with a real touch/release check AT THE STARTING (cap, least
        # noisy) amplitude - characterize_amplitude's first level IS exactly
        # that check (repeated touch/release cycles comparing contact vs.
        # air).  If it shows no real drop, the halt was false; retry the
        # descent (the live monitor re-arms fresh and may not repeat the
        # same nuisance trigger) up to FALSE_HALT_RETRIES times before
        # giving up, instead of ever computing calibration from a fake
        # contact height.
        # Multi-axis arming means more nuisance halts reach the verify pass, and
        # each rejection re-arms strictly below the last - so the budget has to
        # be big enough to walk down past a few before reaching the bed.
        false_halt_retries = gcmd.get_int("FALSE_HALT_RETRIES", 6, minval=0)
        # Per-axis halt floors so the contact-find is gated by each axis's own
        # descent noise (a twitchy cross axis cannot false-halt it) - the same
        # shared characterization the RANK_FREQ/mesh finders use.  Returned too,
        # so SAVE writes halt_sensitivity_x/y/z and the probe uses them at run
        # time.  Characterized at 'speed' so it matches the contact descent.
        noise_reps = gcmd.get_int("CONTACT_NOISE_REPS", 5, minval=2, maxval=10)
        # Characterize the noise DOWN CLOSE TO THE BED, not a millimetre above
        # it.  Noise measured high up structurally under-states the cross axes:
        # the last fraction of a mm - nozzle deflection, the Z motor loading up
        # against the bed - throws transients into them that quiet air never
        # shows (measured: y 13-33% up high, but 27-58% on a real contact
        # descent).  Floors from the quiet region are exactly what made the
        # cross axes false-halt continuously.  Stop just above the paper-gauge
        # zero, still ~0.25-0.35mm clear of true contact.
        air_stop = gcmd.get_float("CONTACT_AIR_FLOOR", 0.15)
        air_floor = max(z_min, min(ceiling - 0.3, air_stop))
        # Keep the contact-FIND MULTI-AXIS: any axis may carry the contact
        # signal, and a false halt is cheap (the verify pass rejects it and
        # re-arms below).  What makes that affordable is characterizing the
        # noise NEAR THE BED (air_floor above) rather than a millimetre above
        # it, so a twitchy cross axis gets a floor that reflects what it
        # actually does on approach instead of one derived from quiet air.
        floors, ceils = self._air_noise_floors(
            gcmd, chip, out_idx, axis, freq, cap, warmup, speed,
            x0, y0, ceiling, air_floor, noise_reps, 1.3, 0.03, 0.9, drip_time)
        # The air-derived floor (noise*1.3 + 3pp) is a NOISE bound only - it
        # knows nothing about how big the real drop is, and here the real drop
        # is small: the excitation cannot fully ring down in the ~0.1mm of
        # travel below contact, so a 26% static damping reaches the live
        # detector as 8-12%.  A floor of 7% against a 9% signal misses, which is
        # exactly how this failed: the finder landed contact at 73.4Hz/x with a
        # 6% floor, then this phase re-characterized, got 7%, and found nothing.
        # So bound the floor by the drop the mode search actually MEASURED on
        # this mode rather than by a hand-tuned constant - halve it, which
        # reproduces the 6% that works here without hard-coding 6%.
        if known_drop:
            capped = round(0.5 * known_drop, 3)
            if capped < floors[out_idx]:
                floors[out_idx] = capped
                if capped <= ceils[out_idx] * 1.1:
                    gcmd.respond_info(
                        "Calibrate: WARNING - %.1f Hz is marginal: half its"
                        " measured %.0f%% contact drop (%.0f%%) is barely above"
                        " the %s noise of %.0f%%.  Detection may be"
                        " unreliable; consider another mode."
                        % (freq, known_drop * 100., capped * 100., accel_axis,
                           ceils[out_idx] * 100.))
        gcmd.respond_info(
            "Calibrate: contact-find floors x=%.0f%% y=%.0f%% z=%.0f%% (all"
            " axes armed; near-bed noise x=%.0f%% y=%.0f%% z=%.0f%%)"
            % (floors[0] * 100., floors[1] * 100., floors[2] * 100.,
               ceils[0] * 100., ceils[1] * 100., ceils[2] * 100.))
        amp = cap / (4. * math.pi**2 * freq)
        m = None
        for attempt in range(false_halt_retries + 1):
            helper = HaltingContactProbe(
                self.printer, chip, out_idx, axis, freq, cap, amp, z_min,
                warmup, cal_sens, cal_halt, 5., 0.01, 0.05, 0.,
                halt_sensitivity_axis=floors)
            contact_z, halted = helper.run(gcmd, ceiling, distance, speed,
                                           lift, lift, x0, y0,
                                           drip_time=drip_time)
            if contact_z is None:
                toolhead.manual_move([x0, y0, ceiling], lift)
                toolhead.wait_moves()
                raise gcmd.error(
                    "Calibrate: no contact detected at the starting"
                    " amplitude (halted=%s) - lower the start height or"
                    " the floor" % (halted,))
            gcmd.respond_info(
                "Calibrate: first contact at z=%.4f (accel_per_hz=%.0f)"
                % (contact_z, cap))
            # Phase 2/3: bounded bidirectional amplitude sweep around the
            # known contact, stepping amplitude down (excitation continuous).
            # Level 0 of this sweep (the cap amplitude) doubles as the
            # verification touch for the false-halt gate below.
            m = helper.characterize_amplitude(
                gcmd, x0, y0, contact_z, lift, up_margin, down_margin,
                cycles, n_levels, min_drop, target_noise)
            if m['verify_drop'] >= 0.08:
                break
            toolhead.manual_move([x0, y0, ceiling], lift)
            toolhead.wait_moves()
            gcmd.respond_info(
                "Calibrate: FALSE HALT at z=%.4f - the strongest-amplitude"
                " verification touch showed only %.0f%% drop (noise %.0f%%);"
                " this was not real contact (attempt %d/%d)"
                % (contact_z, m['verify_drop'] * 100.,
                   m['verify_noise'] * 100., attempt + 1,
                   false_halt_retries + 1))
            m = None
        if m is None:
            raise gcmd.error(
                "Calibrate: every contact-find attempt was a false halt"
                " (verification touch never showed a real drop) after %d"
                " retries - this frequency may not be reliable at this"
                " point, or the start height/floor needs adjusting"
                % false_halt_retries)
        toolhead.manual_move([x0, y0, ceiling], lift)
        toolhead.wait_moves()
        # Now that a REAL contact has been measured on every axis, re-derive the
        # saved floors from each axis's own signal-to-noise.  Axes that can work
        # stay armed (including a cross axis that beats the driven one); axes
        # whose drop cannot clear their own noise disarm themselves.  The
        # contact-find above ran with all axes armed on noise alone - that is
        # deliberately permissive, because the verify pass rejects false halts,
        # whereas a wrongly-disarmed axis silently loses the only usable signal.
        # Refuse to SAVE a mode that leaves the live detector no margin.  The
        # amplitude sweep measures a near-stationary touch; if even its best
        # level cannot clear its own noise by the margin the halt floor needs,
        # the descending probe has no chance, and shipping it produces a config
        # that silently fails to detect.  Better to fail here so the caller can
        # escalate to another frequency.
        if m.get('headroom') is not None and m['headroom'] <= 0.:
            raise gcmd.error(
                "Calibrate: %.1f Hz leaves no live detection margin at any"
                " amplitude (best headroom %.1fpp = 0.5*drop - noise*1.3 - 3pp)"
                " - pick a different mode" % (freq, m['headroom'] * 100.))
        if m.get('all_axes'):
            snr, why = self._snr_floors(ceils, m['all_axes'], 1.3, 0.03, 0.9)
            floors = snr
            gcmd.respond_info("Calibrate: saved halt floors %s"
                              % ", ".join(why))
            if all(f >= 0.95 for f in floors):
                raise gcmd.error(
                    "Calibrate: no axis has a usable contact signal at %.1f Hz"
                    " (every axis's drop is within its own noise) - pick a"
                    " different mode" % freq)
            # Keep accel_axis and the armed floors CONSISTENT.  accel_axis came
            # from the mode search's ranking; the floors come from the amplitude
            # sweep at the chosen frequency.  Those are different measurements
            # and they can disagree - observed on hardware: the ranking said x
            # while the floors armed only z (x=off drop 14% vs noise 7%, y=off
            # 32% vs 15%, z=29% from 57% vs 22%), i.e. the saved config would
            # have halted on z while claiming to detect on x.  The floors are
            # the later and more directly relevant measurement, so let them win,
            # and pick the armed axis with the most margin over its own noise.
            armed = [a for a, f in enumerate(floors) if f < 0.95]
            best = max(armed, key=lambda a: (0.5 * (m['all_axes'].get('xyz'[a])
                                                    or {}).get('drop', 0.)
                                             - ceils[a]))
            if 'xyz'[best] != accel_axis:
                gcmd.respond_info(
                    "Calibrate: detection axis %s -> %s (the amplitude sweep's"
                    " per-axis floors disagree with the mode ranking; the"
                    " floors decide, so the armed axis and accel_axis match)"
                    % (accel_axis, 'xyz'[best]))
                accel_axis, out_idx = 'xyz'[best], best
        aph = m['accel_per_hz']
        baseline = m['baseline']
        rel_noise = m['rel_noise']
        max_drop = m['max_drop']
        if max_drop < 0.08:
            raise gcmd.error("Calibrate: contact drop only %.0f%% - increase the"
                             " amplitude cap or lower the start height"
                             % (max_drop * 100.))
        # Thresholds are fractions of the baseline, so well-validated defaults
        # (fine 0.06, loose 0.15) transfer across machines.  Adapt only at the
        # edges: raise the fine threshold if the measured noise approaches it,
        # and lower the loose halt if the contact drop is too small to reach it.
        sensitivity = round(min(0.12, max(0.06, 4. * rel_noise)), 3)
        halt = round(max(2. * sensitivity, min(0.15, 0.6 * max_drop)), 3)
        gcmd.respond_info(
            "Calibrate: accel_per_hz=%.0f baseline=%.1f rel_noise=%.1f%%"
            " max_drop=%.0f%% contact_z=%.4f -> sensitivity=%.3f"
            " halt_sensitivity=%.3f"
            % (aph, baseline, rel_noise * 100., max_drop * 100., contact_z,
               sensitivity, halt))
        # _plain(): 'floors' and the thresholds are written into the config and
        # assigned to the live probe, so no numpy may survive here - see
        # resonance_probe._plain().
        return _plain({'sensitivity': sensitivity, 'halt': halt,
                       'baseline': baseline, 'rel_noise': rel_noise,
                       'max_drop': max_drop, 'accel_per_hz': aph,
                       'contact_z': contact_z, 'floors': floors,
                       'accel_axis': accel_axis})

    cmd_CALIBRATE_help = ("Automatically determine resonance-probe settings and"
                          " save them to the [resonance_probe] config section")
    def cmd_CALIBRATE(self, gcmd):
        chip = self._lookup_chip(gcmd)
        axis = _parse_axis(gcmd, gcmd.get("AXIS", "x").lower())
        self._check_axis_safety(gcmd, axis)
        self._move_to_point(gcmd)
        # FREQ= fast path: skip the mode search entirely.  The contact phase
        # drives its first descent at the starting (cap) amplitude regardless,
        # so a known-good FREQ is all it needs - handy for repeatable diagnostic
        # runs and when the user already knows the machine's resonance.
        freq = gcmd.get_float("FREQ", None, above=1., maxval=300.)
        if freq is None:
            # Pick the frequency by CONTACT DAMPING across candidate modes (the
            # same high-mode-aware search RANK_FREQ/CALIBRATE_MESH use), not
            # just the single loudest in-air peak - a quiet-but-real high mode
            # damps on contact far better than a loud low mode on some machines
            # (see resonance-nozzle-probe memory) and would never be considered
            # by the old single-peak search.
            accel_axis, freq, ranked, loudest_freq = self._find_best_probe_freq(
                gcmd, chip, axis)
            if ranked:
                known_drop = ranked[0][1]
            if ranked is not None and abs(freq - loudest_freq) > 0.5:
                gcmd.respond_info(
                    "Calibrate: chose %.1f Hz by contact damping over the"
                    " loudest in-air peak (%.1f Hz)" % (freq, loudest_freq))
        else:
            vname = axis.get_name()
            default_axis = vname if vname in ('x', 'y', 'z') else 'x'
            accel_axis = gcmd.get("ACCEL_AXIS", default_axis).lower()
            if accel_axis not in ('x', 'y', 'z'):
                raise gcmd.error("ACCEL_AXIS must be x, y, or z")
            gcmd.respond_info("Calibrate: using FREQ=%.1f accel %s-axis (skipping"
                              " the resonance/amplitude search)"
                              % (freq, accel_axis))
        m = self._calibrate_contact(gcmd, chip, axis, accel_axis, freq,
                                    known_drop=known_drop)
        accel_per_hz = m['accel_per_hz']
        # The contact phase may have moved the detection axis to match the
        # armed floors (see _calibrate_contact) - save what it decided.
        accel_axis = m.get('accel_axis', accel_axis)
        sensitivity, halt = m['sensitivity'], m['halt']
        # One machine-parseable line summarizing this run (for characterization)
        gcmd.respond_info(
            "RPCAL label=%s freq=%.1f aph=%.0f baseline=%.0f relnoise=%.3f"
            " maxdrop=%.3f sens=%.3f halt=%.3f"
            % (gcmd.get("LABEL", ""), freq, accel_per_hz, m['baseline'],
               m['rel_noise'], m['max_drop'], sensitivity, halt))
        if not gcmd.get_int("SAVE", 1):
            return  # dry run: report only, don't touch the config
        # Save into [resonance_probe] (applied on SAVE_CONFIG + restart).
        configfile = self.printer.lookup_object('configfile')
        floors = m['floors']
        vals = {'accel_chip': chip.name, 'accel_axis': accel_axis,
                'vibrate_axis': axis.get_name(),
                'excitation_frequency': "%.1f" % freq,
                'accel_per_hz': "%.0f" % accel_per_hz,
                'sensitivity': "%.3f" % sensitivity,
                'halt_sensitivity': "%.3f" % halt,
                'halt_sensitivity_x': "%.3f" % floors[0],
                'halt_sensitivity_y': "%.3f" % floors[1],
                'halt_sensitivity_z': "%.3f" % floors[2],
                'probe_mode': 'hostdriven'}
        for k, v in vals.items():
            configfile.set('resonance_probe', k, v)
        gcmd.respond_info(
            "Resonance probe calibrated. The following will be saved to"
            " [resonance_probe]:\n"
            + "\n".join("  %s: %s" % (k, vals[k]) for k in vals)
            + "\nRun PROBE_CALIBRATE to set z_offset, then SAVE_CONFIG to keep"
            " these values (the printer will restart)."
            "\nNOTE: SAVE_CONFIG cannot write these if [resonance_probe] lives"
            " in an INCLUDED .cfg - it fails with \"option '...' conflicts with"
            " included value\" AFTER this whole calibration has run.  If you"
            " keep the section in an include, move it into printer.cfg first.")
    # -- standalone contact probe -----------------------------------------

    cmd_CONTACT_help = ("Find the Z height at which the nozzle contacts the bed"
                        " using resonance damping, without configuring"
                        " [resonance_probe] as the machine probe")
    def cmd_CONTACT(self, gcmd):
        import numpy as np
        chip = self._lookup_chip(gcmd)
        axis = _parse_axis(gcmd, gcmd.get("AXIS", "x").lower())
        self._check_axis_safety(gcmd, axis)
        self._move_to_point(gcmd)
        freq = gcmd.get_float("FREQ", None, above=1., maxval=300.)
        if freq is None:
            # No frequency given: locate the resonance (and the responding axis)
            # at the current point first, exactly like CALIBRATE/FIND_FREQ.
            accel_axis, freq = self._find_resonance(gcmd, chip, axis)
        else:
            vname = axis.get_name()
            default_axis = vname if vname in ('x', 'y', 'z') else 'x'
            accel_axis = gcmd.get("ACCEL_AXIS", default_axis).lower()
            if accel_axis not in ('x', 'y', 'z'):
                raise gcmd.error("ACCEL_AXIS must be x, y, or z")
        accel_per_hz = gcmd.get_float("ACCEL_PER_HZ", self.accel_per_hz,
                                      above=0., maxval=self.max_accel_per_hz)
        samples = gcmd.get_int("SAMPLES", 1, minval=1, maxval=20)
        # The descent HALTS in real time on contact (shared HaltingContactProbe),
        # so it is safe on a rigid bed; ZMIN/DISTANCE only bound the worst case
        # if the halt never fires.  Defaults are conservative for an unknown
        # machine - SENSITIVITY can be tightened once a good resonance is known.
        sensitivity = gcmd.get_float("SENSITIVITY", 0.10, above=0., below=1.)
        halt_sens = gcmd.get_float("HALT_SENSITIVITY",
                                   min(0.4, 2. * sensitivity), above=0.,
                                   below=1.)
        # Optional per-axis live-halt floors "x,y,z" (override for testing);
        # otherwise use the main probe's characterized floors if configured.
        hsa = gcmd.get("HALT_SENS_XYZ", None)
        if hsa:
            try:
                halt_axis = [float(v) for v in hsa.split(',')]
            except ValueError:
                raise gcmd.error("HALT_SENS_XYZ must be 'x,y,z' floats")
            if len(halt_axis) != 3:
                raise gcmd.error("HALT_SENS_XYZ must be 'x,y,z'")
        else:
            rp = self.printer.lookup_object('resonance_probe', None)
            halt_axis = getattr(rp, 'halt_sensitivity_axis', None)
        speed = gcmd.get_float("SPEED", 1.0, above=0., maxval=5.)
        # Hard floor: with a standard paper-gauge setup (z=0 ~0.1mm above the
        # bed) true contact is ~-0.1, so a -0.2 floor bounds any detection miss
        # to ~0.1mm of over-travel.  Run only after a paper-gauge level.
        z_min = gcmd.get_float("ZMIN", -0.2)
        warmup = gcmd.get_float("WARMUP", 0.5, above=0.05)
        amp = accel_per_hz / (4. * math.pi**2 * freq)
        out_idx = {'x': 0, 'y': 1, 'z': 2}[accel_axis]
        # Overrun protection for the vibrating descent (see CALIBRATE_MESH /
        # _mode_low_first's docstring) - a high-frequency contact-find is
        # exactly as prone to the "Timer too close" shutdown here.
        drip_time = gcmd.get_float("DRIP_TIME", 0.3, minval=0.) or None
        helper = HaltingContactProbe(
            self.printer, chip, out_idx, axis, freq, accel_per_hz, amp, z_min,
            warmup, sensitivity, halt_sens, 5., 0.01, 0.05, 0.,
            halt_sensitivity_axis=halt_axis)
        toolhead = self.printer.lookup_object('toolhead')
        x, y, ceiling = toolhead.get_position()[:3]
        lift = min(self.move_speed, 10.)
        distance = gcmd.get_float("DISTANCE", max(2.0, ceiling - z_min),
                                  above=0.)
        results = []
        for i in range(samples):
            c, halted = helper.run(gcmd, ceiling, distance, speed, lift, lift,
                                   x, y, drip_time=drip_time)
            if c is None:
                raise gcmd.error(
                    "Resonance contact: no contact detected (halted=%s) at"
                    " SPEED=%.2f; lower SPEED, lower the start height, raise"
                    " ACCEL_PER_HZ, or increase DISTANCE.  SPEED is usually the"
                    " one that matters: the excitation needs TIME to ring down"
                    " once the nozzle is damped, and only ~0.1mm of travel"
                    " exists below contact before the safety floor, so above"
                    " roughly 0.3 mm/s the descent outruns the decay and the"
                    " drop washes out (measured on a working setup: 26%% static"
                    " damping seen live as 9%%/4%%/2%% at 0.3/0.5/0.8 mm/s)."
                    % (halted, speed))
            results.append(c)
            if samples > 1:
                gcmd.respond_info("  sample %d/%d: nozzle contact at Z=%.4f"
                                  % (i + 1, samples, c))
        # run() leaves the nozzle at the last contact; retract to the start.
        toolhead.manual_move([x, y, ceiling], lift)
        toolhead.wait_moves()
        arr = np.array(results)
        median = float(np.median(arr))
        spread = float(arr.max() - arr.min())
        gcmd.respond_info(
            "Resonance contact: nozzle touches the bed at Z=%.4f"
            " (median of %d, spread %.1fum) at X=%.1f Y=%.1f using FREQ=%.1f"
            " accel %s-axis" % (median, samples, spread * 1000., x, y, freq,
                                accel_axis))
        gcmd.respond_info(
            "To set a non-contact probe's z_offset: PROBE at this same X/Y to"
            " read the Z where your probe triggers, then z_offset = (probe"
            " trigger Z) - (%.4f)." % (median,))


    # Resolve the candidate mode list: CANDIDATES=f1,f2,... ranks explicit
    # frequencies (skips the sweep) so a known set of modes can be compared
    # directly; otherwise sweep and find them via _find_candidate_freqs.
    def _resolve_candidates(self, gcmd, chip, axis):
        n_peaks = gcmd.get_int("N_PEAKS", 3, minval=1, maxval=6)
        peak_frac = gcmd.get_float("PEAK_THRESHOLD", 0.1, above=0., below=1.)
        cand_override = gcmd.get("CANDIDATES", None)
        if not cand_override:
            return self._find_candidate_freqs(gcmd, chip, axis, n_peaks,
                                              peak_frac)
        try:
            freqs = [float(x) for x in cand_override.split(',')]
        except ValueError:
            raise gcmd.error("CANDIDATES must be comma-separated frequencies")
        vname = axis.get_name()
        accel_axis = gcmd.get("ACCEL_AXIS",
                              vname if vname in ('x', 'y', 'z')
                              else 'x').lower()
        if accel_axis not in ('x', 'y', 'z'):
            raise gcmd.error("ACCEL_AXIS must be x, y, or z")
        gcmd.respond_info("Mode select: explicit candidates %s (accel %s-axis)"
                          % (freqs, accel_axis))
        return accel_axis, [(fq, 0.) for fq in freqs]

    # Find the best probe frequency among candidate resonance modes by which
    # damps most cleanly ON CONTACT - not just the loudest in air.  Shared by
    # RANK_FREQ (reports only) and CALIBRATE (feeds the chosen freq straight
    # into the existing accel_per_hz/sensitivity calibration), so an ordinary
    # CALIBRATE run gets the same high-mode-aware selection RANK_FREQ has,
    # instead of the single loudest in-air peak the old _find_resonance picked
    # (which is nearly always a low mode - loudest in air != best on contact).
    # Returns (accel_axis, best_freq, ranked_or_None, loudest_candidate_freq).
    # 'ranked' is None when only one candidate was found: nothing to rank
    # against, so the lone candidate IS the answer and contact is not probed
    # here (the caller's own contact phase will still run).
    def _find_best_probe_freq(self, gcmd, chip, axis):
        accel_axis, candidates = self._resolve_candidates(gcmd, chip, axis)
        if len(candidates) < 2:
            gcmd.respond_info(
                "Mode select: only one candidate mode found (%.1f Hz); nothing"
                " to rank against." % (candidates[0][0],))
            return accel_axis, candidates[0][0], None, candidates[0][0]
        best_freq, ranked, _ambiguous, _attempts = self._finder_rank_at_point(
            gcmd, chip, accel_axis, axis, candidates)
        # Use the accelerometer axis the WINNING mode actually damps on, not the
        # axis the PSD scan happened to sweep.  These are different things: the
        # ranking measures the drop on all three axes and a cross-axis mode can
        # win outright (measured here: 104.6 Hz damps 59% on y while the driven
        # x axis shows nothing).  Returning the scan axis meant the contact-find
        # then listened on x for a signal that only exists on y and reported
        # "no contact detected" - the mode search picking a cross-axis mode was
        # self-defeating.  HaltingContactProbe already takes the detect axis
        # (out_idx) and the vibrate axis separately, so this just propagates the
        # right one.
        if ranked:
            best_axis = ranked[0][4]
            if best_axis != accel_axis:
                gcmd.respond_info(
                    "Mode select: %.1f Hz is detected on the %s axis (the scan"
                    " swept %s); probing will read %s"
                    % (best_freq, best_axis, accel_axis, best_axis))
            accel_axis = best_axis
        return accel_axis, best_freq, ranked, candidates[0][0]

    # Extracted from _find_best_probe_freq so a FIXED candidate list (resolved
    # ONCE, e.g. at the mesh center) can be re-evaluated at other points
    # without re-running the swept PSD scan there too - see SURVEY_MESH, which
    # needs exactly this: the same candidates tested at every point to build a
    # point x frequency reliability table, not a fresh candidate search per
    # point.  Returns (best_freq, ranked, ambiguous, attempts) - 'attempts' is
    # the total number of finder descents needed across all escalations, a
    # reliability signal in its own right (see resonance-nozzle-probe memory:
    # a frequency that only lands a clean reading after several retries is
    # objectively less reliable than one that verifies clean on the first
    # try, even if their final detectability numbers end up similar).
    # Descend REPS times through AIR ONLY (per-axis floors = 1.0 so the halt can
    # never fire; the floor 'air_floor' stays above the bed) at freq/out_idx, and
    # return (floors, ceilings): each axis's worst-case live-gradient noise
    # (ceiling) and the halt floor derived from it (ceiling*factor+margin, capped).
    # Shared by RESONANCE_PROBE_CHARACTERIZE_NOISE and the RANK_FREQ/CALIBRATE
    # contact-find, so both gate each axis by its OWN descent noise.
    #
    def _air_noise_floors(self, gcmd, chip, out_idx, axis, freq, accel_per_hz,
                          warmup, speed, x0, y0, ceiling, air_floor, reps,
                          factor, margin, floor_cap, drip_time):
        amp = accel_per_hz / (4. * math.pi**2 * freq)
        helper = HaltingContactProbe(
            self.printer, chip, out_idx, axis, freq, accel_per_hz, amp,
            air_floor, warmup, 0.10, 0.5, 5., 0.01, 0.05, 0.,
            halt_sensitivity_axis=[1., 1., 1.])
        toolhead = self.printer.lookup_object('toolhead')
        lift = min(self.move_speed, 10.)
        distance = max(0.3, ceiling - air_floor)
        ceilings = [0., 0., 0.]
        for _ in range(reps):
            helper._descend_once(gcmd, ceiling, distance, speed, lift, lift,
                                 x0, y0, drip_time=drip_time)
            md = helper.last_maxdrop or [0., 0., 0.]
            for a in range(3):
                ceilings[a] = max(ceilings[a], md[a])
        toolhead.manual_move([x0, y0, ceiling], lift)
        toolhead.wait_moves()
        floors = [min(floor_cap, round(ceilings[a] * factor + margin, 3))
                  for a in range(3)]
        return floors, ceilings

    # Per-axis halt floors derived from each axis's own SIGNAL-TO-NOISE, once a
    # real contact has been measured.  This keeps the live halt MULTI-AXIS - a
    # cross-axis mode is a legitimate and sometimes better detector - while
    # letting an axis that cannot possibly work disarm ITSELF from data rather
    # than being excluded by a blanket rule.
    #
    # For each axis: rejecting noise needs a floor above ceiling*factor+margin;
    # catching contact needs a floor at or below about half the measured drop.
    # When those two demands cross, no floor can do both and the axis is set to
    # an unreachable 0.95 - not as a policy, but because that is what its own
    # numbers say.  Worked example at 73.3 Hz (measured):
    #     x: drop 12%, noise 0.7% -> half-drop 6.0% clears noise  -> armed at 6%
    #     y: drop  0%, noise 11.9% -> half-drop 0%   cannot clear -> 0.95
    #     z: drop  0%, noise  4.7% -> half-drop 0%   cannot clear -> 0.95
    # and pre-belt-tightening, when contact damped Y by 66%, the same rule arms
    # Y instead - which is the behaviour a fixed "excitation axis only" rule
    # would have thrown away.
    def _snr_floors(self, ceilings, all_axes, factor, margin, floor_cap):
        UNREACHABLE, SNR_MARGIN = 0.95, 1.15
        floors, why = [], []
        for a, name in enumerate('xyz'):
            ceil = ceilings[a]
            drop = (all_axes.get(name) or {}).get('drop', 0.)
            f_noise = ceil * factor + margin       # must exceed noise
            f_cap = 0.5 * drop                     # must be catchable
            if f_cap > ceil * SNR_MARGIN:
                floors.append(min(floor_cap, round(max(min(f_noise, f_cap),
                                                       ceil * SNR_MARGIN), 3)))
                why.append("%s=%.0f%% (drop %.0f%% vs noise %.0f%%)"
                           % (name, floors[-1] * 100., drop * 100.,
                              ceil * 100.))
            else:
                floors.append(UNREACHABLE)
                why.append("%s=off (drop %.0f%% cannot clear noise %.0f%%)"
                           % (name, drop * 100., ceil * 100.))
        return floors, why

    # Decide whether a VERIFY-confirmed contact's per-mode ranking is
    # trustworthy.  'trial' is a list of (freq, drop, noise, detectability,
    # axis).  Because the contact was already verified (air-vs-press) by the
    # finder's HaltingContactProbe, a clean mode is a real surface - there is no
    # loudest-in-air "primary" gate (that wrongly rejected contact when the
    # loudest air mode does not damp, e.g. after a belt change).  Returns
    # (verdict, pick): 'clean' + best-detectability clean mode -> accept;
    # 'none' + None when nothing damps (<3% on every candidate) -> escalate to
    # the next finder frequency; 'weak' + best-drop mode when there is some
    # damping but nothing clean -> caller's best-effort / nudge path.  Shared by
    # the RANK_FREQ finder and the CALIBRATE_MESH _mode_low_first finder.
    def _accept_ranked_contact(self, trial, min_drop, target_noise):
        best_drop = max((d for (_f, d, _n, _s, _a) in trial), default=0.)
        clean = [e for e in trial
                 if e[1] >= min_drop and e[2] <= target_noise]
        if clean:
            return 'clean', max(clean, key=lambda e: e[3])
        if best_drop < 0.03:
            return 'none', None
        return 'weak', max(trial, key=lambda e: e[1])

    def _finder_rank_at_point(self, gcmd, chip, accel_axis, axis, candidates):
        toolhead = self.printer.lookup_object('toolhead')
        pos = toolhead.get_position()
        x0, y0, ceiling = pos[0], pos[1], pos[2]
        # Configured accel_per_hz, not the cap - a stronger excitation shrinks
        # the contact drop as a fraction of it (see _calibrate_contact).
        cap = gcmd.get_float("ACCEL_PER_HZ",
                             min(self.accel_per_hz, self.max_accel_per_hz),
                             above=0.)
        # Same paper-gauge safety floor as CALIBRATE (see _calibrate_contact).
        z_min = gcmd.get_float("CONTACT_ZMIN", -0.2)
        if ceiling <= z_min:
            raise gcmd.error("Mode select: start Z %.3f is at/below the floor"
                             " %.3f; raise the nozzle" % (ceiling, z_min))
        warmup = gcmd.get_float("CONTACT_WARMUP", 0.5, above=0.05)
        speed = gcmd.get_float("CONTACT_SPEED", 0.1, above=0., maxval=5.)
        up_margin = gcmd.get_float("CONTACT_UP", 0.10, above=0.)
        down_margin = gcmd.get_float("CONTACT_DOWN", 0.04, above=0.)
        cycles = gcmd.get_int("CONTACT_CYCLES", 4, minval=1)
        min_drop = gcmd.get_float("CONTACT_MIN_DROP", 0.10, above=0., below=1.)
        target_noise = gcmd.get_float("CONTACT_TARGET_NOISE", 0.06,
                                      above=0., below=1.)
        lift = min(self.move_speed, 10.)
        out_idx = {'x': 0, 'y': 1, 'z': 2}[accel_axis]
        # Overrun protection for the vibrating descent (see CALIBRATE_MESH /
        # _mode_low_first's docstring) - the strongest candidate here can easily
        # be a high-frequency mode, exactly as prone to the "Timer too close"
        # shutdown as CALIBRATE_MESH's own descents.
        drip_time = gcmd.get_float("DRIP_TIME", 0.3, minval=0.) or None
        # Escalation ORDER for the contact-find: try order[0] first (the highest
        # mode by default; MODE_ORDER=low for lowest-first) and step through the
        # rest until one lands a VERIFY-confirmed contact.  The old "primary mode
        # must confirm" arbiter is gone - finder.run()'s verify already rejects
        # shallow false halts, so once contact is confirmed the ranking alone
        # picks the best mode (see the acceptance below).
        high_first = gcmd.get("MODE_ORDER", "high").lower() != "low"
        order = sorted(candidates, key=lambda c: c[0])
        if high_first:
            order = list(reversed(order))
        distance = max(2.0, ceiling - z_min)
        contact_z, ranked, ambiguous = None, None, False
        attempts = 0
        noise_reps = gcmd.get_int("CONTACT_NOISE_REPS", 5, minval=2, maxval=10)
        # Characterize noise close to the bed, where the cross axes actually
        # misbehave (see _calibrate_contact).
        air_stop = gcmd.get_float("CONTACT_AIR_FLOOR", 0.15)
        air_floor = max(z_min, min(ceiling - 0.3, air_stop))
        for finder_idx, (finder_f, _pw) in enumerate(order):
            amp = cap / (4. * math.pi**2 * finder_f)
            # Gate each axis by its OWN near-bed descent noise at THIS candidate
            # frequency.  ALL axes stay armed: the finder's whole job is to find
            # out which axis a mode damps, so disarming any of them up front
            # would hide the very cross-axis modes worth discovering.  False
            # halts are handled by the verify pass, not by exclusion.
            floors, ceils = self._air_noise_floors(
                gcmd, chip, out_idx, axis, finder_f, cap, warmup, speed,
                x0, y0, ceiling, air_floor, noise_reps, 1.3, 0.03, 0.9,
                drip_time)
            gcmd.respond_info(
                "Mode select: %.1f Hz floors x=%.0f%% y=%.0f%% z=%.0f%%"
                " (near-bed noise x=%.0f%% y=%.0f%% z=%.0f%%)"
                % (finder_f, floors[0] * 100., floors[1] * 100.,
                   floors[2] * 100., ceils[0] * 100., ceils[1] * 100.,
                   ceils[2] * 100.))
            finder = HaltingContactProbe(
                self.printer, chip, out_idx, axis, finder_f, cap, amp, z_min,
                warmup, 0.06, 0.15, 5., 0.01, 0.05, 0.,
                halt_sensitivity_axis=floors)
            for attempt in range(3):
                attempts += 1
                cz, halted = finder.run(gcmd, ceiling, distance, speed, lift,
                                        lift, x0, y0, drip_time=drip_time)
                if cz is None:
                    gcmd.respond_info(
                        "Mode select: %.1f Hz found no contact (attempt %d/3,"
                        " halted=%s)" % (finder_f, attempt + 1, halted))
                    continue
                trial = self._rank_modes_by_damping(
                    gcmd, chip, accel_axis, candidates, cz, x0, y0, cap, z_min,
                    warmup, lift, up_margin, down_margin, cycles,
                    min_drop, target_noise)
                verdict, pick = self._accept_ranked_contact(
                    trial, min_drop, target_noise)
                if verdict == 'clean':
                    contact_z, ranked, ambiguous = cz, trial, False
                    gcmd.respond_info(
                        "Mode select: contact at z=%.4f confirmed (best %.1f Hz"
                        " drop=%.0f%% headroom=%+.1fpp)"
                        % (cz, pick[0], pick[1] * 100., pick[3] * 100.))
                    break
                if verdict == 'none':
                    # Nothing damps on any candidate at this contact - genuine
                    # contact would show meaningful damping on at least the mode
                    # that couples to it, so this is a false/shallow halt for
                    # this finder frequency; escalate to the next one.
                    gcmd.respond_info(
                        "Mode select: %.1f Hz halts at z=%.4f but NOTHING damps"
                        " there on any candidate - false contact; escalating"
                        % (finder_f, cz))
                    break
                # 'weak': some real damping but nothing clean - keep as an
                # unverified best-effort only if nothing better ever turns up.
                if ranked is None or pick[1] > max(
                        d for (_f, d, _n, _s, _a) in ranked):
                    contact_z, ranked, ambiguous = cz, trial, True
                gcmd.respond_info(
                    "Mode select: contact at z=%.4f shows some damping (best"
                    " %.0f%%) but nothing clean - keeping as unverified"
                    " best-effort, retrying for a cleaner reading"
                    % (cz, pick[1] * 100.))
            if ranked is not None and not ambiguous:
                break
            if finder_idx < len(order) - 1:
                gcmd.respond_info(
                    "Mode select: %.1f Hz never found a confirmed contact;"
                    " escalating the contact-find to %.1f Hz"
                    % (finder_f, order[finder_idx + 1][0]))
        toolhead.manual_move([x0, y0, ceiling], lift)
        toolhead.wait_moves()
        if ranked is None:
            raise gcmd.error("Mode select: no confirmed contact found across"
                             " any candidate after escalating through all of"
                             " them; lower the start height or floor")
        if contact_z < z_min + down_margin:
            raise gcmd.error("Mode select: contact z=%.4f leaves no room"
                             " above the floor %.3f for a %.3fmm contact dwell"
                             " (bed too low or detection failed)"
                             % (contact_z, z_min, down_margin))
        if ambiguous:
            gcmd.respond_info(
                "Mode select: WARNING - no candidate ever read fully clean at"
                " z=%.4f; the ranking below is a best-effort, verify before"
                " trusting it" % contact_z)
        gcmd.respond_info(
            "Mode select: ranking by live headroom (best first):\n"
            + "\n".join("  %.1f Hz: drop=%.0f%% noise=%.1f%% headroom=%+.1fpp"
                        " axis=%s"
                        % (f, d*100., n*100., s*100., a)
                        for (f, d, n, s, a) in ranked))
        return ranked[0][0], ranked, ambiguous, attempts

    cmd_CHARACTERIZE_NOISE_help = (
        "Descend in AIR (no contact, no halt) and measure each accelerometer"
        " axis's live-gradient noise ceiling, then set per-axis halt_sensitivity"
        " floors above it so a quiet axis stays sensitive while a twitchy axis is"
        " ignored. SAVE_CONFIG to persist.")
    def cmd_CHARACTERIZE_NOISE(self, gcmd):
        chip = self._lookup_chip(gcmd)
        axis = _parse_axis(gcmd, gcmd.get("AXIS", "x").lower())
        self._check_axis_safety(gcmd, axis)
        self._move_to_point(gcmd)
        freq = gcmd.get_float("FREQ", None, above=1., maxval=300.)
        if freq is None:
            accel_axis, freq = self._find_resonance(gcmd, chip, axis)
        else:
            vname = axis.get_name()
            default_axis = vname if vname in ('x', 'y', 'z') else 'x'
            accel_axis = gcmd.get("ACCEL_AXIS", default_axis).lower()
            if accel_axis not in ('x', 'y', 'z'):
                raise gcmd.error("ACCEL_AXIS must be x, y, or z")
        accel_per_hz = gcmd.get_float("ACCEL_PER_HZ", self.accel_per_hz,
                                      above=0., maxval=self.max_accel_per_hz)
        speed = gcmd.get_float("SPEED", 0.3, above=0., maxval=5.)
        reps = gcmd.get_int("REPS", 5, minval=2, maxval=20)
        warmup = gcmd.get_float("WARMUP", 0.8, above=0.05)
        # The floor stays ABOVE the bed so the descent never contacts - we are
        # measuring in-air descent noise ONLY.  With a paper-gauge Z=0 the bed is
        # ~-0.2, so a +0.1 air floor keeps ~0.3mm of clearance.
        air_zmin = gcmd.get_float("AIR_ZMIN", 0.1)
        factor = gcmd.get_float("FACTOR", 1.3, minval=1.)
        margin = gcmd.get_float("MARGIN", 0.03, minval=0.)
        floor_cap = gcmd.get_float("MAX_FLOOR", 0.9, above=0., below=1.)
        drip_time = gcmd.get_float("DRIP_TIME", 0.3, minval=0.) or None
        out_idx = {'x': 0, 'y': 1, 'z': 2}[accel_axis]
        toolhead = self.printer.lookup_object('toolhead')
        x, y, ceiling = toolhead.get_position()[:3]
        if ceiling <= air_zmin:
            raise gcmd.error("Characterize noise: start Z %.3f is at/below the air"
                             " floor %.3f; raise POINT's Z or lower AIR_ZMIN"
                             % (ceiling, air_zmin))
        floors, ceilings = self._air_noise_floors(
            gcmd, chip, out_idx, axis, freq, accel_per_hz, warmup, speed,
            x, y, ceiling, air_zmin, reps, factor, margin, floor_cap, drip_time)
        configfile = self.printer.lookup_object('configfile')
        for a, ax in enumerate('xyz'):
            configfile.set('resonance_probe', 'halt_sensitivity_%s' % ax,
                           "%.3f" % floors[a])
        # Apply live too, so the floors take effect for probes THIS session
        # (configfile.set only persists on SAVE_CONFIG + restart).
        rp = self.printer.lookup_object('resonance_probe', None)
        if rp is not None:
            rp.halt_sensitivity_axis = list(floors)
        gcmd.respond_info(
            "Characterize noise: air ceilings x=%.0f%% y=%.0f%% z=%.0f%% over %d"
            " reps -> floors x=%.3f y=%.3f z=%.3f (ceiling*%.2f+%.2f, cap %.2f)."
            " Run SAVE_CONFIG to persist halt_sensitivity_x/y/z."
            % (ceilings[0] * 100., ceilings[1] * 100., ceilings[2] * 100., reps,
               floors[0], floors[1], floors[2], factor, margin, floor_cap))

    cmd_RANK_FREQ_help = ("Sweep for candidate resonance modes and rank them by"
                          " how strongly each damps on bed contact (picks the"
                          " best probe frequency, not just the loudest in air)")
    def cmd_RANK_FREQ(self, gcmd):
        chip = self._lookup_chip(gcmd)
        axis = _parse_axis(gcmd, gcmd.get("AXIS", "x").lower())
        self._check_axis_safety(gcmd, axis)
        self._move_to_point(gcmd)
        # SCAN_ONLY: report the mode landscape (air-only sweep) and stop, with no
        # contact - for surveying how the resonance frequency varies across the
        # bed without touching the platform at every point.
        if gcmd.get_int("SCAN_ONLY", 0):
            accel_axis, candidates = self._resolve_candidates(gcmd, chip, axis)
            label = gcmd.get("LABEL", "")
            gcmd.respond_info("Scan%s: dominant %.1f Hz (accel %s-axis)"
                              % ((" " + label) if label else "",
                                 candidates[0][0], accel_axis))
            return
        accel_axis, best_freq, ranked, loudest_freq = self._find_best_probe_freq(
            gcmd, chip, axis)
        if ranked is None:
            return  # only one candidate; already reported
        gcmd.respond_info(
            "Rank: best probe frequency = %.1f Hz (accel %s-axis). The loudest"
            " in-air peak was %.1f Hz; use FREQ=%.1f for CALIBRATE/probing."
            % (best_freq, accel_axis, loudest_freq, best_freq))

    cmd_SURVEY_MESH_help = (
        "Sweep ONCE at the mesh center for candidate frequencies, then test"
        " EVERY candidate at EVERY [bed_mesh] grid point - reports a full"
        " point x frequency table of drop/noise/detectability plus false-halt"
        " counts, for choosing a frequency with real data instead of trusting"
        " a single point (report-only; does not save config - see"
        " CALIBRATE_MESH to actually commit a frequency and write freq_mesh)")
    def cmd_SURVEY_MESH(self, gcmd):
        chip = self._lookup_chip(gcmd)
        axis = _parse_axis(gcmd, gcmd.get("AXIS", "x").lower())
        self._check_axis_safety(gcmd, axis)
        bed_mesh = self.printer.lookup_object('bed_mesh', None)
        if bed_mesh is None:
            raise gcmd.error("SURVEY_MESH needs a [bed_mesh] section to define"
                             " the probe grid")
        bmc = bed_mesh.bmc
        xmin, ymin = bmc.mesh_min
        xmax, ymax = bmc.mesh_max
        xc, yc = bmc.mesh_config['x_count'], bmc.mesh_config['y_count']
        mesh_z = gcmd.get_float("MESH_Z", 2.0)
        which = gcmd.get("CONTACT_POINTS", "all").lower()
        toolhead = self.printer.lookup_object('toolhead')
        coord = lambda i, lo, hi, n: lo if n <= 1 else lo + (hi - lo) * i / (n - 1)
        cx = coord(xc // 2, xmin, xmax, xc)
        cy = coord(yc // 2, ymin, ymax, yc)
        toolhead.manual_move([cx, cy, mesh_z], self.move_speed)
        toolhead.wait_moves()
        # ONE candidate-finding sweep, at the center - shared across every
        # point below (see _finder_rank_at_point's docstring for why: this is
        # what makes the full point x frequency table affordable instead of
        # re-running a swept PSD scan at every single point too).
        accel_axis, candidates = self._resolve_candidates(gcmd, chip, axis)
        if len(candidates) < 2:
            gcmd.respond_info(
                "SURVEY_MESH: only one candidate mode found (%.1f Hz) -"
                " nothing to compare across points" % (candidates[0][0],))
            return
        gcmd.respond_info(
            "SURVEY_MESH: candidates from the center (%.1f, %.1f): %s Hz"
            % (cx, cy, ", ".join("%.1f" % f for f, _p in candidates)))
        if which == "corners":
            ijs = sorted(set([(xc // 2, yc // 2), (0, 0), (xc - 1, 0),
                              (0, yc - 1), (xc - 1, yc - 1)]))
        else:
            ijs = [(ix, iy) for iy in range(yc)
                   for ix in (range(xc) if iy % 2 == 0
                              else range(xc - 1, -1, -1))]
        table = {}                # (x, y) -> {freq: (drop, noise, snr, axis)}
        ambiguous_points = []
        for (ix, iy) in ijs:
            x = coord(ix, xmin, xmax, xc)
            y = coord(iy, ymin, ymax, yc)
            toolhead.manual_move([x, y, mesh_z], self.move_speed)
            toolhead.wait_moves()
            try:
                _best, ranked, ambiguous, attempts = self._finder_rank_at_point(
                    gcmd, chip, accel_axis, axis, candidates)
            except gcmd.error as e:
                gcmd.respond_info("  point (%.1f, %.1f): FAILED - %s"
                                  % (x, y, str(e)))
                table[(x, y)] = None
                ambiguous_points.append((x, y))
                continue
            table[(x, y)] = {f: (d, n, s, a) for (f, d, n, s, a) in ranked}
            if ambiguous:
                ambiguous_points.append((x, y))
            gcmd.respond_info(
                "  point (%.1f, %.1f) [%d finder attempt(s)]%s:\n"
                % (x, y, attempts,
                   " [AMBIGUOUS - no candidate read fully clean]"
                   if ambiguous else "")
                + "\n".join(
                    "    %.1f Hz: drop=%.0f%% noise=%.1f%%"
                    " headroom=%+.1fpp axis=%s"
                    % (f, d * 100., n * 100., s * 100., a)
                    for (f, d, n, s, a) in ranked))
        gcmd.respond_info(
            "SURVEY_MESH summary (per-candidate, across %d point(s)):"
            % len(ijs))
        for f, _p in candidates:
            drops = [v[f][0] for v in table.values() if v and f in v]
            noises = [v[f][1] for v in table.values() if v and f in v]
            snrs = [v[f][2] for v in table.values() if v and f in v]
            if not drops:
                gcmd.respond_info("  %.1f Hz: no data (every point failed)"
                                  % f)
                continue
            gcmd.respond_info(
                "  %.1f Hz: drop %.0f-%.0f%% (avg %.0f%%), noise"
                " %.1f-%.1f%% (avg %.1f%%), headroom %+.1f..%+.1fpp"
                " (avg %+.1fpp) over %d/%d point(s)"
                % (f, min(drops) * 100., max(drops) * 100.,
                   (sum(drops) / len(drops)) * 100.,
                   min(noises) * 100., max(noises) * 100.,
                   (sum(noises) / len(noises)) * 100.,
                   min(snrs) * 100., max(snrs) * 100.,
                   (sum(snrs) / len(snrs)) * 100.,
                   len(drops), len(ijs)))
        if ambiguous_points:
            gcmd.respond_info(
                "SURVEY_MESH: %d point(s) never read fully clean for any"
                " candidate: %s"
                % (len(ambiguous_points),
                   ", ".join("(%.1f, %.1f)" % p for p in ambiguous_points)))
        toolhead.manual_move([cx, cy, mesh_z], self.move_speed)
        toolhead.wait_moves()

    # Whole-mesh shortcuts applied to already per-row-decided rows: every row a
    # single value AND all equal -> a 1x1 scalar; all rows identical -> a single
    # 1xM row (Y-independent); else the ragged rows as given.
    def _finalize_collapse(self, rows, tol):
        mean = lambda v: sum(v) / len(v)
        if all(len(r) == 1 for r in rows):
            vals = [r[0] for r in rows]
            if max(vals) - min(vals) <= tol:
                return [[mean(vals)]], "constant 1x1"
            return rows, "X-independent %dx1" % (len(rows),)
        w = len(rows[0])
        if w > 1 and all(len(r) == w for r in rows) and all(
                max(rows[k][j] for k in range(len(rows)))
                - min(rows[k][j] for k in range(len(rows))) <= tol
                for j in range(w)):
            return [[mean([rows[k][j] for k in range(len(rows))])
                     for j in range(w)]], "Y-independent 1x%d" % (w,)
        counts = ",".join(str(len(r)) for r in rows)
        return rows, "ragged, rows[%s] cols, %d points" % (
            counts, sum(len(r) for r in rows))

    # Identify candidate resonance modes with a driven scan RESTRICTED to narrow
    # windows around each swept-PSD candidate, instead of the full [f_lo, f_hi]
    # range.  The continuous sweep is fast and already narrows down roughly where
    # the modes are; the slower discrete driven scan only needs to refine/confirm
    # near those peaks and catch nearby distinct modes the sweep's frequency
    # resolution may have blurred together - not re-scan frequencies nothing
    # already pointed to (a full-range discrete scan over the wide 5-135 Hz
    # default is what made this slow).  Overlapping/adjacent windows are merged
    # so a shared frequency is not measured twice.  Peaks are found per-window
    # (never comparing across a gap between two disjoint windows) and then
    # merged/thresholded together, same as the old full-range scan.  Returns the
    # candidate frequencies sorted ascending.
    def _driven_refine(self, gcmd, chip, axis, out_idx, centers, window, f_lo,
                       f_hi, step, aph, dur):
        ranges = sorted((max(f_lo, c - window), min(f_hi, c + window))
                        for c in centers)
        merged_ranges = []
        for lo, hi in ranges:
            if merged_ranges and lo <= merged_ranges[-1][1] + step / 2.:
                merged_ranges[-1] = (merged_ranges[-1][0],
                                     max(merged_ranges[-1][1], hi))
            else:
                merged_ranges.append((lo, hi))
        all_peaks = []
        all_resp = []
        for lo, hi in merged_ranges:
            n = max(1, int(round((hi - lo) / step)))
            band = [lo + k * step for k in range(n + 1)]
            resp = []
            for f in band:
                seq = self._gen_fixed_freq(f, aph * f, dur)
                aclient = self._vibrate_and_collect(_QuietGCmd(gcmd), chip, axis,
                                                     seq)
                resp.append(self._amplitude_at_freq(aclient.get_samples(),
                                                    f)[0][out_idx])
            all_resp.extend(resp)
            if len(band) == 1:
                all_peaks.append((band[0], resp[0]))
                continue
            for j in range(1, len(band) - 1):
                if resp[j] > resp[j-1] and resp[j] >= resp[j+1]:
                    all_peaks.append((band[j], resp[j]))
        mx = max(all_resp) if all_resp else 1e-9
        mx = mx or 1e-9
        chosen = [(f, p) for f, p in all_peaks if p >= 0.3 * mx]
        chosen.sort(key=lambda p: -p[1])
        merged = []
        for f, _r in chosen:
            if all(abs(f - m) > 3. for m in merged):
                merged.append(f)
        if not merged and all_peaks:
            merged = [max(all_peaks, key=lambda p: p[1])[0]]
        gcmd.respond_info(
            "Driven refine (%d window(s) around %s Hz): candidate modes %s Hz"
            % (len(merged_ranges), ", ".join("%.1f" % c for c in centers),
               ", ".join("%.1f" % f for f in sorted(merged))))
        return sorted(merged)

    # Choose the excitation frequency at one point by CONTACT DAMPING, low-first.
    # candidates are sorted ascending.  Contact is FOUND with the lowest mode -
    # the reliable damper - via a real halting descent (so the surface is located
    # safely).  Then each candidate low->high is measured only by the BOUNDED
    # oscillation of characterize_amplitude around that known contact_z (never a
    # blind high-mode descent that could sail through contact and press the bed),
    # and the LOWEST candidate that damps cleanly (drop>=min_drop, noise<=target)
    # is accepted.  Falls back to the strongest damper if none pass.
    #
    # CONTACT-FIND ESCALATION: the lowest candidate is not guaranteed to couple
    # well at every point (a mode's strength can vary across the bed).  If it
    # never detects a real contact (3 attempts each: either no halt at all, or a
    # halt that turns out to be a false early trigger - ~0% damping at every
    # candidate), escalate the FINDER frequency to the next candidate up rather
    # than giving up on the whole point.  This still only touches the descent
    # search itself; the z_min floor bounds over-travel the same way regardless
    # of which candidate is doing the finding, and the per-candidate damping
    # comparison below always prefers the lowest clean mode once contact is
    # located, so escalation here does not bias the final frequency choice.
    # Returns (chosen_freq, contact_z, clean).  'clean' is True only when a
    # candidate genuinely detected (drop>=min_drop AND noise<=target); when no
    # candidate is clean it returns the strongest damper with clean=False (the
    # caller should treat that as unresolved - e.g. nudge off the point and
    # retry - rather than trust it), and (None, None, False) if no candidate
    # ever located the surface at all.
    #
    # DESCENT-HEIGHT TIGHTENING: 'ceiling' is only the WORST-CASE starting
    # height (unknown bed).  Once ANY contact_z is observed for this point -
    # even one later judged a false early halt - it is a much better estimate
    # of where the surface actually is than the original ceiling.  Every
    # subsequent retry/escalation attempt at this point starts from
    # max(z_min, contact_z) + margin instead of the original ceiling, so the
    # descent only re-covers the (small) uncertain band near the surface, not
    # the whole original clearance.  The ceiling only ever tightens (never
    # loosens) within one point's search, and the hard z_min floor is
    # unchanged either way.
    #
    # THE PRIMARY MODE IS THE CONTACT ARBITER (guards against spurious wins at a
    # false halt).  A weak, position-sensitive mode can blip a big "drop" at a
    # SHALLOW FALSE HALT (a Z above the true surface) where the reliable primary
    # mode reads ~0%.  Accepting that blip picks a bad probe frequency at a false
    # height.  So a clean detection is trusted only when the surface is really
    # reached: either the PRIMARY candidate (order[0]) is itself the clean one, or
    # it at least AGREES that we are on the bed (damps >= a small confirm fraction
    # of min_drop).  A halt where ONLY a non-primary mode reads clean while the
    # primary is silent is NOT accepted - we retry for the deeper contact where the
    # primary engages, and if it never does, return unresolved so the caller nudges
    # off this spot.  (We deliberately do NOT accept a primary-silent reading just
    # because its Z repeats: a PERSISTENT shallow false halt is itself Z-repeatable,
    # so Z-repeatability cannot tell it apart from a genuine primary-silent point.)
    #
    # By default the candidates are tested HIGHEST-first (high_first), so the
    # primary is the highest mode - now the most reliable contact detector (it
    # damps strongly on contact across surfaces, including slick plates where the
    # low modes barely couple, so it confirms contact better than the low mode
    # did).  Pass high_first=False to fall back to the original lowest-first order.
    #
    # FALSE-HALT with NO clean candidate: ~0% drop at every candidate is
    # already sufficient evidence of a false/shallow halt for this finder
    # frequency (every candidate is an independent detector, so genuine
    # contact would show meaningful damping on at least one of them) -
    # escalate to the next candidate immediately rather than waiting for the
    # height to repeat.
    # (accel_per_hz, displacement amplitude) for a self-consistent SHM excitation
    # at frequency f.  Klipper's accel_per_hz (aph) fixes peak accel = aph*f, which
    # for SHM means amplitude = aph/(4*pi^2*f) and peak velocity = aph/(2*pi).
    #  excite_amp None/0 -> LEGACY: constant aph=cap, so the displacement amplitude
    #    shrinks as ~1/f and peak velocity is constant across modes.
    #  excite_amp > 0    -> FIXED DISPLACEMENT: hold the lateral amplitude at
    #    excite_amp for EVERY candidate (aph = amp*4*pi^2*f rises with f, and so do
    #    peak velocity ~f and peak accel ~f^2).  This makes the cross-mode damping
    #    comparison fair (equal displacement) and keeps high modes off the
    #    microstep-quantization floor.  aph is capped at the machine limit 'cap';
    #    above f = cap/(amp*4*pi^2) the amplitude falls back to the accel-limited
    #    value (see the clamp warning in cmd_CALIBRATE_MESH).
    def _excite_params(self, f, cap, excite_amp):
        if not excite_amp:
            return cap, cap / (4. * math.pi**2 * f)
        aph = min(cap, excite_amp * 4. * math.pi**2 * f)
        return aph, aph / (4. * math.pi**2 * f)

    # A RECOMMENDED detection-limited descent speed for a mode of frequency f.  The
    # halt window spans (detect_cycles / f) * descend_speed in Z, and must stay
    # well inside the ~0.1mm contact transition for a sharp amplitude edge, so the
    # max reliable speed scales with frequency.  This is only reported (for the
    # user to set CONTACT_SPEED / the probe's descend_speed); descent speed,
    # amplitude and frequency stay independent knobs - the probe never auto-couples
    # speed to frequency.  detect_cycles matches the finder's halt window (5).
    RECOMMEND_TRANSITION_Z = 0.1    # mm, approx contact-transition width
    RECOMMEND_MARGIN = 0.6          # keep the window well inside that transition
    RECOMMEND_DETECT_CYCLES = 5.    # halt-window cycles (matches the finder)
    def _recommend_descend_speed(self, f):
        return (self.RECOMMEND_MARGIN * self.RECOMMEND_TRANSITION_Z
                * f / self.RECOMMEND_DETECT_CYCLES)

    def _mode_low_first(self, gcmd, chip, out_idx, axis, x, y, candidates,
                        ceiling, cap, z_min, warmup, speed, lift, up_margin,
                        down_margin, cycles, min_drop, target_noise,
                        margin=1.0, travel=None, vib_span=None,
                        drip_time=None, excite_amp=None, high_first=True):
        toolhead = self.printer.lookup_object('toolhead')
        cur_ceiling = ceiling
        distance = max(2.0, cur_ceiling - z_min)
        noise_reps = gcmd.get_int("CONTACT_NOISE_REPS", 5, minval=2, maxval=10)
        # Speed for the (lateral, above-bed) move to each point's start height.
        # The near-bed Z lift stays at 'lift'; only this travel is unthrottled.
        start_speed = travel if travel is not None else lift
        overall = None  # (fq, drop, contact_z) - strongest drop seen anywhere
        # Candidates arrive ascending; the finder tries them in 'order' (HIGHEST
        # first by default) and escalates until one lands a VERIFY-confirmed
        # contact.  Each candidate's contact-find is gated by per-axis air-noise
        # floors, and acceptance is decided by _accept_ranked_contact - the same
        # shared logic as the RANK_FREQ finder (verify already rejects shallow
        # false halts, so no loudest-in-air "primary" arbiter is needed).  This
        # is SAFE at any frequency because the descent buffers the MCU further
        # ahead (drip_time) so a high mode's move-feed rate cannot overrun the
        # step pipeline.
        order = list(reversed(candidates)) if high_first else list(candidates)
        for finder_idx, finder_f in enumerate(order):
            f_aph, f_amp = self._excite_params(finder_f, cap, excite_amp)
            # Per-axis near-bed floors, all axes armed (same rationale as the
            # RANK_FREQ finder: this is where cross-axis modes get discovered).
            air_floor = max(z_min, min(cur_ceiling - 0.3, 0.15))
            floors, _c = self._air_noise_floors(
                gcmd, chip, out_idx, axis, finder_f, f_aph, warmup, speed,
                x, y, cur_ceiling, air_floor, noise_reps, 1.3, 0.03, 0.9,
                drip_time)
            for attempt in range(3):
                finder = HaltingContactProbe(
                    self.printer, chip, out_idx, axis, finder_f, f_aph,
                    f_amp, z_min, warmup, 0.06, 0.15, 5., 0.01, 0.05, 0.,
                    halt_sensitivity_axis=floors)
                contact_z, _halted = finder.run(gcmd, cur_ceiling, distance,
                                                speed, lift, start_speed,
                                                x, y, vib_span=vib_span,
                                                drip_time=drip_time)
                if contact_z is None:
                    gcmd.respond_info(
                        "    %.1f Hz: no contact detected (finder attempt"
                        " %d/3)" % (finder_f, attempt + 1))
                    continue
                tighter = max(z_min, contact_z) + margin
                if tighter < cur_ceiling:
                    cur_ceiling = tighter
                    distance = cur_ceiling - z_min
                    gcmd.respond_info(
                        "    contact seen near z=%.4f - tightening the next"
                        " descent to start at z=%.4f" % (contact_z, cur_ceiling))
                # Rank EVERY candidate at this (verify-confirmed) contact, then
                # let the shared decision accept / escalate / best-effort.
                trial = []
                for fq in order:
                    h_aph, h_amp = self._excite_params(fq, cap, excite_amp)
                    helper = HaltingContactProbe(
                        self.printer, chip, out_idx, axis, fq, h_aph,
                        h_amp, z_min, warmup, 0.06, 0.15, 5., 0.01, 0.05, 0.)
                    # Sweep amplitudes per candidate and rank by live headroom,
                    # exactly as the RANK_FREQ finder does - see its comment.
                    m = helper.characterize_amplitude(
                        gcmd, x, y, contact_z, lift, up_margin, down_margin,
                        cycles, gcmd.get_int("RANK_LEVELS", 3, minval=1,
                                             maxval=8),
                        min_drop, target_noise)
                    drop, noise = m['max_drop'], m['rel_noise']
                    det = m.get('headroom')
                    if det is None:
                        det = 0.5 * drop - (noise * 1.3 + 0.03)
                    trial.append((fq, drop, noise, det, m['axis']))
                    clean = drop >= min_drop and noise <= target_noise
                    gcmd.respond_info("    %.1f Hz: drop=%.0f%% noise=%.1f%%"
                                      " (axis=%s)%s"
                                      % (fq, drop * 100., noise * 100.,
                                         m['axis'],
                                         "  <- clean" if clean else ""))
                    if overall is None or drop > overall[1]:
                        overall = (fq, drop, contact_z)
                verdict, pick = self._accept_ranked_contact(
                    trial, min_drop, target_noise)
                if verdict == 'clean':
                    toolhead.manual_move([x, y, ceiling], lift)
                    toolhead.wait_moves()
                    return pick[0], contact_z, True
                if verdict == 'none':
                    gcmd.respond_info(
                        "    NOTHING damps at z=%.4f on any candidate - false"
                        " contact for %.1f Hz; escalating the finder"
                        % (contact_z, finder_f))
                    break
                # 'weak': some real damping but nothing clean -> best-effort so
                # the caller nudges off this spot.
                toolhead.manual_move([x, y, ceiling], lift)
                toolhead.wait_moves()
                return pick[0], contact_z, False
            if finder_idx < len(order) - 1:
                gcmd.respond_info(
                    "    %.1f Hz never found reliable contact; escalating the"
                    " contact-find to %.1f Hz"
                    % (finder_f, order[finder_idx + 1]))
        toolhead.manual_move([x, y, ceiling], lift)
        toolhead.wait_moves()
        # Contacted somewhere but never got a primary-confirmed clean detection ->
        # best-effort (clean=False) so the caller nudges; None only if the
        # surface was never located at all.
        if overall is not None:
            return overall[0], overall[2], False
        return None, None, False

    # Candidate lateral offsets (mm) to retry a point that gave no clean
    # detection: a small step that clears a local low-friction / dead spot
    # while keeping bed height (tilt over ~1mm is negligible) and resonance
    # (varies slowly across the bed) essentially unchanged - a far closer proxy
    # than an adjacent grid point.  Prioritized TOWARD the bed center (always
    # inward, so on-bed and safe); the remaining directions are the two
    # perpendiculars and away-from-center, each clamped to the mesh bounds and
    # skipped if clamping collapses it back onto the original point.
    def _nudge_positions(self, x, y, xmin, xmax, ymin, ymax, radius, tries):
        # Retry points on the circle of the given radius centered on the grid
        # point, at a RANDOM base angle with the tries spread evenly around it
        # (2*pi/tries apart).  Randomizing the direction spreads any surface
        # wear the repeated contacts create over the whole ring instead of one
        # spot, and makes it unlikely that successive nudges re-hit the same
        # plastic blob / dead spot that caused the original bad reading.  Each
        # point is bounds-clamped; one that clamps back onto the grid point
        # (radius fully off the bed) is skipped.
        if tries <= 0:
            return []
        base = random.uniform(0., 2. * math.pi)
        out = []
        for k in range(tries):
            ang = base + 2. * math.pi * k / tries
            nx = min(max(x + radius * math.cos(ang), xmin), xmax)
            ny = min(max(y + radius * math.sin(ang), ymin), ymax)
            if abs(nx - x) > 1e-3 or abs(ny - y) > 1e-3:
                out.append((nx, ny))
        return out

    cmd_CALIBRATE_MESH_help = (
        "Survey the resonance frequency across the [bed_mesh] grid and save a"
        " per-point freq_mesh to [resonance_probe], auto-collapsing any axis that"
        " is constant")
    def cmd_CALIBRATE_MESH(self, gcmd):
        chip = self._lookup_chip(gcmd)
        axis = _parse_axis(gcmd, gcmd.get("AXIS", "x").lower())
        self._check_axis_safety(gcmd, axis)
        bed_mesh = self.printer.lookup_object('bed_mesh', None)
        if bed_mesh is None:
            raise gcmd.error("CALIBRATE_MESH needs a [bed_mesh] section to"
                             " define the probe grid")
        # Reuse the configured [resonance_probe]'s tuned motion so calibration
        # moves like real probing and inherits the user's settings instead of
        # duplicating them.  Only the pure-travel speeds (lift/retract) and the
        # warmup are taken from the probe; the contact-descent speed and the
        # z_min floor stay calibration-specific for accuracy and hands-off
        # safety (see their gcmd defaults below).
        probe = self.printer.lookup_object('resonance_probe', None)
        if probe is None:
            raise gcmd.error("CALIBRATE_MESH needs a [resonance_probe] section")
        probe_params = probe.get_probe_params(gcmd)
        bmc = bed_mesh.bmc
        xmin, ymin = bmc.mesh_min
        xmax, ymax = bmc.mesh_max
        xc, yc = bmc.mesh_config['x_count'], bmc.mesh_config['y_count']
        # Start height for the contact descents (above a paper-gauged bed).
        mesh_z = gcmd.get_float("MESH_Z", 2.0)
        tol = gcmd.get_float("COLLAPSE_TOL", 2.0, minval=0.)
        # When a grid point gives no clean detection (a low-friction / dead
        # spot), retry up to NUDGE_TRIES times at points on a circle of
        # NUDGE_RADIUS mm around it, at random directions spread evenly around
        # the ring (see _nudge_positions), before accepting a best-effort value.
        # A radius ~= the nozzle tip's outer diameter clears the bad spot while
        # keeping the height/resonance change negligible.
        nudge_radius = gcmd.get_float("NUDGE_RADIUS", 1.0, above=0.)
        nudge_tries = gcmd.get_int("NUDGE_TRIES", 4, minval=0)
        # CONTACT_POINTS=all (default) tests every grid point; =corners tests only
        # the bed center + 4 corners (fast - if they agree the bed is uniform).
        which = gcmd.get("CONTACT_POINTS", "all").lower()
        # Driven-scan band for candidate-mode identification (surfaces the higher
        # modes a swept PSD misses).  Defaults span Klipper's own input-shaper
        # sweep range (5-135 Hz for X/Y - see shaper_calibrate.py/Config_Reference
        # .md) rather than a machine-specific band: real structural resonances can
        # legitimately fall anywhere in that range, so outlier/harmonic peaks are
        # rejected by contact-damping + low-first selection below, not by a
        # frequency cutoff.
        f_lo = gcmd.get_float("FREQ_START", 5., above=0.)
        f_hi = gcmd.get_float("FREQ_END", 135., above=f_lo + 1.)
        cand_step = gcmd.get_float("CAND_STEP", 1.0, above=0.)
        cand_dur = gcmd.get_float("CAND_DUR", 0.5, above=0.1)
        # Radius (Hz) of the discrete driven-refine window around each swept-PSD
        # candidate - not the whole [FREQ_START, FREQ_END] range (see
        # _driven_refine).
        cand_window = gcmd.get_float("CAND_WINDOW", 8., above=0.)
        # Safety margin (mm) kept above any already-observed contact height
        # when tightening the descent start - both within one point's retries
        # and across points once the bed's rough height is known (see
        # _mode_low_first and the point loop below).
        contact_margin = gcmd.get_float("CONTACT_MARGIN", 1.0, above=0.)
        # EXCITE_AMP (mm): hold a FIXED lateral displacement amplitude at every
        # candidate instead of a constant accel_per_hz (whose amplitude shrinks
        # ~1/f).  Fairer cross-mode damping comparison + keeps high modes off the
        # microstep floor.  Clamped by the machine's max accel_per_hz at high f
        # (see _excite_params); a warning is printed if that clamp bites.  0 = off.
        # Amplitude, frequency, and CONTACT_SPEED are all independent user knobs;
        # the frequency->descent-speed relationship is only a RECOMMENDATION,
        # reported per chosen mode (see _recommend_descend_speed) for the user to
        # set CONTACT_SPEED, never auto-applied.
        excite_amp = gcmd.get_float("EXCITE_AMP", 0., minval=0.) or None
        # MODE_ORDER=high (default) tests the candidate modes HIGHEST-first (the
        # high modes detect contact most reliably); =low is the original
        # lowest-first order.  See _mode_low_first.
        high_first = gcmd.get("MODE_ORDER", "high").lower() != "low"
        # Overrun protection for the vibrating descent (MCU "Timer too close" on a
        # long / high-frequency / no-contact descent).  Two options:
        #  * DRIP_TIME (default): buffer the MCU further ahead than the stock ~0.1s
        #    drip look-ahead so a busy host cannot starve the step pipeline.  This
        #    does NOT touch the vibrating descent, so it keeps the full-height
        #    ring-up and the best detection SNR.  Costs a little over-travel after
        #    the halt (~DRIP_TIME*CONTACT_SPEED, bounded by CONTACT_ZMIN; the exact
        #    contact Z comes from the anchored analysis, not the stop position).
        #  * VIB_SPAN: the alternative two-stage descent - vibrate only the final
        #    VIB_SPAN mm above the floor, reaching it via a fast non-vibrating
        #    approach.  Bounds the drip regardless of host speed, but the shorter
        #    vibrating run raises the measurement noise, so it is OFF by default
        #    (0).  If set, MUST exceed the bed's height variation.
        drip_time = gcmd.get_float("DRIP_TIME", 0.3, minval=0.)
        drip_time = drip_time if drip_time > 0. else None
        vib_span = gcmd.get_float("VIB_SPAN", 0., minval=0.)
        vib_span = vib_span if vib_span > 0. else None
        cand_aph = gcmd.get_float("MESH_APH", min(120., self.max_accel_per_hz),
                                  above=0., maxval=self.max_accel_per_hz)
        cap = self.max_accel_per_hz
        # Calibration-specific for accuracy/safety (deliberately NOT the probe's
        # faster descent or deeper z_min_position): a slow one-time descent finds
        # contact precisely, and the shallow paper-gauge floor bounds over-travel
        # (see _calibrate_contact).
        z_min = gcmd.get_float("CONTACT_ZMIN", -0.2)
        speed = gcmd.get_float("CONTACT_SPEED", 0.1, above=0., maxval=5.)
        # Warmup reused from the probe (its ring-up time is the same physics).
        warmup = gcmd.get_float("CONTACT_WARMUP", probe.warmup, above=0.05)
        up_margin = gcmd.get_float("CONTACT_UP", 0.10, above=0.)
        down_margin = gcmd.get_float("CONTACT_DOWN", 0.04, above=0.)
        cycles = gcmd.get_int("CONTACT_CYCLES", 4, minval=1)
        min_drop = gcmd.get_float("CONTACT_MIN_DROP", 0.10, above=0., below=1.)
        target_noise = gcmd.get_float("CONTACT_TARGET_NOISE", 0.06, above=0.,
                                      below=1.)
        toolhead = self.printer.lookup_object('toolhead')
        # Z retract/lift reused from the probe's lift_speed (near-bed, slow).
        lift = probe_params['lift_speed']
        # Above-bed lateral travel between probe points: the fast part of real
        # probing happens at the toolhead's max velocity (NOT a probe setting -
        # lift_speed/probe_start_speed are the near-bed speeds).  Default to
        # move_speed, capped at the machine's max velocity.
        vmax = toolhead.get_status(
            self.printer.get_reactor().monotonic())['max_velocity']
        travel_speed = gcmd.get_float("MESH_TRAVEL", min(self.move_speed, vmax),
                                      above=0.)
        coord = lambda i, lo, hi, n: lo if n <= 1 else lo + (hi - lo) * i / (n - 1)
        # Candidates: the swept PSD (reliable, and trends to the contact-friendly
        # low mode - a driven in-air peak can sit ~1 Hz high and miss contact) is
        # the primary source; add only DISTINCT higher modes (>3 Hz away) that a
        # driven scan surfaces.  Merged, sorted ascending for the low-first search.
        cx = coord(xc // 2, xmin, xmax, xc)
        cy = coord(yc // 2, ymin, ymax, yc)
        toolhead.manual_move([cx, cy, mesh_z], self.move_speed)
        toolhead.wait_moves()
        aname, swp = self._find_candidate_freqs(gcmd, chip, axis, 3, 0.1,
                                                quiet=True)
        out_idx = {'x': 0, 'y': 1, 'z': 2}[aname]
        drv = self._driven_refine(gcmd, chip, axis, out_idx,
                                  [f for f, _p in swp], cand_window, f_lo, f_hi,
                                  cand_step, cand_aph, cand_dur)
        # FREQ_START/FREQ_END just bound the search to Klipper's own input-shaper
        # sweep range by default; a bad high mode (weak/false contact damping) is
        # rejected below by _mode_low_first (primary-mode arbiter + clean-drop
        # test), not by narrowing this band.
        candidates = sorted(round(f, 1) for f, _p in swp if f_lo <= f <= f_hi)
        for f in sorted(drv):
            if f_lo <= f <= f_hi and all(abs(f - c) > 3. for c in candidates):
                candidates.append(round(f, 1))
        candidates.sort()
        if not candidates:
            raise gcmd.error("CALIBRATE_MESH: no candidate modes in %.0f-%.0f Hz;"
                             " widen FREQ_START/FREQ_END" % (f_lo, f_hi))
        gcmd.respond_info("Candidate modes (low->high): %s Hz"
                          % ", ".join("%.1f" % f for f in candidates))
        # Suggested (detection-limited) descent speed for EACH candidate - it
        # scales with frequency, so the high modes support proportionally faster
        # probing.  Only a recommendation for setting CONTACT_SPEED; not applied.
        gcmd.respond_info(
            "Suggested descent speed per candidate (mm/s): %s"
            % ", ".join("%.1fHz->%.2f" % (f, self._recommend_descend_speed(f))
                        for f in candidates))
        # Warn if fixed-amplitude excitation cannot be held across the whole
        # candidate range within the accel_per_hz limit (the high modes will run
        # at a reduced, accel-limited amplitude - see _excite_params).
        if excite_amp is not None:
            f_ok = cap / (excite_amp * 4. * math.pi**2)
            clamped = [f for f in candidates if f > f_ok + 1e-6]
            if clamped:
                gcmd.respond_info(
                    "EXCITE_AMP=%.4f mm can only be held up to %.1f Hz within the"
                    " accel_per_hz limit %.0f; %s Hz will use a reduced amplitude"
                    " (lower EXCITE_AMP for a fully fair comparison)"
                    % (excite_amp, f_ok, cap,
                       ", ".join("%.1f" % f for f in clamped)))
        # Points to contact-test: the whole grid (serpentine) or center+corners.
        if which == "corners":
            ijs = sorted(set([(xc // 2, yc // 2), (0, 0), (xc - 1, 0),
                              (0, yc - 1), (xc - 1, yc - 1)]))
        else:
            ijs = [(ix, iy) for iy in range(yc)
                   for ix in (range(xc) if iy % 2 == 0
                              else range(xc - 1, -1, -1))]
        # ONE canonical mode for the WHOLE mesh, chosen ONCE at the center point.
        # Arbitrating "which candidate reads best" independently at every point
        # risks accepting a DIFFERENT physical resonance at different points
        # (e.g. the low mode wins at one spot, a high mode at another) instead
        # of tracking how ONE mode's frequency drifts across the bed.  Low modes
        # tie to large/heavy structure (frame, gantry) and have been observed
        # NOT to drift; a high mode can be more localized - the accelerometer
        # (on the EBB36, not the nozzle tip) may see a slightly different peak
        # frequency there than the nozzle itself experiences - so it may drift
        # point-to-point while still being the SAME mode.  Commit to one mode
        # here via the full candidate arbitration, then at every other point
        # only confirm that committed frequency (fast path) or, on a miss,
        # re-locate that SAME peak in a narrow TRACK_WINDOW band (recovery
        # path) - never re-open the field to a structurally different mode.
        track_window = gcmd.get_float("TRACK_WINDOW", 5., minval=0.)
        gcmd.respond_info("  selecting the mesh's single primary mode at the"
                          " center (%.1f, %.1f)..." % (cx, cy))
        primary_freq, _pcz, p_clean = self._mode_low_first(
            gcmd, chip, out_idx, axis, cx, cy, candidates, mesh_z, cap, z_min,
            warmup, speed, lift, up_margin, down_margin, cycles,
            min_drop, target_noise, margin=contact_margin, travel=travel_speed,
            vib_span=vib_span, drip_time=drip_time, excite_amp=excite_amp,
            high_first=high_first)
        if primary_freq is None:
            raise gcmd.error("CALIBRATE_MESH: no contact found while selecting"
                             " the primary mode at the center point")
        if not p_clean:
            gcmd.respond_info(
                "  WARNING: no clean detection at the center point even for"
                " mode selection; committing to %.1f Hz best-effort - verify"
                % primary_freq)
        gcmd.respond_info("  primary mode for this mesh: %.1f Hz" % primary_freq)
        gcmd.respond_info("Contact-damping mode tracking at %d point(s) (%s),"
                          " primary mode %.1f Hz..."
                          % (len(ijs), which, primary_freq))
        # Each point's FIRST descent always starts from the full mesh_z
        # clearance (bed height at a new X/Y is not yet established there -
        # tilt/warp could put it anywhere within the configured floor).  Only
        # WITHIN one point's own retries/escalations, once a contact_z has
        # actually been observed at that exact spot, does _mode_low_first
        # tighten its own subsequent attempts (see its docstring).
        chosen = {}
        for (ix, iy) in ijs:
            x = coord(ix, xmin, xmax, xc)
            y = coord(iy, ymin, ymax, yc)
            gcmd.respond_info("  point (%.1f, %.1f):" % (x, y))
            point_candidates = [round(primary_freq, 1)]
            f, cz, clean = self._mode_low_first(
                gcmd, chip, out_idx, axis, x, y, point_candidates, mesh_z, cap,
                z_min, warmup, speed, lift, up_margin, down_margin,
                cycles, min_drop, target_noise, margin=contact_margin,
                travel=travel_speed, vib_span=vib_span, drip_time=drip_time,
                excite_amp=excite_amp, high_first=high_first)
            # A miss here is treated as possible DRIFT of the same mode, not as
            # grounds to try a different one: re-locate the committed peak in a
            # narrow band around it (in-air driven scan, safe/cheap) and retry
            # contact-damping restricted to that local set only.
            if not clean and track_window > 0.:
                toolhead.manual_move([x, y, mesh_z], travel_speed)
                toolhead.wait_moves()
                local = self._driven_refine(
                    gcmd, chip, axis, out_idx, [primary_freq], track_window,
                    max(f_lo, primary_freq - track_window),
                    min(f_hi, primary_freq + track_window), cand_step,
                    cand_aph, cand_dur)
                if local:
                    gcmd.respond_info(
                        "    no clean detection at %.1f Hz; re-checking the"
                        " same mode nearby: %s Hz"
                        % (primary_freq,
                           ", ".join("%.1f" % lf for lf in local)))
                    point_candidates = local
                    f, cz, clean = self._mode_low_first(
                        gcmd, chip, out_idx, axis, x, y, point_candidates,
                        mesh_z, cap, z_min, warmup, speed, lift, up_margin,
                        down_margin, cycles, min_drop, target_noise,
                        margin=contact_margin, travel=travel_speed,
                        vib_span=vib_span, drip_time=drip_time,
                        excite_amp=excite_amp, high_first=high_first)
            # No clean detection at the exact grid point (low-friction / marginal
            # spot: a weak or spurious drop, and a shallow false halt can even
            # hand the win to a bad high mode).  Nudge ~1mm off the point to
            # clear the bad spot and take the first clean reading - a far closer
            # proxy than an adjacent grid point, with negligible height/resonance
            # change.  Do NOT substitute a neighbor's value (bed height there is
            # genuinely different - that is what bed_mesh measures).  Nudge
            # retries reuse the same (possibly locally re-located) candidate set
            # - a nudge is a few mm away, not grounds to widen the mode search.
            if not clean and nudge_tries:
                for (nx, ny) in self._nudge_positions(
                        x, y, xmin, xmax, ymin, ymax, nudge_radius,
                        nudge_tries):
                    gcmd.respond_info(
                        "    no clean detection at (%.1f, %.1f); nudging to"
                        " (%.2f, %.2f)" % (x, y, nx, ny))
                    nf, ncz, nclean = self._mode_low_first(
                        gcmd, chip, out_idx, axis, nx, ny, point_candidates,
                        mesh_z, cap, z_min, warmup, speed, lift, up_margin,
                        down_margin, cycles, min_drop, target_noise,
                        margin=contact_margin, travel=travel_speed,
                        vib_span=vib_span, drip_time=drip_time,
                        excite_amp=excite_amp, high_first=high_first)
                    if nclean:
                        f, cz, clean = nf, ncz, True
                        gcmd.respond_info("    resolved at nudged (%.2f, %.2f)"
                                          % (nx, ny))
                        break
            if f is None:
                raise gcmd.error("CALIBRATE_MESH: no contact found at (%.1f,"
                                 " %.1f)" % (x, y))
            if not clean:
                gcmd.respond_info(
                    "    WARNING: (%.1f, %.1f) gave no clean detection even after"
                    " nudging; recording %.1f Hz best-effort - verify this point"
                    % (x, y, f))
            chosen[(ix, iy)] = f
            gcmd.respond_info("    -> chose %.1f Hz (contact z=%.4f)" % (f, cz))
        # Build the mesh from the chosen per-point frequencies.
        if which == "corners":
            vals = list(chosen.values())
            if max(vals) - min(vals) <= tol:
                out = [[sum(vals) / len(vals)]]
                shape = "constant (center+corners agree)"
            else:
                gcmd.respond_info("CALIBRATE_MESH: center+corners disagree (%s Hz)"
                    " -> re-run with CONTACT_POINTS=all for a per-point mesh"
                    % ", ".join("%.1f" % v for v in sorted(vals)))
                return
        else:
            mean = lambda v: sum(v) / len(v)
            finalrows = []
            for iy in range(yc):
                frow = [chosen[(ix, iy)] for ix in range(xc)]
                finalrows.append([mean(frow)] if max(frow) - min(frow) <= tol
                                 else frow)
            out, shape = self._finalize_collapse(finalrows, tol)
        pretty = "\n".join("    " + ", ".join("%.1f" % v for v in r)
                           for r in out)
        gcmd.respond_info("freq_mesh (%s):\n%s" % (shape, pretty))
        # Recommend a detection-limited descent speed for the chosen mode(s): the
        # reliable speed scales with frequency (see _recommend_descend_speed), so
        # the high modes support proportionally faster probing.  This is only a
        # suggestion for setting CONTACT_SPEED / the probe's descend_speed - the
        # probe never couples speed to frequency itself.
        picked = sorted({round(v, 1) for r in out for v in r})
        recs = ", ".join("%.1fHz->%.2f" % (f, self._recommend_descend_speed(f))
                         for f in picked)
        gcmd.respond_info("Recommended descent speed (mm/s, detection-limited,"
                          " scales with frequency): %s" % recs)
        if not gcmd.get_int("SAVE", 1):
            gcmd.respond_info("Dry run (SAVE=0): freq_mesh not written")
            return
        configfile = self.printer.lookup_object('configfile')
        val = "".join("\n  " + ", ".join("%.1f" % v for v in r) for r in out)
        configfile.set('resonance_probe', 'freq_mesh', val)
        gcmd.respond_info("Saved freq_mesh to [resonance_probe]. Run SAVE_CONFIG"
                          " to keep it (the printer will restart).")


def load_config(config):
    return ResonanceProbeCalibrate(config)
