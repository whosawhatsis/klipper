# Resonance-based nozzle probe - contact detection via vibration amplitude
#
# Vibrates a printer axis at a fixed resonant frequency while streaming the
# accelerometer; nozzle contact damps the resonance, dropping the response
# amplitude.  Detection runs entirely on the host (single-bin DFT with the DC
# recomputed each window) - no MCU/firmware changes are required.
#
# Probe modes (probe_mode): "stepwise" descends in increments and detects while
# stationary (safe, desync-proof, resolution = probe_step); "flyby" vibrates
# while descending in one continuous pass and locates contact by post-processing
# the stream (no halt); "hostdriven" vibrates while descending and a host task
# halts the drip move on contact (loose threshold), then post-halt analysis
# finds the precise contact Z.
#
# Copyright (C) 2026
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math
from . import probe, manual_probe
from .homing import HomingMove
from .resonance_tester import TestAxis, ResonanceTestExecutor

AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}

# Required methods on the accelerometer chip
SENSOR_API = ('start_internal_client',)

# Effective sample rate from a (time, ...) sample array's timestamps.
def _sample_rate(times):
    if len(times) < 2 or times[-1] <= times[0]:
        return 1000.
    return (len(times) - 1) / (times[-1] - times[0])

class ResonanceProbe:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.config = config
        # The accelerometer must be defined earlier in the config so it is
        # already instantiated when we look it up here.
        chip_name = config.get('accel_chip')
        self.chip = self.printer.lookup_object(chip_name)
        for m in SENSOR_API:
            if not hasattr(self.chip, m):
                raise config.error(
                    "[%s] accel_chip '%s' does not support resonance probing"
                    " (missing %s)" % (config.get_name(), chip_name, m))
        # Which accelerometer (output) axis to monitor; the streamed samples
        # already have the chip's axes_map applied, so this indexes them
        # directly.
        accel_axis = config.getchoice('accel_axis',
                                      {n: n for n in AXIS_INDEX})
        self.output_index = AXIS_INDEX[accel_axis]
        # Excitation parameters (deliberately small - contact is made by hand
        # in the test and amplitude scales with accel_per_hz)
        self.excitation_freq = config.getfloat('excitation_frequency',
                                               above=1., maxval=300.)
        self.accel_per_hz = config.getfloat('accel_per_hz', 10., above=0.)
        self.sensitivity = config.getfloat('sensitivity', 0.3,
                                           above=0., below=1.)
        # Loose threshold used only to HALT the descent in host-driven mode
        # (bounds over-drive); the precise contact Z still comes from the fine
        # 'sensitivity' post-halt analysis, so this can be generous.
        self.halt_sensitivity = config.getfloat('halt_sensitivity', 0.15,
                                                above=0., below=1.)
        self.allow_z = config.getboolean('allow_z_vibration', False)
        # Parse the printer axis to vibrate (x/y/z or a dx,dy,dz vector)
        raw_axis = config.get('vibrate_axis', 'x')
        self.vibrate_dir = self._parse_config_axis(config, raw_axis)
        self.executor = ResonanceTestExecutor(config)
        # z_min is a hard floor so a failed detection cannot crash the bed.
        self.z_min_position = probe.lookup_minimum_z(config)
        self.warmup = config.getfloat('warmup', 0.3, above=0.05)
        # Probe mode:
        #  stepwise   - descend a small increment, then detect while stationary
        #               (the trigger only notifies; it halts nothing). Safe and
        #               desync-proof, but resolution is one probe_step.
        #  flyby      - vibrate continuously while descending in one pass, then
        #               locate contact by post-processing the streamed data vs
        #               Z (fast, fine resolution, no mid-move halt).
        #  hostdriven - vibrate while descending; a host task watches the proven
        #               host detection and halts the drip move on contact (loose
        #               threshold), then post-halt analysis finds the precise
        #               contact Z.  No MCU trigger; reuses the fly-by analyzer.
        self.probe_mode = config.getchoice(
                'probe_mode', {'stepwise': 'stepwise', 'flyby': 'flyby',
                               'hostdriven': 'hostdriven'}, 'stepwise')
        self.probe_step = config.getfloat('probe_step', 0.05, above=0.)
        self.detect_time = config.getfloat('detect_time', 0.3, above=0.05)
        self.descend_speed = config.getfloat('descend_speed', 1., above=0.)
        # Fly-by: how far below the start to descend, and the sliding-window
        # length (s) used to estimate amplitude vs Z from the stream.
        self.probe_distance = config.getfloat('probe_distance', 0.5, above=0.)
        self.flyby_window = config.getfloat('flyby_window', 0.15, above=0.02)
        # Peak lateral displacement of the excitation (mm); default keeps the
        # sinusoid's acceleration amplitude at accel_per_hz * frequency.
        default_amp = self.accel_per_hz / (4. * math.pi**2 * self.excitation_freq)
        self.probe_amplitude = config.getfloat('probe_amplitude', default_amp,
                                               above=0.)
        self._z_steppers = []
        if self.probe_mode == 'hostdriven':
            probe.LookupZSteppers(config, self._z_steppers.append)
        # Standard probe interface (PROBE command, bed mesh, etc.)
        self.param_helper = probe.ProbeParameterHelper(config)
        self.cmd_helper = probe.ProbeCommandHelper(config, self)
        self.probe_offsets = probe.ProbeOffsetsHelper(config)
        self.probe_session = probe.SampleAveragingHelper(
                config, self.param_helper, self._start_hw_session)
        self.printer.add_object('probe', self)

    # Probe interface used by ProbeCommandHelper / bed mesh / PROBE
    def get_probe_params(self, gcmd=None):
        return self.param_helper.get_probe_params(gcmd)
    def get_offsets(self, gcmd=None):
        return self.probe_offsets.get_offsets(gcmd)
    def start_probe_session(self, gcmd):
        return self.probe_session.start_probe_session(gcmd)
    def get_status(self, eventtime):
        return self.cmd_helper.get_status(eventtime)
    def _start_hw_session(self, gcmd):
        return ResonanceProbeSession(self, gcmd)

    # -- setup helpers -----------------------------------------------------

    def _parse_config_axis(self, config, raw_axis):
        raw_axis = raw_axis.lower()
        if raw_axis in AXIS_INDEX:
            return TestAxis(axis=raw_axis)
        dirs = raw_axis.split(',')
        try:
            vec = [float(d.strip()) for d in dirs]
        except ValueError:
            raise config.error("Invalid vibrate_axis '%s'" % (raw_axis,))
        if len(vec) not in (2, 3):
            raise config.error("Invalid vibrate_axis '%s'" % (raw_axis,))
        return TestAxis(vib_dir=tuple(vec))

    def _check_axis_safety(self, gcmd):
        if self.vibrate_dir.get_dir()[2] and not (
                self.allow_z or gcmd.get_int("ALLOW_Z", 0)):
            raise gcmd.error(
                "Refusing to vibrate along Z (drives the nozzle toward the"
                " platform). Use a lateral vibrate_axis, or set ALLOW_Z=1.")

    # -- motion ------------------------------------------------------------

    # Constant-frequency back-and-forth excitation (see resonance_tester for
    # the (time, accel, freq) segment format consumed by the executor).  Each
    # pulse is a (+a, -a) pair and the sign flips between pulses, matching
    # VibrationPulseTestGenerator: this oscillates symmetrically and returns to
    # the start.  Flipping every single segment instead would only ever
    # accelerate-then-stop in one direction, drifting the toolhead.
    def _gen_fixed_freq(self, freq, accel, duration):
        t_seg = .25 / freq
        n_pulse = max(1, int(round(duration / (2. * t_seg))))
        res, sign, t = [], 1., 0.
        for _ in range(n_pulse):
            t += t_seg
            res.append((t, sign * accel, freq))
            t += t_seg
            res.append((t, -sign * accel, freq))
            sign = -sign
        return res

    def _vibrate(self, gcmd, duration):
        accel = self.accel_per_hz * self.excitation_freq
        test_seq = self._gen_fixed_freq(self.excitation_freq, accel, duration)
        self.executor.run_test(test_seq, self.vibrate_dir, gcmd)

    # -- probing moves -----------------------------------------------------

    # Vibrate in place and return the single-frequency (DFT) response amplitude
    # of the monitored axis, computed on the host.  The DC is recomputed every
    # call (so a contact-induced DC shift can't mask the drop) and only the
    # excitation frequency is kept (rejecting contact friction noise) - this is
    # why it is far cleaner than the MCU's fixed-DC broadband envelope.
    def _measure_amplitude(self, gcmd, duration):
        import numpy as np
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.wait_moves()
        toolhead.dwell(0.100)
        aclient = self.chip.start_internal_client()
        self._vibrate(gcmd, duration)
        aclient.finish_measurements()
        samples = aclient.get_samples()
        if not samples:
            raise gcmd.error("Accelerometer measured no data while probing")
        data = np.asarray(samples, dtype=np.float64)
        t = data[:, 0] - data[0, 0]
        a = data[:, 1 + self.output_index]
        a = a - a.mean()
        ref = np.exp(-2j * np.pi * self.excitation_freq * t)
        return 2.0 / len(t) * abs(np.sum(a * ref))

    # Descend in small steps, detecting contact while stationary at each step
    # via the host-side amplitude measurement above.  Z only moves between
    # measurements, so nothing halts a move and the motion queue cannot desync.
    # Resolution is one probe_step.
    def _stepwise_probe(self, gcmd):
        self._check_axis_safety(gcmd)
        toolhead = self.printer.lookup_object('toolhead')
        speed = self.param_helper.get_probe_params(gcmd)['probe_speed']
        step = gcmd.get_float("PROBE_STEP", self.probe_step, above=0.)
        z_min = self.z_min_position
        x, y, cur_z = toolhead.get_position()[:3]
        if cur_z <= z_min:
            raise gcmd.error("Resonance probe started at or below z_min=%.3f"
                             % (z_min,))
        # No-contact baseline amplitude at the (assumed clear) start height
        baseline = self._measure_amplitude(gcmd, self.detect_time)
        threshold = baseline * (1. - self.sensitivity)
        gcmd.respond_info("Resonance probe baseline %.1f; trigger below %.1f"
                          " (%.0f%% drop)"
                          % (baseline, threshold, self.sensitivity * 100.))
        detected = False
        while cur_z > z_min + 1e-9:
            cur_z = max(z_min, cur_z - step)
            toolhead.manual_move([x, y, cur_z], speed)
            toolhead.wait_moves()
            amp = self._measure_amplitude(gcmd, self.detect_time)
            if amp < threshold:
                detected = True
                break
        if not detected:
            raise gcmd.error("Resonance probe: no contact detected down to"
                             " z_min=%.3f" % (z_min,))
        offsets = self.probe_offsets.get_offsets()
        return manual_probe.create_probe_result(toolhead.get_position(),
                                                offsets)

    # Vibrate the lateral axis at the excitation frequency while descending Z
    # in one continuous pass: each oscillation move target also steps Z down by
    # descend_speed * t.  Reuses the resonance executor's velocity integration
    # (the _gen_fixed_freq pulses pass through v=0 only at segment boundaries,
    # so no mid-segment reversal handling is needed).  Issues normal queued
    # moves (no halt - this is a fly-by), so the motion queue cannot desync.
    def _vibrate_descend(self, gcmd, z_floor):
        reactor = self.printer.get_reactor()
        gcode = self.printer.lookup_object('gcode')
        toolhead = self.printer.lookup_object('toolhead')
        tpos = toolhead.get_position()
        x0, y0, z0 = tpos[0], tpos[1], tpos[2]
        vdir = self.vibrate_dir.get_dir()
        descent = self.descend_speed
        total_t = max((z0 - z_floor) / descent, 0.001)
        accel = self.accel_per_hz * self.excitation_freq
        test_seq = self._gen_fixed_freq(self.excitation_freq, accel, total_t)
        max_v = lv = lt = 0.
        for nt, a, _ in test_seq:
            lv = lv + a * (nt - lt)
            lt = nt
            max_v = max(max_v, abs(lv))
        info = toolhead.get_status(reactor.monotonic())
        old_v, old_a = info['max_velocity'], info['max_accel']
        old_cr = info['minimum_cruise_ratio']
        gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT VELOCITY=%.0f ACCEL=%.0f MINIMUM_CRUISE_RATIO=0"
            % (max_v + 1., accel + 1.))
        ishaper = self.printer.lookup_object('input_shaper', None)
        if ishaper is not None:
            ishaper.disable_shaping()
        try:
            cum_d = last_v = last_t = 0.
            for next_t, a, freq in test_seq:
                v = last_v + a * (next_t - last_t)
                if abs(v) < 1e-6:
                    v = 0.
                cum_d += (v * v - last_v * last_v) * (0.5 / a)
                nx = x0 + vdir[0] * cum_d
                ny = y0 + vdir[1] * cum_d
                nz = max(z0 + vdir[2] * cum_d - descent * next_t, z_floor)
                toolhead.limit_next_junction_speed(abs(last_v))
                toolhead.move([nx, ny, nz] + list(tpos[3:]),
                              max(abs(v), abs(last_v)))
                last_t, last_v = next_t, v
        finally:
            gcode.run_script_from_command(
                "SET_VELOCITY_LIMIT VELOCITY=%.3f ACCEL=%.3f"
                " MINIMUM_CRUISE_RATIO=%.3f" % (old_v, old_a, old_cr))
            if ishaper is not None:
                ishaper.enable_shaping()

    # Locate contact in a fly-by stream: map each sample to its Z via the known
    # linear descent, estimate the excitation-frequency amplitude in sliding
    # windows, and return the Z where it first falls below the baseline by
    # 'sensitivity' (interpolated between windows for sub-window resolution).
    def _analyze_flyby(self, gcmd, samples, t0, z_start, z_floor):
        import numpy as np
        data = np.asarray(samples, dtype=np.float64)
        times = data[:, 0]
        zpos = z_start - self.descend_speed * (times - t0)
        col = data[:, 1 + self.output_index]
        f = self.excitation_freq
        win_n = max(8, int(self.flyby_window * _sample_rate(times)))
        step_n = max(1, win_n // 3)
        amps, zwin = [], []
        i = 0
        while i + win_n <= len(times):
            seg = col[i:i + win_n]
            tt = times[i:i + win_n] - times[i]
            a = seg - seg.mean()
            amps.append(2.0 / win_n * abs(np.sum(a * np.exp(-2j*np.pi*f*tt))))
            zwin.append(zpos[i + win_n // 2])
            i += step_n
        if len(amps) < 4:
            return None
        amps = np.array(amps)
        # The vibration builds from rest over the first windows (the resonance
        # has Q, so amplitude ramps up over several cycles).  Discard that
        # ramp-up region: testing it against the steady-state baseline
        # false-triggers at the very top, since a still-rising window reads
        # below baseline.
        nskip = max(1, min(3, len(amps) // 8))
        nb = max(nskip + 2, len(amps) // 3)
        baseline = float(np.median(amps[nskip:nb]))
        threshold = baseline * (1. - self.sensitivity)
        contact = None
        for k in range(nskip, len(amps)):
            # Stop at the descent floor: windows mapped below z_floor are the
            # post-move stationary tail (vibration has ended, amplitude ~0), not
            # contact.  Ignoring them lets a sweep that never reaches the bed
            # correctly report "no contact" instead of false-triggering there.
            if zwin[k] < z_floor:
                break
            # Require the drop to persist into the next window: real contact
            # keeps the amplitude down, while a lone sub-threshold window is
            # ramp/measurement noise.
            if amps[k] < threshold and (k + 1 >= len(amps)
                                        or amps[k + 1] < threshold):
                if amps[k - 1] > amps[k]:
                    frac = (amps[k - 1] - threshold) / (amps[k - 1] - amps[k])
                    contact = zwin[k - 1] + frac * (zwin[k] - zwin[k - 1])
                else:
                    contact = zwin[k]
                break
        gcmd.respond_info("fly-by: baseline %.1f, trigger below %.1f (%.0f%%),"
                          " contact z=%s"
                          % (baseline, threshold, self.sensitivity * 100.,
                             ("%.4f" % contact) if contact is not None
                             else "none"))
        return contact

    # Like _analyze_flyby, but for the drip descent, whose oscillation ramps up
    # slowly (the lookahead eases into the reversals).  The baseline is taken
    # from the amplitude PLATEAU (the first run of windows, after 'armed_time',
    # whose amplitude has stopped rising) rather than the first third, so the
    # ramp cannot depress it.  Returns the Z where the amplitude first falls
    # below the plateau by 'sensitivity'.
    def _analyze_drip(self, gcmd, samples, anchor_t, anchor_z, z_floor,
                      armed_time):
        import numpy as np
        data = np.asarray(samples, dtype=np.float64)
        if len(data) < 8:
            return None
        times = data[:, 0]
        # Map sample time -> Z anchored at (anchor_t, anchor_z).  Anchoring to
        # the physical halt position (from stepper history) instead of the
        # descent-start time removes the drip startup-delay offset, so the
        # reported contact Z is independent of the probe's start height.
        zpos = anchor_z + self.descend_speed * (anchor_t - times)
        col = data[:, 1 + self.output_index]
        f = self.excitation_freq
        win_n = max(8, int(self.flyby_window * _sample_rate(times)))
        step_n = max(1, win_n // 3)
        amps, zwin, twin = [], [], []
        i = 0
        while i + win_n <= len(times):
            seg = col[i:i + win_n]
            tt = times[i:i + win_n] - times[i]
            a = seg - seg.mean()
            amps.append(2.0 / win_n * abs(np.sum(a * np.exp(-2j*np.pi*f*tt))))
            zwin.append(zpos[i + win_n // 2])
            twin.append(times[i + win_n // 2])
            i += step_n
        amps = np.array(amps)
        # Consider only windows after the warmup gate (skips the startup dwell).
        armed_k = [k for k in range(len(amps)) if twin[k] >= armed_time]
        if len(armed_k) < 4:
            gcmd.respond_info("drip: too few windows after warmup")
            return None
        # Baseline = plateau level: median of the armed windows within 80% of
        # the armed-region peak.  This is robust to the slow drip ramp-up and to
        # run-to-run amplitude noise (the stable-run heuristic occasionally set
        # it too high and the fine search then missed the contact).
        amax = max(amps[k] for k in armed_k)
        plateau = [amps[k] for k in armed_k if amps[k] >= 0.8 * amax]
        baseline = float(np.median(plateau))
        threshold = baseline * (1. - self.sensitivity)
        # Start searching once the amplitude first reaches the plateau (so the
        # rising ramp is skipped), then take the first sustained drop below
        # threshold.
        contact = None
        searching = False
        for k in armed_k:
            if not searching:
                searching = amps[k] >= 0.9 * baseline
                continue
            if zwin[k] < z_floor:
                break
            if amps[k] < threshold and (k + 1 >= len(amps)
                                        or amps[k + 1] < threshold):
                if amps[k - 1] > amps[k]:
                    frac = (amps[k - 1] - threshold) / (amps[k - 1] - amps[k])
                    contact = zwin[k - 1] + frac * (zwin[k] - zwin[k - 1])
                else:
                    contact = zwin[k]
                break
        gcmd.respond_info("drip: baseline %.1f, trigger below %.1f (%.0f%%),"
                          " contact z=%s"
                          % (baseline, threshold, self.sensitivity * 100.,
                             ("%.4f" % contact) if contact is not None
                             else "none"))
        return contact

    # Continuous fly-by probe: descend through the contact region while
    # vibrating and streaming the accelerometer, then locate contact on the
    # host.  Over-drives ~probe_distance into the (sprung) bed, then retracts.
    def _flyby_probe(self, gcmd):
        self._check_axis_safety(gcmd)
        toolhead = self.printer.lookup_object('toolhead')
        x0, y0, z_start = toolhead.get_position()[:3]
        z_floor = max(self.z_min_position, z_start - self.probe_distance)
        if z_start <= z_floor:
            raise gcmd.error("Resonance probe started at or below z_min")
        lift_speed = self.param_helper.get_probe_params(gcmd)['lift_speed']
        toolhead.wait_moves()
        toolhead.dwell(0.100)
        aclient = self.chip.start_internal_client()
        t0 = toolhead.get_last_move_time()
        self._vibrate_descend(gcmd, z_floor)
        toolhead.wait_moves()
        aclient.finish_measurements()
        samples = aclient.get_samples()
        if not samples:
            raise gcmd.error("Accelerometer measured no data while probing")
        contact_z = self._analyze_flyby(gcmd, samples, t0, z_start, z_floor)
        if contact_z is None:
            toolhead.manual_move([x0, y0, z_start], lift_speed)
            toolhead.wait_moves()
            raise gcmd.error("Resonance fly-by probe: no contact detected")
        # Leave the toolhead at the detected contact height (the descent
        # over-drove to z_floor).  This mirrors a real probe's stop-at-trigger
        # behavior so the probe session measures its retract from contact, not
        # from the descent start - otherwise repeated samples walk upward by
        # one retract each and miss the bed.
        toolhead.manual_move([x0, y0, contact_z], lift_speed)
        toolhead.wait_moves()
        epos = list(toolhead.get_position())
        epos[2] = contact_z
        return manual_probe.create_probe_result(epos,
                                                self.probe_offsets.get_offsets())

    # Build the vibrate-while-descend trajectory: lateral oscillation at the
    # excitation frequency superimposed on a slow linear Z descent to z_target.
    # Sampled at half-period waypoints (alternating +/- amplitude) so the
    # reversal rate equals the excitation frequency.  Returns a list of
    # (pos, speed, accel) for toolhead.drip_move_sequence.
    def _gen_descend_segments(self, z_target):
        toolhead = self.printer.lookup_object('toolhead')
        x0, y0, z0 = toolhead.get_position()[:3]
        f = self.excitation_freq
        accel = self.accel_per_hz * f
        amp = self.probe_amplitude
        vdir = self.vibrate_dir.get_dir()
        # Peak velocity of the accel-limited +/-amp swing
        peak_v = max(self.accel_per_hz / (2. * math.pi), 1e-3)
        total_t = max((z0 - z_target) / self.descend_speed, 1e-6)
        n = max(2, int(math.ceil(total_t * 2. * f)))
        segs = []
        for k in range(1, n + 1):
            frac = min(1., (k / (2. * f)) / total_t)
            z = z0 - (z0 - z_target) * frac
            disp = amp if (k % 2) else -amp
            segs.append(([x0 + vdir[0] * disp, y0 + vdir[1] * disp, z],
                         peak_v, accel))
        # Finish centered, exactly at the target height
        segs.append(([x0, y0, z_target], peak_v, accel))
        return segs

    # Host-driven halt: vibrate-while-descend via the drip path, but a host
    # task watches the proven host detection (single-bin DFT, DC recomputed) and
    # halts the move on contact using a loose threshold; the precise contact Z
    # then comes from the fine post-halt analysis of the captured stream.  No
    # MCU trigger is involved.
    def _hostdriven_probe(self, gcmd):
        self._check_axis_safety(gcmd)
        toolhead = self.printer.lookup_object('toolhead')
        reactor = self.printer.get_reactor()
        gcode = self.printer.lookup_object('gcode')
        z_min = self.z_min_position
        pos = toolhead.get_position()
        x0, y0, z_start = pos[0], pos[1], pos[2]
        if z_start <= z_min:
            raise gcmd.error("Resonance probe started at or below z_min=%.3f"
                             % (z_min,))
        lift_speed = self.param_helper.get_probe_params(gcmd)['lift_speed']
        accel = self.accel_per_hz * self.excitation_freq
        peak_v = max(self.accel_per_hz / (2. * math.pi), 1e-3)
        # Match the excitation conditions of the other modes (shaping off, raised
        # limits) so the vibration reaches full amplitude.
        info = toolhead.get_status(reactor.monotonic())
        old_v, old_a = info['max_velocity'], info['max_accel']
        old_cr = info['minimum_cruise_ratio']
        gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT VELOCITY=%.0f ACCEL=%.0f MINIMUM_CRUISE_RATIO=0"
            % (peak_v + self.descend_speed + 1., accel + 1.))
        ishaper = self.printer.lookup_object('input_shaper', None)
        if ishaper is not None:
            ishaper.disable_shaping()
        toolhead.wait_moves()
        t0 = toolhead.get_last_move_time()
        endstop = _HostResonanceEndstop(self, self._z_steppers, t0)
        gen = lambda newpos, speed: self._gen_descend_segments(newpos[2])
        vth = _VibratingToolhead(toolhead, gen)
        hmove = HomingMove(self.printer, [(endstop, "resonance_probe")],
                           toolhead=vth)
        movepos = [x0, y0, z_min] + list(pos[3:])
        try:
            epos = hmove.homing_move(movepos, self.descend_speed,
                                     probe_pos=True, check_triggered=False)
        finally:
            gcode.run_script_from_command(
                "SET_VELOCITY_LIMIT VELOCITY=%.3f ACCEL=%.3f"
                " MINIMUM_CRUISE_RATIO=%.3f" % (old_v, old_a, old_cr))
            if ishaper is not None:
                ishaper.enable_shaping()
        halted = endstop.get_trigger_time() > 0.
        # Precise contact Z from the captured stream (fine 'sensitivity').  Drop
        # samples from before the descent start: the batch stream can include
        # stale/pre-move samples whose timestamps map to nonsensical Z.
        samples = [s for s in endstop.get_samples() if s[0] >= t0]
        # Anchor the Z mapping to the physical halt (trigger time -> epos[2])
        # when it halted; otherwise fall back to the descent start.
        trig_t = endstop.get_trigger_time()
        if halted:
            anchor_t, anchor_z = trig_t, epos[2]
        else:
            anchor_t, anchor_z = t0, z_start
        contact_z = self._analyze_drip(gcmd, samples, anchor_t, anchor_z, z_min,
                                       t0 + self.warmup)
        if contact_z is None:
            toolhead.manual_move([x0, y0, z_start], lift_speed)
            toolhead.wait_moves()
            raise gcmd.error("Resonance host-driven probe: no contact detected"
                             " (halted=%s)" % (halted,))
        # Leave the toolhead at the detected contact height (mirrors a real
        # probe's stop-at-trigger), so a probe session's retract is measured
        # from contact and repeated samples don't walk upward.
        toolhead.manual_move([x0, y0, contact_z], lift_speed)
        toolhead.wait_moves()
        epos = list(toolhead.get_position())
        epos[2] = contact_z
        return manual_probe.create_probe_result(
                epos, self.probe_offsets.get_offsets())


# Host-driven "endstop" for HomingMove: instead of an MCU trigger, a batch
# callback watches the accelerometer resonance amplitude and completes the drip
# completion (halting the move) when it drops past a loose halt threshold.
# HomingMove then derives the halt position from the stepper history at the
# recorded trigger time.  Detection mirrors the fly-by analyzer (single-bin DFT
# with the DC recomputed per window).
class _HostResonanceEndstop:
    def __init__(self, rprobe, steppers, t0):
        import numpy as np
        self._np = np
        self.rprobe = rprobe
        self.printer = rprobe.printer
        self.reactor = self.printer.get_reactor()
        self._steppers = steppers
        self._t0 = t0
        self._armed_time = t0 + rprobe.warmup
        self._completion = None
        self._done = False
        self._trigger_time = 0.
        self._buf = []
        self._amps = []
        self._prev_amp = None
        self._baseline = None
        self._last_analyzed = 0
        self._oidx = rprobe.output_index
        # Window sized lazily from the measured sample rate (first batches).
        self._flyby_window = rprobe.flyby_window
        self._win_n = None
        self._step_n = None
        self._baseline_n = 3
        self._freq = rprobe.excitation_freq
        self._halt_sens = rprobe.halt_sensitivity

    def get_steppers(self):
        return self._steppers

    def get_trigger_time(self):
        return self._trigger_time

    def get_samples(self):
        return self._buf

    def home_start(self, print_time, sample_time, sample_count, rest_time,
                   triggered=True):
        self._completion = self.reactor.completion()
        self.rprobe.chip.batch_bulk.add_client(self._handle_batch)
        return self._completion

    # Batch callback (runs in the reactor while the drip move proceeds).
    def _handle_batch(self, msg):
        if self._done:
            return False
        np = self._np
        # Ignore stale/pre-descent samples (their timestamps map to bad Z).
        self._buf.extend(s for s in msg['data'] if s[0] >= self._t0)
        if self._win_n is None:
            # Size the window once enough span exists to estimate the rate.
            if len(self._buf) >= 2 and self._buf[-1][0] - self._buf[0][0] > 0.05:
                sps = (len(self._buf) - 1) / (self._buf[-1][0] - self._buf[0][0])
                self._win_n = max(8, int(self._flyby_window * sps))
                self._step_n = max(1, self._win_n // 3)
            else:
                return True
        while len(self._buf) - self._last_analyzed >= self._step_n:
            self._last_analyzed += self._step_n
            if len(self._buf) < self._win_n:
                continue
            seg = self._buf[-self._win_n:]
            tc = seg[len(seg) // 2][0]
            if tc < self._armed_time:
                continue  # excitation still ringing up
            t = np.array([r[0] for r in seg])
            a = np.array([r[1 + self._oidx] for r in seg], dtype=np.float64)
            a = a - a.mean()
            ref = np.exp(-2j * np.pi * self._freq * (t - t[0]))
            amp = 2. / len(t) * abs(np.sum(a * ref))
            if self._baseline is None:
                # Establish the baseline only once the amplitude has plateaued
                # (the drip ramps up slowly); skip windows while it is still
                # rising, so the ramp does not depress the baseline.
                if (self._prev_amp is not None and abs(amp - self._prev_amp)
                        < 0.1 * max(self._prev_amp, 1e-9)):
                    self._amps.append(amp)
                    if len(self._amps) >= self._baseline_n:
                        self._baseline = float(np.median(self._amps))
                else:
                    self._amps = []
                self._prev_amp = amp
            elif amp < self._baseline * (1. - self._halt_sens):
                self._trigger_time = tc
                self._done = True
                self._completion.complete(True)
                return False
        return True

    def home_wait(self, home_end_time):
        self._done = True
        if self._completion is not None and not self._completion.test():
            self._completion.complete(False)
        return self._trigger_time


# Toolhead proxy that replaces the single straight drip_move with a vibrating,
# descending move sequence.  All other toolhead calls pass through unchanged, so
# HomingMove's start/halt position bookkeeping works exactly as for a normal
# probing move.
class _VibratingToolhead:
    def __init__(self, toolhead, gen_segments):
        self._toolhead = toolhead
        self._gen_segments = gen_segments
    def drip_move(self, newpos, speed, drip_completion):
        segments = self._gen_segments(newpos, speed)
        self._toolhead.drip_move_sequence(segments, drip_completion)
    def __getattr__(self, name):
        return getattr(self._toolhead, name)


# Inner probe session: one descending probe move per run_probe, dispatched to
# the configured probe_mode.
class ResonanceProbeSession:
    def __init__(self, rprobe, gcmd):
        self.rprobe = rprobe
        self.results = []
    def run_probe(self, gcmd):
        rp = self.rprobe
        if rp.probe_mode == 'flyby':
            self.results.append(rp._flyby_probe(gcmd))
        elif rp.probe_mode == 'hostdriven':
            self.results.append(rp._hostdriven_probe(gcmd))
        else:
            self.results.append(rp._stepwise_probe(gcmd))
    def pull_probed_results(self):
        res = self.results
        self.results = []
        return res
    def end_probe_session(self):
        self.results = []


def load_config(config):
    return ResonanceProbe(config)
