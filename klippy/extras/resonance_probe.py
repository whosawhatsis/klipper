# Resonance-based nozzle probe - contact detection via vibration amplitude
#
# Vibrates a printer axis at a fixed resonant frequency while streaming the
# accelerometer; nozzle contact damps the resonance, dropping the response
# amplitude.  Detection runs entirely on the host (single-bin DFT with the DC
# recomputed each window) - no MCU/firmware changes are required.
#
# Probe modes (probe_mode): "stepwise" descends in increments and detects while
# stationary (safe, desync-proof, resolution = probe_step); "hostdriven" vibrates
# while descending and a host task
# halts the drip move on contact (loose threshold), then post-halt analysis
# finds the precise contact Z.
#
# Copyright (C) 2026
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging, math, os, random
from . import probe, manual_probe, shaper_calibrate, analog_contact
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


# Single-bin DFT magnitude at 'freq', normalized to the sinusoid's amplitude:
# 2/N * |sum((s - mean(s)) * exp(-2*pi*j*f*t))|.  't' is time relative to the
# window start.  Shared by the one-shot amplitude measurements (the windowed
# analyzers precompute the phase term across many windows and inline this).
# Convert numpy scalars/arrays into plain Python types, recursively.
#
# Everything this module measures comes out of numpy, and any numpy value that
# reaches Klipper's status/webhook layer is fatal: json.dumps raises
# "Object of type bool_/float64 is not JSON serializable", Klipper treats that
# as an internal error and SHUTS THE MCU DOWN.  The failure surfaces as an
# unrelated-looking MCU shutdown long after the offending line, so it is very
# expensive to debug (it has bitten this module twice).
#
# Scattering float() calls at each use site does not work - it only takes one
# missed path.  Instead every value crossing OUT of this module (probe results,
# status dicts, measurement dicts handed to the calibration module) goes through
# here once.
# --- halt-floor model -------------------------------------------------------
#
# One definition of "where the halt threshold goes", shared by the live
# detector's per-axis floors, the amplitude-selection headroom score and the
# mode ranking.  These three used to carry their own copies of the same
# arithmetic, which is how accel_axis once ended up naming a different axis
# than the floors actually armed.
#
# A workable floor must sit ABOVE the axis's own noise (or it triggers on
# nothing) and BELOW the contact drop (or it never triggers at all), so the
# usable window is (lo, hi).  Where inside that window to sit is a real
# trade-off, and it is NOT symmetric:
#
#   * too low  -> false halt.  Recoverable: the verify pass rejects it,
#     re-arms strictly below, and retries.  Costs seconds.
#   * too high -> NO halt.  The descent runs to the safety floor, which means
#     the nozzle presses into the bed for the whole remaining travel.  That is
#     the dangerous failure, and it aborts the probe.
#
# So bias toward sensitivity: sit a short way up from the noise bound rather
# than in the middle.  BIAS is that position, 0 = hard against the noise, 1 =
# hard against the drop.
HALT_NOISE_FACTOR = 1.15      # multiple of measured noise the floor must clear
HALT_NOISE_MARGIN = 0.015     # plus a small absolute margin (fraction of 1)
# The stationary characterization OVERSTATES what the live descent sees: the
# excitation cannot fully ring down in the ~0.1mm below contact, so a 21%
# dwell drop reaches the live detector as 8-12%.  DROP_FRACTION is therefore
# not a safety margin - it is the estimate of that live drop, ~half.  Lowering
# it does NOT buy sensitivity; it shrinks the window and starts rejecting modes
# that work (0.35 rejected the 57.8Hz/z mode that measured best on hardware).
# Sensitivity comes from BIAS, the position inside the window.
HALT_DROP_FRACTION = 0.5
HALT_SENSITIVITY_BIAS = 0.15  # position in the usable window (low = sensitive)


def _halt_window(noise, drop):
    """(lo, hi) bounds of the floors that both reject noise and catch contact."""
    return (noise * HALT_NOISE_FACTOR + HALT_NOISE_MARGIN,
            HALT_DROP_FRACTION * drop)


def _halt_headroom(drop, noise):
    """Width of that window: how much room exists to place a working floor.

    <= 0 means no threshold can do both jobs, i.e. this axis/amplitude/mode
    cannot detect contact reliably however it is tuned."""
    lo, hi = _halt_window(noise, drop)
    return hi - lo


def _halt_floor(drop, noise):
    """The floor itself, or None when the window is empty."""
    lo, hi = _halt_window(noise, drop)
    if hi <= lo:
        return None
    return lo + HALT_SENSITIVITY_BIAS * (hi - lo)


# Z span of the live detector's "current" window (see _handle_batch, where this
# caps win_z).  Shared so the calibration sweep measures contact over the SAME
# span the live halt does - a mode judged over a wider span is judged on damping
# the live detector will never get to see.
LIVE_WIN_Z = 0.022


def _moving_stats(amps, zs, contact_z, up_margin, onset_z=LIVE_WIN_Z):
    """Contact drop and noise measured in the MOVING regime.

    The amplitude sweep's down-ramp spans up_margin above contact to
    down_margin below it, so a statistic over the whole ramp is dominated by
    air: with the defaults (0.10/0.04) the median window sits ~30um ABOVE the
    surface.  Split by height instead - windows in the upper half of the air
    side are moving-in-air - and take BOTH the drop and the noise from that
    split, so the two stay in the same regime.

    The contact side is restricted to an ONSET band, the first onset_z below
    contact, because that is all the live detector ever sees: it fires on a
    gradient measured over win_z (<= LIVE_WIN_Z) at the leading edge of
    contact, not on fully-pressed damping.  Measuring everything below
    contact_z instead - up to down_margin (40um) of press - credits a mode for
    damping that develops only as the nozzle pushes in.  That is not
    hypothetical: at 148Hz the whole-below-contact figure rated z at 57% while
    the live descent read 1-7% on z and never triggered, x carrying every halt.
    The air reference (0.5*up_margin = 0.05mm) already matches the live
    detector's ref_z = max(confirm_z, 2*win_z) = 0.05mm.

    Returns (drop, noise); (0., 1.) - i.e. no usable signal - when either
    population is too small to summarise.
    """
    import numpy as np
    amps = np.asarray(amps, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    if amps.shape != zs.shape:
        raise ValueError("amps and zs must be parallel")
    air = amps[zs >= contact_z + 0.5 * up_margin]
    con = amps[(zs <= contact_z) & (zs >= contact_z - onset_z)]
    if len(air) < 3 or len(con) < 2:
        return 0., 1.
    base = float(np.median(air))
    if base <= 0.:
        return 0., 1.
    return (max(0., 1. - float(np.median(con)) / base),
            float(np.std(air)) / base)


def _plain(obj):
    item = getattr(obj, 'item', None)          # numpy scalar -> Python scalar
    if item is not None and getattr(obj, 'ndim', None) == 0:
        return item()
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        vals = [_plain(v) for v in obj]
        if isinstance(obj, list):
            return vals
        make = getattr(obj, '_make', None)     # namedtuple
        if make is not None:
            return make(vals)
        try:
            return type(obj)(vals)             # tuple, and sequence-taking subs
        except TypeError:
            try:
                return type(obj)(*vals)        # subclasses taking positionals
            except TypeError:
                return tuple(vals)             # last resort: plain tuple
    tolist = getattr(obj, 'tolist', None)      # numpy array -> nested lists
    if tolist is not None:
        return tolist()
    return obj


def _dft_amp(t, signal, freq):
    import numpy as np
    s = signal - signal.mean()
    # float(): this is the one place every amplitude leaves numpy.  A numpy
    # amplitude propagates astonishingly far - the retune's parabolic peak
    # interpolation turns it into a numpy FREQUENCY, which becomes segment
    # accels and positions, then move times, then print_time, and finally the
    # temperature callback's read_time; heaters.py then computes
    # can_extrude = (smoothed_temp >= min_extrude_temp) as a numpy bool_, which
    # is not JSON-serializable and SHUTS THE MCU DOWN the next time a client
    # queries the extruder.  Diagnosed 2026-07-22 via RESONANCE_PROBE_AUDIT_STATUS.
    return float(2.0 / len(s) * abs(np.sum(s * np.exp(-2j * np.pi * freq * t))))


# Constant-frequency back-and-forth excitation as (end_time, accel, freq)
# segments: each pulse is a (+a, -a) half-period pair with the sign flipping
# between pulses so the toolhead oscillates symmetrically instead of drifting.
# TIME-parameterized (each segment is a fixed 0.25/f duration) so it plays at
# EXACTLY 'freq' - the basis for the model-free stationary measurements, in
# contrast to the position-parameterized drip path which needs the detuning
# correction.  Consumed by ResonanceTestExecutor.run_test.
def _gen_fixed_freq(freq, accel, duration):
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


# Slide a win_n-sample window by step_n over one accelerometer column, returning
# per-window (amps, zwin, twin) arrays: the single-bin DFT amplitude at 'freq'
# and the Z (from the parallel zpos array) and time at each window's CENTER.
# Used by the Z-mapped drip analyzer (the seg-tag-mapped verify / characterize
# windowers are a separate, multi-axis variant).
def _window_amps(times, col, freq, win_n, step_n, zpos):
    import numpy as np
    amps, zwin, twin = [], [], []
    i, n = 0, len(times)
    while i + win_n <= n:
        amps.append(_dft_amp(times[i:i + win_n] - times[i], col[i:i + win_n],
                             freq))
        c = i + win_n // 2
        zwin.append(zpos[c])
        twin.append(times[c])
        i += step_n
    return np.array(amps), np.array(zwin), np.array(twin)


# Slide a win_n-sample window by step_n and, per window, compute the single-bin
# DFT amplitude at 'freq' for EACH column in 'cols' (the phase term is computed
# once per window and shared across columns), plus the index of the segment whose
# scheduled end time contains the window CENTER (searchsorted into per-segment
# end times 'seg_end').  Returns (wamps, wk): wamps is a list of arrays, one per
# column; wk is an int array of per-window segment indices the caller maps to
# tag / Z / level.  Shared by the moving verify (1 axis) and characterize (3).
def _window_amps_tagged(times, cols, freq, win_n, step_n, seg_end):
    import numpy as np
    seg_end = np.asarray(seg_end, dtype=np.float64)
    nseg = len(seg_end)
    wamps = [[] for _ in cols]
    wk = []
    i, n = 0, len(times)
    while i + win_n <= n:
        ph = np.exp(-2j * np.pi * freq * (times[i:i + win_n] - times[i]))
        for c, col in enumerate(cols):
            seg = col[i:i + win_n]
            wamps[c].append(2.0 / win_n * abs(np.sum((seg - seg.mean()) * ph)))
        wk.append(min(int(np.searchsorted(seg_end, times[i + win_n // 2])),
                      nseg - 1))
        i += step_n
    return [np.array(w) for w in wamps], np.array(wk)


# Per-probe instrumentation, silent unless VERBOSE=1.
#
# These lines were the most valuable debugging asset in getting detection
# working, so they are GATED rather than deleted.  But a single descent emits
# about six of them, so PROBE_ACCURACY SAMPLES=5 prints ~30 lines and a bed
# mesh prints hundreds - unusable as a default.  Every command that reaches
# this module takes the flag, e.g. "PROBE_ACCURACY SAMPLES=5 VERBOSE=1" or
# "BED_MESH_CALIBRATE VERBOSE=1".
#
# Warnings, errors and anything reporting a RECOVERED or REJECTED condition
# stay unconditional - those are things the user must see.
def _dbg(gcmd, msg):
    if gcmd.get_int("VERBOSE", 0):
        gcmd.respond_info(msg)


# gcmd wrapper that swallows respond_info (all other access passes through to
# the originating command) so a fixed-frequency driven-scan test does not print
# a "Testing frequency" line per step during a retune.
class _QuietGCmd:
    def __init__(self, gcmd):
        self._gcmd = gcmd
    def respond_info(self, msg, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return getattr(self._gcmd, name)

# Retry points evenly spread (2*pi/tries apart) on a circle of the given
# radius around (x, y), at a random base angle - same idea as
# ResonanceProbeCalibrate._nudge_positions (mesh nudge), just without mesh
# bounds clamping since a single PROBE has none to honor.  Spreading the tries
# evenly instead of picking each independently at random keeps successive
# retries from clustering back near the same bad spot by chance.
def _nudge_ring(x, y, radius, tries):
    if tries <= 0:
        return []
    base = random.uniform(0., 2. * math.pi)
    return [(x + radius * math.cos(base + 2. * math.pi * k / tries),
             y + radius * math.sin(base + 2. * math.pi * k / tries))
            for k in range(tries)]

class _NudgingSampleAveragingHelper(probe.SampleAveragingHelper):
    # Like probe.SampleAveragingHelper, but when a sample set exceeds
    # samples_tolerance the retry is taken at the next point on an evenly
    # spread ring (see _nudge_ring) of probe_nudge_radius around the requested
    # XY, instead of re-probing the exact same spot.  A bad local spot - a
    # plastic blob, a dead/low-friction patch - is the usual cause of one wild
    # sample, and re-probing it just reproduces the disagreement; nudging a
    # fraction of a millimetre off it clears the spot while keeping the height
    # change negligible (a good radius is ~ the nozzle tip's outer diameter).
    # probe_nudge_radius == 0 reproduces the stock behavior exactly.
    def __init__(self, config, param_helper, start_session_cb):
        super().__init__(config, param_helper, start_session_cb)
        self.nudge_radius = config.getfloat('probe_nudge_radius', 0., minval=0.)
    def run_probe(self, gcmd):
        if self.nudge_radius <= 0.:
            return super().run_probe(gcmd)
        if self.hw_probe_session is None:
            self._probe_state_error()
        params = self.param_helper.get_probe_params(gcmd)
        toolhead = self.printer.lookup_object('toolhead')
        # The XY the caller asked for stays the center of the nudge ring even
        # across multiple retries, so a nudge never walks away from the point.
        home_xy = toolhead.get_position()[:2]
        nudges = _nudge_ring(home_xy[0], home_xy[1], self.nudge_radius,
                             params['samples_tolerance_retries'])
        probexy = list(home_xy)
        retries = 0
        positions = []
        sample_count = params['samples']
        while len(positions) < sample_count:
            pos = self._probe(gcmd)
            positions.append(pos)
            z_positions = [p.bed_z for p in positions]
            if max(z_positions) - min(z_positions) > params['samples_tolerance']:
                if retries >= len(nudges):
                    raise gcmd.error("Probe samples exceed samples_tolerance")
                positions = []
                probexy = list(nudges[retries])
                retries += 1
                gcmd.respond_info(
                    "Probe samples exceed tolerance. Retrying at nudged"
                    " (%.2f, %.2f)..." % (probexy[0], probexy[1]))
            # Retract, moving over to the (possibly nudged) XY for the next probe.
            if len(positions) < sample_count:
                cur_z = toolhead.get_position()[2]
                toolhead.manual_move(
                    probexy + [cur_z + params['sample_retract_dist']],
                    params['lift_speed'])
        epos = probe.calc_probe_z_average(positions, params['samples_result'])
        self.results.append(epos)

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
        # Optional per-axis live-halt floors, populated by the CALIBRATE
        # air-noise pass (SAVE_CONFIG).  Each axis triggers only on a drop above
        # its own descent-noise floor; an axis without one falls back to the
        # scalar halt_sensitivity.  See _HostResonanceEndstop.
        self.halt_sensitivity_axis = [
            config.getfloat('halt_sensitivity_%s' % ax, None,
                            above=0., below=1.) for ax in 'xyz']
        # DERIVATIVE halt threshold: the single-window amplitude down-step that
        # counts as contact, used in PARALLEL with halt_sensitivity above.  See
        # _HostResonanceEndstop for the measurements; briefly, contact is a
        # ~3-window step that the ratio test's median smoothing dilutes away on
        # quieter axes, while the per-window step separates air from contact by
        # >26 sigma.  Set 0 to disable and run on the ratio test alone.
        self.deriv_sensitivity = config.getfloat('deriv_sensitivity', 0.08,
                                                 minval=0., below=1.)
        self.deriv_sensitivity_axis = [
            config.getfloat('deriv_sensitivity_%s' % ax, None,
                            above=0., below=1.) for ax in 'xyz']
        # DRAWDOWN halt test (analog_contact.ContactDetector), the third
        # parallel detector.  Threshold is a FRACTION of the running reference
        # and is derived from each descent's own air noise, so it needs no
        # per-location config - see _HostResonanceEndstop.
        self.drawdown_sensitivity = config.getfloat('drawdown_sensitivity',
                                                    0.10, minval=0., below=1.)
        self.drawdown_nsigma = config.getfloat('drawdown_nsigma', 8.,
                                               minval=1.)
        # 0.06 measured on the descent corpus: a 0.12 lookback lets slow air
        # wander accumulate against a peak set half a window-span ago and fires
        # in mid-air on 5 of 44 never-touched descents; 0.06 drops that to 4
        # (and to 2 if drawdown_nsigma is also raised - but that value is shared
        # with the mode selector, so it is left alone here).  Labelled overshoot
        # traces show the trigger depth is IDENTICAL either way, so the tighter
        # lookback costs nothing in detection.
        self.drawdown_lookback = config.getfloat('drawdown_lookback', 0.06,
                                                 above=0.)
        # Verify/refine sampling.  Each rep is one down ramp plus one up ramp
        # through the candidate contact, and each ramp now yields its OWN
        # estimate, so reps buy independent datapoints (and a measurable
        # up-vs-down bias) rather than a longer pooled sample.  Costs time:
        # roughly one ramp pair per rep at VERIFY_RAMP_SPEED.
        self.verify_reps = config.getint('verify_reps', 1, minval=1, maxval=10)
        # 1 = report the mean of the down and up estimates (default; measured
        # best, and it cancels the positional up/down bias).  0 = down ramp
        # only, which is what this did before the up ramp was analysed at all -
        # kept for A/B, not because it is better.
        self.verify_combine = config.getint('verify_combine', 1, minval=0,
                                            maxval=1)
        # Directory for automatic per-descent trace capture.  Every halting
        # descent writes one CSV (amplitude per axis vs mm below arming) plus a
        # metadata header, building a replay corpus.  Detector changes can then
        # be evaluated offline against real descents instead of by probing
        # again - which matters because probing WEARS the plate: a smooth PEI
        # surface shows visible marking after a few hundred contacts, and a
        # worn spot measurably changes both its noise floor and which axis
        # carries the contact signal.
        self.trace_dir = config.get('trace_dir', None)
        # Free-form tag copied into every trace header.  The physical surface is
        # a measurement condition exactly like freq or accel_per_hz - a textured
        # plate detects differently from a smooth one - but nothing in the
        # machine can read it, so it has to be declared.  Config rather than
        # G-code on purpose: a deploy service-restarts klippy, and a note that
        # lived in RAM would silently vanish mid-session and untag the traces
        # recorded after it.  Absence of the line means "not declared", which is
        # also what the 264 traces recorded before this option say.
        self.trace_note = config.get('trace_note', None)
        # Part-fan speed saved while probing, restored afterwards.  See
        # _quiet_part_fan for why the part fan (but not the heatsink fan) must
        # be off for a measurement to mean anything.
        self._fan_saved = None
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
        #  hostdriven - vibrate while descending; a host task watches the proven
        #               host detection and halts the drip move on contact (loose
        #               threshold), then post-halt analysis finds the precise
        #               contact Z.  No MCU trigger.
        self.probe_mode = config.getchoice(
                'probe_mode', {'stepwise': 'stepwise',
                               'hostdriven': 'hostdriven'}, 'stepwise')
        self.probe_step = config.getfloat('probe_step', 0.05, above=0.)
        self.detect_time = config.getfloat('detect_time', 0.3, above=0.05)
        # The Z descent rate while vibrating.  By default this is the standard
        # probe 'speed' (so there's one speed knob, like every other probe); the
        # Z travelled per lateral vibration is derived from it.  An explicit
        # descend_speed overrides only if you want the descent slower than other
        # probe moves.  Resolved per-probe into self._descend_speed.
        self.descend_speed = config.getfloat('descend_speed', None, above=0.)
        self._descend_speed = self.descend_speed or 1.
        # How far below the start height the vibrating descent may travel.
        self.probe_distance = config.getfloat('probe_distance', 0.5, above=0.)
        # Speed-aware detection: the amplitude window stays a fixed *time* (above)
        # so it always spans enough excitation cycles, but the analysis STEP and
        # the contact-confirmation are expressed in Z *distance* so detection
        # behaves the same at any descent speed.  detect_step_z sets how finely Z
        # is sampled (windows overlap more at higher speed, keeping localization
        # tight); detect_confirm_z is how far the amplitude drop must persist to
        # count as contact (rejects momentary ramp-up/noise dips).
        self.detect_step_z = config.getfloat('detect_step_z', 0.01, above=0.)
        self.detect_confirm_z = config.getfloat('detect_confirm_z', 0.05,
                                                above=0.)
        # Amplitude window length in EXCITATION CYCLES (not fixed seconds), so it
        # is the shortest window that still measures amplitude accurately and is
        # frequency-aware.  Its Z span is detect_cycles/freq*descend_speed, which
        # grows with speed and smears the contact edge; detect_offset_frac
        # subtracts that fraction of the window's Z span from the detected Z to
        # cancel the resulting speed-dependent depth bias (the threshold is
        # crossed when the window's leading edge reaches contact, ~offset_frac of
        # a window above the true contact).  Tune offset_frac so the reported Z
        # is the same across descent speeds.
        self.detect_cycles = config.getfloat('detect_cycles', 5., minval=2.)
        self.detect_offset_frac = config.getfloat('detect_offset_frac', 0.,
                                                  minval=0., maxval=1.)
        # Optional fixed height to rapid-move to before each probe descent.  The
        # vibrating descent then travels at most probe_distance below this, so
        # the probe never queues a huge descent (timer-too-close) nor drives the
        # nozzle toward z_min from an arbitrary start height.  Unset = probe from
        # wherever the toolhead currently is.
        self.probe_start_z = config.getfloat('probe_start_z', None)
        # Speed of the (non-vibrating) rapid move to probe_start_z.  Defaults to
        # the probe lift_speed, which is often slow; set higher for a quick
        # reposition from a parked height.
        self.probe_start_speed = config.getfloat('probe_start_speed', None,
                                                 above=0.)
        # Peak lateral displacement of the excitation (mm); default keeps the
        # sinusoid's acceleration amplitude at accel_per_hz * frequency.
        default_amp = self.accel_per_hz / (4. * math.pi**2 * self.excitation_freq)
        self._amp_is_default = config.get('probe_amplitude', None) is None
        self.probe_amplitude = config.getfloat('probe_amplitude', default_amp,
                                               above=0.)
        # Per-session auto re-tune: before each probe session, run a short
        # continuous frequency sweep spanning +/-retune_range around the
        # *configured* excitation_frequency and adopt the spectral peak on the
        # monitored axis.  0 = disabled.  Tracks environmental drift without a
        # manual FIND_FREQ.
        self._base_freq = self.excitation_freq
        self.retune_range = config.getfloat('retune_range', 0., minval=0.)
        # retune_hz_per_sec is retained for config compatibility; the retune now
        # uses a driven-response scan (see _retune) rather than a swept PSD.
        self.retune_hz_per_sec = config.getfloat('retune_hz_per_sec', 1.,
                                                 above=0., maxval=2.)
        # Driven-scan retune: frequency step (Hz, peak parabolically interpolated)
        # and the excitation dwell per step.
        self.retune_step = config.getfloat('retune_step', 1., above=0.)
        self.retune_time = config.getfloat('retune_time', 0.4, above=0.05)
        # Optional per-point excitation frequency: a mesh of frequencies whose
        # coordinate extents are borrowed from [bed_mesh] (mesh_min/mesh_max), so
        # there is no redundant extent here.  Rows = Y (front->back), columns = X,
        # matching bed_mesh's ordering.  A degenerate dimension broadcasts: an Nx1
        # column varies with Y only, a 1xM row with X only, a 1x1 mesh is constant
        # (== the scalar excitation_frequency).  Unset -> use the scalar (which
        # the per-session retune may still adjust).
        self._freq_mesh = None
        self._freq_mesh_interp = config.getchoice(
                'freq_mesh_interp',
                {'bilinear': 'bilinear', 'nearest': 'nearest'}, 'bilinear')
        if config.get('freq_mesh', None) is not None:
            self._freq_mesh = config.getlists('freq_mesh', seps=(',', '\n'),
                                              parser=float)
            # Rows MAY be ragged: a row with a single value is constant across X
            # at that Y, so only rows that actually vary across X carry multiple
            # X-points (and only those cost extra retunes).
            if not self._freq_mesh or any(len(r) < 1
                                          for r in self._freq_mesh):
                raise config.error(
                    "[%s] freq_mesh must have at least one value per row"
                    % (config.get_name(),))
        self._bed_mesh = None
        self._reported_freq = None
        # Session mesh: a mutable copy of freq_mesh whose cells are retuned lazily
        # (None = use the configured freq_mesh as-is).  _retuned tracks which
        # cells have already been retuned this session.  Both reset per session.
        self._active_mesh = None
        self._retuned = set()
        self.printer.register_event_handler('klippy:connect',
                                            self._handle_connect)
        # Standard probe interface (PROBE command, bed mesh, etc.)
        self.param_helper = probe.ProbeParameterHelper(config)
        self.cmd_helper = probe.ProbeCommandHelper(config, self)
        self.probe_offsets = probe.ProbeOffsetsHelper(config)
        self.probe_session = _NudgingSampleAveragingHelper(
                config, self.param_helper, self._start_hw_session)
        self.printer.add_object('probe', self)

    # Probe interface used by ProbeCommandHelper / bed mesh / PROBE
    def get_probe_params(self, gcmd=None):
        return self.param_helper.get_probe_params(gcmd)
    def get_offsets(self, gcmd=None):
        return self.probe_offsets.get_offsets(gcmd)
    # The part-cooling fan is a MEASURED hazard, not a theoretical one.  Parked
    # on this machine it adds ~25x broadband to the accelerometer plus a hard
    # tone whose fundamental tracks RPM: 100Hz at 25%, 108 at 50%, 136 at 75%,
    # 158 at 100%, with second harmonics at 200-216Hz.  That sweep crosses or
    # nearly hits every excitation mode in use (68Hz~65.5, 216Hz~212.2, and
    # 158Hz is 10Hz off 148), and in band it is up to 10x more noise.  Since the
    # halt threshold is self-derived as max(sens, nsigma*sd) from air noise,
    # that inflates the threshold and silently shrinks detection margin - the
    # same failure as a false halt, from a cause outside the probe.
    #
    # Turning it down is NOT a workaround: 50% produced the LOUDEST tone of all
    # (a mount resonance), so the damage is not monotonic with speed.  Off is
    # the only safe state.
    #
    # The HEATSINK fan is deliberately left alone: measured at +1% broadband
    # with no tones, and it must keep running whenever the nozzle is hot.
    def _quiet_part_fan(self, gcmd):
        if self._fan_saved is not None:
            return                      # already quieted by this session
        fan = self.printer.lookup_object('fan', None)
        if fan is None:
            return
        # get_status() reports the APPLIED speed, and M106 is scheduled through
        # the motion queue - so a fan command issued just before probing is
        # still pending and reads as 0.  Observed on hardware: a probe preceded
        # by "M106 S153" saw speed=0, disabled nothing, and then the fan spun up
        # DURING the descent.  Flush first so the reading is the real state.
        self.printer.lookup_object('toolhead').wait_moves()
        eventtime = self.printer.get_reactor().monotonic()
        speed = float(fan.get_status(eventtime).get('speed', 0.) or 0.)
        if speed <= 0.:
            return
        self._fan_saved = speed
        gcode = self.printer.lookup_object('gcode')
        # Dwell so the fan actually spins DOWN before anything is measured -
        # commanding zero does not stop it instantly.
        gcode.run_script_from_command("M107\nG4 P2000")
        self._fan_msg(gcmd,
            "resonance_probe: part fan was at %.0f%%; disabled for probing"
            " (it adds up to 10x in-band noise) and will be restored"
            % (speed * 100.,))

    # Session teardown has no gcmd, and ResonanceProbe does not hold a gcode
    # object, so route the message through whichever is available.
    def _fan_msg(self, gcmd, msg):
        if gcmd is not None:
            gcmd.respond_info(msg)
        else:
            self.printer.lookup_object('gcode').respond_info(msg)

    def _restore_part_fan(self, gcmd=None):
        speed = self._fan_saved
        if speed is None:
            return
        self._fan_saved = None
        gcode = self.printer.lookup_object('gcode')
        gcode.run_script_from_command("M106 S%d" % (int(round(speed * 255.)),))
        self._fan_msg(gcmd, "resonance_probe: part fan restored to %.0f%%"
                      % (speed * 100.,))

    def start_probe_session(self, gcmd):
        # Once per session: (optionally) re-tune.  With a freq_mesh + retune, each
        # mesh cell is retuned LAZILY at the first probe point that needs it (no
        # upfront detour - see _apply_point_frequency); here we just start a
        # mutable session copy.  Without a mesh, retune the single scalar now.
        self._reported_freq = None
        self._retuned = set()
        # Before anything is measured, including the retune sweep below.
        self._quiet_part_fan(gcmd)
        if self.retune_range > 0. and self._freq_mesh is not None:
            self._check_axis_safety(gcmd)
            self._active_mesh = [list(r) for r in self._freq_mesh]
        else:
            self._active_mesh = None
            if self.retune_range > 0.:
                self._check_axis_safety(gcmd)
                self._goto_start_height(gcmd)
                self._retune(gcmd)
        return self.probe_session.start_probe_session(gcmd)
    def get_status(self, eventtime):
        # Sanitized: this dict goes straight to the webhook/JSON layer.
        return _plain(self.cmd_helper.get_status(eventtime))
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
        return _gen_fixed_freq(freq, accel, duration)

    # -- probing moves -----------------------------------------------------

    # Rapid-move (no vibration) to the configured fixed probe start height so a
    # probe always begins its bounded vibrating descent from a known Z, no
    # matter where the toolhead happened to be (e.g. parked high after homing).
    # Requires Z homed; returns the resulting start Z.
    def _goto_start_height(self, gcmd):
        toolhead = self.printer.lookup_object('toolhead')
        x, y, z = toolhead.get_position()[:3]
        if self.probe_start_z is None:
            return z
        speed = (self.probe_start_speed
                 or self.param_helper.get_probe_params(gcmd)['lift_speed'])
        toolhead.manual_move([x, y, self.probe_start_z], speed)
        toolhead.wait_moves()
        return self.probe_start_z


    # Adopt a new excitation frequency, keeping the displacement default (which
    # depends on frequency) consistent unless probe_amplitude was set explicitly.
    def _set_excitation_freq(self, freq):
        # float() at the STATE boundary too: this value outlives the call that
        # produced it and seeds every later move, so it must not be numpy even
        # if some future caller computes it differently (see _dft_amp).
        freq = float(freq)
        self.excitation_freq = freq
        if self._amp_is_default:
            self.probe_amplitude = (self.accel_per_hz
                                    / (4. * math.pi**2 * freq))

    # Resolve the optional [bed_mesh] binding that anchors freq_mesh's extents.
    # bed_mesh is loaded by connect time; a soft dependency (None = disable the
    # mesh and fall back to the scalar frequency).
    def _handle_connect(self):
        if self._freq_mesh is None:
            return
        self._bed_mesh = self.printer.lookup_object('bed_mesh', None)
        if self._bed_mesh is None:
            logging.warning("resonance_probe: freq_mesh is set but there is no"
                            " [bed_mesh] to anchor its extents; ignoring it")
            self._freq_mesh = None
            return
        bmc = self._bed_mesh.bmc
        nrows = len(self._freq_mesh)
        maxcols = max(len(r) for r in self._freq_mesh)
        xc, yc = bmc.mesh_config['x_count'], bmc.mesh_config['y_count']
        if nrows > yc or maxcols > xc:
            logging.warning("resonance_probe: freq_mesh has up to %dx%d points but"
                            " the bed_mesh grid is %dx%d; re-run calibration if the"
                            " probe grid changed" % (nrows, maxcols, yc, xc))

    # Per-point excitation frequency from freq_mesh, bilinearly interpolated
    # within the bed_mesh extents and CLAMPED to the edges (never extrapolated).
    # A degenerate mesh dimension (count 1) contributes no interpolation on that
    # axis, so an Nx1 column is X-independent and a 1xM row is Y-independent.
    # Returns None when no usable mesh is configured (caller keeps the current
    # frequency, e.g. a per-session retune result).
    def _freq_at(self, x, y):
        if self._freq_mesh is None or self._bed_mesh is None:
            return None
        bmc = self._bed_mesh.bmc
        xmin, ymin = bmc.mesh_min
        xmax, ymax = bmc.mesh_max
        mesh = self._active_mesh if self._active_mesh is not None \
            else self._freq_mesh
        nearest = self._freq_mesh_interp == 'nearest'
        def frac(v, lo, hi, n):
            if n <= 1 or hi <= lo:
                return 0., 0, 0
            t = min(max((v - lo) / (hi - lo) * (n - 1), 0.), n - 1.)
            i0 = int(t)
            return t - i0, i0, min(i0 + 1, n - 1)
        # Evaluate one (possibly single-value) row at x: a 1-value row is constant
        # across X; a multi-value row interpolates across the X extent.
        def row_at_x(row):
            m = len(row)
            if m == 1:
                return row[0]
            fx, ix0, ix1 = frac(x, xmin, xmax, m)
            if nearest:
                return row[ix1 if fx >= 0.5 else ix0]
            return row[ix0] * (1. - fx) + row[ix1] * fx
        fy, iy0, iy1 = frac(y, ymin, ymax, len(mesh))
        if nearest:
            return row_at_x(mesh[iy1 if fy >= 0.5 else iy0])
        return row_at_x(mesh[iy0]) * (1. - fy) + row_at_x(mesh[iy1]) * fy

    # Before each probe point, adopt the per-point excitation frequency (if a
    # freq_mesh is configured).  Reports only when the frequency changes, so a
    # bed-mesh sweep does not print one line per point.
    def _apply_point_frequency(self, gcmd, x, y):
        if self._freq_mesh is None or self._bed_mesh is None:
            return
        # Lazy per-cell retune: the toolhead is already at the probe point, so if
        # retuning and the nearest mesh cell has not been retuned this session,
        # refine it right here (no detour) before reading the frequency.
        if self.retune_range > 0. and self._active_mesh is not None:
            self._retune_cell_if_needed(gcmd, x, y)
        f = self._freq_at(x, y)
        if f is None:
            return
        self._set_excitation_freq(f)
        if self._reported_freq != round(f, 1):
            self._reported_freq = round(f, 1)
            gcmd.respond_info("Resonance probe: point (%.1f, %.1f) using"
                              " %.1f Hz" % (x, y, f))

    # Driven (steady-state) response amplitude of the monitored axis: vibrate in
    # place at 'freq' (default: the current excitation frequency) and return the
    # single-bin DFT magnitude.  Unlike a swept PSD this measures how hard the
    # toolhead actually resonates when DRIVEN at freq - the right basis for the
    # probe frequency (a swept PSD can pick a shoulder mode that is quiet when
    # driven).  The DC is recomputed every call (so a contact-induced DC shift
    # can't mask the drop) and only the excitation bin is kept (rejecting contact
    # friction noise), so it is far cleaner than the MCU's fixed-DC broadband
    # envelope.  quiet=False surfaces the executor's per-step log line;
    # require=True raises instead of returning 0 when no data is captured.
    def _measure_response_at(self, gcmd, freq=None, duration=None,
                             dwell=0.050, quiet=True, require=False):
        import numpy as np
        if freq is None:
            freq = self.excitation_freq
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.wait_moves()
        toolhead.dwell(dwell)
        aclient = self.chip.start_internal_client()
        accel = self.accel_per_hz * freq
        test_seq = _gen_fixed_freq(freq, accel, duration)
        cmd = _QuietGCmd(gcmd) if quiet else gcmd
        try:
            self.executor.run_test(test_seq, self.vibrate_dir, cmd)
        finally:
            aclient.finish_measurements()
        samples = aclient.get_samples()
        if not samples:
            if require:
                raise gcmd.error("Accelerometer measured no data while probing")
            return 0.
        data = np.asarray(samples, dtype=np.float64)
        t = data[:, 0] - data[0, 0]
        return _dft_amp(t, data[:, 1 + self.output_index], freq)

    # Scan the driven response over center +/- retune_range (retune_step Hz) and
    # return (peak_freq, peak_response), the peak parabolically interpolated
    # between steps.  Restores the toolhead position afterward (the fixed-freq
    # pulses drift it laterally).
    def _driven_peak(self, gcmd, center, speed):
        toolhead = self.printer.lookup_object('toolhead')
        start_pos = toolhead.get_position()
        step = self.retune_step
        n = max(1, int(round(self.retune_range / step)))
        freqs = [center + k * step for k in range(-n, n + 1)
                 if center + k * step >= 1.]
        resp = []
        try:
            for f in freqs:
                resp.append(self._measure_response_at(gcmd, f, self.retune_time))
        finally:
            toolhead.manual_move(list(start_pos[:3]), speed)
            toolhead.wait_moves()
        if not resp or max(resp) <= 0.:
            return center, 0.
        i = resp.index(max(resp))
        best = freqs[i]
        if 0 < i < len(freqs) - 1:
            y0, y1, y2 = resp[i-1], resp[i], resp[i+1]
            den = y0 - 2.*y1 + y2
            if den < 0.:
                d = 0.5 * (y0 - y2) / den
                if -1. <= d <= 1.:
                    best = freqs[i] + d * step
        return best, resp[i]

    # Refine the single scalar frequency to the current driven-response peak over
    # a narrow band (retune_range) and adopt it.
    def _retune(self, gcmd):
        speed = (self.probe_start_speed
                 or self.param_helper.get_probe_params(gcmd)['lift_speed'])
        peak, resp = self._driven_peak(gcmd, self._base_freq, speed)
        if resp <= 0.:
            gcmd.respond_info("Resonance re-tune: no response; keeping %.1f Hz"
                              % (self.excitation_freq,))
            return
        gcmd.respond_info("Resonance re-tune: %.1f Hz (was %.1f), driven scan"
                          " %.1f +/- %.1f Hz on accel %s-axis"
                          % (peak, self._base_freq, self._base_freq,
                             self.retune_range, 'xyz'[self.output_index]))
        self._set_excitation_freq(peak)

    # Retune the freq_mesh cell nearest to (x, y) if it has not been retuned this
    # session, refining it at the current probe point (driven-response).  Each
    # cell is retuned once, the first time a probe point falls nearest to it - so
    # the retune count equals the number of freq_mesh cells actually used, and
    # every retune happens where the probe already is (no detour; a flat/collapsed
    # row is one cell = one retune for the whole row).  _driven_peak restores the
    # toolhead to this probe point when done.
    def _retune_cell_if_needed(self, gcmd, x, y):
        bmc = self._bed_mesh.bmc
        xmin, ymin = bmc.mesh_min
        xmax, ymax = bmc.mesh_max
        mesh = self._active_mesh
        def nidx(v, lo, hi, n):
            if n <= 1 or hi <= lo:
                return 0
            t = (v - lo) / (hi - lo) * (n - 1)
            return min(max(int(round(t)), 0), n - 1)
        iy = nidx(y, ymin, ymax, len(mesh))
        ix = nidx(x, xmin, xmax, len(mesh[iy]))
        if (iy, ix) in self._retuned:
            return
        self._retuned.add((iy, ix))
        speed = (self.probe_start_speed
                 or self.param_helper.get_probe_params(gcmd)['lift_speed'])
        old = mesh[iy][ix]
        peak, resp = self._driven_peak(gcmd, old, speed)
        if resp > 0.:
            mesh[iy][ix] = peak
        gcmd.respond_info("  retune cell [%d,%d] @ (%.1f, %.1f): %.1f -> %.1f Hz"
                          % (iy, ix, x, y, old, mesh[iy][ix]))

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
        self._goto_start_height(gcmd)
        x, y, cur_z = toolhead.get_position()[:3]
        # Descend at most probe_distance below the start, never below z_min.
        z_floor = max(z_min, cur_z - self.probe_distance)
        if cur_z <= z_floor:
            raise gcmd.error("Resonance probe start Z %.3f is at or below the"
                             " descent floor %.3f (z_min=%.3f,"
                             " probe_distance=%.3f)"
                             % (cur_z, z_floor, z_min, self.probe_distance))
        # No-contact baseline amplitude at the (assumed clear) start height
        baseline = self._measure_response_at(gcmd, duration=self.detect_time, dwell=0.100,
                                          quiet=False, require=True)
        threshold = baseline * (1. - self.sensitivity)
        gcmd.respond_info("Resonance probe baseline %.1f; trigger below %.1f"
                          " (%.0f%% drop)"
                          % (baseline, threshold, self.sensitivity * 100.))
        detected = False
        while cur_z > z_floor + 1e-9:
            cur_z = max(z_floor, cur_z - step)
            toolhead.manual_move([x, y, cur_z], speed)
            toolhead.wait_moves()
            amp = self._measure_response_at(gcmd, duration=self.detect_time, dwell=0.100,
                                          quiet=False, require=True)
            if amp < threshold:
                detected = True
                break
        if not detected:
            raise gcmd.error("Resonance probe: no contact detected down to the"
                             " descent floor %.3f" % (z_floor,))
        offsets = self.probe_offsets.get_offsets()
        return manual_probe.create_probe_result(toolhead.get_position(),
                                                offsets)

    # Host-driven halt mode delegates to the shared HaltingContactProbe: it
    # vibrates while descending and halts the move the instant contact damps the
    # resonance (loose threshold), then the anchored analysis pins the precise
    # contact Z.  Built fresh per probe so a per-session re-tune of the
    # excitation frequency is reflected.
    def _make_contact_helper(self):
        return HaltingContactProbe(
            self.printer, self.chip, self.output_index, self.vibrate_dir,
            self.excitation_freq, self.accel_per_hz, self.probe_amplitude,
            self.z_min_position, self.warmup, self.sensitivity,
            self.halt_sensitivity, self.detect_cycles, self.detect_step_z,
            self.detect_confirm_z, self.detect_offset_frac,
            halt_sensitivity_axis=self.halt_sensitivity_axis)

    def _hostdriven_probe(self, gcmd):
        # A single probe can also be issued outside a session (and a session
        # that ERRORS never reaches end_probe_session), so the fan is quieted
        # here too and restored on any way out.  Both calls are idempotent.
        self._quiet_part_fan(gcmd)
        try:
            return self._hostdriven_probe_inner(gcmd)
        except Exception:
            self._restore_part_fan(gcmd)
            raise

    def _hostdriven_probe_inner(self, gcmd):
        self._check_axis_safety(gcmd)
        toolhead = self.printer.lookup_object('toolhead')
        params = self.param_helper.get_probe_params(gcmd)
        lift_speed = params['lift_speed']
        descend_speed = self.descend_speed or params['probe_speed']
        x0, y0, curz = toolhead.get_position()[:3]
        ceiling = (self.probe_start_z if self.probe_start_z is not None
                   else curz)
        start_speed = self.probe_start_speed or lift_speed
        contact_z, halted = self._make_contact_helper().run(
                gcmd, ceiling, self.probe_distance, descend_speed, lift_speed,
                start_speed, x0, y0)
        if contact_z is None:
            raise gcmd.error("Resonance host-driven probe: no contact detected"
                             " (halted=%s)" % (halted,))
        epos = list(toolhead.get_position())
        epos[2] = contact_z
        return manual_probe.create_probe_result(
                _plain(epos), self.probe_offsets.get_offsets())


# Host-driven "endstop" for HomingMove: instead of an MCU trigger, a batch
# callback watches the accelerometer resonance amplitude and completes the drip
# completion (halting the move) when it drops past a loose halt threshold.
# HomingMove then derives the halt position from the stepper history at the
# recorded trigger time.  Detection mirrors the drip analyzer (single-bin DFT
# with the DC recomputed per window).
class _HostResonanceEndstop:
    # Monitor ALL THREE accelerometer axes live, not just the configured
    # excitation/output axis - a contact-damping signal is not guaranteed to
    # show up most clearly on the axis being driven (see resonance-nozzle-probe
    # memory, 2026-07-06 #9: on the smooth plate, the low mode reads ~0-3% drop
    # on X but 30-55% on Y/Z, so a single-axis live trigger has essentially no
    # real signal to detect that mode's contact with at all).  Each axis tracks
    # its own baseline/below-run state independently and the halt fires on
    # whichever axis FIRST confirms a persistent drop.  All three axes share
    # the same halt_sensitivity threshold (no per-axis noise floor is known at
    # descent time - that only comes from the separate, offline
    # characterize_amplitude dwell test) - a consistently noisy cross axis
    # could in principle nuisance-trigger slightly early; watch for this in
    # hardware validation before trusting it unattended on a new axis/mode.
    AXIS_COUNT = 3

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
        self._trigger_axis = None
        self._buf = []
        # Streaming GRADIENT contact detection (see _handle_batch): per axis, a
        # short rolling history of recent (time, amplitude) windows used to
        # measure the local amplitude DROP over a short Z span.  Replaces the old
        # plateau/baseline lock, which could never establish a baseline on the
        # height-dependent RISING amplitude profile (a monotonically climbing
        # signal never plateaus, so the baseline never locked / kept re-arming)
        # and so missed the large contact drop entirely.
        self._amp_hist = [[] for _ in range(self.AXIS_COUNT)]
        self._below_run = [0] * self.AXIS_COUNT
        self._first_below_t = [0.] * self.AXIS_COUNT
        self._last_analyzed = 0
        # Diagnostics: the largest per-axis gradient drop actually observed live,
        # and how many windows were scored - to tell whether a non-halting
        # descent even SAW the contact crater in real time (vs the batch/skip
        # timing hiding it) when debugging a missed halt.
        self._dbg_maxdrop = [0.] * self.AXIS_COUNT
        # Sample time at which each axis's max drop occurred, to locate the drop
        # in the descent (startup transient vs mid-descent vs contact).
        self._dbg_maxdrop_t = [0.] * self.AXIS_COUNT
        self._dbg_nwin = 0
        # Absolute amplitude of the first few scored windows (air, well above
        # the bed).  A gradient is a RATIO, so it cannot distinguish "contact
        # damped a strong excitation" from "the excitation itself is weak"; if
        # this falls off as descend_speed rises, the descent is detuning the
        # oscillation rather than the detector missing the crater.
        self._dbg_amp0 = [[] for _ in range(self.AXIS_COUNT)]
        # Full per-window amplitude trace of the descent, all axes, for offline
        # analysis (RESONANCE_PROBE_DUMP_TRACE).  The gradient test reduces each
        # window to one number and throws the shape away, which is exactly what
        # is needed to tell "no signal on this axis" from "a signal the ratio
        # test cannot express" - e.g. a steady slope with a KNEE at contact
        # rather than a step.  Appending a 4-tuple per window is cheap enough
        # for the callback (a couple hundred windows per descent); everything
        # expensive - reconstructing Z, differentiating, writing a file -
        # happens after the descent, never in here.
        self._trace = []
        # Derivative trigger state: previous window's amplitude per axis, the
        # consecutive-step run, and the largest DOWN step seen (diagnostic, the
        # derivative counterpart of _dbg_maxdrop).
        self._prev_amp = [None] * self.AXIS_COUNT
        self._deriv_run = [0] * self.AXIS_COUNT
        self._dbg_minstep = [0.] * self.AXIS_COUNT
        self._trigger_kind = 'gradient'
        self._dbg_win_z = 0.   # Z span one DFT window averages over (see below)
        # Window sized lazily from the measured sample rate (first batches).
        # The window is a fixed time; the step is a fixed Z distance (so the
        # halt behaves the same at any descent speed) and the drop must persist
        # for detect_confirm_z before halting (debounce against transients).
        self._win_cycles = rprobe.detect_cycles
        self._descend_speed = rprobe._descend_speed
        self._step_z = rprobe.detect_step_z
        self._confirm_z = rprobe.detect_confirm_z
        self._win_n = None
        self._step_n = None
        self._confirm_n = 2
        self._grad_n = 3      # slope reference span (windows); set in _handle_batch
        self._grad_persist = 2  # windows the drop must persist to halt
        self._ref_dt = 0.      # gradient reference/smoothing spans (descent
        self._smooth_dt = 0.   # time, s); set from the sample rate in _handle_batch
        self._freq = rprobe.excitation_freq
        self._halt_sens = rprobe.halt_sensitivity
        # Per-axis live-halt floor: an axis triggers only on a drop exceeding its
        # OWN descent-noise floor (from the CALIBRATE air-noise pass), so a
        # twitchy axis is ignored while a quiet one stays sensitive.  Falls back
        # to the scalar halt_sensitivity for any axis without a characterized
        # floor.  This is the per-axis noise floor the class comment above notes
        # was previously missing at descent time.
        axis_floor = getattr(rprobe, 'halt_sensitivity_axis', None)
        self._halt_axis = [
            (axis_floor[i] if axis_floor and axis_floor[i] is not None
             else self._halt_sens) for i in range(self.AXIS_COUNT)]
        # Per-axis DERIVATIVE floor: the single-window down-step that counts as
        # contact.  Measured over 4 descents at 148Hz, the worst air excursion
        # was -1.85% (x) / -2.53% (z) against contact steps of -21% / -23%, so
        # anything in the -5..-10% band has ~3x margin on BOTH sides.  Default
        # 8%, overridable per axis, and floored at 6x the axis's characterized
        # descent noise where that is known - a twitchy axis (y: air sd 8.3%,
        # worst air excursion -23%) then disarms itself from its own numbers
        # rather than by rule, exactly as the ratio floors do.
        self._deriv_sens = getattr(rprobe, 'deriv_sensitivity', 0.08)
        deriv_floor = getattr(rprobe, 'deriv_sensitivity_axis', None)
        # Scale the per-axis derivative floor off that axis's RATIO floor, which
        # calibration already derived from its measured descent noise.  An axis
        # that needed a high ratio floor is noisy and needs a high derivative
        # floor for the same reason.  1.5x because the ratio floor is set
        # against a SMOOTHED drop while this test sees single-window noise,
        # which is larger.
        #
        # Do not reintroduce a `descent_noise_axis` fallback here: no such
        # attribute exists, so an earlier version silently armed every axis at
        # the flat 8% default.  On y (air sd 8.3%) that is a 1-sigma threshold,
        # and it produced a false halt 660um above the bed on the first live
        # test.  Measured floors under this rule: x 10.8%, y 32.3%, z 13.4%,
        # against contact steps of -21% (x), -37% (y), -23% (z) and worst air
        # excursions of -1.9% (x), -23.2% (y), -2.5% (z).
        # (no noise_axis fallback - see above)
        # 0 disables the derivative trigger entirely (ratio test only).  Note
        # this MUST short-circuit the per-axis derivation below - a threshold of
        # 0 would otherwise make "step <= -0" true on every window and halt the
        # descent immediately.
        # A ratio floor at/above DISARM is the established "this axis must
        # never trigger" sentinel (calibration parks unusable axes at 0.95, and
        # CHARACTERIZE_NOISE sets 1.0 on every axis to guarantee a no-halt air
        # descent).  The derivative and drawdown tests MUST honour it too, or
        # they silently break that guarantee - observed on hardware: a
        # "floors=1.0" air descent halted on drawdown anyway, truncating the
        # very noise measurement it existed to collect.
        DISARM = 0.95
        self._disarmed = [self._halt_axis[i] >= DISARM
                          for i in range(self.AXIS_COUNT)]
        self._deriv_axis = [None] * self.AXIS_COUNT
        if self._deriv_sens > 0.:
            for i in range(self.AXIS_COUNT):
                if self._disarmed[i]:
                    continue
                if deriv_floor and deriv_floor[i] is not None:
                    self._deriv_axis[i] = deriv_floor[i]
                else:
                    self._deriv_axis[i] = max(self._deriv_sens,
                                              1.5 * self._halt_axis[i])
        # Two consecutive down-steps, matching _grad_persist.  Contact gave 2-3
        # such windows in every trace, and a single-window rule would be one
        # noise excursion away from a false halt.
        self._deriv_persist = 2
        # DRAWDOWN detectors (analog_contact), built lazily once enough air
        # windows exist to measure this descent's own noise - see _handle_batch.
        # Self-referencing on purpose: surface features wander (removable
        # plates, and a worn spot migrates), so a stored per-location floor goes
        # stale in a way a fresh per-descent estimate cannot.
        self._dd_sens = getattr(rprobe, 'drawdown_sensitivity', 0.10)
        self._dd_nsigma = getattr(rprobe, 'drawdown_nsigma', 8.)
        self._dd_lookback = getattr(rprobe, 'drawdown_lookback', 0.06)
        self._dd_det = [None] * self.AXIS_COUNT
        self._dd_air = [[] for _ in range(self.AXIS_COUNT)]
        self._dd_thresh = [None] * self.AXIS_COUNT
        self._dd_warm = 40      # air windows used for the noise estimate

    def get_steppers(self):
        return self._steppers

    def get_trigger_time(self):
        return self._trigger_time

    def get_trigger_axis(self):
        return self._trigger_axis

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
                spd0 = max(self._descend_speed, 1e-9)
                self._win_n = max(8, int(self._win_cycles / self._freq * sps))
                # ROOT CAUSE of the drop vanishing at higher descend speed: a
                # DFT window is a fixed TIME (win_cycles/freq), so the Z span it
                # averages over grows in proportion to the descent speed -
                # 0.021mm at 0.3 mm/s but 0.056mm at 0.8.  There is only ~0.13mm
                # of travel below contact before the safety floor, so at speed
                # every window straddles the surface and averages air together
                # with contact; the 26% damping showed up as 1-2% (measured
                # 9%/2%/1% at 0.3/0.5/0.8 - and the largest drop was no longer
                # even AT the contact depth, i.e. noise was winning).  So cap the
                # window by its Z EXTENT, not just its cycle count: keep it near
                # the Z resolution that works at 0.3 mm/s.  Fewer cycles makes
                # the single-bin DFT noisier, hence the hard 2-cycle floor (below
                # that the estimate is meaningless); the excitation axis has
                # plenty of SNR for 2 cycles, and a noisier estimate that is
                # sharp in Z is what this detector needs - a false halt is
                # cheaply rejected by the verify-on-halt retry, whereas a diluted
                # crater is not recoverable at all.
                MAX_WIN_Z = LIVE_WIN_Z
                self._win_n = max(8, int(2. / self._freq * sps),
                                  min(self._win_n, int(MAX_WIN_Z * sps / spd0)))
                # Step a fixed Z distance; confirm a drop over detect_confirm_z.
                # But bound the window RATE: a Z-sized step means windows arrive
                # proportionally faster as descend_speed rises, and past ~30/s
                # the reactor cannot analyze them, so the catch-up guard below
                # throws most away (measured: 118/89/24 windows analyzed for the
                # same 1.35mm at 0.3/0.5/0.8 mm/s - at 0.8 that is 1 window per
                # 0.056mm, sampled too sparsely and too erratically to resolve
                # the contact crater).  Flooring the step to MIN_STEP_DT trades Z
                # resolution for windows that are actually scored: coarser but
                # real beats fine but discarded.  Z resolution is recovered
                # afterwards by the offline fine analysis, which sees every
                # sample regardless.
                MIN_STEP_DT = 0.030
                cfg_step_z = self._step_z   # configured, before the rate floor
                self._step_n = max(1, int(round(MIN_STEP_DT * sps)), int(round(
                    self._step_z * sps / max(self._descend_speed, 1e-9))))
                # Effective step in Z after the rate floor - everything below
                # (confirm counts, reference spans) must use this, not the
                # configured detect_step_z, or the spans silently shrink to
                # fewer windows than intended at speed.
                self._step_z = self._step_n / sps * max(self._descend_speed,
                                                        1e-9)
                self._confirm_n = max(2, int(round(
                    self._confirm_z / self._step_z)))
                # Slope reference span ~= detect_confirm_z: the reference window
                # sits a full confirm-distance ABOVE the current one (well clear
                # of the smoothing windows) so the drop measured across a real
                # contact crater is large.  Then require it to persist just a
                # couple of windows (the wide span already carries the evidence;
                # a long persist would run past the shallow contact zone into the
                # z_floor before confirming).
                self._grad_n = max(3, self._confirm_n)
                self._grad_persist = 2
                # Reference/smoothing spans as descent TIME (progress), so the
                # gradient references a window by how far the descent has moved -
                # NOT by window index.  This tolerates windows the catch-up guard
                # skips (which it does at higher descend speed): the reference is
                # found from whatever windows survive near tc-ref_dt, so a gap
                # just defers the gradient a little instead of starving it or
                # firing a phantom across the gap.  Same span (~confirm_z) as the
                # old index scheme, so the history stays the same small size.
                spd = max(self._descend_speed, 1e-9)
                # The reference must sit at least ~2 DFT-window Z-heights above
                # the current window, else at higher descend speed the window's
                # own Z-coverage (win_z = win_n/sps * speed) overlaps the
                # reference and SMEARS the contact drop (measured 9%/3%/1% at
                # 0.3/0.5/0.8 mm/s before this).  Scaling ref_z with win_z keeps
                # the drop un-diluted as speed rises.
                win_z = self._win_n / sps * spd
                self._dbg_win_z = win_z
                # BUDGET THE SPANS IN Z.  There is only ~0.13mm of travel below
                # contact before the safety floor, and the detector needs 'cur'
                # entirely inside the crater while 'ref' is still in air.  That
                # costs smooth_z + win_z/2 (to get cur fully damped) plus ref_z
                # (to lift ref back into air).  Tying these to step_z made them
                # grow with the rate floor above: at 0.5 mm/s smooth_z reached
                # 0.045mm, so the budget wanted ~0.15mm - more than exists, the
                # reference band sat INSIDE the crater too, and the 26% damping
                # measured as 3%.  Fixed Z spans keep the budget at ~0.12mm at
                # every speed.  MIN_SMOOTH_Z floors at one step so the median
                # always has a window to work with.
                # 3 * the CONFIGURED step (what this span was before the rate
                # floor above started inflating step_z), floored at one actual
                # step so the median always has a window to work with.
                smooth_z = max(self._step_z, 3. * cfg_step_z)
                ref_z = max(self._confirm_z, 2. * win_z)
                self._ref_dt = ref_z / spd
                self._smooth_dt = smooth_z / spd
            else:
                return True
        # Bound the worst-case work done in one callback.  Fixing the window-
        # advance bug above (see its comment) means a delayed/bursty batch -
        # e.g. from a deep drip look-ahead - now does a FULL, non-redundant
        # analysis of every backlogged step instead of cheaply re-reading the
        # same tail window; combined with 3-axis analysis (all three axes
        # every window, not just one) that made one real "Timer too close"
        # MCU shutdown reproduce on hardware (2026-07-06/07 - the host fell
        # behind feeding the MCU while the reactor was busy catching up this
        # callback).  If backlog exceeds MAX_CATCHUP_STEPS worth of steps,
        # skip forward to the most recent MAX_CATCHUP_STEPS instead of
        # analyzing the whole gap - a slow reactor callback is exactly what
        # starves the step buffer, so bounding this is a real safety fix, not
        # just a performance one.  Skipped windows are simply never scored;
        # this cannot mask a real contact, since the halt only needs ONE
        # window's confirm_n run to fire and there are always several more
        # windows ahead as the descent continues.
        MAX_CATCHUP_STEPS = 4
        n_pending = (len(self._buf) - self._last_analyzed) // self._step_n
        if n_pending > MAX_CATCHUP_STEPS:
            # Bound the per-callback DFT work (host falling behind is what starves
            # the MCU step buffer -> "Timer too close").  The skipped windows just
            # leave a time gap in the history; the gradient below references a
            # window by tc (descent time), not by index, so it tolerates the gap
            # instead of firing a phantom across it or restarting - which is what
            # keeps detection alive when skips happen (esp. at higher speed).
            self._last_analyzed += (n_pending - MAX_CATCHUP_STEPS) * self._step_n
        while len(self._buf) - self._last_analyzed >= self._step_n:
            self._last_analyzed += self._step_n
            if self._last_analyzed < self._win_n:
                continue
            # Anchor the window at _last_analyzed's own progress through the
            # buffer, not always at the tail - a delayed/bursty batch delivery
            # (e.g. from a deeper drip look-ahead) can hand this loop many
            # steps' worth of new data in one callback, and always reading the
            # tail would re-analyze the same final window on every one of
            # those iterations instead of advancing through each in turn.
            seg = self._buf[self._last_analyzed - self._win_n:
                            self._last_analyzed]
            tc = seg[len(seg) // 2][0]
            if tc < self._armed_time:
                continue  # excitation still ringing up
            # One array per window covering all 3 axes (not 4 separate
            # np.array() conversions) - cheaper per-window overhead, which
            # matters now that every window scores 3 axes instead of 1.
            arr = np.asarray(seg, dtype=np.float64)
            t = arr[:, 0]
            ref = np.exp(-2j * np.pi * self._freq * (t - t[0]))
            cols = arr[:, 1:1 + self.AXIS_COUNT]
            cols = cols - cols.mean(axis=0)
            amps = 2. / len(t) * np.abs(np.sum(cols * ref[:, None], axis=0))
            self._trace.append((float(tc), float(amps[0]), float(amps[1]),
                                float(amps[2])))
            for a_idx in range(self.AXIS_COUNT):
                # GRADIENT contact: append this window's amplitude to a short
                # rolling history, then compare a smoothed CURRENT level to a
                # smoothed level self._grad_n windows (~half detect_confirm_z)
                # ABOVE it.  A steep local DROP (>= halt_sensitivity over that
                # span) that persists confirm_n windows is contact.  The gentle
                # height-dependent RISE has the opposite slope, so it can never
                # trigger - which is exactly what defeated the old plateau lock
                # (a monotonically rising signal never plateaus, so no baseline
                # ever locked and the big contact drop was missed).
                hist = self._amp_hist[a_idx]
                hist.append((float(tc), float(amps[a_idx])))
                if len(self._dbg_amp0[a_idx]) < 8:
                    self._dbg_amp0[a_idx].append(float(amps[a_idx]))
                # Prune by TIME to the reference span (+ a smoothing margin).
                cutoff = tc - self._ref_dt - self._smooth_dt
                while hist and hist[0][0] < cutoff:
                    hist.pop(0)
                # Need history reaching ~ref_dt back to have a reference level.
                if hist[0][0] > tc - self._ref_dt:
                    continue
                ref_t = tc - self._ref_dt
                # Smoothed CURRENT (most recent smooth_dt) vs a smoothed level a
                # full ref_dt (~confirm_z) ABOVE it - selected by tc, so skipped
                # windows just thin the medians rather than shifting the span.
                # Split the (already time-pruned) history at ref_t rather than
                # sampling a narrow band around it.  A band can come up EMPTY
                # whenever windows are sparse or unevenly spaced, and an empty
                # band skipped the comparison entirely - so at higher descend
                # speed the gradient was evaluated only on the windows that
                # happened to line up, and mostly missed contact.  A split
                # always yields both sides (hist[0] is at/older than ref_t by
                # the check above, and this window itself is at tc), so every
                # window gets scored at every speed.
                cur_vals = [a for (t, a) in hist if t >= tc - self._smooth_dt]
                ref_vals = [a for (t, a) in hist if t <= ref_t]
                cur = np.median(cur_vals)
                ref_amp = np.median(ref_vals)
                drop = (ref_amp - cur) / ref_amp if ref_amp > 1e-9 else 0.
                if drop > self._dbg_maxdrop[a_idx]:
                    self._dbg_maxdrop[a_idx] = drop
                    self._dbg_maxdrop_t[a_idx] = float(tc)
                if a_idx == 0:
                    self._dbg_nwin += 1
                # DERIVATIVE trigger, in parallel with the ratio test above.
                # The ratio compares a MEDIAN over smooth_dt against one a
                # ref_dt above, which is robust against the gentle air rise but
                # cannot see a short event: contact is a ~3-window step, so the
                # current-side median is still mostly air when it happens.  On
                # hardware that cost the z axis entirely - z stepped -22..-26%
                # at contact, well above its 9% floor, yet the ratio never read
                # more than 4-7% and z never triggered once in five descents.
                #
                # The per-window change has no such dilution, and separates far
                # better (measured over 4 descents at 148Hz):
                #   x: air sd 0.62-0.70%, worst air -1.85%, contact -21..-24%
                #   z: air sd 0.87-0.92%, worst air -2.53%, contact -23..-27%
                #   y: air sd 8.2-8.8%,   worst air -23.2%, contact -37..-44%
                # i.e. >26 sigma on x and z.  It also needs no smoothing, so the
                # rising air baseline is handled structurally rather than by
                # tuning it out.
                #
                # SIGNED, deliberately: the FIRST contact window steps UP on the
                # cross axes (z +13.8%, y +25.4%), most likely the nozzle
                # scrubbing the platform texture adding a new vibration source
                # before damping takes over.  An absolute-value test would fire
                # on that transient instead of on contact.
                #
                # Kept alongside the ratio rather than replacing it: the ratio
                # is what currently delivers 3.5um repeatability on x, and four
                # descents is not enough evidence to retire a working
                # safety-critical path.  If the derivative keeps proving better,
                # the ratio test (and its smoothing machinery) should go.
                # DRAWDOWN trigger - third parallel path.  Fall from a running
                # extreme (bounded lookback), which unlike the per-window step
                # ACCUMULATES across windows, so it does not weaken when the
                # event is spread over more samples.  Offline replay of 24
                # axis-traces: fires 1-2 windows earlier than the derivative on
                # x, and detects contact on z in four traces where the
                # per-window rule found nothing, with no fire more than 2
                # windows before the real halt.
                val = float(amps[a_idx])
                # Disarmed axes must not TRIGGER, but must still be MEASURED:
                # _dbg_maxdrop below is what CHARACTERIZE_NOISE reads back to
                # compute its air ceilings, so skipping the analysis outright
                # would silently report zero noise on every axis.
                det = None if self._disarmed[a_idx] else self._dd_det[a_idx]
                if self._disarmed[a_idx]:
                    pass
                elif det is None:
                    # Still measuring this descent's own air noise.  Cannot
                    # trigger yet - which is correct, since the first windows
                    # after the warmup gate are the ones most contaminated by
                    # the excitation ringing up.
                    air = self._dd_air[a_idx]
                    air.append(val)
                    if len(air) >= self._dd_warm:
                        base = sum(air) / len(air)
                        sd = (analog_contact.estimate_noise(air) / base
                              if base > 1e-9 else 0.)
                        thr = max(self._dd_sens, self._dd_nsigma * sd)
                        self._dd_thresh[a_idx] = thr
                        self._dd_det[a_idx] = analog_contact.ContactDetector(
                            analog_contact.DRAWDOWN, thr,
                            max(self._descend_speed, 1e-6),
                            1. / max(self._step_z / max(self._descend_speed,
                                                        1e-9), 1e-9),
                            relative=True, persist_mm=2. * self._step_z,
                            lookback_mm=self._dd_lookback)
                elif det.update(val, position=float(tc)):
                    self._trigger_time = float(tc)
                    self._trigger_axis = a_idx
                    self._trigger_kind = 'drawdown'
                    self._done = True
                    self._completion.complete(True)
                    return False
                prev = self._prev_amp[a_idx]
                self._prev_amp[a_idx] = float(amps[a_idx])
                thr = self._deriv_axis[a_idx]
                if prev is not None and prev > 1e-9:
                    step = (float(amps[a_idx]) - prev) / prev
                    if step < self._dbg_minstep[a_idx]:
                        self._dbg_minstep[a_idx] = step
                    if thr and step <= -thr:
                        self._deriv_run[a_idx] += 1
                        if self._deriv_run[a_idx] >= self._deriv_persist:
                            self._trigger_time = float(tc)
                            self._trigger_axis = a_idx
                            self._trigger_kind = 'derivative'
                            self._done = True
                            self._completion.complete(True)
                            return False
                    else:
                        self._deriv_run[a_idx] = 0
                if drop >= self._halt_axis[a_idx]:
                    if self._below_run[a_idx] == 0:
                        # Anchor the trigger to the reference time (~confirm_z
                        # above) = the contact onset, not the deeper confirming
                        # window.  float() keeps a numpy scalar out of status
                        # reporting (a numpy value there crashes Moonraker's JSON
                        # encoder -> MCU shutdown).
                        self._first_below_t[a_idx] = float(ref_t)
                    self._below_run[a_idx] += 1
                    if self._below_run[a_idx] >= self._grad_persist:
                        self._trigger_time = float(self._first_below_t[a_idx])
                        self._trigger_axis = a_idx
                        self._done = True
                        self._completion.complete(True)
                        return False
                else:
                    self._below_run[a_idx] = 0
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
    def __init__(self, toolhead, gen_segments, drip_time=None):
        self._toolhead = toolhead
        self._gen_segments = gen_segments
        self._drip_time = drip_time
    def drip_move(self, newpos, speed, drip_completion):
        segments = self._gen_segments(newpos, speed)
        self._toolhead.drip_move_sequence(segments, drip_completion,
                                          drip_time=self._drip_time)
    def __getattr__(self, name):
        return getattr(self._toolhead, name)


# Real-time-halting resonance contact descent + anchored analysis, factored out
# so it can be reused without a [resonance_probe] config section.  It vibrates
# the lateral axis at the resonance while descending via the drip path; a host
# task halts the move the instant contact damps the resonance (loose threshold),
# then the anchored fine analysis pins the precise contact Z.  Both
# [resonance_probe] (hostdriven mode) and the [resonance_probe_calibrate]
# utility build one of these, so everything a user runs halts in real time (no
# over-drive into the bed).  Construct with the excitation/detection parameters
# and call run() once per touch.
class HaltingContactProbe:
    def __init__(self, printer, chip, output_index, vibrate_dir,
                 excitation_freq, accel_per_hz, probe_amplitude, z_min,
                 warmup, sensitivity, halt_sensitivity, detect_cycles,
                 detect_step_z, detect_confirm_z, detect_offset_frac,
                 halt_sensitivity_axis=None):
        self.printer = printer
        self.chip = chip
        self.output_index = output_index
        self.vibrate_dir = vibrate_dir
        self.excitation_freq = excitation_freq
        self.accel_per_hz = accel_per_hz
        self.probe_amplitude = probe_amplitude
        self.z_min = z_min
        self.warmup = warmup
        self.sensitivity = sensitivity
        self.halt_sensitivity = halt_sensitivity
        # Optional per-axis live-halt floors [x,y,z]; None -> the scalar for all.
        self.halt_sensitivity_axis = halt_sensitivity_axis
        self.detect_cycles = detect_cycles
        self.detect_step_z = detect_step_z
        self.detect_confirm_z = detect_confirm_z
        self.detect_offset_frac = detect_offset_frac
        # Selector thresholds.  Default to the LIVE drawdown detector's values
        # so calibration scores candidates by the same bar the halt will use;
        # fall back to the module defaults when there is no probe object (this
        # helper is also constructed directly by the calibration commands).
        rp = printer.lookup_object('resonance_probe', None)
        self._sel_floor = getattr(rp, 'drawdown_sensitivity', 0.10) or 0.10
        self._sel_nsigma = getattr(rp, 'drawdown_nsigma', 8.) or 8.
        # Verify/refine sampling.  Each rep contributes an independent DOWN and
        # UP estimate, so raising this trades time for datapoints - both for a
        # better refined Z and for measuring the up/down bias.
        self._verify_reps = getattr(rp, 'verify_reps', 1) or 1
        self._verify_combine = getattr(rp, 'verify_combine', 1)
        # Per-direction detail from the last verify, for reporting.
        self._verify_detail = {}
        self._descend_speed = 1.
        self._z_steppers = None
        # Timing-corrected reversal cruise velocity, installed by run() for the
        # duration of a descent (None = use the legacy accel-limited peak_v).
        self._descent_v = None
        # Diagnostics from the last anchored analysis, for calibration: the
        # no-contact plateau amplitude, its relative measurement noise, and the
        # deepest drop reached during the vibrating descent (a lower bound on the
        # contact drop, since the halt stops the descent early by design).
        self.last_baseline = None
        self.last_rel_noise = None
        self.last_max_drop = None
        self.last_maxdrop = None      # per-axis [x,y,z] from the last descent

    # The Z steppers for HomingMove's halt-position bookkeeping.  LookupZSteppers
    # only fires at startup, so gather them on demand (the MCU is long since
    # identified by the time a probe runs).
    def _get_z_steppers(self):
        if self._z_steppers is None:
            kin = self.printer.lookup_object('toolhead').get_kinematics()
            self._z_steppers = [s for s in kin.get_steppers()
                                if s.is_active_axis('z')]
        return self._z_steppers

    # Cruise velocity that makes each +/-amp reversal (a point-to-point move of
    # distance D = 2*probe_amplitude) take exactly one half-period, 1/(2f), so
    # the drip descent oscillates at the COMMANDED frequency.  Left to the
    # planner, each reversal runs at the accel/velocity limit and finishes early
    # (t = 3/(2*pi*f) < 1/(2f)), so the motion outruns the command by ~pi/3 and
    # drives off the resonance (measured 56 -> 59.6 Hz on hardware).  Solving the
    # full-stop trapezoid time t(v) = v/a + D/v = 1/(2f) for v caps the cruise so
    # the reversal fills the half-period.  All terms are live settings
    # (accel_per_hz, excitation_freq, probe_amplitude), so the correction
    # self-adjusts if any change.  run() also clamps square_corner_velocity for
    # the descent so each reversal really stops (making this full-stop solution
    # exact - otherwise the planner blends a little velocity through the near-
    # 180deg turn and the motion stays slightly fast).  Returns None if there is
    # no real solution (excitation too weak to fill the half-period even at a
    # full stop) -> caller keeps the legacy peak_v.
    def _timing_corrected_cruise_v(self, aph=None):
        # aph=None uses the configured cap amplitude; pass a lower level (e.g. a
        # characterize_amplitude sweep step) to correct that level's reversals.
        f = self.excitation_freq
        if aph is None:
            a = self.accel_per_hz * f        # effective accel of the reversal
            D = 2. * self.probe_amplitude    # +amp -> -amp travel (cap amplitude)
        else:
            a = aph * f
            D = 2. * (aph / (4. * math.pi**2 * f))  # travel at this sweep level
        T = 1. / (2. * f)                    # target half-period
        disc = a * a * T * T - 4. * a * D
        if disc <= 0.:
            return None
        return max((a * T - math.sqrt(disc)) / 2., 1e-3)

    # Detuning-corrected reversal cruise, with the accel-limited swing peak as
    # the fallback when no full-stop solution exists.  Central so EVERY bounded
    # (position-parameterized) vibration is on the commanded frequency by
    # construction, not by each caller remembering to apply the correction.
    def _corrected_peakv(self, aph=None):
        vt = self._timing_corrected_cruise_v(aph)
        if vt:
            return vt
        a = self.accel_per_hz if aph is None else aph
        return max(a / (2. * math.pi), 1e-3)

    # Execute a bounded +/-amp vibration (position-parameterized drip) and return
    # its captured accelerometer stream.  This is the ONE place the shared
    # probing conditions live: input shaping off, MINIMUM_CRUISE_RATIO=0, the
    # SQUARE_CORNER_VELOCITY clamp that makes the reversal timing correction
    # exact, and a Z accel/velocity raise covering any superimposed Z ramp.  The
    # segments carry their own (corrected) per-reversal cruise; vel_ceiling /
    # accel_ceiling are the global worst-case caps (drip only ratchets DOWN from
    # them).  Returns (samples, seg_times, t0).  Restores every limit on exit;
    # does NOT move the toolhead afterward (the caller parks as it needs).
    def _run_bounded_vibration(self, gcmd, segs, f, ramp_speed,
                               vel_ceiling, accel_ceiling, drip_time=None):
        toolhead = self.printer.lookup_object('toolhead')
        reactor = self.printer.get_reactor()
        gcode = self.printer.lookup_object('gcode')
        scv = gcmd.get_float("DESCENT_SCV", 1.0, minval=0.01)
        info = toolhead.get_status(reactor.monotonic())
        old_v, old_a, old_cr = (info['max_velocity'], info['max_accel'],
                                info['minimum_cruise_ratio'])
        old_scv = info.get('square_corner_velocity', 5.)
        kin_zsaved = None
        kin = toolhead.get_kinematics()
        if hasattr(kin, 'max_z_accel') and hasattr(kin, 'max_z_velocity'):
            need_za = math.pi**2 * f * max(ramp_speed, 0.1)
            kin_zsaved = (kin, kin.max_z_velocity, kin.max_z_accel)
            kin.max_z_accel = max(kin.max_z_accel, 3. * need_za)
            kin.max_z_velocity = max(kin.max_z_velocity,
                                     max(ramp_speed, vel_ceiling) + 1.)
        ishaper = self.printer.lookup_object('input_shaper', None)
        if ishaper is not None:
            ishaper.disable_shaping()
        gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT VELOCITY=%.3f ACCEL=%.3f MINIMUM_CRUISE_RATIO=0"
            " SQUARE_CORNER_VELOCITY=%.3f"
            % (vel_ceiling, accel_ceiling, scv))
        toolhead.wait_moves()
        toolhead.dwell(0.050)
        aclient = self.chip.start_internal_client()
        t0 = toolhead.get_last_move_time()
        tail = list(toolhead.get_position()[3:])
        seg_times = None
        try:
            completion = reactor.completion()
            seg_times = toolhead.drip_move_sequence(
                [(p + tail, s, a) for p, s, a in segs], completion,
                drip_time=drip_time)
            toolhead.wait_moves()
        finally:
            aclient.finish_measurements()
            gcode.run_script_from_command(
                "SET_VELOCITY_LIMIT VELOCITY=%.3f ACCEL=%.3f"
                " MINIMUM_CRUISE_RATIO=%.3f SQUARE_CORNER_VELOCITY=%.3f"
                % (old_v, old_a, old_cr, old_scv))
            if kin_zsaved is not None:
                k, zv, za = kin_zsaved
                k.max_z_velocity, k.max_z_accel = zv, za
            if ishaper is not None:
                ishaper.enable_shaping()
        return aclient.get_samples(), seg_times, t0

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
        # Cruise velocity of each +/-amp reversal.  Default: the accel-limited
        # swing peak (aph/2pi).  run() may install a timing-corrected cruise
        # (self._descent_v) that stretches each half-cycle to exactly 1/(2f) so
        # the drip motion runs at the COMMANDED frequency instead of outrunning
        # it (see _timing_corrected_cruise_v).
        peak_v = (self._descent_v if getattr(self, '_descent_v', None)
                  else max(self.accel_per_hz / (2. * math.pi), 1e-3))
        total_t = max((z0 - z_target) / self._descend_speed, 1e-6)
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

    # Steady-state driven-axis DFT amplitude while vibrating IN PLACE at height
    # z.  Same bounded-oscillation execution as characterize_amplitude (shaping
    # off, raised limits, drip-fed to survive high-frequency segment counts) but
    # at a single fixed Z.  Used by the halt VERIFY: contact damps this, air does
    # not.
    def _static_amp(self, gcmd, x0, y0, z, duration):
        import numpy as np
        f = self.excitation_freq
        accel = self.accel_per_hz * f
        amp = self.probe_amplitude
        vdir = self.vibrate_dir.get_dir()
        peak_v = self._corrected_peakv()
        n = max(4, int(round(duration * 2. * f)))
        segs = [([x0 + vdir[0] * (amp if k % 2 == 0 else -amp),
                  y0 + vdir[1] * (amp if k % 2 == 0 else -amp), z], peak_v, accel)
                for k in range(n)]
        segs.append(([x0, y0, z], peak_v, accel))
        samples, _, _ = self._run_bounded_vibration(
            gcmd, segs, f, 0., peak_v + 1., accel + 1.)
        data = np.asarray(samples, dtype=np.float64)
        if len(data) < 4:
            return 0.
        t = data[:, 0] - data[0, 0]
        return _dft_amp(t, data[:, 1 + self.output_index], f)

    # Static up/down confirmation of a candidate halt at z_halt: measure the
    # driven-axis amplitude in guaranteed air (z_halt + gap) and with a slight
    # bounded press into the surface (z_halt - engage).  Real contact damps the
    # press; a false halt shows both ~equal.  Returns (air_amp, touch_amp).
    def _verify_contact(self, gcmd, x0, y0, z_halt, lift_speed, gap, engage):
        toolhead = self.printer.lookup_object('toolhead')
        dur = gcmd.get_float("VERIFY_DWELL", 0.4, above=0.1)
        toolhead.manual_move([x0, y0, z_halt + gap], lift_speed)
        toolhead.wait_moves()
        air_amp = self._static_amp(gcmd, x0, y0, z_halt + gap, dur)
        z_touch = max(self.z_min, z_halt - engage)
        toolhead.manual_move([x0, y0, z_touch], min(lift_speed, 2.))
        toolhead.wait_moves()
        touch_amp = self._static_amp(gcmd, x0, y0, z_touch, dur)
        return air_amp, touch_amp, None

    # MOVING up/down confirmation of a candidate halt at z_cand: continuous
    # excitation while Z ramps DOWN through the candidate and back UP (the
    # characterize_amplitude trajectory), WITH the descent detuning fix so the
    # motion stays on the resonance.  Beyond the static air-vs-contact check this
    # yields the amplitude-vs-Z profile of the ramp, so the contact Z is REFINED
    # to where the down-ramp amplitude crosses the air/contact midpoint (a
    # sharper datum than the live halt, which anchors a little high).  Returns
    # (air_amp, contact_amp, refined_z_or_None) on the driven axis.
    def _verify_contact_moving(self, gcmd, x0, y0, z_cand, lift_speed,
                               z_limit=None, up_override=None):
        import numpy as np
        toolhead = self.printer.lookup_object('toolhead')
        up = gcmd.get_float("VERIFY_UP", 0.15, above=0.)
        if up_override is not None:
            up = up_override
        down = gcmd.get_float("VERIFY_DOWN", 0.25, above=0.)
        reps = gcmd.get_int("VERIFY_REPS", self._verify_reps, minval=1,
                            maxval=10)
        ramp_speed = gcmd.get_float("VERIFY_RAMP_SPEED", 0.5, above=0.,
                                    maxval=5.)
        f = self.excitation_freq
        accel = self.accel_per_hz * f
        amp = self.probe_amplitude
        vdir = self.vibrate_dir.get_dir()
        peakv = self._corrected_peakv()  # detuning-corrected reversal cruise
        half_dt = 0.5 / f
        z_hi = z_cand + up
        # z_limit lets a caller forbid going below a floor TIGHTER than the
        # hard z_min - the salvage path runs this at the descent floor, and
        # without the clamp the ramp would press a further 'down' (0.25mm)
        # BELOW the floor the descent was bounded by.  Observed on hardware:
        # a salvage at floor -0.5 ramped toward -0.75.
        z_lo = max(self.z_min, z_cand - down)
        if z_limit is not None:
            z_lo = max(z_lo, z_limit)
        z_travel = max(z_hi - z_lo, 1e-3)
        dwell_t = gcmd.get_float("VERIFY_DWELL", 0.4, above=0.1)
        dwell_segs = max(4, int(round(dwell_t / half_dt)))
        ramp_t = max(z_travel / ramp_speed, 2. * half_dt)
        ramp_segs = max(6, int(round(ramp_t / half_dt)))
        warm_segs = max(4, int(round(self.warmup / half_dt)))
        segs = []
        tags = []
        rep_ids = []
        sign = [1.]
        cur_rep = [-1]
        def emit(z, tag):
            disp = amp if sign[0] > 0 else -amp
            segs.append(([x0 + vdir[0] * disp, y0 + vdir[1] * disp, z],
                         peakv, accel))
            tags.append(tag)
            # Which rep a ramp belongs to.  Each ramp must be estimated on its
            # own: pooling them and sorting by Z interleaves reps and computes
            # steps across a rep boundary, which is noise, not signal.
            rep_ids.append(cur_rep[0])
            sign[0] = -sign[0]
        # Ring up in air (never started while touching), then ramp through.
        toolhead.manual_move([x0, y0, z_hi], lift_speed)
        toolhead.wait_moves()
        for _ in range(warm_segs):
            emit(z_hi, 'warm')
        for r in range(reps):
            cur_rep[0] = r
            for s in range(ramp_segs):
                emit(z_hi - z_travel * (s + 1) / ramp_segs, 'down')
            for _ in range(dwell_segs):
                emit(z_lo, 'contact')
            for s in range(ramp_segs):
                emit(z_lo + z_travel * (s + 1) / ramp_segs, 'up')
            for _ in range(dwell_segs):
                emit(z_hi, 'air')
        cur_rep[0] = -1
        # NOTE: this trailing centering segment does not go through emit(), so
        # every parallel array must be appended to by hand.  Missing rep_ids
        # here desynced it by one and the window index ran off the end mid-probe
        # - an IndexError inside the probe shuts the PRINTER down, so these are
        # checked rather than trusted.
        segs.append(([x0, y0, z_hi], peakv, accel))
        tags.append('air')
        rep_ids.append(-1)
        if not (len(segs) == len(tags) == len(rep_ids)):
            raise gcmd.error("verify: segment bookkeeping desync"
                             " (segs=%d tags=%d reps=%d)"
                             % (len(segs), len(tags), len(rep_ids)))
        samples, seg_times, _ = self._run_bounded_vibration(
            gcmd, segs, f, ramp_speed, peakv + ramp_speed + 1., accel + 1.)
        toolhead.manual_move([x0, y0, z_hi], lift_speed)
        toolhead.wait_moves()
        data = np.asarray(samples, dtype=np.float64)
        if len(data) < 8 or not seg_times:
            return 0., 0., None
        times = data[:, 0]
        sps = (len(times) - 1) / max(times[-1] - times[0], 1e-9)
        win_n = max(8, int(self.detect_cycles / f * sps))
        seg_end = np.asarray(seg_times, dtype=np.float64)
        tag_arr = np.asarray(tags)
        zc = np.asarray([s[0][2] for s in segs], dtype=np.float64)
        col = data[:, 1 + self.output_index]
        (wamp,), wk = _window_amps_tagged(times, [col], f, win_n,
                                          max(1, win_n // 2), seg_end)
        wtag = tag_arr[wk]
        wz = zc[wk]
        wrep = np.asarray(rep_ids)[wk]
        air = wamp[wtag == 'air']
        contact = wamp[wtag == 'contact']
        if len(air) < 2 or len(contact) < 1:
            return 0., 0., None
        air_amp = float(np.median(air))
        contact_amp = float(np.median(contact))
        # Refine.  Every ramp is estimated separately - each rep gives one DOWN
        # and one UP datapoint - so more reps mean more independent estimates
        # instead of a longer pooled list, and so any systematic difference
        # between pressing in and releasing (ring-down, Z backlash, or the press
        # needed before the resonance damps at all) is measurable rather than
        # averaged away unseen.
        down_edges, up_edges = [], []
        step_snr = 0.
        for r in range(reps):
            for tag, bucket in (('down', down_edges), ('up', up_edges)):
                m = (wtag == tag) & (wrep == r)
                if int(m.sum()) < 3:
                    continue
                edge, snr = self._ramp_edge(wz[m], wamp[m], tag == 'down',
                                            air_amp, contact_amp)
                if edge is not None:
                    bucket.append(edge)
                if tag == 'down':
                    step_snr = max(step_snr, snr)
        # MEAN, not median.  Measured over 26 probes: the mean beats the median
        # at every level (down 2.63 vs 2.97um, up 2.88 vs 2.96, all 2.34 vs
        # 2.46), and it beats it EVEN THOUGH one down ramp in three is a ~12um
        # outlier that the median correctly rejects.  With 4 ramps a median is
        # the average of the middle two, so it throws away half the samples; the
        # sqrt(n) gain from averaging them all is worth more than the outlier
        # costs.  A median would only win if the outliers were rarer or bigger.
        def _agg(vals):
            return float(sum(vals) / len(vals)) if vals else None
        def _spread(vals):
            return (max(vals) - min(vals)) if len(vals) > 1 else 0.
        r_down, r_up = _agg(down_edges), _agg(up_edges)
        bias = (r_up - r_down) if (r_down is not None
                                   and r_up is not None) else None
        # Averaging BOTH directions is the default (VERIFY_COMBINE=0 restores
        # the down-only answer for A/B).  Two reasons, both measured:
        #  - repeatability: 2.34um vs 2.92um for down-only, pooled within-point
        #    over 26 probes, and the mean-of-both won on every run tried.
        #  - the up/down bias is POSITIONAL, not a machine constant: +1..+2um at
        #    one point, +15..+20um at another, and it has been seen negative.
        #    Reporting one direction alone inherits that positional error;
        #    averaging the two cancels it by construction, which matters more
        #    than the sd difference.
        combine = gcmd.get_int("VERIFY_COMBINE", self._verify_combine,
                               minval=0, maxval=1)
        if combine and bias is not None:
            refined = 0.5 * (r_down + r_up)
        else:
            refined = r_down if r_down is not None else r_up
        self._verify_detail = {
            'down': r_down, 'up': r_up, 'bias': bias, 'reps': reps,
            'down_n': len(down_edges), 'up_n': len(up_edges),
            'down_spread': _spread(down_edges), 'up_spread': _spread(up_edges),
            'combined': bool(combine and bias is not None),
        }
        # Save the ramp windows.  Which estimator to report (down / up / their
        # mean) is an open question, and it is decidable OFFLINE from one set of
        # probes instead of one hardware run per candidate - but only if the
        # per-ramp windows are kept.  The descent autosave does not cover this
        # path: it saves the drip descent, which ends at the halt.
        self._autosave_verify(wz, wamp, wtag, wrep, air_amp, contact_amp,
                              z_cand, reps, ramp_speed)
        return air_amp, contact_amp, refined, step_snr

    def _autosave_verify(self, wz, wamp, wtag, wrep, air_amp, contact_amp,
                         z_cand, reps, ramp_speed):
        rp = self.printer.lookup_object('resonance_probe', None)
        tdir = getattr(rp, 'trace_dir', None)
        if not tdir or not len(wz):
            return
        try:
            if not os.path.isdir(tdir):
                os.makedirs(tdir)
            # Named apart from descent traces: the two have different columns
            # and different meanings, and the replay tooling keys on filename.
            n = len([f for f in os.listdir(tdir) if f.startswith('verify')])
            path = os.path.join(tdir, "verify%05d.csv" % (n,))
            with open(path, 'w') as fh:
                fh.write("# freq=%.2f accel_per_hz=%.1f ramp_speed=%.4f\n"
                         % (self.excitation_freq, self.accel_per_hz,
                            ramp_speed))
                fh.write("# z_cand=%.5f reps=%d\n" % (z_cand, reps))
                fh.write("# air_amp=%.3f contact_amp=%.3f\n"
                         % (air_amp, contact_amp))
                note = getattr(rp, 'trace_note', None)
                if note:
                    # The replay loaders parse header lines by splitting on
                    # whitespace and then on '=', so a note containing spaces
                    # would silently truncate to its first word.  Emit a
                    # whitespace-free value instead of trusting the operator to
                    # remember - a mangled tag is unrecoverable without
                    # re-probing, and probing wears the plate.
                    fh.write("# note=%s\n" % ("-".join(note.split()),))
                fh.write("z,amp,tag,rep\n")
                for z, a, t, r in zip(wz, wamp, wtag, wrep):
                    fh.write("%.5f,%.3f,%s,%d\n" % (z, a, t, r))
            # The CONFIRMED/REJECTED verdict is not known here - it is decided
            # by the caller from the drop and the threshold - so the outcome is
            # appended afterwards rather than duplicating that logic.
            self._last_verify_path = path
        except (IOError, OSError) as e:
            # Diagnostics must never break probing.
            logging.warning("resonance_probe: verify trace autosave failed: %s",
                            e)

    # Tag the verify trace just written with how it was judged.  A REJECTED
    # trace is not junk: it is a recording of the signal around a false halt,
    # which is the only direct evidence of why the halt fired.  Keeping them
    # labelled means they can be excluded from estimator scoring (where they
    # would be noise) while still being available for studying false halts.
    def _tag_verify_outcome(self, outcome, drop, thresh, step_snr, via=''):
        path = getattr(self, '_last_verify_path', None)
        if not path:
            return
        try:
            with open(path, 'a') as fh:
                fh.write("# outcome=%s drop=%.4f thresh=%.4f step_snr=%.2f%s\n"
                         % (outcome, drop, thresh, step_snr,
                            (" via=%s" % via) if via else ""))
        except (IOError, OSError) as e:
            logging.warning("resonance_probe: verify outcome tag failed: %s", e)
        self._last_verify_path = None

    # One monotonic ramp -> the Z at which the amplitude crosses between air and
    # contact, plus the step SNR.  'descending' picks which way the ramp runs
    # and therefore which sign the contact edge has: pressing in makes the
    # amplitude FALL, releasing makes it RISE.  Returns (edge_z, snr).
    def _ramp_edge(self, zs, amps, descending, air_amp, contact_amp):
        import numpy as np
        zs = np.asarray(zs, dtype=np.float64)
        amps = np.asarray(amps, dtype=np.float64)
        order = np.argsort(-zs if descending else zs)
        dz = zs[order]
        da = amps[order]
        if len(da) < 3:
            return None, 0.
        # DERIVATIVE check, same principle as the live halt test: contact is a
        # STEP, and a step is far better separated from air than a ratio of
        # medians is.  The ramp here is slow enough to resolve it - at
        # VERIFY_RAMP_SPEED 0.5mm/s a window spans ~0.027mm against a ~0.030mm
        # event, where the calibration sweep at 1.0mm/s cannot.
        #
        # This matters because verify is the last line of defence: it is what
        # rejected a false halt 660um above the bed.  A ratio of medians over a
        # short ramp can read a real touch as marginal (that is exactly why
        # VERIFY_DROP had to drop from 15% to 5% on this machine); a step test
        # does not have that dilution problem.
        edge, snr = None, 0.
        steps = (da[1:] - da[:-1]) / np.maximum(da[:-1], 1e-9)
        if len(steps) >= 3:
            # Noise from the first half of the ramp: going down that end is
            # air, going up it is contact - either way it is the flat part,
            # before the edge.
            air_steps = steps[:max(2, len(steps) // 2)]
            sd = float(np.std(air_steps))
            # The contact edge is the most NEGATIVE step on the way down and
            # the most POSITIVE on the way up.
            worst = float(np.min(steps)) if descending else float(np.max(steps))
            found = (worst < 0.) if descending else (worst > 0.)
            if sd > 1e-6 and found:
                snr = abs(worst) / sd
            # The steepest step IS the contact edge, and it localises the
            # transition better than the air/contact midpoint crossing when the
            # two levels are close.  Take the midpoint of the window pair that
            # straddles it.
            if found:
                j = int(np.argmin(steps) if descending else np.argmax(steps))
                edge = float(0.5 * (dz[j] + dz[j + 1]))
        # Prefer the air/contact midpoint crossing when it exists: it is the
        # long-standing, well-tested estimator.  The step-based edge above is
        # the fallback for the case it cannot handle (levels too close for a
        # midpoint to be crossed cleanly).
        if air_amp > contact_amp:
            mid = 0.5 * (air_amp + contact_amp)
            for j in range(1, len(da)):
                prev, cur = da[j - 1], da[j]
                if descending:
                    # falling through mid: air -> contact
                    if not (prev >= mid >= cur):
                        continue
                    frac = (prev - mid) / max(prev - cur, 1e-9)
                else:
                    # rising through mid: contact -> air
                    if not (prev <= mid <= cur):
                        continue
                    frac = (mid - prev) / max(cur - prev, 1e-9)
                edge = float(dz[j - 1] + (dz[j] - dz[j - 1]) * frac)
                break
        return edge, snr

    # A descent that reached the floor without halting is NOT proof that the
    # bed was never touched.  The nozzle may be pressed against it right now,
    # with the detector simply having missed the transition (a drop diluted by
    # ring-down, a floor set slightly high, or contact arriving before enough
    # above-contact history existed to measure a gradient).
    #
    # Pressed contact is a STATE, not an event, so it is still measurable after
    # the fact: lift back up and watch the amplitude RISE as the nozzle
    # releases.  That is exactly the up/down ramp the halt verifier already
    # performs, so run that at the floor and reuse its interpolated crossing as
    # the contact Z.
    #
    # If the bed genuinely is not reachable, both ends of the ramp read air,
    # the drop is ~0, and this reports failure - which is a real hardware or
    # Z-offset problem for the user to fix, not something to paper over.
    def _salvage_on_rise(self, gcmd, x0, y0, z_floor, lift_speed, thresh):
        if not gcmd.get_int("SALVAGE", 1):
            return None
        # NEVER PRESS FURTHER.  Reaching here means the descent already went
        # too deep, so the recovery may only go UP: z_limit pins the bottom of
        # the ramp at the floor we are already sitting on.
        #
        # And lift FAR enough that the top of the ramp is genuinely free air.
        # The default 0.15mm reference is measured from the candidate, which
        # here is the over-travelled floor - if the real surface is further
        # above that than the reference height, both ends of the ramp are still
        # pressed and the "drop" is only a press-DEPTH gradient, which says
        # nothing about contact.  That is exactly what happened on hardware: a
        # salvage at floor -0.53 reported -22% while the true surface was at
        # ~-0.05, i.e. every sample was pressed.
        up = gcmd.get_float("SALVAGE_UP", 0.8, above=0.2)
        air_a, touch_a, refined, step_snr = self._verify_contact_moving(
            gcmd, x0, y0, z_floor, lift_speed, z_limit=z_floor,
            up_override=up)
        drop = (air_a - touch_a) / air_a if air_a > 1e-9 else 0.
        # Salvage deliberately keeps the RATIO as its sole criterion.  It runs
        # when the nozzle is already pressed, so the release is a slow rise
        # over the whole ramp rather than a step at one height - the very shape
        # the derivative is worst at, and a false "recovered contact" here
        # invents a bed position out of nothing.
        if drop < thresh or refined is None:
            gcmd.respond_info(
                "salvage: no contact at the floor either (air=%.0f pressed="
                "%.0f, %.0f%% < %.0f%%) - the bed is genuinely out of reach"
                " from this start height; check the Z endstop offset,"
                " probe_start_z or probe_distance"
                % (air_a, touch_a, drop * 100., thresh * 100.))
            return None
        gcmd.respond_info(
            "salvage: the descent missed its halt but the nozzle IS on the bed"
            " (air=%.0f pressed=%.0f, -%.0f%%) - recovered contact z=%.4f from"
            " the release ramp.  It over-pressed by %.3fmm getting there;"
            " lower halt_sensitivity_%s if this repeats."
            % (air_a, touch_a, drop * 100., refined,
               max(0., refined - z_floor), 'xyz'[self.output_index]))
        return refined

    # Public entry: a VERIFIED contact touch.  Each live halt is only a
    # candidate - confirm it with a static up/down amplitude test, and on a
    # rejected false halt restart the descent with the ceiling lowered below the
    # false height (so the detector re-arms only past it) and retry.  This makes
    # the halt robust without perfectly tuned per-axis floors.  VERIFY=0 falls
    # back to a single unverified descent.
    def run(self, gcmd, ceiling, probe_distance, descend_speed,
            lift_speed, start_speed, x0=None, y0=None, vib_span=None,
            drip_time=None):
        toolhead = self.printer.lookup_object('toolhead')
        cur = toolhead.get_position()
        if x0 is None:
            x0 = cur[0]
        if y0 is None:
            y0 = cur[1]
        z_floor = max(self.z_min, ceiling - probe_distance)
        if not gcmd.get_int("VERIFY", 1):
            return self._descend_once(gcmd, ceiling, probe_distance,
                                      descend_speed, lift_speed, start_speed,
                                      x0, y0, vib_span, drip_time)
        attempts = gcmd.get_int("VERIFY_ATTEMPTS", 6, minval=1, maxval=12)
        gap = gcmd.get_float("VERIFY_GAP", 0.15, above=0.)
        engage = gcmd.get_float("VERIFY_ENGAGE", 0.10, minval=0.)
        thresh = gcmd.get_float("VERIFY_DROP", 0.15, above=0., below=1.)
        cur_ceiling = ceiling
        for _ in range(attempts):
            contact_z, halted = self._descend_once(
                gcmd, cur_ceiling, cur_ceiling - z_floor, descend_speed,
                lift_speed, start_speed, x0, y0, vib_span, drip_time)
            if not halted or contact_z is None:
                # Missed the halt - try to recover it on the way back up before
                # giving up (see _salvage_on_rise).
                z_sal = self._salvage_on_rise(gcmd, x0, y0, z_floor,
                                              lift_speed, thresh)
                if z_sal is not None:
                    return z_sal, True
                return contact_z, halted
            if gcmd.get_int("VERIFY_MOVING", 1):
                air_a, touch_a, refined, step_snr = \
                    self._verify_contact_moving(gcmd, x0, y0, contact_z,
                                                lift_speed)
            else:
                air_a, touch_a, refined = self._verify_contact(
                    gcmd, x0, y0, contact_z, lift_speed, gap, engage)
                step_snr = 0.   # stationary verify has no ramp to differentiate
            drop = (air_a - touch_a) / air_a if air_a > 1e-9 else 0.
            # Confirm on EITHER criterion.  The ratio is the established test;
            # the step test catches the case that forced VERIFY_DROP down to 5%
            # on this machine, where a real touch reads as a marginal ratio
            # because the ramp is short and the medians blend air with contact.
            # Both are computed from the same sweep, so this costs nothing.
            step_min = gcmd.get_float("VERIFY_STEP_SNR", 8., minval=0.)
            by_step = step_min > 0. and step_snr >= step_min
            if drop >= thresh or by_step:
                self._tag_verify_outcome(
                    'confirmed', drop, thresh, step_snr,
                    "ratio" if drop >= thresh else "step")
                final_z = refined if refined is not None else contact_z
                _dbg(gcmd,
                    "verify: CONFIRMED contact z=%.4f%s (air=%.0f contact=%.0f,"
                    " -%.0f%% vs %.0f%%; step SNR %.1f vs %.1f) via %s"
                    % (final_z,
                       (" (refined from %.4f)" % contact_z)
                       if refined is not None else "",
                       air_a, touch_a, drop * 100., thresh * 100.,
                       step_snr, step_min,
                       "ratio" if drop >= thresh else "step"))
                d = self._verify_detail or {}
                if d.get('bias') is not None:
                    # The up/down difference is the interesting number: a
                    # direction-symmetric error (backlash, ring-down) shows up
                    # here and nowhere else in the probe's output.
                    _dbg(gcmd,
                         "verify: down=%.4f (n=%d, spread %.1fum) up=%.4f"
                         " (n=%d, spread %.1fum) bias=%+.1fum%s"
                         % (d['down'], d['down_n'], d['down_spread'] * 1000.,
                            d['up'], d['up_n'], d['up_spread'] * 1000.,
                            d['bias'] * 1000.,
                            " [combined]" if d.get('combined') else ""))
                toolhead.manual_move([x0, y0, final_z], lift_speed)
                toolhead.wait_moves()
                return final_z, True
            self._tag_verify_outcome('rejected', drop, thresh, step_snr)
            gcmd.respond_info(
                "verify: REJECTED false halt z=%.4f (air=%.0f touch=%.0f,"
                " -%.0f%% < %.0f%%; step SNR %.1f < %.1f); re-arming below it"
                % (contact_z, air_a, touch_a, drop * 100., thresh * 100.,
                   step_snr, step_min))
            cur_ceiling = contact_z - gap
            if cur_ceiling <= z_floor + 0.02:
                break
        gcmd.respond_info("verify: no confirmed contact above the floor")
        toolhead.manual_move([x0, y0, ceiling], lift_speed)
        toolhead.wait_moves()
        return None, False

    # Refine near the halt anchor for the drip descent, whose oscillation ramps
    # up slowly (the lookahead eases into the reversals).  The baseline is taken
    # from the amplitude PLATEAU (the first run of windows, after 'armed_time',
    # whose amplitude has stopped rising) rather than the first third, so the
    # ramp cannot depress it.  Returns the Z where the amplitude first falls
    # below the plateau by 'sensitivity'.
    #
    # SPEED CEILING (future work): the un-halted full-stream search detects the
    # first threshold crossing of the windowed amplitude.  The window must span
    # enough excitation cycles to measure amplitude (detect_cycles), so its Z
    # span = detect_cycles/freq * descend_speed grows with speed; above ~1mm/s at
    # a ~47Hz resonance it exceeds the ~0.1mm contact transition and detection
    # smears/fails.  A promising avenue to lift the ceiling is to track the
    # amplitude GRADIENT (the steepest drop, which sits at the true contact
    # independent of window width) with a short, noise-averaged window, instead
    # of a fixed threshold - but it needs more post-contact travel and is a real
    # project.  The anchored (halted) path below is not subject to this ceiling.

    # Shared GRADIENT contact finder for the post-descent fine analysis (both
    # the un-anchored full-stream search and the anchored near-halt refine).
    # Scans the given window 'order' (ascending index = descending Z) and reports
    # the Z where the amplitude begins a STEEP local drop: a smoothed CURRENT
    # level vs a smoothed level grad_n windows (~half detect_confirm_z) ABOVE it,
    # a fall of >= 'sensitivity' sustained confirm_n windows.  Immune to the
    # height-dependent RISING baseline that defeats a flat-plateau threshold (the
    # rise is a gentle opposite slope).  'hi', if given, rejects a drop whose
    # onset sits above it (the anchored refine uses it to keep the drop near the
    # halt).  Returns (contact_z, ref_amp) or (None, None).
    def _gradient_contact(self, amps, zwin, order, z_floor, win_z,
                          sensitivity, hi=None):
        import numpy as np
        confirm_n = max(2, int(round(self.detect_confirm_z
                                     / self.detect_step_z)))
        grad_n = max(3, confirm_n)   # slope span ~= detect_confirm_z
        persist = 2                  # windows the drop must persist
        below = 0
        onset = None
        for j in range(grad_n + 3, len(order)):
            k = order[j]
            if zwin[k] < z_floor:
                break
            cur = np.median([amps[order[j - d]] for d in range(3)])
            ref = np.median([amps[order[j - grad_n - d]] for d in range(3)])
            drop = (ref - cur) / ref if ref > 1e-9 else 0.
            if drop >= sensitivity:
                if below == 0:
                    onset = order[max(0, j - grad_n)]
                below += 1
                if below >= persist:
                    if hi is None or zwin[onset] <= hi:
                        cz = zwin[onset] - self.detect_offset_frac * win_z
                        return float(cz), float(ref)
                    below = 0
                    onset = None
            else:
                below = 0
                onset = None
        return None, None

    def _analyze_drip(self, gcmd, samples, anchor_t, anchor_z, z_floor,
                      armed_time, anchored=False, trigger_axis=None):
        import numpy as np
        data = np.asarray(samples, dtype=np.float64)
        if len(data) < 8:
            return None
        times = data[:, 0]
        # Map sample time -> Z anchored at (anchor_t, anchor_z).  Anchoring to
        # the physical halt position (from stepper history) instead of the
        # descent-start time removes the drip startup-delay offset, so the
        # reported contact Z is independent of the probe's start height.
        zpos = anchor_z + self._descend_speed * (anchor_t - times)
        # Refine on whichever axis actually triggered the live halt (may be a
        # cross-coupled axis - see _HostResonanceEndstop); fall back to the
        # configured output_index for the un-halted (no trigger axis known)
        # full-stream search.
        axis_idx = trigger_axis if trigger_axis is not None else self.output_index
        col = data[:, 1 + axis_idx]
        f = self.excitation_freq
        sps = _sample_rate(times)
        # Window: a fixed *time* (enough excitation cycles for an accurate
        # amplitude).  Step: a fixed Z *distance*, so the amplitude-vs-Z curve is
        # sampled at the same resolution regardless of descent speed (windows
        # just overlap more at higher speed).
        win_n = max(8, int(self.detect_cycles / f * sps))
        # This analysis runs AFTER the halt, so it has no real-time budget and
        # no reason to inherit the live detector's coarse hop.  The live hop is
        # floored by MIN_STEP_DT to bound reactor work (host lag starves the MCU
        # step buffer), but here the samples are already captured and the only
        # cost is host CPU on a stationary machine.  At 0.2mm/s the inherited
        # hop is ~160 samples against a 173-sample window - 7% overlap, i.e.
        # effectively discrete, so contact could only ever be localised to one
        # window (~0.010mm) plus interpolation.  Oversampling slides the same
        # window in finer steps and localises the edge to a few SAMPLES.
        oversample = gcmd.get_int("OVERSAMPLE", 1, minval=1, maxval=32)
        step_n = max(1, int(round(self.detect_step_z * sps
                                  / max(self._descend_speed, 1e-9)))
                     // oversample)
        # Z span of one window; the detected edge sits ~offset_frac of this above
        # the true contact (window-leading-edge effect), more so at higher speed.
        win_z = (win_n / sps) * self._descend_speed
        amps, zwin, twin = _window_amps(times, col, f, win_n, step_n, zpos)
        # Consider only windows after the warmup gate (skips the startup dwell).
        armed_k = [k for k in range(len(amps)) if twin[k] >= armed_time]
        if len(armed_k) < 4:
            gcmd.respond_info("drip: too few windows after warmup")
            return None
        # When the descent was halted on contact we have a strong prior (the
        # halt position) and refine around it instead of re-searching the whole
        # stream - see _contact_near_anchor.  The full-stream search below is
        # only for the un-halted case, which has no such prior.
        if anchored:
            return self._contact_near_anchor(gcmd, amps, zwin, twin, armed_k,
                                             anchor_z, z_floor, win_z)
        # GRADIENT search: locate the steep local amplitude drop (contact),
        # immune to the height-dependent RISING baseline that a flat-plateau
        # threshold cannot handle (see _gradient_contact).
        contact, _ref = self._gradient_contact(
            amps, zwin, armed_k, z_floor, win_z, self.sensitivity)
        gcmd.respond_info("drip(gradient): drop>=%.0f%% over ~%.3fmm,"
                          " win=%dcyc/%.2fmm step=%dsmp, contact z=%s"
                          % (self.sensitivity * 100.,
                             0.5 * self.detect_confirm_z,
                             int(self.detect_cycles), win_z, step_n,
                             ("%.4f" % contact) if contact is not None
                             else "none"))
        return None if contact is None else float(contact)

    # Refine the contact Z using the halt as a strong prior.  HomingMove halted
    # the descent on the loose (halt_sensitivity) threshold and handed us
    # anchor_z (the halt position) plus the trigger time, so contact is known to
    # sit just above anchor_z.  Rather than re-search the whole stream - which
    # needs a fresh plateau "arm" and a post-contact persistence the halt leaves
    # no room for, and so intermittently misses contact entirely - establish the
    # baseline from the plateau ABOVE the contact band and take the first fine
    # (sensitivity) crossing within a short band just above the halt.  This both
    # finds contact on touches the full-stream search misses and removes the
    # deep bias of falling back to the raw halt position (which sits at the
    # looser, later 15% drop rather than the true 6% contact).
    def _contact_near_anchor(self, gcmd, amps, zwin, twin, armed_k,
                             anchor_z, z_floor, win_z):
        import numpy as np
        # Search band: from the halt up to a short distance above it.  The fine
        # crossing precedes the loose halt by only a small descent (the halt is
        # anchored to where the drop began), so keep the band tight - a wide
        # band re-admits the very transients near the plateau that the halt
        # prior is meant to exclude.
        band = max(0.15, 2. * win_z)
        hi = anchor_z + band
        # Baseline from the undisturbed plateau strictly above the search band.
        pre = [amps[k] for k in armed_k if zwin[k] > hi]
        if len(pre) >= 3:
            amax = max(pre)
            plat = [a for a in pre if a >= 0.8 * amax]
            baseline = float(np.median(plat))
        elif pre:
            plat = list(pre)
            baseline = float(np.median(pre))
        else:
            # Halt sits near the top with no plateau above it - nothing to
            # measure a drop against; trust the halt position itself.
            self.last_baseline = self.last_rel_noise = self.last_max_drop = None
            gcmd.respond_info("drip(anchored): no plateau above halt; using"
                              " halt z=%.4f" % (anchor_z,))
            return float(anchor_z)
        # Diagnostics for calibration: plateau noise, and the deepest drop seen
        # among windows at/above the halt (active vibration only - windows below
        # the halt are post-stop, with the excitation ended, so excluding them
        # keeps max_drop from collapsing to ~1 on the decayed tail).
        self.last_baseline = baseline
        self.last_rel_noise = (float(np.std(plat)) / baseline
                               if len(plat) >= 2 and baseline > 0 else 0.)
        descent_amps = [amps[k] for k in armed_k if zwin[k] >= anchor_z]
        self.last_max_drop = (max(0., 1. - min(descent_amps) / baseline)
                              if descent_amps and baseline > 0
                              else self.halt_sensitivity)
        # GRADIENT search within/below the band (a steep local drop), instead of
        # a flat-threshold crossing - see _gradient_contact.  'hi' rejects a drop
        # whose onset is above the band top, keeping it near the halt.
        contact, _ref = self._gradient_contact(
            amps, zwin, armed_k, z_floor, win_z, self.sensitivity, hi=hi)
        if contact is None:
            # No steep drop even within the band: trust the halt position.
            contact = float(anchor_z)
            gcmd.respond_info("drip(anchored): baseline %.1f, no gradient"
                              " contact in band; using halt z=%.4f"
                              % (baseline, anchor_z))
        else:
            gcmd.respond_info("drip(anchored): baseline %.1f, drop>=%.0f%% in"
                              " %.2fmm band above halt, contact z=%.4f"
                              % (baseline, self.sensitivity * 100., band,
                                 contact))
        return float(contact)

    # One real-time-halting contact touch.  'ceiling' is the height at which
    # detection arms (the full 'probe_distance' below it is the usable detection
    # range, floored at z_min); a warmup runway is added ABOVE it so the
    # oscillation is at full amplitude by the time detection arms.  Returns
    # (contact_z, halted): contact_z is None only if nothing was detected before
    # the floor.  Leaves the toolhead at contact_z when found, else at the
    # runway start.  Mirrors a real probe's stop-at-trigger so repeated touches
    # don't walk upward.
    def _descend_once(self, gcmd, ceiling, probe_distance, descend_speed,
                      lift_speed, start_speed, x0=None, y0=None, vib_span=None,
                      drip_time=None):
        toolhead = self.printer.lookup_object('toolhead')
        reactor = self.printer.get_reactor()
        gcode = self.printer.lookup_object('gcode')
        self._descend_speed = descend_speed
        cur = toolhead.get_position()
        if x0 is None:
            x0 = cur[0]
        if y0 is None:
            y0 = cur[1]
        z_floor = max(self.z_min, ceiling - probe_distance)
        if ceiling <= z_floor:
            raise gcmd.error("Resonance probe ceiling %.3f is at or below the"
                             " descent floor %.3f (z_min=%.3f, distance=%.3f)"
                             % (ceiling, z_floor, self.z_min, probe_distance))
        runway = self._descend_speed * self.warmup
        # TWO-STAGE DESCENT (step-feed overrun guard).  The vibrating drip is fed
        # to the MCU just-in-time with a shallow buffer so the host can halt the
        # instant contact damps the resonance - which makes a LONG sustained
        # vibrating descent fragile (it overruns the step pipeline: MCU "Timer too
        # close").  So only vibrate the final 'vib_span' mm just above the floor,
        # where contact actually is, and cover the clearance above that with the
        # fast, deeply-buffered, NON-vibrating approach move that already runs
        # here.  vib_span MUST exceed the bed's height variation so the fast
        # approach always stops safely above the surface; vib_span=None keeps the
        # legacy full-height vibrating descent.  Once a contact_z is known for the
        # point, 'ceiling' is already tightened below this, so z_start follows it.
        if vib_span is not None:
            z_vib_top = min(ceiling, z_floor + vib_span)
        else:
            z_vib_top = ceiling
        z_start = z_vib_top + runway
        toolhead.manual_move([x0, y0, z_start], start_speed)
        toolhead.wait_moves()
        accel = self.accel_per_hz * self.excitation_freq
        peak_v = max(self.accel_per_hz / (2. * math.pi), 1e-3)
        # DESCENT TIMING CORRECTION (option B): cap the reversal cruise so each
        # half-cycle fills a full 1/(2f) period (motion at the commanded freq,
        # not ~pi/3 faster), and clamp square_corner_velocity so the reversals
        # actually stop.  TIMING_FIX=0 restores the legacy behavior for A/B
        # testing; DESCENT_SCV tunes the corner-velocity clamp.
        timing_fix = gcmd.get_int("TIMING_FIX", 1)
        descent_scv = gcmd.get_float("DESCENT_SCV", 1.0, minval=0.01)
        vt = self._timing_corrected_cruise_v() if timing_fix else None
        self._descent_v = vt
        cruise_v = vt if vt else peak_v
        if vt:
            _dbg(gcmd,
                "Descent timing fix: cruise %.2f mm/s (was %.2f), scv=%.2f,"
                " targeting %.1f Hz motion"
                % (vt, peak_v, descent_scv, self.excitation_freq))
        # Match the excitation conditions (shaping off, raised limits) so the
        # vibration reaches full amplitude.
        info = toolhead.get_status(reactor.monotonic())
        old_v, old_a = info['max_velocity'], info['max_accel']
        old_cr = info['minimum_cruise_ratio']
        old_scv = info.get('square_corner_velocity', 5.)
        gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT VELOCITY=%.3f ACCEL=%.0f MINIMUM_CRUISE_RATIO=0"
            " SQUARE_CORNER_VELOCITY=%.3f"
            % (cruise_v + self._descend_speed + 1., accel + 1., descent_scv))
        # ROOT CAUSE of descent detuning: CoreXY/cartesian check_move caps any
        # X/Z move's accel to max_z_accel * (move_d/z_dist).  As the descent
        # speeds up, the Z step per half-cycle grows, that ratio shrinks, the cap
        # falls below the swing's needed accel, and the oscillation slows (its
        # frequency drops off the resonance).  The Z only travels ~microns per
        # half-cycle, so it genuinely needs just ~pi^2*f*descend_speed of accel;
        # raise max_z_accel to cover that (x3 margin) for the descent only.
        kin_zsaved = None
        kin = toolhead.get_kinematics()
        if hasattr(kin, 'max_z_accel') and hasattr(kin, 'max_z_velocity'):
            need_za = math.pi**2 * self.excitation_freq * self._descend_speed
            kin_zsaved = (kin, kin.max_z_velocity, kin.max_z_accel)
            kin.max_z_accel = max(kin.max_z_accel, 3. * need_za)
            kin.max_z_velocity = max(kin.max_z_velocity, peak_v + 1.)
        ishaper = self.printer.lookup_object('input_shaper', None)
        if ishaper is not None:
            ishaper.disable_shaping()
        toolhead.wait_moves()
        t0 = toolhead.get_last_move_time()
        endstop = _HostResonanceEndstop(self, self._get_z_steppers(), t0)
        gen = lambda newpos, speed: self._gen_descend_segments(newpos[2])
        vth = _VibratingToolhead(toolhead, gen, drip_time=drip_time)
        hmove = HomingMove(self.printer, [(endstop, "resonance_probe")],
                           toolhead=vth)
        movepos = [x0, y0, z_floor] + list(toolhead.get_position()[3:])
        try:
            epos = hmove.homing_move(movepos, self._descend_speed,
                                     probe_pos=True, check_triggered=False)
        finally:
            gcode.run_script_from_command(
                "SET_VELOCITY_LIMIT VELOCITY=%.3f ACCEL=%.3f"
                " MINIMUM_CRUISE_RATIO=%.3f SQUARE_CORNER_VELOCITY=%.3f"
                % (old_v, old_a, old_cr, old_scv))
            self._descent_v = None
            if kin_zsaved is not None:
                k, zv, za = kin_zsaved
                k.max_z_velocity, k.max_z_accel = zv, za
            if ishaper is not None:
                ishaper.enable_shaping()
        halted = endstop.get_trigger_time() > 0.
        # Expose the per-axis max live gradient this descent saw, so a no-halt
        # AIR descent (floors=1.0) can be used to characterize each axis's
        # descent-noise ceiling (see RESONANCE_PROBE_CHARACTERIZE_NOISE).
        self.last_maxdrop = [float(endstop._dbg_maxdrop[a])
                             for a in range(endstop.AXIS_COUNT)]
        # Keep the last descent's trace as (mm-below-arm, ax, ay, az).  Depth is
        # reconstructed the same way the "max drop at mm-below-arm" line does
        # it: the descent is a constant-speed drip from _armed_time, so
        # depth = (t - armed) * descend_speed.  Only the newest descent is kept
        # (PROBE_ACCURACY runs several); dump it before running another.
        # Publish on the REGISTERED printer object, not on this per-descent
        # helper - RESONANCE_PROBE_DUMP_TRACE can only reach the former.
        self.last_trace = [
            ((t - endstop._armed_time) * endstop._descend_speed, ax, ay, az)
            for (t, ax, ay, az) in endstop._trace]
        rp = self.printer.lookup_object('resonance_probe', None)
        if rp is not None:
            rp.last_trace = self.last_trace
            self._autosave_trace(rp, endstop)

        _dbg(gcmd,
            "live-halt diag: max gradient drop x=%.0f%% y=%.0f%% z=%.0f%% over"
            " %d live windows (floor x=%.0f%% y=%.0f%% z=%.0f%%)"
            % (endstop._dbg_maxdrop[0] * 100., endstop._dbg_maxdrop[1] * 100.,
               endstop._dbg_maxdrop[2] * 100., endstop._dbg_nwin,
               endstop._halt_axis[0] * 100., endstop._halt_axis[1] * 100.,
               endstop._halt_axis[2] * 100.))
        dfloor = ["%.0f%%" % (t * 100.) if t else "off"
                  for t in endstop._deriv_axis]
        _dbg(gcmd,
            "live-halt diag: biggest 1-window step x=%.0f%% y=%.0f%% z=%.0f%%"
            " (deriv floor x=%s y=%s z=%s) - halt came from the %s test"
            % (endstop._dbg_minstep[0] * 100., endstop._dbg_minstep[1] * 100.,
               endstop._dbg_minstep[2] * 100.,
               dfloor[0], dfloor[1], dfloor[2], endstop._trigger_kind))
        ddt = ["%.0f%%" % (t * 100.) if t else "unset"
               for t in endstop._dd_thresh]
        _dbg(gcmd,
            "live-halt diag: drawdown thresholds x=%s y=%s z=%s"
            " (self-derived from this descent's air noise)"
            % (ddt[0], ddt[1], ddt[2]))
        a0 = [(sum(v) / len(v)) if v else 0. for v in endstop._dbg_amp0]
        _dbg(gcmd,
            "live-halt diag: start-of-descent air amplitude x=%.0f y=%.0f z=%.0f"
            " (win=%s samp/%.3fmm step=%.3fmm)"
            % (a0[0], a0[1], a0[2], endstop._win_n, endstop._dbg_win_z,
               endstop._step_z))
        # Where each axis's max drop landed, as mm descended below the arm
        # height (= (t_max - armed) * descend_speed).  Small = startup/ring-up
        # transient (fixable by arming later); large = mid-descent/contact.
        dz = [(endstop._dbg_maxdrop_t[a] - endstop._armed_time)
              * endstop._descend_speed if endstop._dbg_maxdrop_t[a] else -1.
              for a in range(endstop.AXIS_COUNT)]
        _dbg(gcmd,
            "live-halt diag: max drop at mm-below-arm x=%.2f y=%.2f z=%.2f"
            % (dz[0], dz[1], dz[2]))
        # Precise contact Z from the captured stream (fine 'sensitivity').  The
        # endstop already dropped pre-descent samples (timestamps before t0 map
        # to nonsensical Z), so its buffer is used directly.
        samples = endstop.get_samples()
        # Anchor the Z mapping to the physical halt (trigger time -> epos[2])
        # when it halted; otherwise fall back to the descent start.
        trig_t = endstop.get_trigger_time()
        if halted:
            anchor_t, anchor_z = trig_t, epos[2]
        else:
            anchor_t, anchor_z = t0, z_start
        # Refine using whichever axis actually triggered the live halt (may be
        # a cross-coupled axis, not the configured excitation/output axis - see
        # _HostResonanceEndstop) - the fine crossing must be searched for on the
        # SAME channel that produced the halt, not blindly on output_index,
        # or the refinement can find nothing on an axis that never dropped.
        trig_axis = endstop.get_trigger_axis() if halted else None
        if halted:
            _dbg(gcmd,"Resonance probe: live halt triggered on axis=%s"
                              % 'xyz'[trig_axis])
        contact_z = self._analyze_drip(gcmd, samples, anchor_t, anchor_z,
                                       z_floor, t0 + self.warmup,
                                       anchored=halted, trigger_axis=trig_axis)
        if contact_z is None:
            if halted:
                # The loose halt already confirmed contact (and anchored its
                # trigger to where the amplitude drop began), but the fine
                # post-halt analysis could not confirm a persistent crossing -
                # typically because the move stopped at the halt, leaving too
                # little post-contact travel for the confirm window.  Fall back
                # to the halt position (anchor_z = epos[2]): a slightly looser
                # contact Z is far better than failing the touch outright.
                contact_z = anchor_z
                gcmd.respond_info("Resonance probe: fine analysis inconclusive;"
                                  " using halt position z=%.4f" % (contact_z,))
            else:
                toolhead.manual_move([x0, y0, z_start], lift_speed)
                toolhead.wait_moves()
                return None, halted
        # Leave the toolhead at the detected contact height (mirrors a real
        # probe's stop-at-trigger), so a session's retract is measured from
        # contact and repeated samples don't walk upward.
        toolhead.manual_move([x0, y0, contact_z], lift_speed)
        toolhead.wait_moves()
        return contact_z, halted

    # Calibration amplitude search (Path A).  With the contact height already
    # known (from a halting descent at the starting amplitude), keep the
    # excitation running continuously and oscillate Z up/down THROUGH the contact
    # point by a small bounded margin while STEPPING the amplitude down.  Each Z
    # cycle reads both an air sample (top) and a contact sample (bottom) -
    # bidirectional - so the resonance drop is measured at every amplitude with
    # no per-step warm-up and no unbounded descent (motion is bounded around the
    # known contact: safe on a rigid bed).  The excitation is rung up in air at
    # the top before the first dip to contact - never started while touching.
    # Returns the lowest amplitude whose drop is cleanly detectable, with its
    # diagnostics (accel_per_hz, baseline, rel_noise, max_drop).

    # Persist every descent to trace_dir, so detector changes can be replayed
    # offline instead of re-probed.  Contacts are a consumable: the plate wears,
    # and a worn spot changes both its noise floor and which axis detects.
    def _autosave_trace(self, rp, endstop):
        tdir = getattr(rp, 'trace_dir', None)
        if not tdir or not self.last_trace:
            return
        try:
            if not os.path.isdir(tdir):
                os.makedirs(tdir)
            # Sequence by what is already there - the host clock may be unset
            # on a headless boot, and a collision would silently overwrite a
            # descent we cannot re-create without wearing the plate again.
            n = len([f for f in os.listdir(tdir) if f.endswith('.csv')])
            path = os.path.join(tdir, "descent%05d.csv" % (n,))
            with open(path, 'w') as fh:
                # Metadata first: a trace is only replayable if the conditions
                # that produced it are known.
                fh.write("# freq=%.2f accel_per_hz=%.1f speed=%.4f\n"
                         % (self.excitation_freq, self.accel_per_hz,
                            endstop._descend_speed))
                fh.write("# win_n=%s step_z=%.5f\n"
                         % (endstop._win_n, endstop._step_z))
                # Warm-up gates when detection arms AND where the trace starts,
                # so two sets recorded at different warmups are not comparable
                # in their air baseline.  Unrecorded, that difference is
                # invisible in the corpus and reads as a real effect.
                fh.write("# warmup=%.3f\n" % (self.warmup,))
                fh.write("# halt_floor=%s\n"
                         % (",".join("%.4f" % v for v in endstop._halt_axis),))
                fh.write("# deriv_floor=%s\n"
                         % (",".join(("%.4f" % v) if v else "off"
                                     for v in endstop._deriv_axis),))
                fh.write("# drawdown_thresh=%s\n"
                         % (",".join(("%.4f" % v) if v else "unset"
                                     for v in endstop._dd_thresh),))
                fh.write("# trigger_kind=%s trigger_axis=%s\n"
                         % (endstop._trigger_kind, endstop._trigger_axis))
                note = getattr(rp, 'trace_note', None)
                if note:
                    # The replay loaders parse header lines by splitting on
                    # whitespace and then on '=', so a note containing spaces
                    # would silently truncate to its first word.  Emit a
                    # whitespace-free value instead of trusting the operator to
                    # remember - a mangled tag is unrecoverable without
                    # re-probing, and probing wears the plate.
                    fh.write("# note=%s\n" % ("-".join(note.split()),))
                fh.write("mm_below_arm,amp_x,amp_y,amp_z\n")
                for depth, ax, ay, az in self.last_trace:
                    fh.write("%.5f,%.3f,%.3f,%.3f\n" % (depth, ax, ay, az))
        except (IOError, OSError) as e:
            # Diagnostics must never break probing.
            logging.warning("resonance_probe: trace autosave failed: %s", e)

    def characterize_amplitude(self, gcmd, x0, y0, contact_z, lift_speed,
                               up_margin, down_margin,
                               cycles_per_level, n_levels, min_drop,
                               target_noise):
        import numpy as np
        toolhead = self.printer.lookup_object('toolhead')
        gcode = self.printer.lookup_object('gcode')
        reactor = self.printer.get_reactor()
        f = self.excitation_freq
        vdir = self.vibrate_dir.get_dir()
        z_hi = contact_z + up_margin
        # SAFETY: never let the bottom of the oscillation go below the hard floor
        # (guards against a bad contact_z, e.g. from a mode that detected poorly).
        z_lo = max(contact_z - down_margin, self.z_min)
        if z_lo >= contact_z - 0.005:
            raise gcmd.error("Calibrate: no room below contact %.3f above the"
                             " floor %.3f for the amplitude sweep; lower the"
                             " floor or raise the bed" % (contact_z, self.z_min))
        # SAFETY: lift OUT of contact into air before any excitation, so the
        # oscillation is never rung up while touching the bed.
        toolhead.manual_move([x0, y0, z_hi], lift_speed)
        toolhead.wait_moves()
        half_dt = 0.5 / f                       # one lateral half-cycle
        # Continuous lateral excitation throughout; the Z profile per level is a
        # DWELL at the top (air) -> ramp DOWN through contact -> short DWELL at
        # the bottom (in contact) -> ramp UP, repeated.  The dwells give clean,
        # Z-stationary air/contact references (no window Z-smear); the down ramp
        # is the MOVING measurement that matches how the real probe detects.  The
        # excitation is rung up ONCE in air (warm-up) and never stops, so it is
        # never started while touching the bed.  The contact dwell is kept short
        # (bed/nozzle wear) since the moving ramp is the probe-relevant metric.
        dwell_t = gcmd.get_float("CONTACT_DWELL", 0.35, above=0.1)
        # Ramp speed and span are chosen TOGETHER; changing one alone breaks
        # the other.  Segments per ramp are (z_travel/speed)/(0.5/f), so the
        # span pays for the speed:
        #   1.0mm/s over 0.14mm -> 59 segments @212Hz, but a 0.054mm window,
        #       twice the ~0.030mm contact event - it smears it away entirely.
        #   0.3mm/s over 0.14mm -> 198 segments, and this SHUT THE MCU DOWN
        #       ("Timer too close", measured).
        #   0.5mm/s over 0.05mm -> 42 segments (fewer than today) AND a 0.027mm
        #       window, just inside the event.  Both better at once.
        # Hence the narrow span below.  Do not slow this further "for
        # resolution": a step detector gets WORSE with finer sampling, because
        # a fixed drop split across more windows shrinks each per-window step
        # toward the noise.  Measured on the verify ramp: 0.5 -> 0.3mm/s took
        # step SNR from 9.9-24.2 down to 3.0-10.7.  ~2-3 windows across the
        # event is the target, which is where 0.5mm/s sits.
        # Scale the ramp speed WITH the excitation frequency.  A window spans
        # (detect_cycles/f) seconds, so at a fixed mm/s its Z width changes
        # threefold across the candidate range - 0.5mm/s gives 0.061mm at
        # 65.5Hz but 0.019mm at 212Hz, i.e. a fixed speed cannot resolve the
        # contact event at both ends.  Deriving speed from f pins the window at
        # RAMP_WIN_Z regardless of frequency.
        #
        # It also fixes the segment budget for free: segments per ramp are
        # span*2f/speed, so with speed proportional to f the frequency cancels
        # and every candidate costs the same ~40 segments.
        RAMP_WIN_Z = 0.020
        auto_speed = RAMP_WIN_Z * f / max(self.detect_cycles, 1e-9)
        ramp_speed = gcmd.get_float("CONTACT_RAMP_SPEED",
                                    min(max(auto_speed, 0.05), 2.0),
                                    above=0., maxval=10.)
        # Same MCU step-buffer overrun protection as the halting descent (see
        # HaltingContactProbe.run) - a bounded oscillation at high excitation
        # frequency is exactly as prone to "Timer too close".
        drip_time = gcmd.get_float("DRIP_TIME", 0.3, minval=0.) or None
        z_travel = z_hi - z_lo
        # SEGMENT BUDGET GUARD.  Every segment is one lateral half-cycle, and
        # the host must keep the MCU step buffer fed for all of them; overrun
        # is an MCU shutdown ("Timer too close"), which aborts calibration and
        # leaves the printer needing FIRMWARE_RESTART.  This has bitten twice,
        # both times from changing ONE parameter without redoing the
        # arithmetic: ramp speed 1.0 -> 0.3mm/s (475 -> 1583 segments/level),
        # and cycles 4 -> 10 (339 -> 848).  Measured on this machine: ~475 per
        # level is fine, 848 is not.
        #
        # So compute it and cap the reps, rather than trusting the caller to.
        # Reps are the right thing to give up: fewer ramps means noisier
        # per-level statistics, which degrades the result, where an overrun
        # destroys the whole run.
        MAX_SEGS_PER_LEVEL = 500
        ramp_t_est = max(z_travel / max(ramp_speed, 1e-3), 2. * half_dt)
        segs_per_ramp = max(2, int(round(ramp_t_est / half_dt)))
        max_reps = max(1, MAX_SEGS_PER_LEVEL // (2 * segs_per_ramp))
        reps = cycles_per_level
        if reps > max_reps:
            gcmd.respond_info(
                "Calibrate: capping CONTACT_CYCLES %d -> %d at %.1fHz (%d"
                " segments/ramp x2 would exceed the %d-segment budget and risk"
                " an MCU 'Timer too close' shutdown)"
                % (reps, max_reps, f, segs_per_ramp, MAX_SEGS_PER_LEVEL))
            reps = max_reps
        ramp_t = max(z_travel / max(ramp_speed, 1e-3), 2. * half_dt)
        air_dwell_segs = max(4, int(round(dwell_t / half_dt)))
        contact_dwell_segs = max(3, int(round(max(0.2, 0.55 * dwell_t)
                                              / half_dt)))
        ramp_segs = max(2, int(round(ramp_t / half_dt)))
        warm_segs = max(4, int(round(self.warmup / half_dt)))
        start_aph = self.accel_per_hz           # the starting (cap) amplitude
        levels = [start_aph / (2. ** i) for i in range(n_levels)]
        segs = []
        # 'warm'|'settle'|'air'|'down'|'contact'|'up' per segment ('settle' and
        # 'warm' windows are excluded from the analysis - see the emit() calls).
        seg_tag = []
        level_ranges = []       # (seg_start, seg_end, aph) for analysis
        sign = [1.]
        amp0 = start_aph / (4. * math.pi**2 * f)
        accel0 = start_aph * f
        # Detuning fix: cap each level's reversal cruise so the oscillation
        # stays on the commanded frequency (see _timing_corrected_cruise_v);
        # fall back to the accel-limited swing peak if there is no solution.
        peakv0 = self._corrected_peakv(start_aph)
        def emit(z, disp_amp, peakv, accel, tag):
            disp = disp_amp if sign[0] > 0 else -disp_amp
            segs.append(([x0 + vdir[0]*disp, y0 + vdir[1]*disp, z],
                         peakv, accel))
            seg_tag.append(tag)
            sign[0] = -sign[0]
        for _ in range(warm_segs):
            emit(z_hi, amp0, peakv0, accel0, 'warm')
        for aph in levels:
            amp = aph / (4. * math.pi**2 * f)
            accel = aph * f
            peakv = self._corrected_peakv(aph)
            seg_start = len(segs)
            # SETTLE (not measured): after the amplitude step DOWN the high-Q
            # resonance is still ringing at the previous, higher amplitude and
            # takes a few cycles to decay.  Skip this block so the measured air
            # baseline is at the settled amplitude (this was the dominant noise).
            for _ in range(air_dwell_segs):
                emit(z_hi, amp, peakv, accel, 'settle')
            for _ in range(reps):
                for s in range(ramp_segs):          # moving descent through contact
                    emit(z_hi - z_travel * (s + 1) / ramp_segs, amp, peakv,
                         accel, 'down')
                for _ in range(contact_dwell_segs): # short in-contact reference
                    emit(z_lo, amp, peakv, accel, 'contact')
                for s in range(ramp_segs):          # release edge (moving up)
                    emit(z_lo + z_travel * (s + 1) / ramp_segs, amp, peakv,
                         accel, 'up')
                for _ in range(air_dwell_segs):     # back to air
                    emit(z_hi, amp, peakv, accel, 'air')
            level_ranges.append((seg_start, len(segs), aph))
        segs.append(([x0, y0, z_hi], peakv, accel))  # finish in air, centered
        seg_tag.append('air')
        # Execute the whole bounded up/down sweep via the shared drip executor
        # (probing conditions + the detuning-exact corner clamp); its deep MCU
        # look-ahead is what survives the high segment counts at high excitation
        # frequency.  Worst-case ceiling is the level-0 (cap) segment; each
        # segment's own accel only ratchets it down.  Then park in air.
        samples, seg_times, t0 = self._run_bounded_vibration(
            gcmd, segs, f, ramp_speed, peakv0 + 1., accel0 + 1.,
            drip_time=drip_time)
        toolhead.manual_move([x0, y0, z_hi], lift_speed)
        toolhead.wait_moves()
        t_end = seg_times[-1] if seg_times else t0
        if not samples:
            raise gcmd.error("Calibrate: accelerometer measured no data during"
                             " the amplitude sweep")
        # Optional diagnostic: dump the raw stream + the planned schedule + the
        # ACTUAL per-segment end times, so the air-vs-contact analysis can be
        # debugged offline without re-running on hardware.
        dump = gcmd.get("DUMP", None)
        if dump:
            self._dump_sweep(dump, samples, segs, seg_times, seg_tag, t0, t_end,
                             contact_z, z_hi, z_lo, level_ranges, f)
            gcmd.respond_info("Calibrate: wrote sweep diagnostic to %s" % (dump,))
        # --- analysis: window the excitation amplitude and tag each window by
        # the phase (air dwell / moving down / contact dwell) of the segment it
        # falls in.  Per level: air dwell = clean baseline + noise; contact dwell
        # = clean in-contact reference (dwell_drop); the down ramp = the MOVING
        # drop the real (moving) probe would see. ---
        data = np.asarray(samples, dtype=np.float64)
        times = data[:, 0]
        sps = _sample_rate(times)
        win_n = max(8, int(self.detect_cycles / f * sps))
        # The onset drop can only be measured if a window is SHORTER in Z than
        # the onset band; otherwise every window straddles the surface, the
        # onset population comes back empty and every mode reads as no-signal -
        # a silent and very confusing failure.  Say so instead.
        win_z_span = (win_n / sps) * ramp_speed
        if win_z_span > LIVE_WIN_Z:
            gcmd.respond_info(
                "Calibrate: WARNING - at %.2f mm/s one analysis window spans"
                " %.3fmm of Z, wider than the %.3fmm onset band, so the moving"
                " numbers below are smeared across the surface; lower"
                " CONTACT_RAMP_SPEED to about %.2f mm/s"
                % (ramp_speed, win_z_span, LIVE_WIN_Z,
                   LIVE_WIN_Z * sps / win_n))
        # Map each sample time to the ACTUAL segment (via its real scheduled end
        # time) using the real per-segment times (an averaged seg_dt would drift
        # over the later levels and mislabel windows).
        seg_end_t = np.asarray(seg_times, dtype=np.float64)
        seg_tag_arr = np.asarray(seg_tag)
        # Compute the windowed on-resonance amplitude on ALL THREE accelerometer
        # axes, not just the configured excitation/detection axis - contact
        # damping is not guaranteed to show up most clearly on the axis being
        # driven; a cross-coupled axis can read a cleaner (or even stronger)
        # drop at some frequencies (see resonance-nozzle-probe memory).
        axis_names = ('x', 'y', 'z')
        cols = [data[:, 1 + a] for a in range(3)]
        wamp_axes, wk = _window_amps_tagged(times, cols, f, win_n,
                                            max(1, win_n // 2), seg_end_t)
        wtag = seg_tag_arr[wk]
        # Z of each window, from the segment it fell in.  Needed to split the
        # down-ramp by height: the ramp spans up_margin ABOVE contact to
        # down_margin BELOW it, and with the defaults (0.10 / 0.04) only 29% of
        # it is in contact, so a median over the whole ramp lands ~30um ABOVE
        # the surface - in air.  The old moving_drop did exactly that, making it
        # a measure of the gentle height-dependent air rise rather than of
        # contact damping, which is why it read 0-6% almost everywhere.
        seg_z = np.asarray([s[0][2] for s in segs], dtype=np.float64)
        wz = seg_z[wk]
        results = []
        for (s_start, s_end, aph) in level_ranges:
            m = (wk >= s_start) & (wk < s_end)
            per_axis = []
            for a in range(3):
                wamp = wamp_axes[a]
                air = wamp[m & (wtag == 'air')]
                contact = wamp[m & (wtag == 'contact')]
                downw = wamp[m & (wtag == 'down')]
                if len(air) < 3 or len(contact) < 2:
                    per_axis.append(None)
                    continue
                baseline = float(np.median(air))
                drop = max(0., 1. - float(np.median(contact)) / max(baseline, 1e-9))
                noise = float(np.std(air)) / max(baseline, 1e-9)
                # MOVING regime, measured against its own reference.  The live
                # halt happens while descending, where ring-down dilutes the
                # drop, so the stationary dwell numbers above do not predict it:
                # measured on hardware, the axis with a 68% dwell drop and +31pp
                # of dwell headroom reached only 1-6% live and never triggered,
                # while the axis ranked WORST on dwell carried every halt.
                #
                # Split the ramp by height and compare like with like: windows
                # below contact_z are moving-in-contact, windows in the upper
                # half of the air side are moving-in-air.  The gap between them
                # is left unassigned so the transition itself pollutes neither
                # population.  Deriving the noise from the moving-air windows
                # (not the stationary dwell) keeps drop and noise in the SAME
                # regime - pairing a moving drop with dwell noise would just be
                # a differently-mismatched metric.
                mdrop, mnoise = _moving_stats(
                    downw, wz[m & (wtag == 'down')], contact_z, up_margin)
                per_axis.append((baseline, drop, noise, mdrop, mnoise))
            valid_axes = [a for a, r in enumerate(per_axis) if r is not None]
            if not valid_axes:
                results.append((aph, None, None, None, None, per_axis))
                continue
            # Prefer the configured/primary axis when it is itself clean (keeps
            # behavior unchanged in the common case); otherwise take whichever
            # axis shows the strongest clean drop, so a cross-coupled axis can
            # win when the excitation axis itself does not damp cleanly.
            def is_clean(a):
                return (per_axis[a][1] >= min_drop
                        and per_axis[a][2] <= target_noise)
            # Pick the axis with the most halt-floor headroom, NOT the
            # configured one whenever it merely passes.  Preferring
            # output_index hid dramatically better detectors: measured at
            # 131.0Hz, x read 21%/0.9% (headroom 8.2pp) and was chosen, while
            # y read 89%/4.9% (headroom 37pp) on the same touch and was
            # discarded.  Ranking modes, choosing the amplitude and setting the
            # floors all use headroom, so the axis must too or the pipeline
            # disagrees with itself about what "detectable" means.
            clean_axes = [a for a in valid_axes if is_clean(a)]
            pool = clean_axes or valid_axes
            best_axis = max(pool, key=lambda a: _halt_headroom(per_axis[a][1],
                                                               per_axis[a][2]))
            baseline, drop, noise, mdrop, mnoise = per_axis[best_axis]
            results.append((aph, baseline, drop, noise, best_axis, per_axis))
            # Report the MOVING headroom next to the dwell one.  The live halt
            # is decided by the moving figures, so a mode/amplitude/axis with
            # big dwell headroom and negative moving headroom is a config that
            # looks excellent and cannot detect - exactly the 148Hz/z case.
            m_head = _halt_headroom(mdrop, mnoise)
            gcmd.respond_info(
                "Calibrate sweep: accel_per_hz=%.0f baseline=%.0f dwell_drop=%.0f%%"
                " noise=%.1f%% moving=%.0f%%/%.1f%% (head %+.1fpp) axis=%s"
                " (all axes: %s)"
                % (aph, baseline, drop * 100., noise * 100., mdrop * 100.,
                   mnoise * 100., m_head * 100.,
                   axis_names[best_axis],
                   ", ".join(
                       # dwell drop/noise then moving drop/noise, per axis -
                       # the pair that decides whether an axis will actually
                       # halt is the MOVING one, so both must be visible when
                       # comparing candidates.
                       "%s=%.0f/%.1f mov %.0f/%.1f" % (
                           axis_names[a], r[1] * 100., r[2] * 100.,
                           r[3] * 100., r[4] * 100.) if r is not None
                       else "%s=n/a" % axis_names[a]
                       for a, r in enumerate(per_axis))))
        # Pick the amplitude that leaves the LIVE detector the most HEADROOM.
        #
        # The old rule took the lowest amplitude clearing two hard thresholds
        # (drop >= min_drop and noise <= target_noise) and then stepped up a
        # couple of levels.  That is a THRESHOLD CROSSING, so ordinary run-to-run
        # measurement scatter decides which level crosses first and the pick
        # jumps around: the same machine at the same point chose accel_per_hz
        # 100, then 120, then 60 on three runs, dragging the derived thresholds
        # with it (sensitivity 0.060/0.060/0.097, halt 0.135/0.128/0.194).  An
        # unlucky run shipped a config with almost no live margin.
        #
        # Score instead by the headroom the halt floor will actually have, using
        # the SAME arithmetic the floor is later derived from (see the
        # calibration module's SNR floor rule): the floor must sit above
        # noise*1.3 + 3pp to reject noise, and at or below half the drop to catch
        # contact, so what is left over is
        #     headroom = 0.5*drop - (noise*1.3 + 0.03)
        # This is a smooth function of amplitude with an interior maximum - it
        # penalises high amplitude (drop shrinks as a fraction of a bigger
        # baseline) and low amplitude (noise grows) - so its argmax is stable,
        # and it optimises exactly the quantity that decides whether the probe
        # detects at all.  On the measured sweep it picks 100 (headroom 6.4pp)
        # over 120 (4.3), 60 (3.1) and 200 (2.1).
        valid = [i for i, r in enumerate(results) if r[1] is not None]
        if not valid:
            raise gcmd.error("Calibrate: amplitude sweep saw no usable contact"
                             " drop; check the start height/floor")

        # Score by the DETECTOR'S margin on the WEAKEST axis, not by headroom
        # on the best one.  Both changes were derived by simulating the live
        # detector over recorded descents (scripts/selector_eval.py):
        #
        #  - Weakest, not best: which axis carries contact varies with bed
        #    position, so a level chosen because one axis looked excellent here
        #    can go blind elsewhere.  Simulated all-axis trigger rate over the
        #    corpus: 172.9Hz 100%, 212.2Hz 75%, 65.5Hz 50%, 148Hz 27%.
        #  - Detector arithmetic, not headroom: headroom charges 1.15*noise
        #    where the live threshold is max(floor, nsigma*sd).  The gentler
        #    term over-rewards high-drop/high-noise candidates - it ranked
        #    aph 60 SECOND where simulation ranks it LAST (0% trigger rate),
        #    because low amplitude grows the drop on the good axes while
        #    raising the marginal axis's noise faster still.
        sel_floor = gcmd.get_float("SELECT_FLOOR", self._sel_floor, above=0.)
        sel_nsigma = gcmd.get_float("SELECT_NSIGMA", self._sel_nsigma,
                                    minval=1.)

        def headroom(i):
            # Retained under its old name so the reporting below is unchanged;
            # the VALUE is now a margin (x over threshold), not percentage
            # points - see the report string.
            per_axis = results[i][5]
            return analog_contact.robust_axis_margin(
                [None if r is None else (r[1], r[2]) for r in per_axis],
                sel_floor, sel_nsigma)
        # NO separate noise cap.  Headroom already subtracts the noise term
        # (noise*1.15 + 0.015), so filtering by noise a second time
        # double-counts it and can override the metric outright: on hardware a
        # 3% cap made this pick 6.7pp over 15.3pp at 65.5Hz and 8.2pp over
        # 19.0pp at 131Hz, both times dropping to a much lower amplitude.
        # That was audible - aph 30 is ~12um of displacement against ~42um at
        # 120 - and quieter is not safer: it is the low-amplitude end where the
        # live signal gets too small to detect.
        clean = [i for i in valid
                 if results[i][2] >= min_drop and results[i][3] <= target_noise]
        pool = clean or valid
        # Strict argmax, with the gentler (lower) amplitude winning exact ties.
        # Deliberately NO "within a whisker, prefer lower amplitude" band: the
        # low-amplitude end is precisely where noise explodes, so a tolerance
        # band just re-creates the trap this replaces.
        pick = max(pool, key=lambda i: (headroom(i), i))
        chosen = results[pick]
        if not clean:
            gcmd.respond_info(
                "Calibrate: WARNING - no amplitude met drop>=%.0f%% &"
                " noise<=%.1f%%; using the best available headroom"
                % (min_drop * 100., target_noise * 100.))
        gcmd.respond_info(
            "Calibrate: amplitude by 2nd-best-axis detector margin:"
            " %s -> accel_per_hz=%.0f (margin %.1fx)"
            % (", ".join("%.0f:%.1fx" % (results[i][0], headroom(i))
                         for i in valid),
               chosen[0], headroom(pick)))
        chosen_headroom = headroom(pick)
        # At 1.0x the second axis only just reaches its threshold, so ordinary
        # run-to-run variation leaves detection resting on one axis.  On the
        # trace corpus the worst usable mode scored 1.6x; the best scored 6.9x.
        if chosen_headroom < 2.5:
            gcmd.respond_info(
                "Calibrate: WARNING - only %.1fx margin on the SECOND-best"
                " axis, so detection leans on a single axis and a different"
                " bed position could shift the signal off it.  Consider"
                " another frequency" % (chosen_headroom,))
        aph, baseline, drop, noise, best_axis, per_axis = chosen
        gcmd.respond_info("Calibrate: chose accel_per_hz=%.0f (drop=%.0f%%,"
                          " noise=%.1f%%, axis=%s)"
                          % (aph, drop * 100., noise * 100.,
                             axis_names[best_axis]))
        # float() every value: these come straight out of numpy arrays, and a
        # numpy scalar that reaches status reporting crashes Moonraker's JSON
        # encoder ("Object of type bool_/float64 is not JSON serializable"),
        # which takes the MCU down with it.  Coerce at the boundary where the
        # numbers leave numpy, not at each downstream use.
        all_axes = {axis_names[a]: {'drop': r[1], 'noise': r[2]}
                    for a, r in enumerate(per_axis) if r is not None}
        # Verification signal for the CALLER to judge whether contact_z was
        # real: results[0] is the STARTING (cap/strongest) amplitude level -
        # the one most likely to show a clean drop if contact is real.  A
        # false halt (live descent stopped on noise/cross-axis coupling, not
        # actual bed contact) shows ~0% drop on EVERY axis even here, since
        # there is no real touch to damp regardless of how hard the nozzle
        # is driven (see resonance-nozzle-probe memory: 142Hz z-axis false
        # halt, contact_z off by 0.5mm from a moments-earlier scan, drop=0%
        # on x/y/z at accel_per_hz=200 - proof the "contact" was never real).
        cap_aph, cap_baseline, cap_drop, cap_noise, cap_axis, _ = results[0]
        verify_drop = cap_drop if cap_baseline is not None else 0.
        verify_noise = cap_noise if cap_baseline is not None else 1.
        # _plain(): this dict leaves the module for the calibration code, which
        # writes parts of it into the config and reports it - see _plain().
        return _plain({'accel_per_hz': aph, 'baseline': baseline,
                       'rel_noise': noise, 'max_drop': drop,
                       'axis': axis_names[best_axis], 'all_axes': all_axes,
                       'verify_drop': verify_drop,
                       'verify_noise': verify_noise,
                       'headroom': chosen_headroom})

    # Write a self-contained JSON diagnostic of one amplitude sweep: the raw
    # accelerometer stream plus everything needed to reconstruct what the motion
    # SHOULD have been.  seg_plan_z/seg_plan_aph are the planned Z and amplitude
    # of each waypoint; seg_times are the ACTUAL scheduled end times of those
    # waypoints (so time->segment->Z can be checked against the averaged seg_dt
    # the analysis currently assumes).  Analyzed offline.
    def _dump_sweep(self, path, samples, segs, seg_times, seg_tag, t0, t_end,
                    contact_z, z_hi, z_lo, level_ranges, f):
        import json
        # segs[i] = ([x, y, z], peak_v, accel); accel = aph * f.
        seg_plan_z = [float(s[0][2]) for s in segs]
        seg_plan_aph = [float(s[2]) / f for s in segs]
        meta = {
            'f': float(f), 'contact_z': float(contact_z),
            'z_hi': float(z_hi), 'z_lo': float(z_lo),
            't0': float(t0), 't_end': float(t_end),
            'output_index': int(self.output_index),
            'detect_cycles': float(self.detect_cycles),
            'levels': [[int(a), int(b), float(c)] for (a, b, c) in level_ranges],
            'seg_times': [float(x) for x in seg_times],
            'seg_plan_z': seg_plan_z, 'seg_plan_aph': seg_plan_aph,
            'seg_tag': list(seg_tag),
        }
        with open(path, 'w') as fh:
            json.dump({'meta': meta,
                       'samples': [[float(s[0]), float(s[1]), float(s[2]),
                                    float(s[3])] for s in samples]}, fh)


# Inner probe session: one descending probe move per run_probe, dispatched to
# the configured probe_mode.
class ResonanceProbeSession:
    def __init__(self, rprobe, gcmd):
        self.rprobe = rprobe
        self.results = []
    def run_probe(self, gcmd):
        rp = self.rprobe
        # The toolhead is already at the probe point; adopt its per-point
        # excitation frequency before the descent (no-op unless freq_mesh is set).
        x, y = rp.printer.lookup_object('toolhead').get_position()[:2]
        rp._apply_point_frequency(gcmd, x, y)
        if rp.probe_mode == 'hostdriven':
            self.results.append(rp._hostdriven_probe(gcmd))
        else:
            self.results.append(rp._stepwise_probe(gcmd))
    def pull_probed_results(self):
        # Sanitized: these results are consumed by PROBE_ACCURACY, bed_mesh and
        # probe status, all of which end up in JSON - see _plain().
        res = _plain(self.results)
        self.results = []
        return res
    def end_probe_session(self):
        self.results = []
        # Normal end of a session.  An ABORTED probe does not reach here, which
        # is why _hostdriven_probe also restores on the way out of an error -
        # leaving a print's part fan off would be far worse than a stray probe.
        self.rprobe._restore_part_fan()


def load_config(config):
    return ResonanceProbe(config)
