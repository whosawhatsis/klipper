# Vibration overlay - excitation as a position modulation, not as queued moves.
#
# PROTOTYPE.  The C side (chelper/kin_vibration.c) compiles clean and its
# kinematics mirror kin_shaper.c; this module is the wiring and has NOT been
# run on hardware.
#
# Why this exists: the resonance probe emits one lookahead move per excitation
# quarter-cycle (~290/s at 72Hz).  That exhausts the MCU step queue ("Timer
# too close"), forcing a 2000-segment budget in which reps, Z resolution and
# air headroom all compete; it cannot span a move boundary, so vibration stops
# between a descent and its ascent; and it must run with input shaping
# disabled.  Overlaying the oscillation on the stepper position removes all
# three at once.
#
# THE ONE THING THIS CANNOT DO: Klipper generates steps only while moves exist
# in the trapq, so an overlay modulates motion - it cannot create it.  A dwell
# is not a move, so vibration during a stationary dwell needs a "carrier" move
# spanning the window.  That is still ONE move instead of thousands, which is
# the whole point, but it is not free and the caller must arrange it.
import chelper


class VibrationOverlay:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.steppers = []
        self.vib_sks = []
        self.printer.register_event_handler("klippy:connect", self._connect)

    def _connect(self):
        # Install OUTSIDE any input shaper, so the excitation is applied after
        # shaping and the shaper does not filter it away.  Ordering is by
        # install time: whatever is on the stepper now becomes orig_sk.
        ffi_main, ffi_lib = chelper.get_ffi()
        kin = self.printer.lookup_object('toolhead').get_kinematics()
        for s in kin.get_steppers():
            sk = s.get_stepper_kinematics()
            vib_sk = ffi_main.gc(ffi_lib.vibration_alloc(), ffi_lib.free)
            if ffi_lib.vibration_set_sk(vib_sk, sk) < 0:
                continue  # stepper this overlay cannot drive (e.g. extruder)
            s.set_stepper_kinematics(vib_sk)
            self.steppers.append(s)
            self.vib_sks.append((vib_sk, sk))

    def arm(self, freq, amp, start_time, end_time, direction=(1., 0., 0.)):
        """Oscillate at 'freq' with amplitude 'amp' over [start,end) print time.

        amp=0 disables.  Times are absolute print times, so the caller arms the
        window around whatever motion it is about to queue - including across a
        descent and the ascent that follows, which is impossible today.
        """
        ffi_main, ffi_lib = chelper.get_ffi()
        ax, ay, az = direction
        for vib_sk, _orig in self.vib_sks:
            ffi_lib.vibration_set_params(vib_sk, freq, amp, start_time,
                                         end_time, ax, ay, az)

    def disarm(self):
        self.arm(1., 0., 0., 0.)


def load_config(config):
    return VibrationOverlay(config)
