// Vibration overlay - excitation as a modulation of stepper position rather
// than as queued moves.
//
// The resonance probe drives the toolhead at a structural resonance by
// emitting one lookahead move per quarter-cycle.  At 72Hz that is ~290 moves
// per second, which (a) exhausts the MCU step queue and raises "Timer too
// close", forcing a 2000-segment budget that trades reps against resolution
// against air headroom, (b) cannot span a move boundary, so the vibration
// necessarily stops between a descent and the ascent that follows it, and
// (c) has to run with input shaping disabled, since the shaper would filter
// the excitation away.
//
// This wraps a stepper's kinematics the same way kin_shaper.c does: intercept
// calc_position_cb, evaluate the underlying position, and add a sinusoidal
// offset along a chosen axis.  Installed OUTSIDE the input shaper, the
// excitation lands after shaping and needs no disable/re-enable.  It consumes
// no lookahead entries at all, so the segment budget stops existing.
//
// IMPORTANT: Klipper generates steps only while moves exist in the trapq, so
// an overlay modulates motion, it cannot create it.  Continuous vibration
// therefore still needs a "carrier" move spanning the vibration window - but
// ONE long move, not thousands of short ones.
#define DUMMY_T 500.0

#include <math.h>   // sin
#include <stddef.h> // offsetof
#include <stdlib.h> // malloc
#include <string.h> // memset
#include "compiler.h" // __visible
#include "itersolve.h" // struct stepper_kinematics
#include "trapq.h" // struct move

struct vibration {
    struct stepper_kinematics sk;
    struct stepper_kinematics *orig_sk;
    struct move m;
    double freq, amp;          // Hz, mm (zero amp = pass-through)
    double start_time, end_time;
    double ax, ay, az;         // unit vector of the excitation direction
};

static double
vib_offset(struct vibration *v, double t)
{
    if (v->amp == 0. || t < v->start_time || t > v->end_time)
        return 0.;
    return v->amp * sin(2. * M_PI * v->freq * (t - v->start_time));
}

static double
vib_calc_position(struct stepper_kinematics *sk, struct move *m
                  , double move_time)
{
    struct vibration *v = container_of(sk, struct vibration, sk);
    double t = m->print_time + move_time;
    double off = vib_offset(v, t);
    if (off == 0.)
        return v->orig_sk->calc_position_cb(v->orig_sk, m, move_time);
    // Collapse the move to the offset position and let the underlying
    // kinematics map it, exactly as kin_shaper.c does.  Doing it this way
    // keeps the overlay kinematics-agnostic: on corexy the stepper position
    // is a combination of x and y, and only orig_sk knows that mapping.
    struct coord pos = move_get_coord(m, move_time);
    v->m.start_pos.x = pos.x + off * v->ax;
    v->m.start_pos.y = pos.y + off * v->ay;
    v->m.start_pos.z = pos.z + off * v->az;
    return v->orig_sk->calc_position_cb(v->orig_sk, &v->m, DUMMY_T);
}

int __visible
vibration_set_sk(struct stepper_kinematics *sk
                 , struct stepper_kinematics *orig_sk)
{
    struct vibration *v = container_of(sk, struct vibration, sk);
    if (!(orig_sk->active_flags & (AF_X | AF_Y | AF_Z)))
        return -1;
    v->sk.calc_position_cb = vib_calc_position;
    v->sk.active_flags = orig_sk->active_flags;
    v->orig_sk = orig_sk;
    v->sk.commanded_pos = orig_sk->commanded_pos;
    v->sk.last_flush_time = orig_sk->last_flush_time;
    v->sk.last_move_time = orig_sk->last_move_time;
    return 0;
}

// Arm the overlay for [start_time, end_time).  amp=0 disables it.  The step
// generation window is widened by one full cycle either side so the generator
// reconsiders steps that the oscillation can still move.
int __visible
vibration_set_params(struct stepper_kinematics *sk, double freq, double amp
                     , double start_time, double end_time
                     , double ax, double ay, double az)
{
    struct vibration *v = container_of(sk, struct vibration, sk);
    if (freq <= 0. || amp < 0.)
        return -1;
    v->freq = freq;
    v->amp = amp;
    v->start_time = start_time;
    v->end_time = end_time;
    v->ax = ax;
    v->ay = ay;
    v->az = az;
    double window = (amp > 0.) ? 1. / freq : 0.;
    v->sk.gen_steps_pre_active = window;
    v->sk.gen_steps_post_active = window;
    return 0;
}

struct stepper_kinematics * __visible
vibration_alloc(void)
{
    struct vibration *v = malloc(sizeof(*v));
    memset(v, 0, sizeof(*v));
    v->m.move_t = 2. * DUMMY_T;
    return &v->sk;
}
