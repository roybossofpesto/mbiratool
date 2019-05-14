'use strict';

class MbiraInstrument {
    constructor(effects, storage) {
        // create grid
        this.grid = new GridWidget(storage);

        // Mbira synth
        this.mbira_synth = new Tone.PolySynth(24, Tone.Synth).fan(effects.auto_panner, effects.buzzer);
        this.mbira_synth.set({
            envelope: {
                attack: 1e-2,
                attackCurve: "exponential",
                decay: 1.,
                decayCurve: "exponential",
                sustain: 0,
                release: .25,
                releaseCurve: "ripple",
            },
            volume: -20,
        });
        console.log("mbira_synth", this.mbira_synth.get());

        // Mbira loop
        const mbira_loop = new Tone.Sequence((time, widget) => {
            if (this.grid.playing) {
                // const transpose = transpose_coarse + transpose_fine / 100.;
                const next_note = widget.enabled && widget.note ? widget.note.note : null;
                const next_pitch = next_note ? Tone.Frequency(next_note) : null;
                const next_velocity = next_pitch ? .5  - (next_pitch - 520) / 1750 : null;
                // console.log('hhhhh', next_velocity)
                if (next_pitch && next_velocity) this.mbira_synth.triggerAttackRelease(next_pitch, "1n", time, next_velocity);
            }
            Tone.Draw.schedule(() => widget.ping(), time);
        }, this.grid.widgets, "8t").start();
    }


    set decay(value) {
        this.mbira_synth.set({
            envelope: {
                decay: value / 1000.,
            },
        });
        // console.log(value, "mbira_synth", this.mbira_synth.get('envelope.decay'));
    }
}
