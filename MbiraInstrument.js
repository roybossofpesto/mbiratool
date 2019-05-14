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
                sustain: .7,
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
                if (next_note) this.mbira_synth.triggerAttackRelease(Tone.Frequency(next_note), "16n", time);
                // if (note) this.mbira_synth.triggerAttackRelease(Tone.Frequency(note).transpose(transpose), "16n", time);
            }
            Tone.Draw.schedule(() => widget.ping(), time);
        }, this.grid.widgets, "8t").start();
    }


    set release(value) {
        this.mbira_synth.set({
            envelope: {
                release: value / 1000.,
            },
        });
        console.log(value, "mbira_synth", this.mbira_synth.get());
    }
}
