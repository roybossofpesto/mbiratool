'use strict';

class MbiraInstrument {
    constructor(effects, storage) {
        // create grid
        this.grid = new GridWidget(storage);

        // Mbira synth
        const mbira_synth = new Tone.PolySynth(24, Tone.Synth).fan(effects.auto_panner, effects.buzzer);
        mbira_synth.set({
            envelope: {
                attack: 1e-2,
                attackCurve: "exponential",
                decay: 1.,
                sustain: .3,
                release: 1.5,
                releaseCurve: "ripple",
            },
            volume: -20,
        });
        console.log("mbira_synth", mbira_synth.get());

        // Mbira loop
        const mbira_loop = new Tone.Sequence((time, widget) => {
            if (this.grid.playing) {
                // const transpose = transpose_coarse + transpose_fine / 100.;
                const next_note = widget.enabled && widget.note ? widget.note.note : null;
                if (next_note) mbira_synth.triggerAttackRelease(Tone.Frequency(next_note), "16n", time);
                // if (note) mbira_synth.triggerAttackRelease(Tone.Frequency(note).transpose(transpose), "16n", time);
            }
            Tone.Draw.schedule(() => widget.ping(), time);
        }, this.grid.widgets, "8t").start();
    }
}
