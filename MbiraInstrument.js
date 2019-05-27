'use strict';

class MbiraInstrument {
    constructor(effects, storage) {
        this.transpose = 0;

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
        console.info("mbira_synth", this.mbira_synth.get());

        // Mbira loop
        const mbira_loop = new Tone.Pattern((time, widget) => {
            if (this.grid.playing) {
                // const transpose = transpose_coarse + transpose_fine / 100.;
                const next_note = widget.enabled && widget.note ? widget.note.note : null;
                const next_pitch = next_note ? Tone.Frequency(next_note).transpose(this.transpose) : null;
                const next_velocity = next_pitch ? Math.max(.2, Math.min(1., .5 - (next_pitch - 520) / 1750 + .5 * (Math.random() - .5))) : null;
                const next_length = next_pitch ? Math.max(.02, Math.min(.5, .5 - (next_pitch - 600) / 1200)) : null;
                // console.log('hhhhh', next_pitch, next_velocity, next_length, this.transpose);
                if (next_pitch && next_velocity && next_length) this.mbira_synth.triggerAttackRelease(next_pitch, next_length, time, next_velocity);
            }
            Tone.Draw.schedule(() => widget.ping(), time);
        }, this.grid.widgets, "up");
        mbira_loop.interval = "8t";
        mbira_loop.humanize = true;
        mbira_loop.start();
        console.info('mbira_loop', mbira_loop)
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
