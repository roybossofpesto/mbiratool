const wrap = (xx) => {
    return xx % 7;
};

const letters = "CDEFGAB";
const numbers = "1234567";
const tunings = [
    "Ionian",
    "Dorian",
    "Phyrgian",
    "Lydian",
    "Myxolydian",
    "Aeolian",
    "Locrian",
];
const song_names = []

const root_key_colors = ["#1c96fe", "#feb831", "#aee742", "#b75ac4", "#fbed00", "#d73535", "#ff5986"]
    .map((color) => {
        return chroma(color);
    });

$(document).ready(() => {

    // const synth = new Tone.Synth().toMaster()
    const synth = new Tone.FMSynth().toMaster();
    // synth.triggerAttackRelease('C4', 0.5, 0)
    // synth.triggerAttackRelease('E4', 0.5, 1)
    // synth.triggerAttackRelease('G4', 0.5, 2)
    // synth.triggerAttackRelease('B4', 0.5, 3)
    // var synth = new Tone.PolySynth(6, Tone.Synth).toMaster();
    // synth.set("detune", -1200);
    // synth.triggerAttackRelease(["C4", "E4", "A4"], "4n");

    // $(document)
    //     .keydown((evt) => {
    //         Tone.context.resume().then(() => {
    //             console.log('down', evt.key)
    //             synth.triggerAttack('C5', )
    //         })
    //     })
    //     .keyup((evt) => {
    //         Tone.context.resume().then(() => {
    //             console.log('up', evt.key)
    //             synth.triggerRelease();
    //         })
    //     })
    const base_size = 360;
    const knob_size = 70;
    const pen_width = 2;

    let tuning = parseInt($('#tuning-knob').val());
    let mode = parseInt($('#mode-knob').val());
    let use_letter_alphabet = $('#letters-checkbox').prop('checked');

    const mbira_callbacks = $('div.mbira').map(function() {
        const paper = Raphael(this, base_size, base_size);

        paper.rect(pen_width, pen_width, base_size - 2 * pen_width, base_size - 2 * pen_width, 10).attr({
            "stroke-width": 2 * pen_width,
            "fill": "#eee"
        });
        paper.rect(20, 40, base_size - 40, base_size - 80, 0).attr({
            "stroke-width": 2 * pen_width,
            "fill": "#777"
        });
        paper.circle(base_size - 55, base_size - 75, 20).attr({
            "stroke-width": 2 * pen_width,
            "fill": "#fff"
        });

        const reset_colors = () => {
            keys.forEach((foo) => {
                foo.attr('fill', root_key_colors[wrap(foo.key + tuning)])
            })
        };
        const highlight_colors = function() {
            const key = this.key
            const key_third = (key + 2) % 7;
            const key_fifth = (key + 4) % 7;
            keys.forEach((foo, kk) => {
                const base_color = root_key_colors[wrap(key + tuning)];
                const color =
                    foo.key == key ? base_color :
                    foo.key == key_third ? base_color.brighten(2.5) :
                    foo.key == key_fifth ? base_color.brighten(1) :
                    "white";
                foo.attr('fill', color);
            })

        };
        const keys = [];
        const create_key = (width, height, key) => {
            const path_string = "M0,0l0," + height + "c0," + width + "," + width + "," + width + "," + width + ",0l0," + (-height) + "z";
            const path_object = paper.path(path_string);
            path_object.attr({
                "stroke-width": 2 * pen_width,
                "fill": root_key_colors[wrap(key + tuning)],
            });
            path_object.key = key;
            keys.push(path_object);
            path_object.hover(highlight_colors, reset_colors);
            return path_object;
        };

        const big_key_width = 24;
        const small_key_width = 15.35;
        const key_offset = 15;
        const lower_left_keys = [1, 6, 5, 4, 3, 2, 0];
        const upper_left_keys = [1, 0, 6, 5, 3, 4, 0];
        const right_keys = [0, 1, 2, 3, 4, 5, 6, 0, 1];
        lower_left_keys.forEach((key, kk) => {
            create_key(big_key_width, 210 + 10 * kk, key).translate(key_offset + big_key_width * (kk + .5), key_offset);
        })
        upper_left_keys.forEach((key, kk) => {
            create_key(big_key_width, 150 + 8 * kk, key).translate(key_offset + big_key_width * kk, key_offset);
        })
        create_key(big_key_width, 170, 2).translate(key_offset + big_key_width * upper_left_keys.length, key_offset);
        right_keys.forEach((key, kk) => {
            create_key(small_key_width, 150 - 5 * kk, key).translate(base_size - key_offset + small_key_width * (kk - right_keys.length), key_offset);
        })

        paper.rect(pen_width, 35, base_size - 2 * pen_width, 20, 0).attr({
            "stroke-width": 2 * pen_width,
            "fill": "#777"
        });

        return {
            highlight: highlight_colors,
            reset: reset_colors,
            update: reset_colors,
        };
    }).get();

    const dial_callbacks = $('div.dial').map(function() {
        const paper = Raphael(this, base_size, base_size);
        paper.customAttributes.arc = function(centerX, centerY, startAngle, endAngle, innerR, outerR) {
            var radians = Math.PI / 180,
                largeArc = +(endAngle - startAngle > 180);
            // calculate the start and end points for both inner and outer edges of the arc segment
            // the -90s are about starting the angle measurement from the top get rid of these if this doesn't suit your needs
            outerX1 = centerX + outerR * Math.cos((startAngle - 90) * radians),
                outerY1 = centerY + outerR * Math.sin((startAngle - 90) * radians),
                outerX2 = centerX + outerR * Math.cos((endAngle - 90) * radians),
                outerY2 = centerY + outerR * Math.sin((endAngle - 90) * radians),
                innerX1 = centerX + innerR * Math.cos((endAngle - 90) * radians),
                innerY1 = centerY + innerR * Math.sin((endAngle - 90) * radians),
                innerX2 = centerX + innerR * Math.cos((startAngle - 90) * radians),
                innerY2 = centerY + innerR * Math.sin((startAngle - 90) * radians);

            // build the path array
            var path = [
                ["M", outerX1, outerY1], //move to the start point
                ["A", outerR, outerR, 0, largeArc, 1, outerX2, outerY2], //draw the outer edge of the arc
                ["L", innerX1, innerY1], //draw a line inwards to the start of the inner edge of the arc
                ["A", innerR, innerR, 0, largeArc, 0, innerX2, innerY2], //draw the inner arc
                ["z"] //close the path
            ];
            return {
                path: path
            };
        };
        paper.circle(base_size / 2, base_size / 2, base_size / 2 - pen_width).attr({
            'fill': '#eee',
            'stroke-width': 2 * pen_width,
            'stroke': 'black',
        });

        const sectors = []
        for (let kk = 0; kk < 48; kk++) {
            const sector = paper
                .path()
                .attr({
                    "stroke-width": 0,
                    'stroke': "#f0f",
                    'fill': root_key_colors[Math.floor(kk / 4) % 7],
                    'arc': [base_size / 2, base_size / 2, 0, 360 / 48 + .5, 50, 176],
                })
                .rotate(360 * kk / 48, base_size / 2, base_size / 2);
            sector.hover(function() {
                mbira_callbacks.forEach((foo) => foo.highlight.call({
                    key: this.chord - tuning,
                }));
            }, () => mbira_callbacks.forEach((foo) => foo.reset()));
            sectors.push(sector);
        }

        const loop = new Tone.Pattern(function(time, sector) {
            synth.triggerAttackRelease(sector.note, "16n", time);
            Tone.Draw.schedule(function() {
                sector.attr({
                    opacity: 0
                }).animate({
                    opacity: 1
                }, 500)
            }, time);
        }, sectors).start(0);
        loop.interval = '8n';
        Tone.Transport.lookAhead = 0.5;

        {
            const center_back = paper.circle(base_size / 2, base_size / 2, 50 - pen_width).attr({
                'fill': '#eee',
                'stroke-width': 2 * pen_width,
                'stroke': "black"
            })
            const center_symbol = paper.path("M30,0L-15,-26L-15,26z").attr({
                'fill': 'black',
                'stroke-width': 0,
            }).translate(base_size / 2, base_size / 2)
            let playback = false;
            const toggle_playback = () => {
                playback = !playback;
                if (playback) {
                    center_symbol.attr('path', "M15,26l0,-52l-10,0l0,52zM-15,26l0,-52l10,0l0,52z");
                    Tone.Transport.start();
                } else {
                    center_symbol.attr('path', "M30,0L-15,-26L-15,26z");
                    Tone.Transport.stop();
                }
            };
            center_back.click(toggle_playback);
            center_symbol.click(toggle_playback);
        }

        const expand_chord = (aa, bb, cc) => {
            const helper = (chord, delta, octave) => ({
                note: `${letters[wrap(chord+delta)]}${octave}`,
                chord: wrap(chord),
                octave: octave,
                delta: delta,
            });
            return [
                helper(aa, 0, 4), helper(aa, 0, 5), helper(aa, 5, 3), helper(aa, 5, 5),
                helper(bb, 0, 4), helper(bb, 0, 5), helper(bb, 5, 3), helper(bb, 5, 5),
                helper(cc, 0, 4), helper(cc, 0, 5), helper(cc, 5, 3), helper(cc, 5, 5),
            ];
        }

        const update = () => {
            let chords = [];
            const value = mode + tuning;
            first_song_blocks.each(function(index) {
                const aa = wrap(value + 0 + (index > 2));
                const bb = wrap(value + 2 + (index > 1));
                const cc = wrap(value + 4 + (index > 0));
                chords = chords.concat(expand_chord(aa, bb, cc));
            })
            sectors.forEach((sector, index) => {
                const chord = chords[index];
                sector.note = chord.note;
                sector.attr('fill', root_key_colors[chord.chord].brighten(chord.delta == 5 ? 1 : 0))
            })
        }
        return {
            update: update,
        };
    }).get();

    const first_song_blocks = $('#first-song>div>span');
    const second_song_blocks = $('#second-song>div>span');
    const third_song_blocks = $('#third-song>div>span');
    const update_songs = () => {
        const alphabet = use_letter_alphabet ? letters : numbers;
        const value = mode + use_letter_alphabet * tuning;
        const value_ = mode + tuning;
        first_song_blocks.each(function(index) {
            const aa = wrap(value + 0 + (index > 2));
            const bb = wrap(value + 2 + (index > 1));
            const cc = wrap(value + 4 + (index > 0));
            const aa_ = wrap(value_ + 0 + (index > 2));
            $(this)
                .text(`${alphabet[aa]}${alphabet[bb]}${alphabet[cc]}`)
                .css('background-color', root_key_colors[aa_]);
        })
        second_song_blocks.each(function(index) {
            const aa = wrap(value + 2 + (index < 2));
            const bb = wrap(value + 4 + (index != 2));
            const cc = wrap(value + 0 + (index == 0));
            const aa_ = wrap(value_ + 2 + (index < 2));
            $(this)
                .text(`${alphabet[aa]}${alphabet[bb]}${alphabet[cc]}`)
                .css('background-color', root_key_colors[aa_]);
        })
        third_song_blocks.each(function(index) {
            const aa = wrap(value + 4 + (index < 3));
            const bb = wrap(value + 0 + (index == 1));
            const cc = wrap(value + 2 + (index < 2));
            const aa_ = wrap(value_ + 4 + (index < 3));
            $(this)
                .text(`${alphabet[aa]}${alphabet[bb]}${alphabet[cc]}`)
                .css('background-color', root_key_colors[aa_]);
        })
    };

    const update_all = () => {
        update_songs();
        mbira_callbacks.forEach((foo) => foo.update());
        dial_callbacks.forEach((foo) => foo.update());
    }

    // knob demo page http://anthonyterrien.com/demo/knob/

    $('#tuning-knob').knob({
        'width': knob_size,
        'height': knob_size,
        'min': 0,
        'max': 7,
        'displayInput': false,
        'cursor': 360 / 7,
        'fgColor': 'black',
        'thickness': .5,
        'format': (foo) => {
            tuning = foo
            $('div.view.tuning').text(tunings[wrap(tuning)]);
            update_all();
            return foo;
        },
    })
    $('#mode-knob').knob({
        'width': knob_size,
        'height': knob_size,
        'min': 0,
        'max': 7,
        'displayInput': false,
        'cursor': 360 / 7,
        'fgColor': 'black',
        'thickness': .5,
        'format': (foo) => {
            mode = foo
            $('div.view.mode').text(numbers[wrap(mode)]);
            update_all();
            return foo;
        },
    })
    $('#letters-checkbox').on('change', function() {
        use_letter_alphabet = this.checked;
        update_all();
    });

})
