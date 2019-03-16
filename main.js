'use strict';

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

const root_key_colors = ["#1c96fe", "#feb831", "#aee742", "#b75ac4", "#15cdc2", "#fa2424", "#ff5986"]
    .map((color) => {
        return chroma(color);
    });

let mbira_synth = undefined;

// create note from chord, delta (=0 root, =2 third, =4 fifth) and octave on keyboard
const create_note = (chord, delta, octave) => ({
    note: `${letters[wrap(chord+delta)]}${octave}`,
    chord: wrap(chord),
    octave: octave,
    delta: delta,
});

const helper_single = (chords, nn = 4, octave = 4) => {
    const octaves = [];
    while (octaves.length < chords.length)
        octaves.push(octave);
    return helper_standard(chords, octaves, nn);
}

const helper_standard = (chords, octaves, nn = 4) => {
    const foo = chords.map((chord, index) => {
        const octave = octaves[index];
        return {
            note: `${letters[wrap(chord)]}${octave}`,
            chord: wrap(chord),
            octave: octave,
            delta: 0,
        };
    })
    while (foo.length < nn)
        foo.push(null);
    return foo;
}

const expands_chord = [
    // 0
    (aa, bb, cc) => [
        helper_standard([aa, aa], [5, 4], 3),
        helper_standard([aa, aa], [5, 4], 3),
        helper_standard([bb, bb], [5, 4], 3),
        helper_standard([cc, cc], [5, 4], 3),
    ].flat(),
    // 1
    (aa, bb, cc) => [
        helper_single([aa], 3, 5),
        helper_single([aa], 3),
        helper_single([bb], 3),
        helper_single([cc], 3),
    ].flat(),
    (aa, bb, cc) => [
        helper_single([aa], 3),
        helper_single([aa], 3),
        helper_single([bb], 3),
        helper_single([cc], 3),
    ].flat(),
    (aa, bb, cc) => {
        const foo = [
            helper_single([aa], 6),
            helper_single([bb], 3),
            helper_single([cc], 3),
        ].flat();
        return foo;
    },
    (aa, bb, cc) => [
        helper_single([aa], 3),
        helper_single([aa], 3),
        helper_single([bb], 3),
        helper_single([cc], 3),
    ].flat(),
    (aa, bb, cc) => {
        const foo = [
            helper_single([aa], 3),
            helper_single([bb], 3),
            helper_single([aa], 3),
            helper_single([cc], 3),
        ].flat();
        return foo;
    },
    (aa, bb, cc) => {
        const foo = [
            helper_single([aa]),
            helper_single([bb]),
            helper_single([cc]),
        ].flat();
        return foo;
    },
    (aa, bb, cc) => {
        return [
            create_note(aa, 0, 4), create_note(aa, 0, 5), create_note(aa, 0, 3), create_note(aa, 4, 5),
            create_note(bb, 0, 4), create_note(bb, 0, 5), create_note(bb, 0, 3), create_note(bb, 4, 5),
            create_note(cc, 0, 4), create_note(cc, 0, 5), create_note(cc, 0, 3), create_note(cc, 4, 5),
        ];
    },
    (aa, bb, cc) => {
        return [
            create_note(aa, 0, 4), create_note(aa, 2, 5), create_note(aa, 0, 3), create_note(aa, 4, 5),
            create_note(bb, 0, 4), create_note(bb, 2, 5), create_note(bb, 0, 3), create_note(bb, 4, 5),
            create_note(cc, 0, 4), create_note(cc, 2, 5), create_note(cc, 0, 3), create_note(cc, 4, 5),
        ];
    },
    (aa, bb, cc) => {
        return [
            create_note(aa, 0, 5), null, create_note(aa, 0, 5), null,
            create_note(bb, 0, 5), null, create_note(bb, 0, 5), null,
            create_note(cc, 0, 5), null, create_note(cc, 0, 5), null,
        ];
    },
    // 10 - Nemamoussassa on 4??
    (aa, bb, cc) => {
        return [
            create_note(aa, 0, 6),
            create_note(aa, 0, 5),
            create_note(aa, 4, 5),
            create_note(aa, 0, 4),
            ///////////////////////////////
            create_note(aa, 0, 6),
            create_note(bb, 0, 5),
            create_note(bb, 2, 5),
            create_note(bb, 0, 4),
            ///////////////////////////////
            null, null, null, null,
        ];

    }
]

$(document).ready(() => {

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

    // const mbira_synth = new Tone.Synth().toMaster()
    // const mbira_synth = new Tone.FMSynth().toMaster();
    mbira_synth = new Tone.PolySynth(24, Tone.Synth).toMaster();
    mbira_synth.voices.forEach((voice) => {
        voice.envelope.attack = 0.0005;
        voice.envelope.release = 10;
    })

    const hosho_synth = new Tone.NoiseSynth({
        noise: {
            type: 'pink',
        },
        envelope: {
            attack: 0.0005,
            decay: 0.1,
            sustain: .05,
            release: .1,
        },
    }).toMaster();
    hosho_synth.volume.value = -20;

    const base_size = 360;
    const knob_size = 50;
    const pen_width = 2;

    let tuning = parseInt($('#tuning-knob').val());
    let mode = parseInt($('#mode-knob').val());
    let use_letter_alphabet = $('#letters-checkbox').prop('checked');
    let transpose_coarse = parseInt($('#transpose-knob-coarse').val())
    let transpose_fine = parseInt($('#transpose-knob-fine').val())

    const createButton = (paper, row, col, label_str, callback, button_radius = 18, button_separation = 2) => {
        const step = 2 * button_radius + button_separation
        const cx = col >= 0 ? col * step + button_radius : base_size + col * step + step - button_radius;
        const cy = row >= 0 ? row * step + button_radius : base_size + row * step + step - button_radius;
        const cursor = callback ? "pointer" : "default";
        const button = paper.circle(cx, cy, button_radius).attr({
            'fill': 'black',
            'stroke-width': 0,
            'cursor': cursor,
        })
        const label = paper.text(cx, cy, label_str).attr({
            'fill': 'white',
            'cursor': cursor,
        })
        button.click(callback);
        label.click(callback);
        return label;
    }

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
            // calculate the start and end points for both inner and outer edges of the arc segment
            // the -90s are about starting the angle measurement from the top get rid of these if this doesn't suit your needs
            const radians = Math.PI / 180,
                largeArc = +(endAngle - startAngle > 180),
                outerX1 = centerX + outerR * Math.cos((startAngle - 90) * radians),
                outerY1 = centerY + outerR * Math.sin((startAngle - 90) * radians),
                outerX2 = centerX + outerR * Math.cos((endAngle - 90) * radians),
                outerY2 = centerY + outerR * Math.sin((endAngle - 90) * radians),
                innerX1 = centerX + innerR * Math.cos((endAngle - 90) * radians),
                innerY1 = centerY + innerR * Math.sin((endAngle - 90) * radians),
                innerX2 = centerX + innerR * Math.cos((startAngle - 90) * radians),
                innerY2 = centerY + innerR * Math.sin((startAngle - 90) * radians);

            // build the path array
            const path = [
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

        const center = base_size / 2;
        const radius_inside = 50;
        const hosho_thickness = 14;
        const radius_outside = center - hosho_thickness - 4 * pen_width;

        paper.circle(center, center, radius_outside + 2 * pen_width + hosho_thickness / 2).attr({
            'fill': '#eee',
            'stroke-width': 4 * pen_width + hosho_thickness,
            'stroke': 'black',
        });

        const mbira_sectors = [];
        for (let kk = 0; kk < 48; kk++) {
            const sector = paper
                .path()
                .attr({
                    "stroke-width": 0,
                    'stroke': "#f0f",
                    'fill': root_key_colors[Math.floor(kk / 4) % 7],
                    'arc': [center, center, 0, 360 / 48 + .5, radius_inside, radius_outside],
                    'cursor': 'pointer',
                })
                .rotate(360 * kk / 48, center, center);
            sector.hover(function() {
                mbira_callbacks.forEach((foo) => foo.highlight.call({
                    key: wrap(this.chord + 7 - tuning),
                }));
            }, () => mbira_callbacks.forEach((foo) => foo.reset()));
            mbira_sectors.push(sector);
        }

        const grow = (arr, nn, vv = false) => {
            while (arr.length < nn)
                arr.push(vv);
            return arr;
        }

        let hosho_position = 0;
        const hosho_sectors = [];
        const hosho_on_color = '#eee';
        const hosho_off_color = '#555';
        const hosho_size = 12;
        const color_hosho = (index) => {
            const bar = {
                'checked': hosho_position < 3 ? index % 3 == hosho_position ? true : false : hosho_position == 3 ? false : hosho_position == 4 ? grow([true, false, true, false, false, true], 12)[index % 12] : hosho_position < 6 ? index % 12 == hosho_position ? true : false : hosho_position < 12 ? index % 12 < hosho_position ? true : false : false,
                'stroke-width':
                    //(index %3 )* pen_width,
                    0,
            };
            bar.fill = bar.checked ? hosho_on_color : hosho_off_color;
            return bar;
        };

        // hosho sectors init
        const hosho_ring = paper.circle(center, center, radius_outside + 2 * pen_width + hosho_thickness / 2).attr({
            'fill': null,
            'stroke-width': pen_width + hosho_thickness,
            'stroke': hosho_off_color,
            'cursor': 'pointer',
        });

        const inc_hosho_pattern = () => {
            hosho_position += 1;
            hosho_position %= 12;
            update_hosho();
        };
        const dec_hosho_pattern = () => {
            hosho_position += 11;
            hosho_position %= 12;
            update_hosho();
        };

        hosho_ring.click(inc_hosho_pattern)

        const square_path = (size) => `M${-size/2},${-size/2}l${size},0l0,${size}l${-size},0z`
        const diamond_path = (size) => `M${-size/Math.sqrt(2)},0L0,${-size/Math.sqrt(2)}L${size/Math.sqrt(2)},0L0,${size/Math.sqrt(2)}z`
        // const triangle_path = (size) => `M${-Math.cos(Math.PI/6)*size},${Math.sin(Math.PI/6)*size}L0,-${size}L${Math.cos(Math.PI/6)*size},${Math.sin(Math.PI/6)*size}z`
        const update_hosho = () => hosho_sectors.forEach((sector, index) => sector.attr(color_hosho(index)));
        for (let kk = 0; kk < 48; kk++) {
            const sector = paper
                .path()
                .attr(Object.assign({
                    'stroke-width': 1,
                    path: square_path(hosho_size),
                    cursor: 'pointer',
                }, color_hosho(kk)))
                .rotate(360 * (kk + .5) / 48, center, center)
                .translate(center, center - radius_outside - hosho_thickness / 2 - 2 * pen_width);
            sector.index = kk;
            sector.click(() => {
                hosho_position += 1;
                hosho_position %= 12;
                update_hosho();
            })
            hosho_sectors.push(sector);
        }

        // Tone.js patterns -- loops
        const mbira_loop = new Tone.Pattern(function(time, sector) {
            const transpose = transpose_coarse + transpose_fine / 100.;
            if (sector.note) mbira_synth.triggerAttackRelease(Tone.Frequency(sector.note).transpose(transpose), "16n", time);
            Tone.Draw.schedule(function() {
                sector.attr({
                    opacity: 0
                }).animate({
                    opacity: 1
                }, 500)
            }, time);
        }, mbira_sectors).start(0);
        mbira_loop.interval = '8n';

        const hosho_loop = new Tone.Pattern(function(time, sector) {
            if (sector.attr('fill') == hosho_on_color)
                hosho_synth.triggerAttackRelease("16n", time);
            Tone.Draw.schedule(function() {
                sector.attr({
                        path: diamond_path(hosho_size),
                    })
                    .animate({
                        path: square_path(hosho_size),
                    }, 300)
            }, time);
        }, hosho_sectors).start(0);
        hosho_loop.interval = '8n';

        // Tone.Transport.lookAhead = .5;

        {
            const center_back = paper.circle(center, center, radius_inside - pen_width).attr({
                'fill': '#eee',
                'stroke-width': 2 * pen_width,
                'stroke': "black",
                'cursor': 'pointer',
            })
            const center_symbol = paper.path("M30,0L-15,-26L-15,26z").attr({
                'fill': 'black',
                'stroke-width': 0,
                'cursor': 'pointer',
            }).translate(center, center)
            let playback = false;
            const toggle_playback = () => {
                playback = !playback;
                if (playback) {
                    center_symbol.attr('path', "M18,26l0,-52l-14,0l0,52zM-18,26l0,-52l14,0l0,52z");
                    Tone.Transport.start();
                } else {
                    center_symbol.attr('path', "M30,0L-15,-26L-15,26z");
                    Tone.Transport.stop();
                }
            };
            center_back.click(toggle_playback);
            center_symbol.click(toggle_playback);
            $(document).keydown((evt) => {
                if (evt.key != ' ') return;
                toggle_playback();
                evt.preventDefault();
            })
        }


        let current_expand_chord_index = 0;

        { // expands_chord
            const label = createButton(paper, 1, 0, '0')
            const inc_callback = () => {
                current_expand_chord_index++;
                current_expand_chord_index %= expands_chord.length;
                label.attr('text', current_expand_chord_index);
                update_chords();
            }
            const dec_callback = () => {
                current_expand_chord_index += expands_chord.length - 1;
                current_expand_chord_index %= expands_chord.length;
                label.attr('text', current_expand_chord_index);
                update_chords();
            }
            createButton(paper, 0, 0, 'EC+', inc_callback);
            createButton(paper, 0, 1, 'EC-', dec_callback);
            mbira_sectors.forEach(sector => sector.click(inc_callback))
        }

        { // hosho_volume
            let volume = 0;
            const callback = () => {
                volume += 10;
                if (volume > 20) volume = -20;
                hosho_synth.volume.value = volume - 20;
                label.attr('text', `HV ${volume}`);
            }
            const label = createButton(paper, -2, 0, `HV ${volume}`, callback)
        }

        { // hosho_pattern
            createButton(paper, -1, 0, 'HP+', inc_hosho_pattern)
            createButton(paper, -1, 1, 'HP-', dec_hosho_pattern)
        }

        const update_chords = () => {
            let chords = [];
            const value = mode + tuning;
            first_song_blocks.each(function(index) {
                const aa = wrap(value + 0 + (index > 2));
                const bb = wrap(value + 2 + (index > 1));
                const cc = wrap(value + 4 + (index > 0));
                chords = chords.concat(expands_chord[current_expand_chord_index](aa, bb, cc));
            })
            mbira_sectors.forEach((sector, index) => {
                const chord = chords[index];
                if (chord == null) {
                    sector.attr('fill', 'black')
                    sector.note = null;
                    sector.chord = null;
                    return;
                }
                sector.note = chord.note;
                sector.chord = chord.chord;
                sector.attr('fill', root_key_colors[chord.chord]
                    .brighten(chord.delta == 4 ? 1 : chord.delta == 2 ? 2.5 : 0))
            })
        }
        return {
            update_chords: update_chords,
            update_hosho: update_hosho,
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
        dial_callbacks.forEach((foo) => foo.update_chords());
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
    });

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
    });

    $('#bpm-knob').knob({
        'width': knob_size,
        'height': knob_size,
        'min': 60,
        'max': 360,
        'fgColor': 'black',
        'thickness': .5,
        'displayInput': false,
        'format': (foo) => {
            Tone.Transport.bpm.value = foo;
            $('div.view.bpm').text(foo);
            return foo;
        },
    });

    $('#transpose-knob-coarse').knob({
        'width': knob_size,
        'height': knob_size,
        'min': -12,
        'max': 12,
        'fgColor': 'black',
        'thickness': .5,
        'displayInput': false,
        'format': (foo) => {
            transpose_coarse = foo;
            const transpose = transpose_coarse + transpose_fine / 100.;
            $('div.view.transpose').text(transpose);
            return foo;
        },
    });

    $('#transpose-knob-fine').knob({
        'width': knob_size,
        'height': knob_size,
        'min': -100,
        'max': 100,
        'fgColor': 'black',
        'thickness': .5,
        'displayInput': false,
        'format': (foo) => {
            transpose_fine = foo;
            const transpose = transpose_coarse + transpose_fine / 100.;
            $('div.view.transpose').text(transpose);
            return foo;
        },
    });

    $('#letters-checkbox').on('change', function() {
        use_letter_alphabet = this.checked;
        update_all();
    });

})
