'use strict';

const wrap = xx => xx % 7;
const letters = "CDEFGAB";

// create note from chord, delta (=0 root, =2 third, =4 fifth) and octave on keyboard
// delta < 0 return an empty note (null)
const create_note = (chord, delta, octave) => delta < 0 ? null : {
    note: `${letters[wrap(chord+delta)]}${octave}`,
    chord: wrap(chord),
    octave: octave,
    delta: delta,
};

const chord_colors = ["#1c96fe", "#feb831", "#aee742", "#b75ac4", "#15cdc2", "#fa2424", "#ff5986"]
    .map(color => chroma(color));

const delta_brighten = (color, delta) => color.brighten(
    delta == 4 ? 1 :
    delta == 3 ? 1.7 :
    delta == 2 ? 2.5 :
    delta == 0 ? 0 :
    -10);
