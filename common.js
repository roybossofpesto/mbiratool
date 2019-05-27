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

const delta_back_brighten = (color, delta) => color.brighten(
    delta == 4 ? 1 :
    delta == 3 ? 1.7 :
    delta == 2 ? 2.5 :
    delta == 0 ? 0 :
    -10);

const delta_front_brighten = (color, delta) =>
    color == chord_colors[2] && delta == 4 ? chroma('#87be1f') :
    color == chord_colors[1] && delta == 4 ? chroma('#d49317') : 
    delta <= 0 || delta == 4 ? chroma("white") :
    chroma.mix(color, "black");

const compute_sparse_score = (score) => {
    let sparse_score = score.map(elem => elem.enabled ? elem.note == null ? '__' : elem.note.note : '&nbsp;&nbsp;');
    sparse_score.splice(12, 0, "<br/>");
    sparse_score.splice(25, 0, "<br/>");
    sparse_score.splice(38, 0, "<br/>");
    sparse_score = sparse_score.join(' ');
    return sparse_score;
}

const format_label = (xx) => xx < 1000 ? xx.toFixed(0).padStart(4, '0') : `${(xx/1000).toFixed(1).padStart(4, '0')}k`;
