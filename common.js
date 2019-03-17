
// create note from chord, delta (=0 root, =2 third, =4 fifth) and octave on keyboard
const create_note = (chord, delta, octave) => ({
    note: `${letters[wrap(chord+delta)]}${octave}`,
    chord: wrap(chord),
    octave: octave,
    delta: delta,
});
