"use strict";

const nema_left_hand = (aa, bb, cc, index) => { // Nemamoussassa left hand
    console.log('NemamoussassaLeftHand', aa, bb, cc, index);
    if (index != 0) return [];
    let foo = [
        //////
        create_note(aa, 0, 4), null,
        create_note(aa, 0, 4), null,
        create_note(bb, 0, 4), null,
        create_note(bb, 0, 4), null,
        create_note(cc, 0, 4), null,
        create_note(cc, 0, 4), null,
        ///////
        create_note(aa, 0, 4), null,
        create_note(aa, 0, 4), null,
        create_note(bb, 0, 4), null,
        create_note(bb, 0, 4), null,
        create_note(cc+1, 0, 4), null,
        create_note(cc+1, 0, 4), null,
        ///////
        create_note(aa, 0, 4), null,
        create_note(aa, 0, 4), null,
        create_note(bb+1, 0, 4), null,
        create_note(bb+1, 0, 4), null,
        create_note(cc+1, 0, 4), null,
        create_note(cc+1, 0, 4), null,
        ///////
        create_note(aa+1, 0, 4), null,
        create_note(aa+1, 0, 4), null,
        create_note(bb+1, 0, 4), null,
        create_note(bb+1, 0, 4), null,
        create_note(cc+1, 0, 4), null,
        create_note(cc+1, 0, 4), null,
        ///////
        // { note: 'A5', chord: 5, delta: 0, octave: 5 },
    ];
    // while (foo.length < 12) foo.push(null);
    return foo;
};

const nema_full = (aa, bb, cc, index) => { // Nemamoussassa left hand & right hand
    console.log('NemamoussassaFull', aa, bb, cc, index);
    if (index != 0) return [];
    let foo = [
        //////
        create_note(aa, 0, 5),   create_note(aa, 4, 5),
        create_note(aa, 0, 4),   create_note(aa, 0, 6),
        create_note(bb, 0, 5),   create_note(bb, 2, 5),
        create_note(bb, 0, 4),   create_note(bb, 4, 5),
        create_note(cc, 0, 4),   create_note(cc, 0, 5),
        create_note(cc, 0, 3),   create_note(cc, 3, 6),
        ///////
        create_note(aa, 0, 5),   create_note(aa, 4, 5),
        create_note(aa, 0, 4),   create_note(aa, 0, 6),
        create_note(bb, 0, 5),   create_note(bb, 2, 5),
        create_note(bb, 0, 4),   create_note(bb, 4, 5),
        create_note(cc+1, 4, 5), create_note(cc+1, 0, 5),
        create_note(cc+1, 0, 4), create_note(cc+1, 2, 6),
        ///////
        create_note(aa, 0, 5),   create_note(aa, 4, 5),
        create_note(aa, 0, 4),   create_note(aa, 0, 6),
        create_note(bb+1, 0, 5), create_note(bb+1, 2, 5),
        create_note(bb+1, 0, 4), create_note(bb+1, 4, 6),
        create_note(cc+1, 4, 5), create_note(cc+1, 0, 5),
        create_note(cc+1, 0, 4), create_note(cc+1, 3, 6),
        ///////
        create_note(aa+1, 0, 5), create_note(aa+1, 4, 5),
        create_note(aa+1, 0, 4), create_note(aa+1, 0, 6),
        create_note(bb+1, 0, 5), create_note(bb+1, 2, 5),
        create_note(bb+1, 0, 4), create_note(bb+1, 4, 6),
        create_note(cc+1, 4, 5), create_note(cc+1, 0, 5),
        create_note(cc+1, 0, 4), create_note(cc+1, 2, 6),
        ///////
        // { note: 'A5', chord: 5, delta: 0, octave: 5 },
    ];
    // while (foo.length < 12) foo.push(null);
    return foo;
};
