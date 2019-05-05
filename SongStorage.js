'use strict';

const getScoreHash = (score) => {
    let sparse_score = score.map(elem => elem.enabled ? elem.note == null ? '_' : elem.note.note : ' ').join('');

    const encoder = new TextEncoder();
    const data = encoder.encode(sparse_score);

    return window.crypto.subtle
        .digest('SHA-1', data)
        .then((buffer) => {
            const array = new Uint8Array(buffer);

            const codes = [...array].map(value => {
                const code = value.toString(16);
                const padded_code = code.padStart(2, '0');
                return padded_code;
            });

            return codes.join('');
        });
}

class SongStorage {
    constructor() {
        this.songs = []
    }

    addSong(song) {
        getScoreHash(song.score).then((score_hash) => {
            console.log('addSong', song, score_hash);
        })
    }
}
