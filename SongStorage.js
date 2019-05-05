'use strict';

const getStringHash = async (str) => {
    const encoder = new TextEncoder();
    const data = encoder.encode(str);

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
};

const getScoreCategoryHash = async (score) => {
    const sparse_score = score.map(elem => !elem.enabled || elem.note == null ? ' ' : elem.note.note).join('');
    return getStringHash(sparse_score);
};

const getSongHash = async (song) => {
    return getStringHash(JSON.stringify(song));
}

const isObject = (a) => {
    return (!!a) && (a.constructor === Object);
};

class SongStorage {
    constructor() {
        this.songs = JSON.parse(localStorage.getItem('mbira_songs')) || {};
        if (!isObject(this.songs)) this.songs = {};
        console.log(isObject(this.songs), this.songs)
    }

    addSong(song) {
        getScoreCategoryHash(song.score).then(async (category_hash) => {
            if (!this.songs.hasOwnProperty(category_hash)) this.songs[category_hash] = {};
            const song_hash = await getSongHash(song);
            song.song_hash = song_hash;
            song.category_hash = category_hash;
            this.songs[category_hash][song_hash] = song;
            console.log('addSong', song, category_hash, this.songs);
            this.sync();
        })
    }

    sync() {
        localStorage.setItem('mbira_songs', JSON.stringify(this.songs));
    }
}
