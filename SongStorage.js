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

const getCategoryHash = async (score) => {
    const sparse_score = score.map(elem => !elem.enabled || elem.note == null ? ' ' : elem.note.note).join('');
    return getStringHash(sparse_score);
};

const getSongHash = async (song) => {
    return getStringHash(JSON.stringify(song));
}

class SongStorage {
    constructor() {
        this.songs = JSON.parse(localStorage.getItem('mbira_songs')) || {};
        if (!isObject(this.songs)) this.songs = {};
        // console.log(isObject(this.songs), this.songs)
    }

    forEachSong(cb) {
        Object.values(this.songs).forEach(category => Object.values(category).forEach(cb));
    }

    get searchSettings() {
        return {
            responseAsync: (settings, cb) => {
                const query = settings.urlData.query.toLowerCase();

                const results = Object.values(this.songs)
                    .map(category => Object.values(category).filter(song => song.title.toLowerCase().startsWith(query)))
                    .flat();

                const response = {
                    success: results.length > 0,
                    results: results,
                };
                cb(response);
            },
        }
    }

    async removeSong(song) {
        delete this.songs[song.category_hash][song.song_hash];
        return this.synchronise();
    }

    async addSong(song) {
        return getCategoryHash(song.score)
            .then(async (category_hash) => {
                if (!this.songs.hasOwnProperty(category_hash)) this.songs[category_hash] = {};
                const song_hash = await getSongHash(song);
                song.song_hash = song_hash;
                song.category_hash = category_hash;
                this.songs[category_hash][song_hash] = song;
                const stats = await this.synchronise();
                stats.song = song;
                if (this.onAddedSong) this.onAddedSong(song);
                return stats;
            })
    }

    async synchronise() {
        localStorage.setItem('mbira_songs', JSON.stringify(this.songs));
        const category_keys = Object.keys(this.songs);
        return {
            ncategory: category_keys.length,
            nsong: category_keys.reduce((previous, key) => previous + Object.keys(this.songs[key]).length, 0),
        }
    }

    async getCategory(score) {
        return getCategoryHash(score)
            .then((category_hash) => {
                const category = {
                    hash: category_hash,
                    songs: [],
                };
                if (this.songs.hasOwnProperty(category_hash))
                    category.songs = Object.values(this.songs[category_hash]);
                return category;
            })
    }
}
