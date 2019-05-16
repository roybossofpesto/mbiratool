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
};

class SongStorage {
    constructor() {
        this.songs = JSON.parse(localStorage.getItem('mbira_songs')) || {};
        if (!_.isObject(this.songs)) this.songs = {};
        // console.log(isObject(this.songs), this.songs)
    }

    clear() {
        localStorage.removeItem('mbira_songs');
    }

    forEachSong(cb) {
        _.values(this.songs).forEach(category => category.forEach(cb));
    }

    get searchSettings() {
        return {
            responseAsync: (settings, cb) => {
                const query = settings.urlData.query.toLowerCase();
                const start_match_query = song => song.title.toLowerCase().startsWith(query);
                const results = _.flatten(_.values(this.songs).map(category => category.filter(start_match_query)));
                const response = {
                    success: results.length > 0,
                    results: results,
                };
                console.log(`found ${response.results.length} match`);
                cb(response);
            },
        };
    }

    async removeSong(song) {
        this.songs[song.category_hash] = this.songs[song.category_hash].filter(song_ => song != song_);
        return this.synchronise();
    }

    async addSong(song_) {
        const song = _.cloneDeep(song_);
        console.log('addingSong', song)
        return getCategoryHash(song.score)
            .then(async (category_hash) => {
                if (!_.has(this.songs, category_hash)) this.songs[category_hash] = []
                song.song_hash = await getSongHash(song);
                song.category_hash = category_hash;
                this.songs[category_hash].push(song);
                const sync_promise = this.synchronise();
                if (this.onAddedSong) this.onAddedSong(song);
                const stats = await sync_promise;
                stats.song = song;
                return stats;
            });
    }

    async synchronise() {
        localStorage.setItem('mbira_songs', JSON.stringify(this.songs));
        const category_keys = _.keys(this.songs);
        return {
            ncategory: category_keys.length,
            nsong: category_keys.reduce((previous, key) => previous + _.keys(this.songs[key]).length, 0),
        };
    }

    async getCategory(score) {
        return getCategoryHash(score)
            .then((category_hash) => {
                const category = {
                    hash: category_hash,
                    songs: [],
                };
                if (_.has(this.songs, category_hash))
                    category.songs = this.songs[category_hash];
                return category;
            });
    }

    async getSongs(score) {
        return this.getCategory(score)
            .then(category => {
                let songs = [];
                if (_.has(this.songs, category.hash))
                    songs = this.songs[category.hash];
                return songs;
            });
    }
}
