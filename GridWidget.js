"use strict";

class GridWidget {
    constructor() {
        const menus = $($.parseHTML(`
        <div class="ui top attached menu">
            <div class="ui dropdown icon item">
                Chords
                <div class="menu">
                    <div class="set_chords_633 item">Pattern 633</div>
                    <div class="set_chords_444 item">Pattern 444</div>
                    <div class="divider"></div>
                    <div class="increment_chords item">Transpose +</div>
                    <div class="decrement_chords item">Transpose -</div>
                </div>
            </div>
            <div class="ui dropdown icon item">
                Octaves
                <div class="menu">
                    <div class="set_binary_octaves item">Repeat 0+</div>
                    <div class="clear_octaves item">All 0</div>
                </div>
            </div>
            <div class="ui dropdown icon item">
                Deltas
                <div class="menu">
                    <div class="set_all_first_deltas item">All 1st</div>
                    <div class="clear_deltas item">All &emptyset;</div>
                </div>
            </div>
            <div class="ui dropdown icon item">
                Songs
                <div class="menu">
                    <div class="set_nemafull_song item">Nemamoussassa</div>
                    <div class="set_nemaleft_song item">Nemamoussassa left hand</div>
                </div>
            </div>
            <div class="right menu">
                <!--<div class="ui item">
                    <div class="ui transparent icon input">
                        <input type="text" placeholder="Song"/>
                        <i class="music icon"></i>
                    </div>
                </div>-->
                <div class="item icon add_song button"><i class="save icon"></i></div>
                <div class="ui tiny add_song modal">
                    <div class="icon header">
                        <i class="save icon"></i>
                        Save song
                    </div>
                    <div class="content">
                        <div class="ui form">
                            <div class="field">
                                <label>Title</label>
                                <input name="title" type="text">
                            </div>
                            <div class="field">
                                <label>Description</label>
                                <textarea name='description'></textarea>
                            </div>
                            <button class="ui primary submit button">Save</button>
                        </div>
                    </div>
                </div>
                <div class="ui song search item">
                    <div class="ui transparent icon input">
                        <input class="prompt" type="text" placeholder="Search songs..."/>
                        <i class="music link icon search_all"></i>
                    </div>
                    <div class="results"></div>
                </div>
            </div>
        </div>
        `));
        const grid = $('<div>', {
            class: "ui twelve column center aligned grid segment",
        });
        const label = $('<div>', {
            class: "ui bottom attached left aligned score_label segment",
            style: "font-family: monospace;",
            text: "coucou",
        });

        const chords_633 = [
            0, 0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4,
            0, 0, 0, 0, 0, 0, 2, 2, 2, 5, 5, 5,
            0, 0, 0, 0, 0, 0, 3, 3, 3, 5, 5, 5,
            1, 1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5
        ];

        const chords_444 = [];
        [0, 2, 4, 0, 2, 5, 0, 3, 5, 1, 3, 5].forEach((value, index) => {
            for (let kk = 0; kk < 4; kk++)
                chords_444.push(value);
        });

        this.widgets = chords_444.map((chord, index) => {
            const widget = new NoteWidget();
            widget.chord = chord;
            widget.onUpdate = () => {
                // console.log('widget.note', widget.note);
                // console.log('widget.index', widget.index);
                // console.log('widget.chord', widget.chord);
                this.score[index] = widget.note;
                this.update();
            }
            grid.append(widget.elem);
            return widget;
        });
        this.score = this.widgets.map(widget => widget.note);

        { // Add Song
            const add_song_button = menus.find('.add_song.button');
            const add_song_modal = menus.find('.add_song.modal');
            const add_song_form = add_song_modal.find('.ui.form');
            add_song_modal.find('.ui.form').form({
                fields: {
                    title: 'empty',
                    description: 'maxLength[256]',
                },
                onSuccess: (evt, song) => {
                    song.notes = this.score;

                    const songs = JSON.parse(localStorage.getItem('mbira_songs')) || [];
                    songs.push(song);
                    localStorage.setItem('mbira_songs', JSON.stringify(songs));

                    add_song_form.form('clear');
                    add_song_modal.modal('hide');
                }
            });
            add_song_button.click(() => {
                add_song_modal.modal('show');
            });
        }

        { // Search song
            const song_search = menus.find('.song.search');
            song_search.search({
                apiSettings: {
                    responseAsync: (settings, cb) => {
                        const query = settings.urlData.query.toLowerCase();

                        let songs = JSON.parse(localStorage.getItem('mbira_songs')) || [];
                        songs = songs.filter(song => song.title.toLowerCase().startsWith(query));

                        const response = {
                            success: songs.length > 0,
                            results: songs,
                        };
                        cb(response);
                    },
                },
                minCharacters: 0,
                cache: false,
                onSelect: (selection) => {
                    // console.log('got song', selection.title, selection.notes)
                    this.widgets.forEach((widget, index) => widget.note = selection.notes[index]);
                },
            });

            /*song_search.find('.search_all').click(() => {
                song_search.search('set value', '');
                song_search.search('query');
            })*/
        }

        const widget_action = cb => () => this.widgets.forEach(cb);
        const set_action = (key, values) => widget_action((widget, index) => widget[key] = values[index]);

        menus.find('.ui.dropdown').dropdown();

        { // chord tools
            const set_chords_action = chords => set_action('chord', chords);
            menus.find('.increment_chords').click(widget_action((widget, index) => widget.chord++));
            menus.find('.decrement_chords').click(widget_action((widget, index) => widget.chord += 6));
            menus.find('.set_chords_444').click(set_chords_action(chords_444));
            menus.find('.set_chords_633').click(set_chords_action(chords_633));
        }

        { // octave tools
            menus.find('.clear_octaves').click(widget_action((widget, index) => widget.octave = 5));
            menus.find('.set_binary_octaves').click(widget_action((widget, index) => widget.octave = index % 2 == 0 ? 5 : 6));
        }

        { // delta tools
            menus.find('.clear_deltas').click(widget_action((widget, index) => widget.delta = -1));
            menus.find('.set_all_first_deltas').click(widget_action((widget, index) => widget.delta = 0));
        }

        { // song tools
            const set_notes_action = notes => set_action('note', notes);
            menus.find('.set_nemafull_song').click(set_notes_action(nema_full(0, 2, 4)));
            menus.find('.set_nemaleft_song').click(set_notes_action(nema_left_hand(0, 2, 4)));
        }

        this.elem = $('<div>', {
            class: "ui segments"
        });
        this.elem.append(menus);
        this.elem.append(grid);
        this.elem.append(label);

        this.update();
    }

    update() {
        let sparse_score = this.score.map(elem => elem == null ? '__' : elem.note);
        sparse_score.splice(12, 0, "<br/>");
        sparse_score.splice(25, 0, "<br/>");
        sparse_score.splice(38, 0, "<br/>");
        sparse_score = sparse_score.join(' ');
        // console.log(sparse_score);

        this.elem.find('.score_label').html(sparse_score);
    }
}
