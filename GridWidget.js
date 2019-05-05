"use strict";

class GridWidget {
    constructor() {
        this.__visible = false;
        this.__playing = true;

        const menus = $($.parseHTML(`
        <div class="ui top attached menu">
            <div class="ui dropdown icon item">
                Chords
                <div class="menu">
                    <div class="set_chords_1111_3333_5555 item">1111 3333 5555</div>
                    <div class="set_chords_111_111_333_555 item">111 111 333 555</div>
                    <div class="set_chords_111_333_333_555 item">111 333 333 555</div>
                    <div class="set_chords_111_333_555_555 item">111 333 555 555</div>
                    <div class="set_chords_111_333_111_555 item">111 333 111 555</div>
                    <div class="set_chords_111_333_555_333 item">111 333 555 333</div>
                    <div class="set_chords_111_555_333_555 item">111 555 333 555</div>
                    <div class="divider"></div>
                    <div class="increment_chords item">Transpose +</div>
                    <div class="decrement_chords item">Transpose -</div>
                </div>
            </div>
            <div class="ui dropdown icon item">
                Octaves
                <div class="menu">
                    <div class="increment_octaves item"><i class="ui up arrow icon"></i>Shift Up</div>
                    <div class="decrement_octaves item"><i class="ui down arrow icon"></i>Shift Down</div>
                    <div class="shift_left_octaves item"><i class="ui left arrow icon"></i>Shift Left</div>
                    <div class="shift_right_octaves item"><i class="ui right arrow icon"></i>Shift Right</div>
                    <div class="divider"></div>
                    <div class="clear_octaves item">All &#x2582;</div>
                    <div class="set_minus_zero_octaves item">Repeat &#x2582;&#x2585;</div>
                    <div class="dropdown item">
                        3 Pattern
                        <i class="dropdown icon"></i>
                        <div class="menu">
                            <div class="set_minus_zero_plus_octaves item">Repeat &#x2582;&#x2585;&#x2588;</div>
                            <div class="set_plus_zero_minus_octaves item">Repeat &#x2588;&#x2585;&#x2582;</div>
                            <div class="set_zero_zero_minus_octaves item">Repeat &#x2585;&#x2585;&#x2582;</div>
                            <div class="set_minus_minus_zero_octaves item">Repeat &#x2582;&#x2582;&#x2585;</div>
                        </div>
                    </div>
                    <div class="dropdown item">
                        4 Pattern
                        <i class="dropdown icon"></i>
                        <div class="menu">
                            <div class="set_minus_zero_plus_zero_octaves item">Repeat &#x2582;&#x2585;&#x2588;&#x2585;</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="ui dropdown icon item">
                Deltas
                <div class="menu">
                    <div class="shift_left_deltas item"><i class="ui left arrow icon"></i>Shift Left</div>
                    <div class="shift_right_deltas item"><i class="ui right arrow icon"></i>Shift Right</div>
                    <div class="divider"></div>
                    <div class="set_all_first_deltas item">1</div>
                    <!--    <div class="clear_deltas item">All &emptyset;</div> -->
                    <div class="set_deltas_repeat_one_three item">13</div>
                    <div class="set_deltas_repeat_one_five item">15</div>
                    <div class="set_deltas_repeat_one_one_three item">113</div>
                    <div class="set_deltas_repeat_one_one_five item">115</div>
                    <div class="set_deltas_repeat_one_three_five item">135</div>
                    <div class="set_deltas_repeat_one_five_three item">153</div>
                    <div class="set_deltas_repeat_one_one_five_five item">1155</div>
                    <div class="set_deltas_repeat_one_one_three_five item">1135</div>
                    <div class="set_deltas_repeat_one_one_five_five item">1153</div>
                    <div class="set_deltas_repeat_one_three_one_five item">1315</div>
                    <div class="set_deltas_repeat_one_five_one_three item">1513</div>
                    <div class="set_deltas_repeat_one_five_one_three_doubletime item">11551133</div>


                </div>
            </div>
            <div class="ui dropdown icon item">
                Gates
                <div class="menu">
                    <div class="set_gates item">1</div>
                    <div class="set_on_off_gates item">10</div>
                    <div class="set_off_on_gates item">01</div>
                    <div class="set_on_off_on_gates item">101</div>
                    <div class="set_on_on_off_gates item">110</div>
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
                <div class="item icon active mute button"><i class="volume up icon"></i></div>
                <div class="item icon active collapse button"><i class="eye icon"></i></div>
            </div>
        </div>
        `));
        this.grid = $('<div>', {
            class: "ui twelve column center aligned grid segment",
        });
        const label = $('<div>', {
            class: "ui bottom attached left aligned score_label segment",
            style: "font-family: monospace;",
            text: "coucou",
        });

        const chords_111_111_333_555 = [
            0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4,
            0, 0, 0, 0, 0, 0, 2, 2, 2, 5, 5, 5,
            0, 0, 0, 0, 0, 0, 3, 3, 3, 5, 5, 5,
            1, 1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5, 0,
        ];

        const chords_111_333_333_555 = [
            0, 0, 0, 2, 2, 2, 2, 2, 2, 4, 4, 4,
            0, 0, 0, 2, 2, 2, 2, 2, 2, 5, 5, 5,
            0, 0, 0, 3, 3, 3, 3, 3, 3, 5, 5, 5,
            1, 1, 1, 3, 3, 3, 3, 3, 3, 5, 5, 5,
        ];

        const chords_111_333_555_555 = [
         5, 0, 0, 0, 2, 2, 2, 4, 4, 4, 4, 4, 4,
            0, 0, 0, 2, 2, 2, 5, 5, 5, 5, 5, 5,
            0, 0, 0, 3, 3, 3, 5, 5, 5, 5, 5, 5,
            1, 1, 1, 3, 3, 3, 5, 5, 5, 5, 5,
        ];

        const chords_111_333_111_555 = [
            0, 0, 0, 2, 2, 2, 0, 0, 0, 4, 4, 4,
            0, 0, 0, 2, 2, 2, 0, 0, 0, 5, 5, 5,
            0, 0, 0, 3, 3, 3, 0, 0, 0, 5, 5, 5,
            1, 1, 1, 3, 3, 3, 1, 1, 1, 5, 5, 5,
        ];

        const chords_111_333_555_333 = [
            0, 0, 0, 2, 2, 2, 4, 4, 4, 2, 2, 2,
            0, 0, 0, 2, 2, 2, 5, 5, 5, 2, 2, 2,
            0, 0, 0, 3, 3, 3, 5, 5, 5, 3, 3, 3,
            1, 1, 1, 3, 3, 3, 5, 5, 5, 3, 3, 3,
        ];

        const chords_111_555_333_555 = [
            0, 0, 0, 4, 4, 4, 2, 2, 2, 4, 4, 4,
            0, 0, 0, 5, 5, 5, 2, 2, 2, 5, 5, 5,
            0, 0, 0, 5, 5, 5, 3, 3, 3, 5, 5, 5,
            1, 1, 1, 5, 5, 5, 3, 3, 3, 5, 5, 5,
        ];


        const chords_1111_3333_5555 = [];
        [0, 2, 4, 0, 2, 5, 0, 3, 5, 1, 3, 5].forEach((value, index) => {
            for (let kk = 0; kk < 4; kk++)
                chords_1111_3333_5555.push(value);
        });

        this.widgets = chords_1111_3333_5555.map((chord, index) => {
            const widget = new NoteWidget();
            widget.chord = chord;
            widget.onUpdate = (note, enabled) => {
                this.score[index] = {
                    note: note,
                    enabled: enabled
                };
                this.update();
            }
            this.grid.append(widget.elem);
            return widget;
        });
        this.score = this.widgets.map(widget => ({
            note: widget.note,
            enabled: widget.enabled,
        }));

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

            song_search.find('.search_all').click(() => {
                song_search.search('set value', '');
                song_search.search('query');
            })
        }

        const widget_action = cb => () => this.widgets.forEach(cb);
        const set_action = (key, values) => widget_action((widget, index) => widget[key] = values[index % values.length]);

        menus.find('.ui.dropdown').dropdown({
            on: 'hover'
        });

        this.mute_button = menus.find('.mute.button');
        this.mute_button.click(() => {
            this.playing = !this.mute_button.hasClass('active');
        });

        this.collapse_button = menus.find('.collapse.button');
        this.collapse_button.click(() => {
            this.visible = !this.collapse_button.hasClass('active');
        })

        { // chord tools
            const set_chords_action = chords => set_action('chord', chords);
            menus.find('.increment_chords').click(widget_action((widget, index) => widget.chord++));
            menus.find('.decrement_chords').click(widget_action((widget, index) => widget.chord += 6));
            menus.find('.set_chords_1111_3333_5555').click(set_chords_action(chords_1111_3333_5555));
            menus.find('.set_chords_111_111_333_555').click(set_chords_action(chords_111_111_333_555));
            menus.find('.set_chords_111_333_333_555').click(set_chords_action(chords_111_333_333_555));
            menus.find('.set_chords_111_333_555_555').click(set_chords_action(chords_111_333_555_555));
            menus.find('.set_chords_111_333_111_555').click(set_chords_action(chords_111_333_111_555));
            menus.find('.set_chords_111_333_555_333').click(set_chords_action(chords_111_333_555_333));
            menus.find('.set_chords_111_555_333_555').click(set_chords_action(chords_111_555_333_555));
        }

        { // octave tools
            const repeat_octaves_action = octaves => set_action('octave', octaves);
            menus.find('.clear_octaves').click(widget_action((widget, index) => widget.octave = 5));
            menus.find('.set_minus_zero_octaves').click(repeat_octaves_action([4, 5]));
            menus.find('.set_minus_zero_plus_octaves').click(repeat_octaves_action([4, 5, 6]));
            menus.find('.set_plus_zero_minus_octaves').click(repeat_octaves_action([6, 5, 4]));
            menus.find('.set_zero_zero_minus_octaves').click(repeat_octaves_action([5, 5, 4]));
            menus.find('.set_minus_minus_zero_octaves').click(repeat_octaves_action([4, 4, 5]));
            menus.find('.set_minus_zero_plus_zero_octaves').click(repeat_octaves_action([4, 5, 6, 5]));
            menus.find('.increment_octaves').click(widget_action((widget, index) => widget.octave = 4 + (widget.octave - 3) % 3));
            menus.find('.decrement_octaves').click(widget_action((widget, index) => widget.octave = 4 + (widget.octave - 2) % 3));
            menus.find('.shift_right_octaves').click(() => {
                let prev = this.widgets[this.widgets.length - 1].octave;
                for (let kk = 0; kk < this.widgets.length; kk++) {
                    const current = this.widgets[kk].octave;
                    this.widgets[kk].octave = prev;
                    prev = current;
                }
            });
            menus.find('.shift_left_octaves').click(() => {
                let prev = this.widgets[0].octave;
                for (let kk = this.widgets.length - 1; kk >= 0; kk--) {
                    const current = this.widgets[kk].octave;
                    this.widgets[kk].octave = prev;
                    prev = current;
                }
            });

        }

        { // delta tools
            const repeat_deltas_action = deltas => set_action('delta', deltas);
            //menus.find('.clear_deltas').click(widget_action((widget, index) => widget.delta = -1));
            menus.find('.set_deltas_repeat_one_three').click(repeat_deltas_action([0, 2]));
            menus.find('.set_deltas_repeat_one_five').click(repeat_deltas_action([0, 4]));
            menus.find('.set_deltas_repeat_one_one_three').click(repeat_deltas_action([0, 0, 2]));
            menus.find('.set_deltas_repeat_one_one_five').click(repeat_deltas_action([0, 0, 4]));
            menus.find('.set_deltas_repeat_one_three_five').click(repeat_deltas_action([0, 2, 4]));
            menus.find('.set_deltas_repeat_one_five_three').click(repeat_deltas_action([0, 4, 2]));
            menus.find('.set_deltas_repeat_one_one_three_five').click(repeat_deltas_action([0, 0, 2, 4]));
            menus.find('.set_deltas_repeat_one_one_five_three').click(repeat_deltas_action([0, 0, 4, 2]));
            menus.find('.set_deltas_repeat_one_one_five_five').click(repeat_deltas_action([0, 0, 4, 4]));
            menus.find('.set_deltas_repeat_one_three_one_five').click(repeat_deltas_action([0, 2, 0, 4]));
            menus.find('.set_deltas_repeat_one_five_one_three').click(repeat_deltas_action([0, 4, 0, 2]));
            menus.find('.set_deltas_repeat_one_five_one_three_doubletime').click(repeat_deltas_action([0, 0, 4, 4, 0, 0, 2, 2]));
            menus.find('.set_all_first_deltas').click(widget_action((widget, index) => widget.delta = 0));
            menus.find('.shift_right_deltas').click(() => {
                let prev = this.widgets[this.widgets.length - 1].delta;
                for (let kk = 0; kk < this.widgets.length; kk++) {
                    const current = this.widgets[kk].delta;
                    this.widgets[kk].delta = prev;
                    prev = current;
                }
            });
            menus.find('.shift_left_deltas').click(() => {
                let prev = this.widgets[0].delta;
                for (let kk = this.widgets.length - 1; kk >= 0; kk--) {
                    const current = this.widgets[kk].delta;
                    this.widgets[kk].delta = prev;
                    prev = current;
                }
            });
        }

        { // gate tools
            const set_enabled_action = gates => set_action('enabled', gates);
            menus.find('.set_gates').click(widget_action((widget, index) => widget.enabled = true));
            menus.find('.set_on_off_gates').click(set_enabled_action([true, false]));
            menus.find('.set_off_on_gates').click(set_enabled_action([false, true]));
            menus.find('.set_on_off_on_gates').click(set_enabled_action([true, false, true]));
            menus.find('.set_on_on_off_gates').click(set_enabled_action([true, true, false]));
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
        this.elem.append(this.grid);
        this.elem.append(label);

        this.update();
    }

    get collapsed() {
        return !this.__visible;
    }

    set collapsed(enabled) {
        this.visible = !enabled;
    }

    get visible() {
        return this.__visible;
    }

    set visible(enabled) {
        this.__visible = enabled;
        // console.log('GridWidget.visible', this.__visible)
        this.collapse_button.toggleClass('active', this.__visible);
        this.collapse_button.find('i').attr('class', this.__visible ? 'eye icon' : 'eye slash icon');
        if (this.__visible) this.grid.show();
        else this.grid.hide();
    }

    get muted() {
        return !this.__playing;
    }

    set muted(enabled) {
        this.playing = !enabled;
    }

    get playing() {
        return this.__playing;
    }

    set playing(enabled) {
        this.__playing = enabled;
        // console.log('GridWidget.playing', this.__playing, this.onMute);
        this.mute_button.toggleClass('active', this.__playing)
        this.mute_button.find('i').attr('class', this.__playing ? 'icon volume up' : 'icon volume off');
        if (this.onMute) this.onMute(this.__playing);
    }

    update() {
        let sparse_score = this.score.map(elem => elem.enabled ? elem.note == null ? '__' : elem.note.note : '&nbsp;&nbsp;');
        sparse_score.splice(12, 0, "<br/>");
        sparse_score.splice(25, 0, "<br/>");
        sparse_score.splice(38, 0, "<br/>");
        sparse_score = sparse_score.join(' ');
        // console.log(sparse_score);

        this.elem.find('.score_label').html(sparse_score);
    }
}
