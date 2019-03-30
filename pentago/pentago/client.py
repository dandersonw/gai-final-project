import curses
import os

from . import agent, game, render, view


class ConsoleClient(agent.Agent, view.View):
    key = 'human'

    def __init__(self):
        pass

    @classmethod
    def load_params(cls, config):
        return dict()

    def to_params(self):
        raise NotImplementedError

    def move_made(self, move, explanation):
        pass

    def render(self, model: game.Game):
        top_axis_label_h = 2
        left_axis_label_w = 4
        self.scr.clear()
        rendering = render.render_board(model.board).split(os.linesep)
        # Top axis label
        self.scr.addch(0, left_axis_label_w + len(rendering[0]) // 4, 'E')
        self.scr.addch(0, left_axis_label_w + 3 * len(rendering[0]) // 4, 'W')
        for j in range(game.BOARD_W):
            self.scr.addch(1, j * 2 + left_axis_label_w, str(j + 1))
        # left axis label
        self.scr.addch(top_axis_label_h + 1, 0, 'N')
        self.scr.addch(top_axis_label_h + 5, 0, 'S')
        for i in range(game.BOARD_H):
            if i >= game.QUADRANT_H:
                h_adj = 1
            else:
                h_adj = 0
            label = chr(ord('f') - i)
            self.scr.addch(i + top_axis_label_h + h_adj, 2, label)
        # rendered strings
        for i in range(len(rendering)):
            self.scr.addstr(i + top_axis_label_h,
                            left_axis_label_w,
                            rendering[i])
        self.scr.refresh()

    def game_ended(self, winner):
        if winner == 0:
            message = 'DRAW!'
        elif winner == 1:
            message = 'WHITE WINS!'
        elif winner == -1:
            message = 'BLACK WINS!'

        self.scr.clear()
        self.scr.addstr(0, 0, message)
        self.scr.addstr(2, 0, 'Press any key to quit')
        self.scr.getch()
        curses.endwin()

    def make_move(self, model: game.Game) -> agent.Explanation:
        extra_message = '{} Turn'.format('White' if model.turn == 1 else 'Black')
        while True:
            i, j, q_i, q_j, d = self._get_input(extra_message)
            i -= ord('a')
            j -= ord('1')
            q_i = 1 if q_i == ord('N') else 0
            q_j = 1 if q_j == ord('W') else 0
            d = 1 if d == ord('R') else -1
            if model.can_place_in_square(i, j):
                break
            else:
                extra_message = 'Cannot place at that location!'
        placement = game.generate_placement_from_indices(i, j)
        rotation = game.generate_rotation(q_i, q_j, d)
        move = (placement, rotation)
        explanation = None
        return move, explanation

    def _get_input(self, extra_message):
        i = None
        j = None
        q_i = None
        q_j = None
        d = None
        while any([i is None, j is None, q_i is None, q_j is None, d is None]):
            self.input_win.clear()
            self.input_win.addstr(0, 0, extra_message)
            self.input_win.addstr(1, 0, 'Enter keys for row and column')
            self.input_win.addstr(2,
                                  0,
                                  'Placement Row: {}, Placement Column: {}'
                                  .format(chr(i) if i is not None else 'None',
                                          chr(j) if j is not None else 'None'))
            self.input_win.addstr(3,
                                  0,
                                  'Rotation Row: {}, Rotation Column: {}\nDirection (R/L): {}'
                                  .format(chr(q_i) if q_i is not None else 'None',
                                          chr(q_j) if q_j is not None else 'None',
                                          chr(d) if d is not None else 'None'))
            c = self.input_win.getch()
            if c == ord('q'):
                # TODO - quit the game
                pass
            elif c >= ord('a') and c <= ord('f'):
                i = c
            elif c >= ord('1') and c <= ord('6'):
                j = c
            elif c == ord('N') or c == ord('S'):
                q_i = c
            elif c == ord('E') or c == ord('W'):
                q_j = c
            elif c == ord('R') or c == ord('L'):
                d = c
        return i, j, q_i, q_j, d

    def __enter__(self):
        self.scr = curses.initscr()
        curses.noecho()
        curses.curs_set(False)
        self.input_win = curses.newwin(6, 50, game.BOARD_H + 3, 0)
        return self

    def __exit__(self, type, value, traceback):
        curses.endwin()
