import pygame
import random
import sys
from math import sqrt
from collections import deque
import numpy

# color constants
BLACK = (0, 0, 0)
CYAN_GREY = (20, 20, 20)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 128)
LIGHTGREEN = (0, 255, 0)


EMPTY = 0
SNAKE_HEAD = 1
SNAKE = 2
FOOD = 3


class ListDict:
    """
    Parallel list/dict
    Used to hold empty spaces
    Make selecting new food position O(1)
    """

    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def add_item(self, item):
        """add unique item to the structure"""

        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove_item(self, item):
        """remove from struct by value of item"""

        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        """get random item"""

        choice = random.choice(self.items)
        return choice

    def clear(self):
        """reset struct to clear"""

        self.items.clear()
        self.item_to_position.clear()

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class Graphics:
    """
    Used to render the game state
    """

    WINDOW_SIZE = WINDOW_WIDTH, WINDOW_HEIGHT = (421, 471)
    GRID_SIZE = GRID_WIDTH, GRID_HEIGHT = (20, 20)

    def __init__(
        self, window_size=None, grid_size=None, margin=1, edge_margin=2, top_margin=50
    ):
        pygame.init()
        self.font = pygame.font.SysFont("monospace", 19)

        # set window size
        if not window_size:
            self.window_size = self.window_width, self.window_height = (
                Graphics.WINDOW_WIDTH,
                Graphics.WINDOW_HEIGHT,
            )
        else:
            self.window_size = self.window_width, self.window_height = window_size

        # set grid size
        if not grid_size:
            self.grid_size = self.grid_width, self.grid_height = (
                Graphics.GRID_WIDTH,
                Graphics.GRID_HEIGHT,
            )
        else:
            self.grid_size = self.grid_width, self.grid_height = grid_size

        # set margins
        self.margin = margin
        self.edge_margin = edge_margin
        self.top_margin = top_margin

        # set square size
        self.square_size = self.square_width, self.square_height = (
            self.window_width
        ) // (self.grid_width) - self.margin, (
            self.window_height - self.top_margin
        ) // (
            self.grid_height
        ) - self.margin

        self.screen = pygame.display.set_mode(self.window_size)

    def gen_rect(self, x, y):
        """generate rectangle dimensions list"""

        return [
            (self.margin + self.square_width) * x,
            (self.margin + self.square_height) * y + self.top_margin,
            self.square_width,
            self.square_height,
        ]

    def draw_square(self, x, y, color):
        """draw square on screen"""

        if self.screen:
            pygame.draw.rect(self.screen, color, self.gen_rect(x, y))

    def draw_background(self):
        """fill in board background"""

        if self.screen:
            self.screen.fill(CYAN_GREY)

    def draw_header(self):
        """draw header rectangle"""

        game_width = (self.margin + self.grid_width) * self.square_width
        pygame.draw.rect(self.screen, BLACK, [0, 0, game_width, self.top_margin])

    def draw_header_text(self, text):
        """draw the text in the header"""

        img = self.font.render(text, True, LIGHTGREEN)
        self.screen.blit(img, (2, 2))  # (10, self.top_margin/2)

    def draw_food(self, game):
        """draw food square"""

        self.draw_square(game.food_pos[0], game.food_pos[1], RED)

    def draw_snake(self, snake_coords):
        """draw snake on board"""

        if snake_coords:
            self.draw_square(*snake_coords[0], LIGHTGREEN)
            for i in range(1, len(snake_coords)):
                prev_coord = snake_coords[i - 1]
                curr_coord = snake_coords[i]
                prev_vec = (
                    curr_coord[0] - prev_coord[0],
                    curr_coord[1] - prev_coord[1],
                )

                rect = self.gen_rect(*curr_coord)
                """
                [x,y, width, height]
                [(self.margin + self.square_width) * x + self.margin,
                          (self.margin + self.square_height) * y + self.margin,
                          self.square_width,
                          self.square_height]
                """

                if prev_vec == (0, 1):
                    "coming from above"
                    # increase height by margin
                    rect[3] += self.margin
                    # move up by margin
                    rect[1] -= self.margin
                elif prev_vec == (0, -1):
                    "coming from below"
                    # increase height by margin
                    rect[3] += self.margin
                elif prev_vec == (1, 0):
                    "coming from left"
                    # increase width by margin
                    rect[2] += self.margin
                    # move left by margin
                    rect[0] -= self.margin
                elif prev_vec == (-1, 0):
                    "coming from right"
                    # increase width by margin
                    rect[2] += self.margin

                pygame.draw.rect(self.screen, LIGHTGREEN, rect)

    def update(self):
        """refresh the display"""

        pygame.display.flip()


class Grid:
    """
    Grid board holds game state
    NOTE: indexing is based on board[x][y]
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = [[EMPTY for i in range(0, width)] for i in range(0, height)]

    
    def get(self, x, y):
        """get space type at (x, y)"""

        # indexing is based on board[y][x]
        if y >= 0 and y < self.height:
            if x >= 0 and x < self.width:
                return self.board[y][x]
        return None

    def set(self, x, y, val):
        """set space type at (x, y)"""
        
        # indexing is based on board[y][x]
        if y >= 0 and y < self.height:
            if x >= 0 and x < self.width:
                self.board[y][x] = val


class Game:
    def __init__(self, size=(20, 20), graphics=False):
        self.score = 0
        # snake queue [HEAD, ...]
        self.snake = deque()
        self.grid = Grid(*size)
        self.max_dist = sqrt((size[1] ** 2 + size[1] ** 2) / 2)
        self.graphics = None
        if graphics:
            self.graphics = Graphics(grid_size=size)

        self.empty_spaces = ListDict()
        for row in range(0, self.grid.width):
            for col in range(0, self.grid.height):
                self.empty_spaces.add_item((row, col))

        new_head = self.empty_spaces.choose_random_item()

        self.empty_spaces.remove_item(new_head)
        self.grid.set(*new_head, SNAKE_HEAD)
        self.snake.appendleft(new_head)

        self.last_move = (1, 0)

        self.food_pos = self.add_food()

        self.action_space_n = 4

        self.food_reward = 1
        self.death_reward = -1
        self.def_reward = 0
        self.win_reward = 5

    def reset(self):
        self.score = 0
        # snake queue [HEAD, ...]
        self.snake.clear()
        self.empty_spaces.clear()

        self.last_move = (1, 0)

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                self.empty_spaces.add_item((x, y))
                self.grid.set(x, y, EMPTY)

        new_head = self.empty_spaces.choose_random_item()

        self.empty_spaces.remove_item(new_head)
        self.grid.set(*new_head, SNAKE_HEAD)
        self.snake.appendleft(new_head)

        self.food_pos = self.add_food()

    def sample_action_space(self):
        # state = random.getstate()
        # random.seed()
        choice = random.choice([(0, 1), (0, -1), (-1, 0), (1, 0)])
        # random.setstate(state)
        return choice

    def get_state(self):
        return str(self.snake) + str(self.food_pos)

    def get_state_adv(self):
        """
        return an array of:
        [(head_dir (rather where it came from)) =
            UP
            DOWN
            LEFT
            RIGHT
        (tail_dir (rather where it will go next)) =
            UP
            DOWN
            LEFT
            RIGHT
        (food_dir (which segment is the food in)) =
            UP_RIGHT
            DOWN_RIGHT
            DOWN_LEFT
            UP_LEFT
        (vision) =
            (for each cardinal direction)
            SEE_FOOD
            SEE_SNAKE
            SEE_WALL
            DISTANCE (normalized to max amount (board_width x board_height))
            (for each diagonal direction)
            SEE_FOOD
            SEE_SNAKE
            SEE_WALL
            DISTANCE (normalized to max amount (board_width x board_height))
        ]
        """
        state_keys = {
            "HEAD": {
                "UP": 0,
                "DOWN": 1,
                "LEFT": 2,
                "RIGHT": 3,
            },
            "TAIL": {
                "UP": 4,
                "DOWN": 5,
                "LEFT": 6,
                "RIGHT": 7,
            },
            "FOOD_DIR": {
                "UP": 8,
                "UP_RIGHT": 9,
                "RIGHT": 10,
                "DOWN_RIGHT": 11,
                "DOWN": 12,
                "DOWN_LEFT": 13,
                "LEFT": 14,
                "UP_LEFT": 15,
            },
            "SIGHT": {
                "UP": {
                    "vec": (0, -1),
                    "FOOD": 16,
                    "SNAKE": 17,
                    "WALL": 18,
                    "DIST": 19,
                },
                "UP-RIGHT": {
                    "vec": (1, -1),
                    "FOOD": 20,
                    "SNAKE": 21,
                    "WALL": 22,
                    "DIST": 23,
                },
                "RIGHT": {
                    "vec": (1, 0),
                    "FOOD": 24,
                    "SNAKE": 25,
                    "WALL": 26,
                    "DIST": 27,
                },
                "DOWN-RIGHT": {
                    "vec": (1, 1),
                    "FOOD": 28,
                    "SNAKE": 29,
                    "WALL": 30,
                    "DIST": 31,
                },
                "DOWN": {
                    "vec": (0, 1),
                    "FOOD": 32,
                    "SNAKE": 33,
                    "WALL": 34,
                    "DIST": 35,
                },
                "DOWN-LEFT": {
                    "vec": (-1, 1),
                    "FOOD": 36,
                    "SNAKE": 37,
                    "WALL": 38,
                    "DIST": 39,
                },
                "LEFT": {
                    "vec": (-1, 0),
                    "FOOD": 40,
                    "SNAKE": 41,
                    "WALL": 42,
                    "DIST": 43,
                },
                "UP-LEFT": {
                    "vec": (-1, -1),
                    "FOOD": 44,
                    "SNAKE": 45,
                    "WALL": 46,
                    "DIST": 47,
                },
            },
        }
        state = [0 for i in range(4 + 4 + 8 + 4 * 8)]

        # set head one-hots
        if len(self.snake) > 1:
            # head coord - second coord
            dir_vec = tuple(numpy.subtract(self.snake[0], self.snake[1]))
        else:
            dir_vec = self.last_move

        if dir_vec == (0, -1):
            state[state_keys["HEAD"]["UP"]] = 1
        elif dir_vec == (0, 1):
            state[state_keys["HEAD"]["DOWN"]] = 1
        elif dir_vec == (-1, 0):
            state[state_keys["HEAD"]["LEFT"]] = 1
        elif dir_vec == (1, 0):
            state[state_keys["HEAD"]["RIGHT"]] = 1

        # set food_dir
        food_vec = tuple(numpy.subtract(self.snake[0], self.food_pos))
        if food_vec[0] < 0:
            # food on left side
            if food_vec[1] > 0:
                # food above
                state[state_keys["FOOD_DIR"]["UP_LEFT"]] = 1
            elif food_vec[1] < 0:
                # food below
                state[state_keys["FOOD_DIR"]["DOWN_LEFT"]] = 1
            else:
                # same y level as food
                state[state_keys["FOOD_DIR"]["LEFT"]] = 1
        elif food_vec[0] > 0:
            # food on right side
            if food_vec[1] > 0:
                # food above
                state[state_keys["FOOD_DIR"]["UP_RIGHT"]] = 1
            elif food_vec[1] < 0:
                # food below
                state[state_keys["FOOD_DIR"]["DOWN_RIGHT"]] = 1
            else:
                # same y level as food
                state[state_keys["FOOD_DIR"]["RIGHT"]] = 1
        else:
            # food on same x level as food
            if food_vec[1] > 0:
                # food above
                state[state_keys["FOOD_DIR"]["UP"]] = 1
            elif food_vec[1] < 0:
                # food below
                state[state_keys["FOOD_DIR"]["DOWN"]] = 1

        # set tail one-hots
        if len(self.snake) > 1:
            # second-to-last coord - tail coord
            dir_vec = tuple(numpy.subtract(self.snake[-2], self.snake[-1]))
        else:
            dir_vec = self.last_move

        if dir_vec == (0, -1):
            state[state_keys["TAIL"]["UP"]] = 1
        elif dir_vec == (0, 1):
            state[state_keys["TAIL"]["DOWN"]] = 1
        elif dir_vec == (-1, 0):
            state[state_keys["TAIL"]["LEFT"]] = 1
        elif dir_vec == (1, 0):
            state[state_keys["TAIL"]["RIGHT"]] = 1

        # handle sight calculation
        for dir_key in state_keys["SIGHT"]:
            dir_vec = state_keys["SIGHT"][dir_key]["vec"]
            dist = 1
            pos = numpy.add(self.snake[0], dir_vec)
            pos_val = self.grid.get(pos[0], pos[1])

            while pos_val != None and pos_val == 0:
                dist += 1
                pos = numpy.add(pos, dir_vec)
                pos_val = self.grid.get(pos[0], pos[1])

            if not pos_val:
                # we have hit a wall
                state[state_keys["SIGHT"][dir_key]["WALL"]] = 1
            elif pos_val == FOOD:
                # we have seen food!
                state[state_keys["SIGHT"][dir_key]["FOOD"]] = 1
            elif pos_val == SNAKE or pos_val == SNAKE_HEAD:
                # we have seen snake or head somehow (shouldnt be possible)
                state[state_keys["SIGHT"][dir_key]["SNAKE"]] = 1

            state[state_keys["SIGHT"][dir_key]["DIST"]] = dist / self.max_dist

        return state

    def draw_game(self, text):
        if self.graphics:
            self.graphics.draw_background()
            self.graphics.draw_header()
            self.graphics.draw_header_text(text)
            self.graphics.draw_food(self)
            self.graphics.draw_snake(self.snake)
            self.graphics.update()

    def add_food(self):
        new_pos = self.empty_spaces.choose_random_item()
        self.empty_spaces.remove_item(new_pos)
        self.grid.set(*new_pos, FOOD)
        return new_pos

    def move_snake(self, pos_vec):
        # pos_vec is tuple position vector ex: (1,0)
        curr_head = self.snake[0]
        self.last_move = pos_vec
        proj_pos = (curr_head[0] + pos_vec[0], curr_head[1] + pos_vec[1])
        if proj_pos == curr_head:
            return self.get_state_adv(), self.death_reward, True
        if (
            (proj_pos[0] < self.grid.width)
            and (proj_pos[0] >= 0)
            and (proj_pos[1] < self.grid.height)
            and (proj_pos[1] >= 0)
        ):
            proj_val = self.grid.get(*proj_pos)
            if (
                (proj_val != SNAKE) or (proj_pos == self.snake[-1])
            ) and proj_val != SNAKE_HEAD:
                # update snake head
                self.snake.appendleft(proj_pos)
                self.grid.set(*proj_pos, SNAKE_HEAD)
                if len(self.snake) > 1:
                    self.grid.set(*self.snake[1], SNAKE)

                # check for food in next space
                if proj_val == FOOD:
                    self.score += 1
                    if len(self.empty_spaces) == 0:
                        return self.get_state_adv(), self.win_reward, True
                    else:
                        self.food_pos = self.add_food()
                        return self.get_state_adv(), self.food_reward, False
                else:
                    # remove last element
                    last_pos = self.snake.pop()
                    if proj_val != SNAKE:
                        self.empty_spaces.remove_item(proj_pos)
                        self.empty_spaces.add_item(last_pos)
                        self.grid.set(*last_pos, EMPTY)
            else:
                # game over :(
                return self.get_state_adv(), self.death_reward, True
        else:
            # game over :(
            return self.get_state_adv(), self.death_reward, True
        return self.get_state_adv(), self.def_reward, False


def main():
    g = Game(size=(20, 20), graphics=True)
    g.draw_game("Score: " + str(g.score))

    next_move = (1, 0)
    clock = pygame.time.Clock()

    setup = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                setup = True
                if event.key == pygame.K_LEFT:
                    next_move = (-1, 0)
                elif event.key == pygame.K_UP:
                    next_move = (0, -1)
                elif event.key == pygame.K_DOWN:
                    next_move = (0, 1)
                elif event.key == pygame.K_RIGHT:
                    next_move = (1, 0)

        if setup:
            _, _, out = g.move_snake(next_move)
            if out:
                g.reset()
                setup = False

        g.draw_game("Score: " + str(g.score))

        
        # print(tuple(numpy.subtract(g.snake[0], g.food_pos)))
        clock.tick(5)


if __name__ == "__main__":
    main()
