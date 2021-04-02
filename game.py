import pygame
import random
import sys
from collections import deque

# color constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0 )
BLUE = (0, 0, 128)
LIGHTGREEN = (0, 255, 0 )

EMPTY = 0
SNAKE_HEAD = 1
SNAKE = 2
FOOD = 3

class ListDict(object):
    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove_item(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return random.choice(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

# COLORS = BACK_COLOR, SNAKE_COLOR, EMPTY_COLOR, FOOD_COLOR = ((255,255,255), )

class Graphics:
    def __init__(self, window_size=None, grid_size=None, margin=1):
        pygame.init()

        # set window size
        if not window_size:
            self.window_size = self.window_width, self.window_height = 421,421
        else:
            self.window_size = self.window_width, self.window_height = window_size

        # set grid size
        if not grid_size:
            self.grid_size = self.grid_width, self.grid_height = 20,20
        else:
            self.grid_size = self.grid_width, self.grid_height = grid_size


        # set margin
        self.margin = margin

        # set square size
        self.square_size = self.square_width, self.square_height = (self.window_width-self.margin)/(self.grid_width+self.margin), (self.window_height-self.margin)/(self.grid_height+self.margin)


        self.screen = pygame.display.set_mode(self.window_size)

    def gen_rect(self, x, y):
        return [(self.margin + self.square_width) * x + self.margin,
                  (self.margin + self.square_height) * y + self.margin,
                  self.square_width,
                  self.square_height]

    def draw_square(self, x, y, color):
        if self.screen:
            pygame.draw.rect(self.screen, color, self.gen_rect(x, y))

    def draw_background(self):
        if self.screen:
            self.screen.fill(WHITE)

    def draw_game(self, game):
        if self.screen:
            for x in range(0, game.grid.width):
                for y in range(0, game.grid.height):
                    type = game.grid.get(x, y)
                    if type == EMPTY:
                        self.draw_square(x, y, BLACK)
                    elif type == SNAKE_HEAD or type == SNAKE:
                        self.draw_square(x, y, LIGHTGREEN)
                    elif type == FOOD:
                        self.draw_square(x, y, RED)

    def draw_snake(self, snake_coords):
        if snake_coords:
            for i in range(0, len(snake_coords)):
                draw_square(*snake_coords[i])
                # draw_face(snake_coords[])

    def update(self):
        pygame.display.flip()


class Grid:
    # initialize board of width and height
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = [[EMPTY for i in range(0, width)] for i in range(0, height)]

    # indexing is based on board[y][x]
    def get(self, x, y):
        if y>=0 and y<self.height:
            if x>=0 and x<self.width:
                return self.board[y][x]
        return None

    def set_val(self, x, y, val):
        if y>=0 and y<self.height:
            if x>=0 and x<self.width:
                self.board[y][x] = val

class Game:
    def __init__(self, size=(20,20), graphics=False):
        self.score = 0
        # snake queue [HEAD, ...]
        self.snake = deque()
        self.grid = Grid(*size)
        self.graphics = None
        if graphics:
            self.graphics = Graphics(grid_size = size)

        self.empty_spaces = ListDict()
        for row in range(0, self.grid.width):
            for col in range(0, self.grid.height):
                self.empty_spaces.add_item((row, col))

        self.empty_spaces.remove_item((0,0))
        self.grid.set_val(0, 0, SNAKE_HEAD)
        self.snake.appendleft((0,0))

        self.add_food()

        self.action_space_n = 4

    def reset(self, size=(20,20)):
        self.score = 0
        # snake queue [HEAD, ...]
        self.snake = deque()
        self.grid = Grid(*size)

        self.empty_spaces = ListDict()
        for row in range(0, self.grid.width):
            for col in range(0, self.grid.height):
                self.empty_spaces.add_item((row, col))

        self.empty_spaces.remove_item((0,0))
        self.grid.set_val(0, 0, SNAKE_HEAD)
        self.snake.appendleft((0,0))

        self.add_food()
        return str(self.grid.board)

    def sample_action_space(self):
        return random.choice([(0,1),(0,-1),(-1,0),(1,0)])

    def draw_game(self):
        if self.graphics:
            self.graphics.draw_background()
            self.graphics.draw_game(self)
            self.graphics.update()

    def add_food(self):
        new_pos = self.empty_spaces.choose_random_item()
        self.empty_spaces.remove_item(new_pos)
        self.grid.set_val(*new_pos, FOOD)

    def move_snake(self, pos_vec):
        # pos_vec is tuple position vector ex: (1,0)
        curr_head = self.snake[0]
        proj_pos = (curr_head[0] + pos_vec[0], curr_head[1] + pos_vec[1])
        if proj_pos == curr_head:
            return str(self.grid.board), -300, True
        if ((proj_pos[0] < self.grid.width) and
                (proj_pos[0] >= 0) and
                (proj_pos[1] < self.grid.height) and
                (proj_pos[1] >= 0)):
            proj_val = self.grid.get(*proj_pos)
            if ((proj_val != SNAKE) or (proj_pos == self.snake[-1])) and proj_val != SNAKE_HEAD:
                # update snake head
                self.snake.appendleft(proj_pos)
                self.grid.set_val(*proj_pos, SNAKE_HEAD)
                if len(self.snake) > 1:
                    self.grid.set_val(*self.snake[1], SNAKE)

                # check for food in next space
                if proj_val == FOOD:
                    self.score += 1
                    if len(self.empty_spaces) == 0:
                        return str(self.grid.board), 500, True
                    else:
                        self.add_food()
                        return str(self.grid.board), 100, False
                else:
                    # remove last element
                    last_pos = self.snake.pop()
                    if proj_val != SNAKE:
                        self.empty_spaces.remove_item(proj_pos)
                        self.empty_spaces.add_item(last_pos)
                        self.grid.set_val(*last_pos, EMPTY)
            else:
                # game over :(
                return str(self.grid.board), -300, True
        else:
            # game over :(
            return str(self.grid.board), -300, True
        return str(self.grid.board), -1, False


def main():
    g = Game(size=(3,3),graphics=True)
    g.draw_game()

    next_move = (1,0)
    clock = pygame.time.Clock()
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    next_move = (-1,0)
                elif event.key == pygame.K_UP:
                    next_move = (0,-1)
                elif event.key == pygame.K_DOWN:
                    next_move = (0,1)
                elif event.key == pygame.K_RIGHT:
                    next_move = (1,0)

        _, _, out = g.move_snake(next_move)
        g.draw_game()

        if out:
            sys.exit()

        clock.tick(5)

if __name__ == "__main__":
    main()
