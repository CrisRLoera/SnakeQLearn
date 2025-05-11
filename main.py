import pygame
import random
import numpy as np
import sys

# Constantes
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
COLS, ROWS = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
FPS = 60
MAX_EPISODES = 500
LEARNING_RATE = 0.2
DISCOUNT = 0.95
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.01
RELATIVE_ACTIONS = [0, 1, 2]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

pygame.init()
font = pygame.font.SysFont("Arial", 24)

def turn_left(dir): return [2, 3, 1, 0][dir]
def turn_right(dir): return [3, 2, 0, 1][dir]

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = 1.0

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 3
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 3
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 3
        old_value = self.q_table[state][action]
        future = max(self.q_table[next_state])
        new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT * future)
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

class Snake:
    def __init__(self):
        self.body = [(COLS // 2, ROWS // 2)]
        self.direction = random.randint(0, 3)

    def apply_action(self, action):
        if action == 0:
            return self.direction
        elif action == 1:
            return turn_left(self.direction)
        else:
            return turn_right(self.direction)

    def move(self, direction):
        dx, dy = ACTIONS[direction]
        new_head = (self.body[0][0] + dx, self.body[0][1] + dy)
        return new_head

    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < COLS and 0 <= y < ROWS and pos not in self.body

    def get_state(self, food):
        head = self.body[0]
        left = self.move(turn_left(self.direction))
        right = self.move(turn_right(self.direction))
        front = self.move(self.direction)
        return (
            int(not self.is_valid(left)),
            int(not self.is_valid(right)),
            int(not self.is_valid(front)),
            int(food[0] < head[0]),
            int(food[0] > head[0]),
            int(food[1] < head[1]),
            int(food[1] > head[1]),
        )

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Q-Learning")
        self.clock = pygame.time.Clock()
        self.agent = QLearningAgent()

    def place_food(self, snake_body):
        while True:
            food = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
            if food not in snake_body:
                return food

    def draw(self, snake, food, score, episode):
        self.screen.fill((0, 0, 0))
        for s in snake.body:
            pygame.draw.rect(self.screen, (0, 255, 0), (s[0]*GRID_SIZE, s[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, (255, 0, 0), (food[0]*GRID_SIZE, food[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
        text = font.render(f"Score: {score}  Ep: {episode+1}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        pygame.display.flip()

    def run(self):
        for episode in range(MAX_EPISODES):
            snake = Snake()
            food = self.place_food(snake.body)
            score = 0
            done = False

            while not done:
                self.draw(snake, food, score, episode)
                state = snake.get_state(food)
                action = self.agent.get_action(state)
                new_dir = snake.apply_action(action)
                new_head = snake.move(new_dir)

                if not snake.is_valid(new_head):
                    reward = -10
                    done = True
                elif new_head == food:
                    reward = 10
                    score += 1
                    snake.body.insert(0, new_head)
                    food = self.place_food(snake.body)
                else:
                    reward = -0.1
                    snake.body.insert(0, new_head)
                    snake.body.pop()

                next_state = snake.get_state(food)
                self.agent.update(state, action, reward, next_state)
                snake.direction = new_dir

                self.clock.tick(FPS)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            self.agent.decay_epsilon()

        pygame.quit()

# Ejecutar el juego
if __name__ == "__main__":
    SnakeGame().run()
