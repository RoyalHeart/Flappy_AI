import pygame
import random
import keyboard
import math
import numpy as np

pygame.init()
isRun = True
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 20)
width = 600
height = 800

screen = pygame.display.set_mode((width, height))
icon = pygame.image.load('ProgramPython/SelfProject/PyGame/FlappyAI/Icon.png')
pygame.display.set_icon(icon)
pygame.display.set_caption('Flappy AI')


class Pipe:
    def __init__(self):
        self.width = 200
        self.height = 70
        self.speed = 5
        self.x = width
        self.y = random.randint(0, height - self.height)

    def display(self):
        self.hitboxTop = (self.x, 0, self.width, self.y)
        self.hitboxBot = (self.x, self.y + self.height, self.width,
                          height - self.y - self.height)
        self.boxTop = pygame.Rect(self.x, 0, self.width, self.y)
        self.boxBot = pygame.Rect(self.x, self.y + self.height, self.width,
                                  height - self.y - self.height)
        pygame.draw.rect(screen, "WHITE", self.hitboxTop, 10)
        pygame.draw.rect(screen, "WHITE", self.hitboxBot, 10)

    def update(self):
        self.x -= self.speed
        if (self.x < -150):
            self.respawn()

    def respawn(self):
        self.x = width
        self.y = random.randint(0, height - self.height)


def sigmoid(x):
    return float(1 / (1 + math.exp(-x)))


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


class FlappyAI:
    def __init__(self):
        self.x = 100
        self.y = height / 2
        self.v = 0
        self.g = 0.9
        self.radius = 32
        self.upSpeed = 6
        self.lengthUp = 0
        self.lengthDown = 0
        self.score = 0
        self.input = np.array([self.lengthUp, self.lengthDown],
                              dtype=np.float32)
        self.box = pygame.Rect(self.x, self.y, self.radius, self.radius)
        self.weightsI_L = np.random.randint(low=0, high=10, size=[4, 2])
        self.weightsL_O = np.random.randint(low=0, high=10, size=[1, 4])
        self.weightsI_L = self.weightsI_L * 0.1
        self.weightsL_O = self.weightsL_O * 0.1

        self.layer = np.empty([4, 1], dtype=np.float32)
        self.output = np.empty([1], dtype=np.float32)

    def display(self):
        pygame.draw.rect(screen, "WHITE",
                         (self.x, self.y, self.radius, self.radius), 1)

    def displayScore(self):
        score = font.render(f'{round(self.score)}', True, 'WHITE')
        screen.blit(score, (width / 2, height / 2))

    def update(self):
        self.box = pygame.Rect(self.x, self.y, self.radius, self.radius)
        self.v += self.g
        self.v *= self.g
        self.y += self.v
        self.lengthUp = math.sqrt(
            pow(self.x - pipe.x, 2) + pow(self.y - pipe.y, 2))
        self.lengthDown = math.sqrt(
            pow(self.x - pipe.x, 2) + pow(self.y - pipe.y - pipe.height, 2))
        self.input = np.array([self.lengthUp, self.lengthDown],
                              dtype=np.float32)

    def respawn(self):
        self.y = height / 2
        self.weightsI_L = np.random.randint(low=0, high=10000, size=[4, 2])
        self.weightsL_O = np.random.randint(low=0, high=10000, size=[1, 4])
        self.weightsI_L = self.weightsI_L * 0.0001
        self.weightsL_O = self.weightsL_O * 0.0001

    def __del__(self):
        return 0

    def hit(self):
        if (self.y >= height - self.radius):
            self.y = height - self.radius
        if (self.y < 0):
            self.y = 0
        if (self.box.colliderect(pipe.boxTop)):
            self.score = 0
            return True
        if (self.box.colliderect(pipe.boxBot)):
            self.score = 0
            return True

    def passPipe(self):
        if (self.x == pipe.x + pipe.width):
            self.score += 1

    def sensorDisplay(self):
        pygame.draw.line(screen, 'White', (self.x, self.y), (pipe.x, pipe.y),
                         1)
        pygame.draw.line(screen, 'White', (self.x, self.y),
                         (pipe.x, pipe.y + pipe.height), 1)
        # draw input circles
        pygame.draw.circle(screen, 'WHITE', (30, 40), 15, 1)
        pygame.draw.circle(screen, 'WHITE', (30, 80), 15, 1)
        # draw layer circles
        pygame.draw.circle(screen, 'WHITE', (150, 20), 15, 1)
        pygame.draw.circle(screen, 'WHITE', (150, 50), 15, 1)
        pygame.draw.circle(screen, 'WHITE', (150, 80), 15, 1)
        pygame.draw.circle(screen, 'WHITE', (150, 110), 15, 1)
        # draw output circles
        pygame.draw.circle(screen, 'WHITE', (270, 60), 15, 1)
        # draw lines from input to layer
        pygame.draw.line(screen, 'WHITE', (30, 40), (150, 20),
                         round(2 * self.weightsI_L[0][0]))
        pygame.draw.line(screen, 'WHITE', (30, 40), (150, 50),
                         round(2 * self.weightsI_L[1][0]))
        pygame.draw.line(screen, 'WHITE', (30, 40), (150, 80),
                         round(2 * self.weightsI_L[2][0]))
        pygame.draw.line(screen, 'WHITE', (30, 40), (150, 110),
                         round(2 * self.weightsI_L[3][0]))
        pygame.draw.line(screen, 'WHITE', (30, 80), (150, 20),
                         round(2 * self.weightsI_L[0][1]))
        pygame.draw.line(screen, 'WHITE', (30, 80), (150, 50),
                         round(2 * self.weightsI_L[1][1]))
        pygame.draw.line(screen, 'WHITE', (30, 80), (150, 80),
                         round(2 * self.weightsI_L[2][1]))
        pygame.draw.line(screen, 'WHITE', (30, 80), (150, 110),
                         round(2 * self.weightsI_L[3][1]))
        # draw lines from layer to output
        pygame.draw.line(screen, 'WHITE', (150, 20), (270, 60),
                         round(2 * self.weightsL_O[0][0]))
        pygame.draw.line(screen, 'WHITE', (150, 50), (270, 60),
                         round(2 * self.weightsL_O[0][1]))
        pygame.draw.line(screen, 'WHITE', (150, 80), (270, 60),
                         round(2 * self.weightsL_O[0][2]))
        pygame.draw.line(screen, 'WHITE', (150, 110), (270, 60),
                         round(2 * self.weightsL_O[0][3]))
        distanceUP = font.render(f'{round(self.lengthUp)}', True, 'WHITE')
        distanceDown = font.render(f'{round(self.lengthDown)}', True, 'WHITE')
        screen.blit(distanceUP, (21, 33))
        screen.blit(distanceDown, (21, 73))

    def feedforward(self):
        self.layer = np.dot(self.input, self.weightsI_L.transpose())
        self.layer = softmax(self.layer)
        for i in range(len(self.layer)):
            self.layer[i] = round(self.layer[i], 2)
            # self.layer[i] = sigmoid(self.layer[i])

        self.output = np.dot(self.layer, self.weightsL_O.transpose())
        # self.output = softmax(self.output)
        self.output[0] = round(self.output[0], 2)

        # print(self.layer)
        # self.ouput = np.dot(self.layer, self.weightsL_O.transpose())
        # self.ouput = sigmoid(self.output[0])
        # print(self.output[0])
        print(self.output)
        return self.output

    def backward(self):
        if (self.lengthUp + self.lengthDown < 1000):
            return False

    def move(self):
        self.v -= self.upSpeed


pipe = Pipe()


def pipeDisplay():
    pipe.display()
    pipe.update()


def crossover(player1=FlappyAI, player2=FlappyAI(), crossover_rate=0.9):
    player1_new = player1.copy()
    player2_new = player2.copy()

    for i in range(4):
        for j in range(2):
            if random.random() < crossover_rate:
                player1_new.weightsI_L[i][j] = player2.weightsI_L[i][j]
                player2_new.weightsI_L[i][j] = player1.weightsI_L[i][j]

    for i in range(4):
        if random.random() < crossover_rate:
            player1_new.weightsL_O[i] = player2.weightsL_O[i]
            player2_new.weightsL_O[i] = player1.weightsL_O[i]

    return player1_new, player2_new


def mutate(player=FlappyAI(), mutation_rate=0.05):
    playerMutate = player.copy()

    for i in range(4):
        for j in range(2):
            if random.random() < mutation_rate:
                playerMutate.weightsI_L[i][j] = random.random()
    for i in range(4):
        if random.random() < mutation_rate:
            playerMutate.weightsL_O[i] = random.random()

    return playerMutate


def selection(sorted_old_population):
    index1 = random.randint(0, m - 1)
    while True:
        index2 = random.randint(0, m - 1)
        if index2 != index1:
            break

    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s


m = 10
players = []
for _ in range(10):
    players.append(FlappyAI())
fitnesses = [player.score for player in players]


def compute_fitness(individual):
    return individual.score


def create_new_population(old_population, elitism=2, gen=1):
    sorted_population = sorted(old_population, key=lambda player: player.score)

    if gen % 1 == 0:
        fitnesses.append(compute_fitness(sorted_population[m - 1]))
        print("BEST:", compute_fitness(sorted_population[m - 1]))

    new_population = []
    while len(new_population) < m - elitism:
        # selection
        individual_s1 = selection(sorted_population)
        individual_s2 = selection(sorted_population)  # duplication

        # crossover
        individual_c1, individual_c2 = crossover(individual_s1, individual_s2)

        # mutation
        individual_m1 = mutate(individual_c1)
        individual_m2 = mutate(individual_c2)

        new_population.append(individual_m1)
        new_population.append(individual_m2)

    for ind in sorted_population[m - elitism:]:
        new_population.append(ind.copy())

    return new_population


player = FlappyAI()


def flappyAIDisplay():
    if (not (player.hit())):
        player.update()
        player.displayScore()
        player.display()
        player.passPipe()
        player.sensorDisplay()
        player.feedforward()
        if (player.output > 0.5):
            player.move()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                player.move()


def flappysAIDisplay():
    i = 0
    while i < len(players):
        if (not (players[i].hit())):
            players[i].displayScore()
            players[i].update()
            players[i].display()
            players[i].passPipe()
            players[i].sensorDisplay()
            players[i].feedforward()
            if (players[i].output > 0.5):
                players[i].move()
            i += 1
        elif (i < len(players)):
            del players[i]
            i -= 1
        i += 1
        if (len(players) == 1):
            np.save(f'ProgramPython/SelfProject/PyGame/FlappyAI/weightI_L.npy',
                    players[i - 1].weightsI_L)
            np.save(f'ProgramPython/SelfProject/PyGame/FlappyAI/weightL_O.npy',
                    players[i - 1].weightsL_O)


while isRun:
    # setup fps, background
    dt = (clock.tick(60))
    fps = font.render(f'FPS: {dt}', True, 'WHITE')
    screen.fill('black')
    screen.blit(fps, (10, 10))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRun = False

    pipeDisplay()
    flappysAIDisplay()
    if (len(players) == 0):
        players = [FlappyAI() for _ in range(m)]
    pygame.display.update()
