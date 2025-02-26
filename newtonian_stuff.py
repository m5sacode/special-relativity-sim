import numpy as np
import matplotlib.pyplot as plt
import pygame
from scipy.constants import speed_of_light

WHITE = (255, 255, 255)
LTGREY = (150,150,150)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
LTYELLOW = (150, 150, 0)
GREY = (100, 100, 100)
BLUE = (0, 0, 255)
LTBLUE = (0, 0, 150)

class World:
    def __init__(self, scaling_factor, speedoflight=speed_of_light, screen_size=(800, 600)):
        self.scaling_factor = scaling_factor
        self.speedoflight = speedoflight
        pygame.init()
        self.screen_WIDTH, self.screen_HEIGHT = screen_size
        self.screen = pygame.display.set_mode((self.screen_WIDTH, self.screen_HEIGHT))
        pygame.display.set_caption("Your 1D pocket world")
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 24)
        self.particles = []
        self.photons = []
        self.photon_emiters = []
        self.time = 0
        self.reference_frame_velocity = 0.0

        self.meters_per_pixel = (self.speedoflight * self.scaling_factor) / self.screen_WIDTH
        self.slider_rect = pygame.Rect(50, self.screen_HEIGHT - 50, 300, 10)

    def draw(self):
        self.screen.fill(BLACK)
        pygame.draw.line(self.screen, GREY, (0, self.screen_HEIGHT // 2), (self.screen_WIDTH, self.screen_HEIGHT // 2), 2)
        meters_text = self.font.render(f"{self.meters_per_pixel:.2e} m/px", True, WHITE)
        speed_text = self.font.render(f"v = {self.reference_frame_velocity:.2e} m/s", True, WHITE)
        self.screen.blit(meters_text, (15, 15))
        self.screen.blit(speed_text, (15, 40))
        pygame.draw.rect(self.screen, WHITE, self.slider_rect)
        slider_x = int(50 + (self.reference_frame_velocity + self.speedoflight) / (2 * self.speedoflight) * 300)
        pygame.draw.circle(self.screen, RED, (slider_x, self.screen_HEIGHT - 45), 8)

        for particle in self.particles:
            particle.draw(self.screen)
        for photon in self.photons:
            photon.draw(self.screen)
        for photon_emiter in self.photon_emiters:
            photon_emiter.draw(self.screen)
        pygame.display.flip()

    def update(self, dt):
        self.time += dt
        for particle in self.particles:
            particle.update(dt, self.meters_per_pixel, self.reference_frame_velocity)
        for photon in self.photons:
            photon.update(dt, self.meters_per_pixel, self.reference_frame_velocity)
        for photon_emiter in self.photon_emiters:
            photon_emiter.update(dt, self.meters_per_pixel, self.reference_frame_velocity, self.time)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.slider_rect.collidepoint(event.pos):
            self.update_slider(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
            self.update_slider(event.pos[0])

    def update_slider(self, mouse_x):
        t = max(0, min(1, (mouse_x - 50) / 300))
        self.reference_frame_velocity = (t * 2 - 1) * self.speedoflight

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_event(event)
            self.update(dt)
            self.draw()
        pygame.quit()

    class Particle:
        def __init__(self, world, setup_position, velocity):
            self.position = setup_position
            self.velocity = velocity
            self.world = world

        def update(self, dt, meters_per_pixel, reference_frame_velocity):
            adjusted_velocity = self.velocity - reference_frame_velocity
            self.position = (self.position + (adjusted_velocity * dt) / meters_per_pixel) % self.world.screen_WIDTH

        def draw(self, screen):
            adjusted_velocity = self.velocity - self.world.reference_frame_velocity
            if abs(adjusted_velocity) < 1e-5:
                x1 = x2 = self.position
            else:
                slope = -self.world.speedoflight / adjusted_velocity
                delta_y = self.world.screen_HEIGHT
                delta_x = delta_y / slope
                x1 = self.position - delta_x / 2
                x2 = self.position + delta_x / 2
            pygame.draw.line(screen, LTGREY, (x1, 0), (x2, self.world.screen_HEIGHT), 2)
            pygame.draw.circle(screen, WHITE, (int(self.position), self.world.screen_HEIGHT // 2), 5)

    class Photon:
        def __init__(self, world, emission_position, direction_of_motion):
            self.position = emission_position
            self.world = world
            self.velocity = direction_of_motion * world.speedoflight

        def update(self, dt, meters_per_pixel, reference_frame_velocity):
            adjusted_velocity = self.velocity - reference_frame_velocity
            self.position = (self.position + (adjusted_velocity * dt) / meters_per_pixel) % self.world.screen_WIDTH

        def draw(self, screen):
            adjusted_velocity = self.velocity - self.world.reference_frame_velocity
            if abs(adjusted_velocity) < 1e-5:
                x1 = x2 = self.position
            else:
                slope = -self.world.speedoflight / adjusted_velocity
                delta_y = self.world.screen_HEIGHT
                delta_x = delta_y / slope
                x1 = self.position - delta_x / 2
                x2 = self.position + delta_x / 2
            pygame.draw.line(screen, LTYELLOW, (x1, 0), (x2, self.world.screen_HEIGHT), 2)
            pygame.draw.circle(screen, YELLOW, (int(self.position), self.world.screen_HEIGHT // 2), 5)
    class Photon_emiter:
        def __init__(self, world, setup_position, velocity, period):
            self.position = setup_position
            self.velocity = velocity
            self.world = world
            self.emission_interval = period
            self.last_emission_time = -period

        def update(self, dt, meters_per_pixel, reference_frame_velocity, current_time):
            adjusted_velocity = self.velocity - reference_frame_velocity
            self.position = (self.position + (adjusted_velocity * dt) / meters_per_pixel) % self.world.screen_WIDTH
            if current_time - self.last_emission_time >= self.emission_interval:
                self.world.photons.append(World.Photon(self.world, self.position, 1))
                self.world.photons.append(World.Photon(self.world, self.position, -1))
                self.last_emission_time = current_time

        def draw(self, screen):
            adjusted_velocity = self.velocity - self.world.reference_frame_velocity
            if abs(adjusted_velocity) < 1e-5:
                x1 = x2 = self.position
            else:
                slope = -self.world.speedoflight / adjusted_velocity
                delta_y = self.world.screen_HEIGHT
                delta_x = delta_y / slope
                x1 = self.position - delta_x / 2
                x2 = self.position + delta_x / 2
            pygame.draw.line(screen, LTBLUE, (x1, 0), (x2, self.world.screen_HEIGHT), 2)
            pygame.draw.circle(screen, BLUE, (int(self.position), self.world.screen_HEIGHT // 2), 5)


world = World(scaling_factor=20)
# particle = world.Particle(world, setup_position=100, velocity=0.5*world.speedoflight)
# photon = world.Photon(world, emission_position=200, direction_of_motion=1)
emitter = world.Photon_emiter(world, setup_position=500, velocity=0.2*world.speedoflight, period=100)
# world.particles.append(particle)
# world.photons.append(photon)
world.photon_emiters.append(emitter)
world.run()
