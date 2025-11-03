import pygame
from scipy.constants import speed_of_light
import numpy as np

WHITE = (255, 255, 255)
LTGREY = (150,150,150)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
LTYELLOW = (150, 150, 0)
GREY = (100, 100, 100)
BLUE = (0, 0, 255)
LTBLUE = (0, 0, 150)
GREEN = (0, 255, 0)
LTGREEN = (0, 150, 0)

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
        self.photon_sensors =[]
        self.time = 0
        self.reference_frame_velocity = 0.0

        self.meters_per_pixel = (self.speedoflight * self.scaling_factor) / self.screen_WIDTH
        self.slider_rect = pygame.Rect(50, self.screen_HEIGHT - 50, 300, 10)

    def draw(self):
        self.screen.fill(BLACK)
        pygame.draw.line(self.screen, GREY, (0, self.screen_HEIGHT // 2), (self.screen_WIDTH, self.screen_HEIGHT // 2), 2)
        meters_text = self.font.render(f"{self.meters_per_pixel:.2e} m/px", True, WHITE)
        speed_text = self.font.render(f"v = {self.reference_frame_velocity/self.speedoflight:.2} c", True, WHITE)
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
        for photon_sensor in self.photon_sensors:
            photon_sensor.draw(self.screen, self.time)
        pygame.display.flip()

    def update(self, dt):
        self.time += dt
        for particle in self.particles:
            particle.update(dt, self.meters_per_pixel, self.reference_frame_velocity)
        for photon in self.photons:
            photon.update(dt, self.meters_per_pixel, self.reference_frame_velocity)
        for photon_emiter in self.photon_emiters:
            photon_emiter.update(dt, self.meters_per_pixel, self.reference_frame_velocity, self.time)
        for photon_sensor in self.photon_sensors:
            photon_sensor.update(dt, self.meters_per_pixel, self.reference_frame_velocity, self.time)

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

    # Add this helper to World (as a method)
    def to_screen_x(self, position_pixels, t_world):
        """
        Convert stored position in pixels (lab frame) at lab time t_world (s)
        to screen x (pixels) in the current reference frame self.reference_frame_velocity.
        """
        # convert pixel -> meters
        x_m = position_pixels * self.meters_per_pixel
        V = self.reference_frame_velocity
        c = self.speedoflight
        if abs(V) >= c:
            V = np.sign(V) * (c * 0.999999999)  # safety
        gamma = 1.0 / np.sqrt(1 - (V / c) ** 2)
        xprime_m = gamma * (x_m - V * t_world)
        xprime_pixels = xprime_m / self.meters_per_pixel
        # wrap to screen width in pixels
        return xprime_pixels % self.screen_WIDTH

    # ---- Particle ----
    class Particle:
        def __init__(self, world, setup_position, velocity):
            # setup_position is pixels (as in your code)
            self.position = float(setup_position)
            self.velocity = float(velocity)  # in m/s
            self.world = world
            self.speedoflight = world.speedoflight

        def update(self, dt, meters_per_pixel, reference_frame_velocity):
            # Update in lab frame (no Lorentz mixing here)
            delta_px = (self.velocity * dt) / meters_per_pixel
            self.position = (self.position + delta_px) % self.world.screen_WIDTH
            # gamma of particle's own rest (if needed later)
            v = self.velocity
            self.gamma = 1.0 / np.sqrt(1 - (v / self.speedoflight) ** 2)

        def draw(self, screen):
            # compute screen x in the chosen reference frame
            screen_x = int(self.world.to_screen_x(self.position, self.world.time))
            # simple small 'line' to indicate direction using relativistic velocity addition:
            V = self.world.reference_frame_velocity
            c = self.speedoflight
            # velocity addition formula for u' (particle velocity seen in moving frame)
            u = self.velocity
            u_prime = (u - V) / (1 - (u * V) / (c ** 2))
            # draw a short line segment proportional to u_prime (visual cue only)
            length_px = max(2,
                            min(self.world.screen_WIDTH // 4, int(abs(u_prime) * 1e-8)))  # scaling constant for visual
            x1 = screen_x - length_px // 2
            x2 = screen_x + length_px // 2
            pygame.draw.line(screen, LTGREY, (x1, 0), (x2, self.world.screen_HEIGHT), 2)
            pygame.draw.circle(screen, WHITE, (screen_x, self.world.screen_HEIGHT // 2), 5)

    # ---- Photon ----
    class Photon:
        def __init__(self, world, emission_position_in_current_frame, direction_of_motion):
            self.world = world
            self.velocity = direction_of_motion * world.speedoflight  # ±c

            # Get current reference frame velocity
            V = world.reference_frame_velocity
            c = world.speedoflight
            gamma = 1.0 / np.sqrt(1 - (V / c) ** 2)

            # Convert from pixels (current frame) → meters
            x_prime_m = emission_position_in_current_frame * world.meters_per_pixel
            t = world.time

            # Inverse Lorentz transform: current frame → lab frame
            x_lab_m = gamma * (x_prime_m + V * t)

            # Store lab-frame position in pixels (for updates and collisions)
            self.position = (x_lab_m / world.meters_per_pixel) % world.screen_WIDTH

        def update(self, dt, meters_per_pixel, reference_frame_velocity):
            # Photon motion in the LAB frame — never affected by reference frame velocity.
            # Always moves at ±c in lab coordinates.
            delta_px = (self.velocity * dt) / meters_per_pixel
            self.position = (self.position + delta_px) % self.world.screen_WIDTH

        def draw(self, screen):
            # Apply Lorentz transformation to draw photon position in current reference frame.
            V = self.world.reference_frame_velocity
            c = self.world.speedoflight
            if abs(V) >= c:
                V = np.sign(V) * (c * 0.999999999)  # avoid singularities


            # Convert lab position (pixels → meters)
            x_lab = self.position * self.world.meters_per_pixel
            t = self.world.time

            # Lorentz transform (x', t') for current frame
            x_prime_m = x_lab

            # Convert back to pixels for rendering
            x_prime_px = (x_prime_m / self.world.meters_per_pixel) % self.world.screen_WIDTH

            # Draw photon in the transformed frame
            pygame.draw.line(screen, LTYELLOW, (x_prime_px, 0), (x_prime_px, self.world.screen_HEIGHT), 2)
            pygame.draw.circle(screen, YELLOW, (int(x_prime_px), self.world.screen_HEIGHT // 2), 5)

    # ---- Photon_emiter ----
    class Photon_emiter:
        def __init__(self, world, setup_position, velocity, period):
            self.position = float(setup_position)  # pixels
            self.velocity = float(velocity)  # m/s
            self.world = world
            self.emission_interval = period  # if this is proper period (emitter rest), we will dilate
            self.speedoflight = world.speedoflight
            self.last_emission_time = -period

        def update(self, dt, meters_per_pixel, reference_frame_velocity, current_time):
            # Update emitter position in lab frame
            delta_px = (self.velocity * dt) / meters_per_pixel
            self.position = (self.position + delta_px) % self.world.screen_WIDTH
            # compute gamma for emitter relative to lab
            v = self.velocity
            self.gamma = 1.0 / np.sqrt(1 - (v / self.speedoflight) ** 2)
            # If period is proper period (in emitter rest frame) then lab interval is gamma * period
            lab_interval = self.gamma * self.emission_interval
            if current_time - self.last_emission_time >= lab_interval:
                # emit photons at emitter's current lab-frame position
                self.world.photons.append(World.Photon(self.world, self.position, 1))
                self.world.photons.append(World.Photon(self.world, self.position, -1))
                self.last_emission_time = current_time

        def draw(self, screen):
            screen_x = int(self.world.to_screen_x(self.position, self.world.time))
            pygame.draw.line(screen, LTBLUE, (screen_x, 0), (screen_x, self.world.screen_HEIGHT), 2)
            pygame.draw.circle(screen, BLUE, (screen_x, self.world.screen_HEIGHT // 2), 5)

    # ---- PhotonSensor ----
    class PhotonSensor:
        def __init__(self, world, setup_position, velocity, detection_radius=10, activation_time=0.5):
            self.position = float(setup_position)
            self.velocity = float(velocity)
            self.world = world
            self.detection_radius = detection_radius
            self.activated_until = 0.0
            self.activation_time = activation_time
            self.speedoflight = world.speedoflight

        def update(self, dt, meters_per_pixel, reference_frame_velocity, current_time):
            # update sensor position in lab frame
            delta_px = (self.velocity * dt) / meters_per_pixel
            self.position = (self.position + delta_px) % self.world.screen_WIDTH
            # detection: compare pixel distance in lab frame (or use transformed positions; here we keep it simple)
            photons_to_remove = []
            for photon in self.world.photons:
                # use lab-frame positions for detection radius (consistent)
                if abs(self.position - photon.position) < self.detection_radius:
                    self.activated_until = current_time + self.activation_time
                    photons_to_remove.append(photon)
            for p in photons_to_remove:
                try:
                    self.world.photons.remove(p)
                except ValueError:
                    pass

        def draw(self, screen, current_time):
            screen_x = int(self.world.to_screen_x(self.position, self.world.time))
            color = GREEN if current_time < self.activated_until else WHITE
            pygame.draw.line(screen, LTGREEN, (screen_x, 0), (screen_x, self.world.screen_HEIGHT), 2)
            pygame.draw.circle(screen, color, (screen_x, self.world.screen_HEIGHT // 2), 5)
world = World(scaling_factor=40)
# particle = world.Particle(world, setup_position=100, velocity=0.5*world.speedoflight)
# photon = world.Photon(world, emission_position=200, direction_of_motion=1)

sensor_arrangement_speed = 0.*world.speedoflight
sensor1 = world.PhotonSensor(world, setup_position=580, velocity=sensor_arrangement_speed)
sensor2 = world.PhotonSensor(world, setup_position=420, velocity=sensor_arrangement_speed)
emitter = world.Photon_emiter(world, setup_position=500, velocity=sensor_arrangement_speed, period=5)
# world.particles.append(particle)
# world.photons.append(photon)
world.photon_emiters.append(emitter)
world.photon_sensors.append(sensor1)
world.photon_sensors.append(sensor2)
world.reference_frame_velocity = 0.*world.speedoflight
world.run()
