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
    def __init__(self, scaling_factor, speedoflight=speed_of_light, screen_size=(1000, 600)):
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
        V = self.reference_frame_velocity
        c = self.speedoflight
        self.gamma = 1.0 / np.sqrt(1 - (V / c) ** 2)
        self.time += (dt / self.gamma)
        self.time += (dt)
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

    def _display_x_to_lab_px(self, x_display_px, t_world):
        """
        Convert x' (display/current-frame pixels) at lab time t_world -> lab-frame pixels.
        Inverse Lorentz: x = gamma * (x' + V t)
        """
        V = self.reference_frame_velocity
        c = self.speedoflight
        if abs(V) >= c:
            V = np.sign(V) * (c * 0.999999999)
        gamma = 1.0 / np.sqrt(1 - (V / c) ** 2)

        xprime_m = x_display_px * self.meters_per_pixel
        x_lab_m = gamma * (xprime_m + V * t_world)
        return (x_lab_m / self.meters_per_pixel) % self.screen_WIDTH

    def _vel_display_to_lab(self, u_display):
        """
        Convert a velocity u' given in the current/display frame -> lab-frame velocity u.
        Formula: u = (u' + V) / (1 + u' V / c^2)
        All inputs/outputs in m/s.
        """
        V = self.reference_frame_velocity
        c = self.speedoflight
        denom = 1.0 + (u_display * V) / (c ** 2)
        if abs(denom) < 1e-15:
            # pathological near-light denominator: clamp to keep things safe
            return np.sign(u_display + V) * (c * 0.999999999)
        return (u_display + V) / denom

    # ---- Particle ----
    class Particle:
        def __init__(self, world, setup_position, velocity):
            # setup_position is pixels (as in your code)
            self.position = float(setup_position)
            self.velocity = float(velocity)  # in m/s
            self.world = world
            self.speedoflight = world.speedoflight

        def update(self, dt, meters_per_pixel, reference_frame_velocity):
            # gamma of particle's own rest (if needed later)
            v = self.velocity
            self.gamma = 1.0 / np.sqrt(1 - (v / self.speedoflight) ** 2)
            # Update in lab frame (no Lorentz mixing here)
            dt = (dt/self.gamma)

            delta_px = (self.velocity * dt) / meters_per_pixel
            self.position = (self.position + delta_px) % self.world.screen_WIDTH

        def draw(self, screen):
            screen_x = self.world.to_screen_x(self.position, self.world.time)
            V = self.world.reference_frame_velocity
            c = self.speedoflight
            # relativistic velocity addition: u' = (u - V) / (1 - u V / c^2)
            u = self.velocity
            denom = 1.0 - (u * V) / (c ** 2)
            if abs(denom) < 1e-12:
                u_prime = np.sign(u - V) * (c * 0.999999999)
            else:
                u_prime = (u - V) / denom

            if abs(u_prime) < 1e-9:
                x1 = x2 = screen_x
            else:
                if abs(u_prime) >= c:
                    u_prime = np.sign(u_prime) * (c * 0.999999999)
                slope = -c / u_prime
                delta_y = self.world.screen_HEIGHT
                delta_x = delta_y / slope
                x1 = screen_x - delta_x / 2
                x2 = screen_x + delta_x / 2

            pygame.draw.line(screen, LTGREY, (int(x1), 0), (int(x2), self.world.screen_HEIGHT), 2)
            pygame.draw.circle(screen, WHITE, (int(screen_x), self.world.screen_HEIGHT // 2), 5)

    # ---- Photon ----
    class Photon:
        def __init__(self, world, emission_position, direction_of_motion):
            self.position = emission_position
            self.world = world
            self.velocity = direction_of_motion * world.speedoflight

        def update(self, dt, meters_per_pixel, reference_frame_velocity):
            adjusted_velocity = self.velocity
            self.position = (self.position + (adjusted_velocity * dt) / meters_per_pixel) % self.world.screen_WIDTH

        def draw(self, screen):
            adjusted_velocity = self.velocity
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
            # gamma of particle's own rest (if needed later)
            v = self.velocity
            self.gamma = 1.0 / np.sqrt(1 - (v / self.speedoflight) ** 2)
            # Update in lab frame (no Lorentz mixing here)
            dt = (dt / self.gamma)

            delta_px = (self.velocity * dt) / meters_per_pixel
            self.position = (self.position + delta_px) % self.world.screen_WIDTH
            # If period is proper period (in emitter rest frame) then lab interval is gamma * period
            lab_interval = self.gamma * self.emission_interval
            if current_time - self.last_emission_time >= lab_interval:
                # emit photons at emitter's current lab-frame position
                screen_x = int(self.world.to_screen_x(self.position, self.world.time))
                self.world.photons.append(World.Photon(self.world, screen_x, 1))
                self.world.photons.append(World.Photon(self.world, screen_x, -1))
                self.last_emission_time = current_time

        def draw(self, screen):
            screen_x = self.world.to_screen_x(self.position, self.world.time)
            V = self.world.reference_frame_velocity
            c = self.speedoflight
            u = self.velocity
            denom = 1.0 - (u * V) / (c ** 2)
            if abs(denom) < 1e-12:
                u_prime = np.sign(u - V) * (c * 0.999999999)
            else:
                u_prime = (u - V) / denom

            if abs(u_prime) < 1e-9:
                x1 = x2 = screen_x
            else:
                if abs(u_prime) >= c:
                    u_prime = np.sign(u_prime) * (c * 0.999999999)
                slope = -c / u_prime
                delta_y = self.world.screen_HEIGHT
                delta_x = delta_y / slope
                x1 = screen_x - delta_x / 2
                x2 = screen_x + delta_x / 2

            pygame.draw.line(screen, LTBLUE, (int(x1), 0), (int(x2), self.world.screen_HEIGHT), 2)
            pygame.draw.circle(screen, BLUE, (int(screen_x), self.world.screen_HEIGHT // 2), 5)

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
            # compute gamma for emitter relative to lab
            v = self.velocity
            self.gamma = 1.0 / np.sqrt(1 - (v / self.speedoflight) ** 2)
            delta_px = (self.velocity * dt/self.gamma) / meters_per_pixel
            self.position = (self.position + delta_px) % self.world.screen_WIDTH
            # detection: compare pixel distance in lab frame (or use transformed positions; here we keep it simple)
            photons_to_remove = []
            for photon in self.world.photons:
                # use lab-frame positions for detection radius (consistent)
                if abs(self.world.to_screen_x(self.position, self.world.time) - photon.position) < self.detection_radius:
                    self.activated_until = current_time + self.activation_time
                    photons_to_remove.append(photon)
            for p in photons_to_remove:
                try:
                    self.world.photons.remove(p)
                except ValueError:
                    pass

        def draw(self, screen, current_time):
            screen_x = self.world.to_screen_x(self.position, self.world.time)
            V = self.world.reference_frame_velocity
            c = self.speedoflight
            u = self.velocity
            denom = 1.0 - (u * V) / (c ** 2)
            if abs(denom) < 1e-12:
                u_prime = np.sign(u - V) * (c * 0.999999999)
            else:
                u_prime = (u - V) / denom

            if abs(u_prime) < 1e-9:
                x1 = x2 = screen_x
            else:
                if abs(u_prime) >= c:
                    u_prime = np.sign(u_prime) * (c * 0.999999999)
                slope = -c / u_prime
                delta_y = self.world.screen_HEIGHT
                delta_x = delta_y / slope
                x1 = screen_x - delta_x / 2
                x2 = screen_x + delta_x / 2

            pygame.draw.line(screen, LTGREEN, (int(x1), 0), (int(x2), self.world.screen_HEIGHT), 2)
            color = GREEN if current_time < self.activated_until else WHITE
            pygame.draw.circle(screen, color, (int(screen_x), self.world.screen_HEIGHT // 2), 5)


world = World(scaling_factor=10)
# particle = world.Particle(world, setup_position=100, velocity=0.5*world.speedoflight)
# photon = world.Photon(world, emission_position=200, direction_of_motion=1)

sensor_arrangement_speed = 0.*world.speedoflight
sensor1 = world.PhotonSensor(world, setup_position=580, velocity=sensor_arrangement_speed)
sensor2 = world.PhotonSensor(world, setup_position=420, velocity=sensor_arrangement_speed)
emitter = world.Photon_emiter(world, setup_position=500, velocity=sensor_arrangement_speed, period=5)

gate_right = world.Particle(world, setup_position=580, velocity=0)
gate_left = world.Particle(world, setup_position=420, velocity=0)

# world.particles.append(particle)
# world.photons.append(photon)
world.particles.append(gate_right)
world.particles.append(gate_left)
world.photon_emiters.append(emitter)
world.photon_sensors.append(sensor1)
world.photon_sensors.append(sensor2)
world.reference_frame_velocity = 0.*world.speedoflight
world.run()
