import os
import pygame
import sys

# Run in headless dummy video mode
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

from game import Game

def capture():
    g = Game()
    # Run for a few frames to let entities render
    for _ in range(10):
        g.clock.tick(60)
        g._handle_events()
        g._update(1.0/60.0)
        g._draw()
    
    # Save the screen surface to a PNG file
    os.makedirs("results", exist_ok=True)
    pygame.image.save(g.screen, "results/game_screenshot.png")
    print("Screenshot saved to results/game_screenshot.png")
    pygame.quit()

if __name__ == "__main__":
    capture()
