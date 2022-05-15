import pygame

stack = [1]  # accelerate


def update_joystick() -> int:
    events = pygame.event.get()

    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                stack.append(0)
            if event.key == pygame.K_RIGHT:
                stack.append(2)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                stack.remove(0)
            if event.key == pygame.K_RIGHT:
                stack.remove(2)

    return stack[-1]
