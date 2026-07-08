import math


def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def target_angle(current_x, current_y, target_x, target_y):
    return math.atan2(target_y - current_y, target_x - current_x)


def bearing_to_target(current_x, current_y, target_x, target_y):
    return target_angle(current_x, current_y, target_x, target_y)


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def distance_to_target(current_x, current_y, target_x, target_y):
    return euclidean_distance(current_x, current_y, target_x, target_y)


def relative_target(current_x, current_y, current_z, target_x, target_y, target_z):
    return (
        target_x - current_x,
        target_y - current_y,
        target_z - current_z,
    )


def relative_vector(current_x, current_y, current_z, target_x, target_y, target_z):
    return relative_target(current_x, current_y, current_z, target_x, target_y, target_z)
