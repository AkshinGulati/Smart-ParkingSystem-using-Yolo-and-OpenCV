# preprocessing.py

from car_parking_coordinate_data import car_park_coordinate

def get_parking_polygons(option):
    """
    Returns parking slot polygons based on selected layout.
    """
    return car_park_coordinate(option)