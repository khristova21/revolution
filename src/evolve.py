"""
evolve.py - Responsible for evolving a set of trained robots. 
Author: Krisi Hristova
Date: May 2025
"""
from utils.morphology import *
import copy

""" 
Mutates a given robot by creating a copy of the robot and augmenting features with 
randomness of a specific intensity and/or towards a certain target robot. 
Mutations include changing the number of links and joints, their positions 

:param robot: The robot to mutate
:param intensity: The amount of randomness to put into mutation
:param target: The target robot that mutations of the robot to mutate should be moved towards
:returns: the mutated robot
"""
def mutateRobot(robot: Robot, randomness: float = 0.2, target: Robot = None) -> Robot:

    # Create a copy of the robot that will be mutated. 
    mutatedRobot = copy.deepcopy(robot)

    # Changing positioning of a limb. 
    # 1. Something like either remove a limb that is connected to the torso and 
    #    add it to a different position. 
    # 2. Add another joint to one or many of the legs of the robot. 

    if target is None:
        # Apply only random changes
        pass
    else:
        # Apply randomness AND mutating towards the target robot. 
        pass

    return mutatedRobot

# Add either a leg to a torso or a torso to a torso. 
def addLimb(robot: Robot, randomness: float = 0.1) -> Robot:    
    # Create a copy of the robot that will be mutated. 
    mutatedRobot = copy.deepcopy(robot)

    # Add a limb link to one of the existing links with a new joint.
    # Get torso link. 
    # Create a new link.
    # Create a new joint.
    # Add to the robot torso as a child.
    return mutatedRobot

# Remove a limb from the torso. 
def removeLimb(robot: Robot, randomness: float = 0.1) -> Robot:
    # Create a copy of the robot that will be mutated. 
    mutatedRobot = copy.deepcopy(robot)

    # Get torso link
    # 1. Remove a link and all of its children for a link which has torso as its parent. (including connecting joints)
    # (Note): Removing a torso from a torso for something like a worm. 
    return mutatedRobot

### TODO - Evolve a set of Robots based on a given set and some data
# Input: List of Tuple of robot object and float. where the float is the score from training. 
# Output: a list of evolved robot objects. 
# Create max number of robots we can create.
def evolve(robots: list[tuple[Robot, float]]) -> list[Robot]:
    # Duplicate the best one three times, each that changes a random variable.
    # second best we make 2 duplicates of and we do random and change towards the best one.
    # worst performed: least amount of randomness, changes trying to imitate the best performing one. 

    # Sort robots into best, medium, worst performing. 
    # Sort robots based on largest to smallest score. 
    print("Robots before sorting.")
    print(robots)   # DEBUGGING
    robots.sort(key=lambda x: x[1])
    print("Robots after sorting.")
    print(robots)   # DEBUGGING

    n = len(robots)
    # Best: 20% of the robots
    # Worst: 80% of the robots
    bestLength = n * 0.2
    worstLength = n - bestLength

    bestRobots = robots[:bestLength]
    worstRobots = robots[-worstLength:]

    # Keep a list of evolved robots to send back
    evolvedRobots = []

    for robot, _ in bestRobots:
        # Add a limb
        # Remove a limb
        # Change position of limb
        evolvedRobots.append(addLimb(robot, randomness=0.2))
        evolvedRobots.append(removeLimb(robot, randomness=0.2))
        evolvedRobots.append(mutateRobot(robot, randomness=0.2))
    
    for robot, _ in worstRobots:
        evolvedRobots.append(mutateRobot(robot, randomness=0.05, target=bestRobots[0][0]))

    return evolvedRobots

if __name__ == "__main__":
    print("Testing main")

