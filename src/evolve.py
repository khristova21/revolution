"""
evolve.py - Responsible for evolving a set of trained robots. 
Author: Krisi Hristova
Date: May 2025
"""
from utils.morphology import *
import copy
import math
import uuid

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

    # Find torso link. 
    links = mutatedRobot.getLinks()
    torsoLink = mutatedRobot.getTorso()
    if torsoLink is None:
        raise ValueError("Robot has no torso link")
    
    limbId = uuid.uuid4()
    limbName = "link_" + str(limbId) + "_limb"

    # TODO Find a location/origin on the torso so the new link won't collide with another. 
    # (is not at the same location as another existing link that is connected to the torso).
    # Make sure it doesn't collide with another limb
    # xyz will be used to set the xyz of Origin objects
    xyz = (0.03, 0.0, 0.0)

    # Visual needs the following: Origin , Geometr, Material - doesn't matter
    limbVisuals = Visual(
        origin = Origin(xyz, rpy = [0, math.pi/2, 0]),
        geometry = Capsule(radius = 0.02, length = 0.06),
        material = Material("red", [1.0, 0, 0, 0.9])
    )
    # Collision should be the same as Visual (without Material)
    limbCollisions = Collision(
        origin = Origin(xyz, rpy = [0, math.pi/2, 0]),
        geometry = Capsule(radius = 0.02, length = 0.06),
    )
    limbInertial = Inertial(
        mass = 0.1,
        inertia = (2.0e-4, 0, 0, 2.0e-4, 0, 1.0e-4),
        origin = Origin(xyz, (0, 0, 0))
    )
    # Create a new limb link.
    limbLink = Link(
        name = limbName, 
        inertial = limbInertial, 
        visuals = [limbVisuals], 
        collisions = [limbCollisions])

    # Create intermediate link. Needs only inertial element. 
    intermediateId = uuid.uuid4()
    intermediateLinkName = "intermediate_link_" + str(intermediateId) + "_limb" 
    intermediateInertial = Inertial(
        mass = 0.001,
        inertia = (1.0e-6, 0, 0, 1.0e-6, 0, 1.0e-6),
        origin = Origin((0, 0, 0), (0, 0, 0))
    )

    intermediateLink = Link(
        name = intermediateLinkName, 
        inertial = intermediateInertial)
    
    # Create a new joint (torso to intermediate)
    jointTorsoToIntermediateName = "joint_" + torsoLink.getName() + "_to_" + intermediateLink.getName()
    jointTorsoToIntermediate = Joint(
        name = jointTorsoToIntermediateName,
        type = Type.REVOLUTE,
        parent = torsoLink,          # parent
        child = intermediateLink,   # child
        origin = Origin(xyz, (0, 0, 0)),
        axis = (0, 0, 1), # Rotation around z-axis for pitch
        limit = Limit(1.0, 2.0, -1.5, 1.5)
    )
    # Create a new joint (intermediate to limb)
    jointIntermediateToLimbName = "joint_" + intermediateLink.getName() + "_to_" + limbLink.getName()
    jointIntermediateToLimb = Joint(
        name = jointIntermediateToLimbName,
        type = Type.REVOLUTE,
        parent = intermediateLink,   # parent
        child = limbLink,           # child
        origin = Origin((0, 0, 0), (0, 0, 0)),
        axis = (0, 1, 0), # Rotation around y-axis for yaw
        limit = Limit(1.0, 2.0, -1.5, 1.5)
    )
    
    # Add to the robot torso as a child.
    mutatedRobot.addLink(limbLink)
    mutatedRobot.addLink(intermediateLink)
    mutatedRobot.addJoint(jointTorsoToIntermediate)
    mutatedRobot.addJoint(jointIntermediateToLimb)

    return mutatedRobot

# TODO Add a limb link to one of the existing links with a new joint.
# Make a joint on a limb. 
def mutateLimb():
    pass

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

    # Sort robots into best, worst performing. 
    print("Robots before sorting.")
    print(robots)   # DEBUGGING
    robots.sort(key=lambda x: x[1])
    print("Robots after sorting.")
    print(robots)   # DEBUGGING

    n = len(robots)
    # Best: 20% of the robots,  Worst: 80% of the robots
    if n < 1:
        print("No robots provided")
        return 
    
    bestLength = max(1, int(n * 0.2))
    print("best length: " + str(bestLength))
    worstLength = n - bestLength

    bestRobots = robots[:bestLength]
    worstRobots = robots[-worstLength:] if worstLength > 0 else []

    evolvedRobots = []

    for robot, _ in bestRobots:
        evolvedRobots.append(addLimb(robot, randomness=0.2))
        #evolvedRobots.append(removeLimb(robot, randomness=0.2))
        #evolvedRobots.append(mutateRobot(robot, randomness=0.2))           # Change position of limb

    for robot, _ in worstRobots:
        evolvedRobots.append(mutateRobot(robot, randomness=0.05, target=bestRobots[0][0]))

    return evolvedRobots

if __name__ == "__main__":
    print("Testing main")
    # Generate Robot from test.urdf
    initialRobot = convertUrdfToRobot("test.urdf")
    print("Running evolve")
    mutatedRobots = evolve([(initialRobot, 1)])
    print("Converting evolved robots into to URDF")
    for robot in mutatedRobots:
        convertRobotToUrdf(robot)

