"""
evolve.py - Responsible for evolving a set of trained robots. 
Author: Krisi Hristova
Date: May 2025
"""
from utils.morphology import *
import copy
import math
import uuid
import random

""" 
Mutates a given robot by creating a copy of the robot and augmenting features with 
randomness of a specific intensity and/or towards a certain target robot. 
Mutations include changing the number of links and joints, their positions 

@param robot: The robot to mutate
@param intensity: The amount of randomness to put into mutation
@param target: The target robot that mutations of the robot to mutate should be moved towards
@return the mutated robot
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

# Helper for rounding in addLimb()
def rounded(pos, decimals=3):
    return tuple(round(x, decimals) for x in pos)

"""
Add a leg to one of 4 corners of a torso. 
If all corners and center are occupied, returns the original robot. 
@param robot The robot to add a limb to.
@returns a mutated robot that has a new limb
"""
def addLimb(robot: Robot, randomness: float = 0.1) -> Robot:    
    # Create a copy of the robot that will be mutated. 
    mutatedRobot = copy.deepcopy(robot)
    # Find torso link. 
    torsoLink = mutatedRobot.getTorso()
    if torsoLink is None:
        raise ValueError("Robot has no torso link.")
    
    # Find locations that are already occupied.
    occupiedPositions = set()
    for joint in mutatedRobot.getJoints():
        if joint.getParent() == torsoLink:
            origin = joint.getOrigin()
            if origin:
                xyz = rounded(origin.getXyz())
                occupiedPositions.add(xyz)
    
    # Find a location/origin on the torso so the new link won't collide with another.     
    torsoGeometry = torsoLink.getVisuals()[0].getGeometry()
    torsoLength = torsoGeometry.getAttributes()['length']
    torsoRadius = torsoGeometry.getAttributes()['radius']
    halfLength = torsoLength / 2
    halfWidth = torsoRadius
    z = -torsoRadius + 0.005

    offset = torsoRadius * 0.3
    print(offset)
    cornerCandidates = [
        (offset*2, offset, z), #front right (looking from the horizontal side)
        (-offset*2, offset, z), # front left
        (offset*2, -offset, z), # back right
        (-offset*2, -offset, z) # back left
    ]

    xyz = None
    for position in cornerCandidates:
        if rounded(position) not in occupiedPositions:
            xyz = position
            print("Placing limb at: ", xyz)
            break
    if xyz is None:
        print("No free torso corners.")
        if rounded(position) not in occupiedPositions:
            xyz = (0, 0, z)
            print("Placing limb at torso center bottom")
        else:
            print("No modifications were made. Returning copy of original robot.")
            return mutatedRobot
        
    limbId = uuid.uuid4()
    limbName = "link_" + str(limbId) + "_limb"

    # xyz will be used to set the xyz of Origin objects
    limbLength = torsoRadius * 2
    limbRadius = torsoRadius / 3

    # Visual needs the following: Origin , Geometry, Material - doesn't matter
    limbVisuals = Visual(
        origin = Origin(xyz=xyz, rpy = (0, 0, 0)), #aligned down
        geometry = Capsule(radius = limbRadius, length = limbLength),
        material = Material("red", [1.0, 0, 0, 0.9])
    )
    # Collision should be the same as Visual (without Material)
    limbCollisions = Collision(
        origin = Origin(xyz=xyz, rpy = (0, 0, 0)), #aligned down
        geometry = Capsule(radius = limbRadius, length = limbLength)
    )
    # Create a new limb link.
    limbLink = Link(
        name = limbName, 
        visuals = [limbVisuals], 
        collisions = [limbCollisions])

    # Create intermediate link. Needs only inertial element. 
    intermediateId = uuid.uuid4()
    intermediateLinkName = "link_intermediate_" + str(intermediateId) + "_limb" 

    intermediateLink = Link(
        name = intermediateLinkName,
    )
    
    # Create a new joint (torso to intermediate)
    jointTorsoToIntermediate = Joint(
        name =  "joint_" + torsoLink.getName() + "_to_" + intermediateLink.getName(),
        type = Type.REVOLUTE,
        parent = torsoLink,
        child = intermediateLink,
        origin = Origin(xyz, (0, 0, 0)),
        axis = (0, 0, 1), # Rotation around z-axis for pitch
        limit = Limit(1.0, 2.0, -1.4, 1.4)
    )
    
    # Create a new joint (intermediate to limb)
    jointIntermediateToLimb = Joint(
        name =  "joint_" + intermediateLink.getName() + "_to_" + limbLink.getName(),
        type = Type.REVOLUTE,
        parent = intermediateLink,
        child = limbLink,
        origin = Origin((0, 0, 0), (0, 0, 0)),
        axis = (0, 1, 0), # Rotation around y-axis for yaw
        limit = Limit(1.0, 2.0, -1.4, 1.4)
    )
    print(jointTorsoToIntermediate.getParent().getName() )
    print(jointIntermediateToLimb.getParent().getName() )

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

"""
Evolve a set of robots based on their scores from training. 
@param robots List of Tuples with robot object and float where the float is the score from training. 
@return a list of evolved robots. 
"""
def evolve(robots: list[tuple[Robot, float]]) -> list[Robot]:
    # Duplicate the best one three times, each that changes a random variable.
    # second best we make 2 duplicates of and we do random and change towards the best one.
    # worst performed: least amount of randomness, changes trying to imitate the best performing one. 

    # Sort robots into best, worst performing. 
    robots.sort(key=lambda x: x[1])

    n = len(robots)
    # Best: 20% of the robots,  Worst: 80% of the robots
    if n < 1:
        print("No robots provided")
        return 
    # Max robots that we are allowed to create. 
    maxRobots = n**2
    
    bestLength = max(1, int(n * 0.2))
    print("best length: " + str(bestLength))
    worstLength = n - bestLength

    bestRobots = robots[:bestLength]
    worstRobots = robots[-worstLength:] if worstLength > 0 else []

    evolvedRobots = []

    for robot, _ in bestRobots:
        evolvedRobots.append(addLimb(robot, randomness=0.2))
        #evolvedRobots.append(removeLimb(robot, randomness=0.2))
        #evolvedRobots.append(mutateRobot(robot, randomness=0.2))   # Change position of limb
    for robot, _ in worstRobots:
        evolvedRobots.append(mutateRobot(robot, randomness=0.05, target=bestRobots[0][0]))

    return evolvedRobots

if __name__ == "__main__":
    # Generate Robot from test.urdf
    initialRobot = convertUrdfToRobot("simple.urdf")
    #initialRobot = convertUrdfToRobot("simple_cyl.urdf")
    #initialRobot = convertUrdfToRobot("stompy-5.urdf")

    print("Running evolve")
    mutatedRobots = evolve([(initialRobot, 1)])
    print("Converting evolved robots into to URDF")
    for robot in mutatedRobots:
        convertRobotToUrdf(robot)

